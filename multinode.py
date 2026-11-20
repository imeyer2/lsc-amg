"""Multi-node ES population evaluation (exact; `multinode` speedup).

The main process (train_es_HPC.py on node 0) keeps its local worker pool and
additionally serves a task queue over TCP (multiprocessing.managers) to one
node agent per extra node. Every agent loads the same train+test samples
(pickled once by the main process, so the data are bit-identical), runs its
own worker pool with the unchanged worker functions (_worker_init,
_worker_eval_one) against its own copy of the per-epoch shared block
(eps / theta / sigma / stag row), pulls (member, sample) tasks and pushes
(member, sample, reward, iters) results. The main process forms the same
per-member means from the same per-sample values in the same order as the
single-node code, so the training history is identical to a single-node run
on the same CPU type. Only the population (and, with `overlap`, the
speculative next population) is distributed; centre-first `stag` scheduling,
eval/test, SoC and diagnostics stay on node 0.

Protocol (all calls go through one BaseManager served by a thread of the
main process):
  hello(node, n_workers)        -> ack
  pull(n_free)                  -> list of (big_v, small_v, j, sid) or "DONE"
  block(big_v)                  -> (pop_size, n_params, eps|theta|sigma bytes)
  small(small_v)                -> (n_rows, bytes of the stag row prefix)
  push(list of (big_v, j, sid, r, it)) -> None

Results carry the block version of their task; collect() ignores results of
an older version, so a late duplicate of a re-queued task from a previous
epoch can never be taken for the same (member, sample) of the current one.
"""
from __future__ import annotations

import collections
import json
import os
import pickle
import queue
import sys
import threading
import time

import numpy as np
from multiprocessing.managers import BaseManager


class _State:
    """Shared state behind the manager's registered callables."""

    def __init__(self):
        self.lock = threading.Lock()
        self.tasks = collections.deque()
        self.results = queue.Queue()
        self.big_v = 0
        self.big_data = None
        self.small_v = 0
        self.small_data = None
        self.done = False
        self.nodes = {}
        self.pulled_remote = 0
        self.outstanding = {}      # node -> set of (j, sid) pulled, not yet pushed
        self.last_result = time.time()

    # ---- remote side ----
    def hello(self, node, n_workers):
        with self.lock:
            self.nodes[node] = n_workers
        print(f"  [multinode] agent {node} connected with {n_workers} workers", flush=True)
        return "ok"

    def pull(self, n_free, node=""):
        out = []
        with self.lock:
            if self.done and not self.tasks:
                return "DONE"
            while self.tasks and len(out) < n_free:
                out.append(self.tasks.popleft())
            self.pulled_remote += len(out)
            self.outstanding.setdefault(node, set()).update((t[2], t[3]) for t in out)
        return out

    def block(self, big_v):
        with self.lock:
            if big_v != self.big_v:
                return None
            return self.big_data

    def small(self, small_v):
        with self.lock:
            if small_v != self.small_v:
                return None
            return self.small_data

    def push(self, results, node=""):
        with self.lock:
            out = self.outstanding.get(node)
            if out is not None:
                out.difference_update((r[1], r[2]) for r in results)
            self.last_result = time.time()
        for r in results:
            self.results.put(tuple(r))
        return None


class _Manager(BaseManager):
    pass


class Dispatcher:
    """Main-process side: local pool + remote agents share one task deque."""

    def __init__(self, pool, worker_fn, shm_name, pop_size, n_params, n_local,
                 port, authkey, inflight_extra=4):
        self.pool = pool
        self.worker_fn = worker_fn
        self.shm_name = shm_name
        self.pop_size = pop_size
        self.n_params = n_params
        self.st = _State()
        self.sem = threading.Semaphore(max(1, n_local) + inflight_extra)
        self.local_done = 0
        self._stop = False
        st = self.st
        # One shared object; its methods are called through a proxy, so the
        # return values come back as plain pickled values.
        _Manager.register("state", callable=lambda: st,
                          exposed=("hello", "pull", "block", "small", "push"))
        self.manager = _Manager(address=("", port), authkey=authkey.encode())
        self.server = self.manager.get_server()
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        self.feeder = threading.Thread(target=self._feed_local, daemon=True)
        self.feeder.start()
        print(f"  [multinode] task server on port {port}", flush=True)

    # ---- block management (called by the training loop) ----
    def set_block(self, es_rows, n_small):
        """Publish eps/theta/sigma (es_rows[:pop+2]) as a new big version."""
        with self.st.lock:
            self.st.big_v += 1
            self.st.big_data = (self.pop_size, self.n_params,
                                np.ascontiguousarray(es_rows[:self.pop_size + 2]).tobytes())
            self.st.small_v += 1
            self.st.small_data = (n_small, np.ascontiguousarray(es_rows[self.pop_size + 2][:n_small]).tobytes())

    def set_small(self, es_rows, n_small):
        with self.st.lock:
            self.st.small_v += 1
            self.st.small_data = (n_small, np.ascontiguousarray(es_rows[self.pop_size + 2][:n_small]).tobytes())

    def submit(self, pairs):
        """Enqueue (j, sid) tasks under the current versions; returns the
        expected key set for collect()."""
        with self.st.lock:
            bv, sv = self.st.big_v, self.st.small_v
            for j, sid in pairs:
                self.st.tasks.append((bv, sv, j, sid))
        return set(pairs)

    def collect(self, expected, got=None):
        """Drain results until every expected (j, sid) has one. If no result
        arrives for LSC_MN_STALL seconds while tasks are outstanding on a
        remote agent (agent died), those tasks are re-queued and evaluated by
        whoever is alive; duplicate results are ignored, so this is exact."""
        got = {} if got is None else got
        n = 0
        stall = float(os.environ.get("LSC_MN_STALL", "600"))
        with self.st.lock:
            cur_v = self.st.big_v
        while expected:
            try:
                bv, j, sid, r, it = self.st.results.get(timeout=5.0)
            except queue.Empty:
                with self.st.lock:
                    stalled = time.time() - self.st.last_result > stall
                    lost = [(nd, set(k for k in ks if k in expected))
                            for nd, ks in self.st.outstanding.items()]
                    lost = [(nd, ks) for nd, ks in lost if ks]
                    if stalled and lost:
                        bv, sv = self.st.big_v, self.st.small_v
                        for nd, ks in lost:
                            print(f"  [multinode] WARNING: {len(ks)} tasks outstanding on {nd} "
                                  f"for > {stall:.0f}s; re-queueing them", flush=True)
                            for (jj, ss) in ks:
                                self.st.tasks.appendleft((bv, sv, jj, ss))
                            self.st.outstanding[nd] = set()
                        self.st.last_result = time.time()
                continue
            key = (j, sid)
            if bv == cur_v and key in expected:
                expected.discard(key)
                got[key] = (r, it)
                n += 1
            # else: duplicate or stale (older block version) -- ignored
        return got

    # ---- local feeder ----
    def _feed_local(self):
        st = self.st
        while not self._stop:
            task = None
            with st.lock:
                if st.tasks:
                    task = st.tasks.popleft()
            if task is None:
                time.sleep(0.002)
                continue
            self.sem.acquire()
            bv, _sv, j, sid = task
            self.pool.apply_async(self.worker_fn,
                                  ((self.shm_name, self.pop_size, self.n_params, j, sid),),
                                  callback=lambda res, bv=bv: self._cb(bv, res),
                                  error_callback=self._err)

    def _cb(self, bv, res):
        self.sem.release()
        self.local_done += 1
        j, sid, r, it = res
        self.st.last_result = time.time()
        self.st.results.put((bv, j, sid, r, it))

    def _err(self, exc):
        self.sem.release()
        print(f"  [multinode] local task failed: {exc!r}", flush=True)

    def wait_for_nodes(self, n, timeout=1800.0):
        """Block until n agents have said hello (or timeout), so that a run
        does not start with the remote pools still initialising."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self.st.lock:
                k = len(self.st.nodes)
            if k >= n:
                print(f"  [multinode] {k} agent(s) connected after {time.time()-t0:.0f}s", flush=True)
                return True
            time.sleep(1.0)
        print(f"  [multinode] WARNING: only {k} of {n} agents connected after {timeout:.0f}s; continuing", flush=True)
        return False

    def stats(self):
        return {"remote_tasks": self.st.pulled_remote, "local_tasks": self.local_done,
                "nodes": dict(self.st.nodes)}

    def close(self):
        with self.st.lock:
            self.st.done = True
        self._stop = True
        time.sleep(1.0)
        try:
            self.server.stop_event.set()
        except Exception:
            pass


# ----------------------------------------------------------------------------
# Node agent
# ----------------------------------------------------------------------------
def agent_main():
    import argparse
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--authkey", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_workers", type=int, required=True)
    args = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    node = os.uname().nodename
    cfg_path = os.path.join(args.out_dir, "node_config.json")
    pkl_path = os.path.join(args.out_dir, "node_samples.pkl")
    t0 = time.time()
    while not (os.path.exists(cfg_path) and os.path.exists(pkl_path + ".ready")):
        if time.time() - t0 > 3 * 3600:
            print(f"[agent {node}] gave up waiting for {pkl_path}", flush=True)
            sys.exit(1)
        time.sleep(5)
    cfg = json.load(open(cfg_path))
    os.environ["LSC_SPEEDUPS"] = cfg["speedups"]
    for k, v in cfg.get("env", {}).items():
        os.environ[k] = v
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import torch
    torch.set_num_threads(1)
    import train_es_HPC as T
    from shared_cache import create_shared_cache, load_from_shared_cache, cleanup_shared_cache
    from multiprocessing.shared_memory import SharedMemory
    from amg_core import FeatureConfig
    t1 = time.time()
    with open(pkl_path, "rb") as f:
        samples = pickle.load(f)
    print(f"[agent {node}] loaded {len(samples)} samples in {time.time()-t1:.1f}s", flush=True)
    shm_meta, shm_block = create_shared_cache(samples)
    del samples
    fast_cfg = None
    if cfg["fast_policy"] == "all":
        fast_cfg = FeatureConfig.from_preset(cfg["feature_preset"])
        if cfg.get("num_relaxed_probes") is not None:
            fast_cfg.num_relaxed_probes = cfg["num_relaxed_probes"]
    model_state = torch.load(os.path.join(args.out_dir, "node_model_init.pt"), map_location="cpu", weights_only=False)
    pop_size, n_params = cfg["pop_size"], cfg["n_params"]
    es_shm = SharedMemory(create=True, size=(pop_size + T._ES_ROWS_EXTRA) * n_params * 4)
    es_rows = np.ndarray((pop_size + T._ES_ROWS_EXTRA, n_params), dtype=np.float32, buffer=es_shm.buf)
    es_rows[:] = 0.0
    t1 = time.time()
    pool = mp.Pool(processes=args.n_workers, initializer=T._worker_init,
                   initargs=(model_state, cfg["model_kwargs"], shm_meta, cfg["k_per_row"],
                             cfg["selection_mode"], cfg["reward_lambda_cx"], fast_cfg))
    print(f"[agent {node}] pool of {args.n_workers} ready in {time.time()-t1:.1f}s", flush=True)

    _Manager.register("state")
    mgr = _Manager(address=(args.host, args.port), authkey=args.authkey.encode())
    for attempt in range(600):
        try:
            mgr.connect()
            break
        except Exception as e:
            time.sleep(2)
    else:
        print(f"[agent {node}] could not connect to {args.host}:{args.port}", flush=True)
        sys.exit(1)
    S = mgr.state()
    S.hello(node, args.n_workers)

    big_v = 0
    small_v = 0
    inflight = [0]
    lock = threading.Lock()
    results = []
    n_done = [0]

    def cb(bv, res):
        with lock:
            inflight[0] -= 1
            results.append((bv,) + tuple(res))
            n_done[0] += 1

    def err(exc):
        with lock:
            inflight[0] -= 1
        print(f"[agent {node}] task failed: {exc!r}", flush=True)

    def flush_results(force=False):
        with lock:
            if results and (force or len(results) >= 4):
                batch = list(results)
                results.clear()
            else:
                batch = None
        if batch:
            S.push(batch, node)

    last_flush = time.time()
    t_start = time.time()
    while True:
        with lock:
            free = args.n_workers + 2 - inflight[0]
        if free <= 0:
            flush_results()
            time.sleep(0.005)
            continue
        tasks = S.pull(free, node)
        if tasks == "DONE":
            break
        if not tasks:
            flush_results(force=(time.time() - last_flush > 0.2))
            if time.time() - last_flush > 0.2:
                last_flush = time.time()
            time.sleep(0.01)
            continue
        for bv, sv, j, sid in tasks:
            if bv != big_v or sv != small_v:
                # Wait until nothing of the previous version is in flight,
                # then refresh the block(s).
                while True:
                    with lock:
                        if inflight[0] == 0:
                            break
                    time.sleep(0.002)
                flush_results(force=True)
                if bv != big_v:
                    blk = S.block(bv)
                    if blk is None:
                        raise RuntimeError(f"block version {bv} no longer available")
                    p, npar, raw = blk
                    es_rows[:pop_size + 2] = np.frombuffer(raw, dtype=np.float32).reshape(pop_size + 2, n_params)
                    big_v = bv
                if sv != small_v:
                    sm = S.small(sv)
                    if sm is None:
                        raise RuntimeError(f"small version {sv} no longer available")
                    n_small, raw = sm
                    es_rows[pop_size + 2] = 0.0
                    es_rows[pop_size + 2][:n_small] = np.frombuffer(raw, dtype=np.float32)
                    small_v = sv
            with lock:
                inflight[0] += 1
            pool.apply_async(T._worker_eval_one, ((es_shm.name, pop_size, n_params, j, sid),),
                             callback=lambda res, bv=bv: cb(bv, res), error_callback=err)
        flush_results()
    while True:
        with lock:
            if inflight[0] == 0:
                break
        time.sleep(0.01)
    flush_results(force=True)
    print(f"[agent {node}] done: {n_done[0]} tasks in {time.time()-t_start:.0f}s", flush=True)
    pool.close(); pool.join()
    es_shm.close(); es_shm.unlink()
    cleanup_shared_cache(shm_block)


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("forkserver", force=True)
    mp.set_forkserver_preload(["torch", "torch_geometric", "scipy.sparse", "scipy.sparse.linalg",
                               "pyamg", "amg_core", "fast_policy", "shared_cache"])
    agent_main()
