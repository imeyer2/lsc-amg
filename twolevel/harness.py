"""Two-level timing harness (sub-study "bar", with the "coarse" and "reward" methods merged in).

Protocol: both methods two-level -- pyamg SA with strength=("predefined", {"C": C}), smooth='jacobi',
default SA smoothers, coarse solver splu (LU factorisation forced and timed inside setup), PCG (scipy
cg, b = ones, x0 = 0, rtol 1e-8, maxiter 500) preconditioned by one V-cycle. Every worker process is
pinned to ONE core and all BLAS/OMP/torch pools are 1 thread.

  learned total   = features + forward (+ selection) + C assembly + SA setup + coarse LU + PCG
  classical total = C (rule + symmetrisation) + SA setup + coarse LU + PCG

Methods (strings):
  theta:<t> | sym:<t> | topk:<k> | srowq:<q>   classical rules (twolevel.rules)
  lsc:<model_best.pt>@cpu|@cuda      paper GAT policy via the exact FastPolicy path (features on 1 CPU
                                     thread; forward on the CPU or the GPU)
  lscref:<model_best.pt>@<dev>       paper reference path (build_sample_features + AMGEdgePolicy.forward)
  lin:<json>                         learned per-row threshold model (twolevel.rowthr, train_lin.py output)
                                     or, for a train_node.py train.json, the node policy (same as node:)
  node:<train.json>                  learned node-modulated policy (twolevel.nodepol)
  two:<th_hard>,<th_bulk>,<jump_t>,<posfrac_t>   two-class hand rule (twolevel.rowthr.two_C)
  mask:<dir>/<problem>               precomputed oracle mask <dir>/<problem>_<feature_seed>.npy
Suffix options after '|', comma separated:
  perm=<permc_spec>,dpt=<x>,sym=1    splu options; 'perm=MMD_AT_PLUS_A,dpt=0,sym=1' = "improved splu"
  thr=<z>          (lsc) override the policy's learned z-score threshold
  gprobe=1         (lsc@cuda) relaxed probes computed on the GPU

Usage: python -m twolevel.harness run --problem anisotropy --n 50000 --seed 1000 --count 16 \
           --methods theta:0.1 sym:0.02 --workers 2 --reps 3 --out x.json
       python -m twolevel.harness gen --problem anisotropy --n 500000 --seed 1000 --count 8
       python -m twolevel.harness summary x.json [y.json ...]
       python -m twolevel.harness fingerprint --problem ... --methods ... --out f.json   (no timing:
           sha1 of C, iterations, coarse size; used by tests/test_twolevel.py)
"""
import os
import time
import json
import socket
import argparse
import hashlib

from . import common
common.single_thread()
import numpy as np  # noqa: E402

from .common import gen_set, load_A, cpu_model  # noqa: E402
from .rules import classical_keep, C_from_keep, offdiag  # noqa: E402


# ----------------------------------------------------------------------------- learned (paper GAT)
_MODELS = {}


def get_fast(path, dev):
    key = (path, dev)
    if key not in _MODELS:
        import torch
        from amg_core import FeatureConfig
        from models import AMGEdgePolicy
        from fast_policy import FastPolicy
        cfg = json.load(open(os.path.join(os.path.dirname(path), "config.json")))
        fc = FeatureConfig.from_preset(cfg.get("feature_preset", "new_algebraic_plus_relaxed"))
        m = AMGEdgePolicy(in_channels=fc.node_dim, hidden=cfg.get("hidden", 64), learn_B=cfg.get("learn_B", False),
                          edge_raw_dim=fc.edge_raw_dim, adaptive_k=(cfg.get("selection_mode", "threshold") == "threshold"))
        m.load_state_dict(torch.load(path, map_location="cpu"))
        m.eval()
        _MODELS[key] = (FastPolicy(m, device=dev), m.to(dev), fc)
    return _MODELS[key]


def _sync(dev):
    if dev.startswith("cuda"):
        import torch
        torch.cuda.synchronize()


def gpu_prepare(A, fc, fseed, dev, omega=2.0 / 3.0):
    """FastPolicy.prepare with the relaxed probes computed on the device (same random start, same Jacobi
    sweeps, float64; differs from pyamg's CPU sweeps only by rounding)."""
    import torch
    from fast_policy import Graph, Prepared
    from amg_core import compute_row_stats, build_node_features
    g = Graph(A, dev)
    K = fc.num_relaxed_probes
    X0 = np.random.default_rng(fseed).standard_normal((A.shape[0], K))
    Ad = torch.sparse_csr_tensor(g.indptr, g.indices, torch.from_numpy(g.A.data).to(dev), size=A.shape)
    dinv = torch.from_numpy(1.0 / g.A.diagonal()).to(dev)[:, None]
    X = torch.from_numpy(X0).to(dev)
    for _ in range(fc.relax_iters):
        X = X - omega * dinv * (Ad @ X)
    R = (X / (X.abs().max() + 1e-6)).cpu().numpy()
    rs = compute_row_stats(g.A)
    x = build_node_features(g.A, None, fc, row_stats=rs, R=R)
    return Prepared(g, x, R, rs)


def learned_C(A, fseed, path, dev, ref=False, thr=None, gprobe=False):
    fp, m, fc = get_fast(path, dev)
    fp.threshold = float(m.threshold.item()) if thr is None else float(thr)
    if not ref:
        t0 = time.perf_counter()
        prep = gpu_prepare(A, fc, fseed, dev) if gprobe else fp.prepare(A, fc, fseed)
        _sync(dev)
        t1 = time.perf_counter()
        sel = fp.directed_selection(prep)          # forward + z-score + threshold, back on host
        t2 = time.perf_counter()
        C = fp.strength_from_selection(prep, sel)
        t3 = time.perf_counter()
        keep = float(sel[prep.g.np_off].mean())
        return C, {"t_feat": t1 - t0, "t_fwd": t2 - t1, "t_C": t3 - t2, "keep": keep}
    import torch
    from amg_core import build_sample_features, select_edges_by_threshold, C_from_selected_edges
    from torch_geometric.utils import from_scipy_sparse_matrix
    t0 = time.perf_counter()
    ei, ew = from_scipy_sparse_matrix(A)
    ew = ew.float()
    ew = ew / ew.abs().max()
    x, ef = build_sample_features(A, None, ei, fc, seed=fseed)
    t1 = time.perf_counter()
    with torch.no_grad():
        lg, _ = m(x.to(dev), ei.to(dev), ew.to(dev), raw_edge_features=ef.to(dev))
        lg = lg.cpu()
    sel, _ = select_edges_by_threshold(logits=lg, edge_index=ei, threshold=m.threshold.item())
    t2 = time.perf_counter()
    C = C_from_selected_edges(A, ei, sel.numpy())
    t3 = time.perf_counter()
    return C, {"t_feat": t1 - t0, "t_fwd": t2 - t1, "t_C": t3 - t2,
               "keep": float(sel.numpy()[(ei[0] != ei[1]).numpy()].mean())}


# ----------------------------------------------------------------------------- small learned policies
_POL = {}


def _policy(kind, path):
    key = (kind, path)
    if key not in _POL:
        J = json.load(open(path))
        if "policy" in J:            # twolevel.train_node output
            from .nodepol import load_node
            _POL[key] = ("node", load_node(path))
        else:                        # twolevel.train_lin output
            from .rowthr import load_lin
            _POL[key] = ("lin", load_lin(path))
    return _POL[key]


# ----------------------------------------------------------------------------- pipeline
def parse_method(spec):
    base, _, opt = spec.partition("|")
    opts = dict(kv.split("=", 1) for kv in opt.split(",") if kv) if opt else {}
    kind, _, arg = base.partition(":")
    return kind, arg, opts


def build_C(A, fseed, spec):
    kind, arg, opts = parse_method(spec)
    if kind in ("lsc", "lscref"):
        path, _, dev = arg.rpartition("@")
        return learned_C(A, fseed, path, dev, ref=(kind == "lscref"), thr=opts.get("thr"),
                         gprobe=opts.get("gprobe") == "1")
    if kind in ("lin", "node"):
        what, obj = _policy(kind, arg)
        if what == "node":
            from .nodepol import node_C
            return node_C(A, *obj)
        from .rowthr import lin_C
        return lin_C(A, obj)
    if kind == "two":
        from .rowthr import two_C
        return two_C(A, *[float(v) for v in arg.split(",")])
    if kind == "mask":
        from .rowthr import mask_C
        return mask_C(A, np.load(f"{arg}_{fseed}.npy"))
    p = float(arg)
    t0 = time.perf_counter()
    keep = classical_keep(A, kind, p)
    C = C_from_keep(A, keep)
    t1 = time.perf_counter()
    row, off = offdiag(A)
    return C, {"t_feat": 0.0, "t_fwd": 0.0, "t_C": t1 - t0, "keep": float(keep[off].mean())}


def splu_opts(opts):
    """splu keyword options from the method suffix."""
    kw = {}
    if "perm" in opts:
        kw["permc_spec"] = opts["perm"]
    if "dpt" in opts:
        kw["diag_pivot_thresh"] = float(opts["dpt"])
    if opts.get("sym") == "1":
        kw["options"] = {"SymmetricMode": True}
    return kw


def setup_solve(A, C, kw=None):
    import pyamg
    from amg_core import fixed_np_rng, vcycle_preconditioner, pcg_iteration_count
    cs = ("splu", kw) if kw else "splu"
    t0 = time.perf_counter()
    with fixed_np_rng(0):
        ml = pyamg.smoothed_aggregation_solver(A, strength=("predefined", {"C": C}), smooth="jacobi",
                                               coarse_solver=cs, keep=True)
    t1 = time.perf_counter()
    Ac = ml.levels[-1].A
    ml.coarse_solver(Ac, np.zeros(Ac.shape[0]))           # force the (otherwise lazy) splu factorisation
    t2 = time.perf_counter()
    it = pcg_iteration_count(A, vcycle_preconditioner(ml), tol=1e-8, maxiter=500)
    t3 = time.perf_counter()
    return ml, {"t_sa": t1 - t0, "t_lu": t2 - t1, "t_solve": t3 - t2, "iters": int(it)}


def hier_stats(ml, A, cycle_reps=5):
    L = ml.levels
    d = {"n": int(A.shape[0]), "nnz": int(A.nnz), "n_levels": len(L), "coarse_n": int(L[-1].A.shape[0]),
         "coarse_nnz": int(L[-1].A.nnz), "oc": sum(lv.A.nnz for lv in L) / L[0].A.nnz,
         "P_nnz": int(L[0].P.nnz) if len(L) > 1 else 0}
    cs = ml.coarse_solver
    lu = getattr(cs, "LU", None)
    d["lu_nnz"] = int(lu.L.nnz + lu.U.nnz) if lu is not None else 0
    d["lu_ratio"] = d["lu_nnz"] / A.nnz
    if hasattr(L[0], "AggOp"):
        agg = np.asarray(L[0].AggOp.sum(axis=0)).ravel()
        d["agg_mean"] = float(agg.mean())
        d["agg_max"] = int(agg.max())
        d["singletons"] = int((agg == 1).sum())
    if cycle_reps:
        # per-V-cycle cost: total vs coarse direct solve
        from amg_core import vcycle_preconditioner
        M = vcycle_preconditioner(ml)
        b = np.random.default_rng(0).standard_normal(A.shape[0])
        bc = np.random.default_rng(1).standard_normal(L[-1].A.shape[0])
        tv, tc = [], []
        for _ in range(cycle_reps):
            t0 = time.perf_counter()
            M.matvec(b)
            tv.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            cs(L[-1].A, bc)
            tc.append(time.perf_counter() - t0)
        d["t_cycle"] = float(np.median(tv))
        d["t_cycle_coarse"] = float(np.median(tc))
        d["t_cycle_fine"] = d["t_cycle"] - d["t_cycle_coarse"]
    return d


def run_pipeline(A, fseed, spec):
    kind, arg, opts = parse_method(spec)
    C, r = build_C(A, fseed, spec)
    ml, s = setup_solve(A, C, splu_opts(opts))
    r.update(s)
    r["t_setup"] = r["t_sa"] + r["t_lu"]
    r["total"] = r["t_feat"] + r["t_fwd"] + r["t_C"] + r["t_setup"] + r["t_solve"]
    return r, ml


TIMED = ("t_feat", "t_fwd", "t_C", "t_sa", "t_lu", "t_setup", "t_solve", "total")
G = {}


def _init(lock, core_list, counter):
    common.single_thread()
    try:
        import torch
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    except ImportError:
        pass
    with counter.get_lock():
        k = counter.value
        counter.value += 1
    if core_list and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {core_list[k % len(core_list)]})
    G["lock"] = lock
    G["core"] = core_list[k % len(core_list)] if core_list else None


def task(t):
    path, spec, reps, warm = t
    A, fseed = load_A(path)
    gpu = "@cuda" in spec
    reps_out = []
    for r in range(reps + (1 if warm else 0)):
        if gpu and G.get("lock") is not None:
            G["lock"].acquire()
        try:
            res, ml = run_pipeline(A, fseed, spec)
        finally:
            if gpu and G.get("lock") is not None:
                G["lock"].release()
        if warm and r == 0:
            continue
        reps_out.append(res)
    med = {k: float(np.median([x[k] for x in reps_out])) for k in TIMED}
    out = dict(reps_out[0])
    out.update(med)
    out["reps_total"] = [x["total"] for x in reps_out]
    out.update(hier_stats(ml, A))
    out["path"] = os.path.basename(path)
    out["method"] = spec
    out["core"] = G.get("core")
    return out


def _cores():
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    return list(range(os.cpu_count() or 1))


def cmd_run(a):
    import multiprocessing as mp
    paths = gen_set(a.problem, a.n, a.seed, a.count)
    methods = a.methods
    done = set()
    rows = []
    if a.resume and os.path.exists(a.out):
        J = json.load(open(a.out))
        rows = [r for r in J["rows"] if r["method"] in methods]
        done = {(r["path"], r["method"]) for r in rows}
    tasks = []
    for i, p in enumerate(paths):
        for m in methods:
            if (os.path.basename(p), m) not in done:
                tasks.append((p, m, a.reps, i == 0 or a.warm_all))
    ctx = mp.get_context("spawn")
    cores = _cores()
    W = min(a.workers, len(cores))
    cores = [cores[(i * len(cores)) // W] for i in range(W)] if W else []   # spread over sockets (memory bandwidth)
    meta = {"host": socket.gethostname(), "cpu": cpu_model(), "workers": W, "cores": cores[:W],
            "slurm_job": os.environ.get("SLURM_JOB_ID"), "problem": a.problem, "n_target": a.n,
            "seed": a.seed, "count": a.count, "reps": a.reps, "methods": methods, "time": time.ctime()}
    try:
        import torch
        meta["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        meta["gpu"] = None
    print(json.dumps(meta), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    def results():
        if W == 0:              # --workers 0: in this process, unpinned (tests only)
            G["lock"] = G["core"] = None
            for t in tasks:
                yield task(t)
            return
        lock = ctx.Lock()
        counter = ctx.Value("i", 0)
        with ctx.Pool(W, initializer=_init, initargs=(lock, cores[:W], counter)) as pool:
            yield from pool.imap_unordered(task, tasks, chunksize=1)
    for r in results():
        rows.append(r)
        print(f"{r['path']} {r['method'][-60:]:60s} it={r['iters']:3d} oc={r['oc']:.2f} cn={r['coarse_n']} "
              f"lu/nnz={r['lu_ratio']:.1f} feat={r['t_feat']:.3f} fwd={r['t_fwd']:.3f} C={r['t_C']:.3f} "
              f"sa={r['t_sa']:.3f} lu={r['t_lu']:.3f} solve={r['t_solve']:.3f} total={r['total']:.3f}", flush=True)
        json.dump({"meta": meta, "rows": rows}, open(a.out + ".tmp", "w"), indent=0)
        os.replace(a.out + ".tmp", a.out)
    json.dump({"meta": meta, "rows": rows}, open(a.out + ".tmp", "w"), indent=0)
    os.replace(a.out + ".tmp", a.out)
    summarize([a.out])


SUMKEYS = ("iters", "oc", "coarse_n", "lu_ratio", "keep", "agg_mean", "t_feat", "t_fwd", "t_C", "t_sa", "t_lu",
           "t_setup", "t_solve", "t_cycle", "t_cycle_coarse", "total")


def summarize(files, quiet=False):
    rows = []
    for f in files:
        J = json.load(open(f))
        rows += [dict(r, _n=J["meta"]["n_target"], _p=J["meta"]["problem"]) for r in J["rows"]]
    groups = {}
    for r in rows:
        groups.setdefault((r["_p"], r["_n"], r["method"]), []).append(r)
    out = {}
    for k, g in sorted(groups.items()):
        out[k] = {q: float(np.mean([r[q] for r in g])) for q in SUMKEYS if q in g[0]}
        out[k]["count"] = len(g)
        out[k]["total_sd"] = float(np.std([r["total"] for r in g]))
        out[k]["fail"] = sum(r["iters"] >= 500 for r in g)
    if not quiet:
        print(f"{'problem':12s} {'n':>7s} {'method':58s} {'cnt':>3s} {'iters':>6s} {'oc':>5s} {'coarse_n':>8s} {'lu/A':>5s} "
              f"{'keep':>5s} {'feat':>6s} {'fwd':>6s} {'C':>6s} {'SA':>6s} {'LU':>6s} {'solve':>6s} {'total':>6s}")
        for (p, n, m), v in out.items():
            print(f"{p:12s} {n:7d} {m[-58:]:58s} {v['count']:3d} {v['iters']:6.1f} {v['oc']:5.2f} {v['coarse_n']:8.0f} "
                  f"{v['lu_ratio']:5.1f} {v.get('keep', float('nan')):5.3f} {v['t_feat']:6.3f} {v['t_fwd']:6.3f} "
                  f"{v['t_C']:6.3f} {v['t_sa']:6.3f} {v['t_lu']:6.3f} {v['t_solve']:6.3f} {v['total']:6.3f}"
                  + (f" FAIL{v['fail']}" if v['fail'] else ""))
    return out


# ----------------------------------------------------------------------------- fingerprints (no timing)
def C_sha1(C):
    C = C.tocsr()
    h = hashlib.sha1()
    for arr in (C.indptr.astype(np.int64), C.indices.astype(np.int64), (C.data != 0).astype(np.int8)):
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def fingerprint(A, fseed, spec):
    """C hash + iteration count + coarse statistics of one (matrix, method); deterministic."""
    kind, arg, opts = parse_method(spec)
    C, r = build_C(A, fseed, spec)
    ml, s = setup_solve(A, C, splu_opts(opts))
    h = hier_stats(ml, A, cycle_reps=0)
    return {"C_sha1": C_sha1(C), "C_nnz": int(C.nnz), "iters": s["iters"], "coarse_n": h["coarse_n"],
            "lu_nnz": h["lu_nnz"], "keep": r["keep"]}


def cmd_fingerprint(a):
    paths = gen_set(a.problem, a.n, a.seed, a.count)
    rows = []
    for p in paths:
        A, fseed = load_A(p)
        for m in a.methods:
            r = fingerprint(A, fseed, m)
            r.update(path=os.path.basename(p), method=m, problem=a.problem, n_target=a.n, seed=a.seed)
            rows.append(r)
            print(f"{r['path']} {m[-60:]:60s} it={r['iters']} cn={r['coarse_n']} {r['C_sha1'][:12]}", flush=True)
    if a.out:
        old = json.load(open(a.out)) if os.path.exists(a.out) else []
        json.dump(old + rows, open(a.out, "w"), indent=1)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "fingerprint"):
        r = sub.add_parser(name)
        r.add_argument("--problem", required=True)
        r.add_argument("--n", type=int, required=True)
        r.add_argument("--seed", type=int, default=1000)
        r.add_argument("--count", type=int, default=16)
        r.add_argument("--methods", nargs="+", required=True)
        r.add_argument("--out", required=(name == "run"))
        if name == "run":
            r.add_argument("--reps", type=int, default=3)
            r.add_argument("--workers", type=int, default=1, help="pinned worker processes (0: run in this process)")
            r.add_argument("--warm_all", action="store_true")
            r.add_argument("--resume", action="store_true", help="keep rows already in --out")
    g = sub.add_parser("gen")
    g.add_argument("--problem", required=True)
    g.add_argument("--n", type=int)
    g.add_argument("--seed", type=int, default=1000)
    g.add_argument("--count", type=int, default=16)
    g.add_argument("--sets", nargs="*", default=[], metavar="N:SEED:COUNT", help="several sets at once")
    s = sub.add_parser("summary")
    s.add_argument("files", nargs="+")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        cmd_run(a)
    elif a.cmd == "fingerprint":
        cmd_fingerprint(a)
    elif a.cmd == "gen":
        sets = [tuple(int(x) for x in v.split(":")) for v in a.sets] + ([(a.n, a.seed, a.count)] if a.n else [])
        for n, seed, count in sets:
            gen_set(a.problem, n, seed, count)
    else:
        summarize(a.files)


if __name__ == "__main__":
    main()
