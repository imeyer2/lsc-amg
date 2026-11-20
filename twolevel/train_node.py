"""ES training of the node-modulated strength policy (sub-study "reward", twolevel.nodepol.NodePolicy) for
the two-level SA, with a selectable reward:
  R        -log(iters) - lam * log(oc)                         (the paper's reward, lam = 0.1)
  cost     -log(cost-model seconds at the training size)       (coarse LU nnz / flops included)
  cost_tgt -log(cost-model seconds extrapolated to --target_n) (LU nnz ~ (N/n)^a_nnz, flops ~ (N/n)^a_flop)
  time     -log(measured C + setup + LU + solve seconds)       (noisy; workers share the node)
Optimiser: sep-CMA-ES (diagonal) with common random numbers (all members see the same minibatch).
Initialisation: exactly the given classical rule (v = u = 0, tau = -log theta): theta:<t> gives the row
base ('noderow'), sym:<t> the abs base ('nodeabs').
With a cost reward the whole run is deterministic (iteration counts and LU sizes only), so a rerun with the
same cost model reproduces the policy exactly.

usage: python -m twolevel.train_node --problem perforated_l --init theta:0.3 --reward cost_tgt \
           --cm results/twolevel_2026-10/cost_model.json --pop 32 --mb 12 --epochs 150 --n_val 16 --out DIR
"""
import argparse
import json
import math
import os
import time
import multiprocessing as mp

import numpy as np

from .common import load_set
from .nodepol import CSRG, node_features, NodePolicy, node_stats, two_level, cost_units, NODE_NAMES
from .rules import C_from_keep

G = {}


class SepCMA:
    def __init__(self, m, scale, lam, seed):
        self.n = len(m)
        self.m = m.copy()
        self.scale = scale.copy()
        self.sigma = 1.0
        self.lam = lam
        self.mu = lam // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1 / (self.w ** 2).sum()
        n = self.n
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff) * (n + 2) / 3)
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.pc = np.zeros(n)
        self.ps = np.zeros(n)
        self.C = np.ones(n)
        self.t = 0
        self.chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        self.rng = np.random.default_rng(seed)

    def ask(self):
        z = self.rng.standard_normal((self.lam, self.n))
        self.y = z * np.sqrt(self.C)
        return self.m[None] + self.sigma * self.y * self.scale[None]

    def tell(self, R):
        idx = np.argsort(-R)[:self.mu]
        yw = self.w @ self.y[idx]
        self.m = self.m + self.sigma * yw * self.scale
        self.t += 1
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * yw / np.sqrt(self.C)
        hs = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * self.t)) / self.chiN < 1.4 + 2 / (self.n + 1)
        self.pc = (1 - self.cc) * self.pc + hs * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * yw
        self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * (self.pc ** 2 + (1 - hs) * self.cc * (2 - self.cc) * self.C) \
            + self.cmu * (self.w @ (self.y[idx] ** 2))
        self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))


def reward_of(d, a):
    it = d["iters"]
    if "fail" in d or it >= a.maxiter:
        return -10.0
    if a.reward == "R":
        return -math.log(max(it, 1)) - a.lam * math.log(d["oc"])
    if a.reward == "time":
        return -math.log(d["C_s"] + d["setup_s"] + d["lu_s"] + d["solve_s"])
    cm = G["cm"]
    if a.reward == "cost":
        return -math.log(cost_units(d, cm))
    if a.reward == "cost_tgt":
        f = a.target_n / d["n"]
        e = dict(d)
        for k in ("n", "nnzA", "nnzP", "nnz_c"):
            e[k] = d[k] * f
        e["lu_nnz"] = d["lu_nnz"] * f ** a.a_nnz
        e["lu_flops"] = d["lu_flops"] * f ** a.a_flop
        return -math.log(cost_units(e, cm))
    raise ValueError(a.reward)


def task(t):
    j, th, which, i = t
    g, feats = G[which][i]
    a = G["a"]
    t0 = time.perf_counter()
    keep = G["pol"].mask(th, g, feats)
    C = C_from_keep(g.A, keep)
    tc = time.perf_counter() - t0
    d = two_level(g.A, C, splu=a.splu, want_lu_flops=a.reward in ("cost", "cost_tgt"), maxiter=a.maxiter)
    d["C_s"] = tc
    d["keep"] = float(keep[g.off].mean())
    return (j, which, i), d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problem", required=True)
    ap.add_argument("--n", type=int, default=5000, help="training matrix size")
    ap.add_argument("--n_train", type=int, default=64)
    ap.add_argument("--n_val", type=int, default=8)
    ap.add_argument("--train_seed", type=int, default=0)
    ap.add_argument("--val_seed", type=int, default=1)
    ap.add_argument("--reward", default="cost", choices=["R", "cost", "cost_tgt", "time"])
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--cm", default="results/twolevel_2026-10/cost_model.json", help="cost model JSON (cost rewards)")
    ap.add_argument("--target_n", type=float, default=50000)
    ap.add_argument("--a_nnz", type=float, default=1.1)
    ap.add_argument("--a_flop", type=float, default=1.5)
    ap.add_argument("--init", default="theta:0.3", help="theta:<t> (row base) | sym:<t> (abs base)")
    ap.add_argument("--node_feats", default=",".join(NODE_NAMES))
    ap.add_argument("--splu", default="default", choices=["default", "sym"],
                    help="coarse splu: pyamg default (COLAMD) or MMD_AT_PLUS_A + diag_pivot_thresh 0 + SymmetricMode")
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--mb", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--sig_w", type=float, default=0.15)
    ap.add_argument("--sig_tau", type=float, default=0.25)
    ap.add_argument("--maxiter", type=int, default=500)
    ap.add_argument("--workers", type=int, default=46)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    G["a"] = a
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, "config.json"), "w"), indent=1)
    if a.reward in ("cost", "cost_tgt"):
        from .common import _rel
        G["cm"] = json.load(open(_rel(a.cm)))[a.problem]["coef"]
    t0 = time.time()
    tr = load_set(a.problem, a.n, a.train_seed, a.n_train)
    va = load_set(a.problem, a.n, a.val_seed, a.n_val)
    names = [x for x in a.node_feats.split(",") if x]
    base = "row" if a.init.startswith("theta") else "abs"
    mu, sd = node_stats(tr[:16], names)
    pol = NodePolicy(names, mu, sd, base)
    G["pol"] = pol
    G["tr"] = [(g, node_features(g, names)) for g in (CSRG(A) for A in tr)]
    G["va"] = [(g, node_features(g, names)) for g in (CSRG(A) for A in va)]
    tau = -math.log(float(a.init.split(":")[1]))
    th = pol.init_theta(tau)
    k = pol.n_params - 1
    scale = np.concatenate([np.full(k, a.sig_w), [a.sig_tau]])
    print(f"data n={tr[0].shape[0]} train={len(tr)} val={len(va)} in {time.time() - t0:.0f}s; params={pol.n_params} "
          f"init tau={tau:.3f} base={base}", flush=True)
    pool = mp.get_context("fork").Pool(a.workers)
    nev = [0]

    def run(tasks):
        out = {}
        for key, d in pool.imap_unordered(task, tasks, chunksize=1):
            out[key] = d
        nev[0] += len(tasks)
        return out

    def evaluate(th, which, idx):
        o = run([(0, th, which, i) for i in idx])
        ds = [o[(0, which, i)] for i in idx]
        R = float(np.mean([reward_of(d, a) for d in ds]))
        summ = {k2: float(np.mean([d.get(k2, np.nan) for d in ds]))
                for k2 in ("iters", "oc", "n_c", "lu_nnz", "keep", "setup_s", "lu_s", "solve_s", "C_s")}
        summ["R"] = R
        return summ

    vidx = list(range(len(va)))
    ev = evaluate(th, "va", vidx)
    print("init val:", json.dumps(ev), flush=True)
    best = {"R": ev["R"], "theta": th.copy(), "epoch": -1}
    hist = [{"epoch": -1, "evals": 0, "wall": 0.0, "val": ev, "theta": th.tolist()}]
    es = SepCMA(th, np.where(scale > 0, scale, 1e-12), a.pop, a.seed)
    rng = np.random.default_rng(a.seed + 11)
    tstart = time.time()
    nev[0] = 0

    def save():
        json.dump({"hist": hist, "best": {"R": best["R"], "epoch": best["epoch"], "theta": best["theta"].tolist()},
                   "policy": dict(pol.to_json(), probes=None), "wall_s": time.time() - tstart, "evals": nev[0],
                   "workers": a.workers, "config": vars(a)}, open(os.path.join(a.out, "train.json"), "w"))

    for ep in range(a.epochs):
        te = time.time()
        mb = rng.choice(len(tr), a.mb, replace=False)
        X = es.ask()
        X[:, scale == 0] = th[scale == 0]
        o = run([(j, X[j], "tr", int(i)) for j in range(a.pop) for i in mb])
        R = np.array([np.mean([reward_of(o[(j, "tr", int(i))], a) for i in mb]) for j in range(a.pop)])
        es.tell(R)
        th = es.m.copy()
        th[scale == 0] = np.array(hist[0]["theta"])[scale == 0]
        ev = evaluate(th, "va", vidx)
        if ev["R"] > best["R"]:
            best.update(R=ev["R"], theta=th.copy(), epoch=ep)
        hist.append({"epoch": ep, "evals": nev[0], "wall": time.time() - tstart, "val": ev, "pop_mean": float(R.mean()),
                     "pop_max": float(R.max()), "sigma": float(es.sigma), "theta": th.tolist()})
        print(f"ep {ep:3d} evals={nev[0]:6d} {time.time() - te:5.1f}s pop={R.mean():.4f}/{R.max():.4f} sig={es.sigma:.3f} | "
              f"val R={ev['R']:.4f} it={ev['iters']:.1f} oc={ev['oc']:.2f} n_c={ev['n_c']:.0f} lu={ev['lu_nnz']:.3g} "
              f"keep={ev['keep']:.3f} tau={math.exp(th[k]):.3f}", flush=True)
        if ep % 10 == 0 or ep == a.epochs - 1:
            save()
    save()
    print(f"done best epoch {best['epoch']} val R={best['R']:.4f} wall={time.time() - tstart:.0f}s evals={nev[0]}", flush=True)
    pool.close()


if __name__ == "__main__":
    main()
