"""Train the learned per-row threshold model (sub-study "coarse") jointly on tuning matrices by separable
CMA-ES on MEASURED total time.

  theta_i = sigmoid(w0 + w . z(phi_i))       base 'theta': s_ij >= theta_i
  theta_i = 0.5 sigmoid(w0 + w . z(phi_i))   base 'symm' : -a_ij >= theta_i sqrt(a_ii a_jj)
objective: mean over matrices of log(total_i / ref_i), ref_i = best global classical rule (theta or symm
grid, median of 3) on matrix i. Initialised at the best single classical rule of the same base.
The objective is wall-clock time, so a rerun gives a statistically similar but not identical model.

usage: python -m twolevel.train_lin --problem perforated_l --base theta --out lin_perforated_l_50k_theta.json
"""
import argparse
import json
import os
import time
import multiprocessing as mp

import numpy as np

from .common import load_set
from .rowthr import Mat, solve, row_features_fast, SepCMA, sel_lin, logit, LIN_NAMES

G = {}


def phi_of(m):
    F = row_features_fast(m, G["nb"])
    Phi = np.column_stack([F[k] for k in G["names"]])
    return (Phi - Phi.mean(0)) / (Phi.std(0) + 1e-12)


def ev(args):
    i, kind, p = args
    m = G["mats"][i]
    t0 = time.perf_counter()
    if G.get("time_feat"):
        m = Mat(m.A)   # rebuild edge arrays as the deployed method does
    if kind == "theta":
        sel = m.sel_theta(p)
    elif kind == "symm":
        sel = m.sel_symm(p)
    else:
        base, x = p
        phi = phi_of(m) if G.get("time_feat") else G["phi"][i]
        sel = sel_lin(m, phi, base, x)
    C = m.C_from_sel(sel)
    t_rule = time.perf_counter() - t0
    try:
        d = solve(m.A, C, t_rule)
    except Exception:
        d = {"iters": 500, "total": 99.0, "agg": 0.0}
    d["keep"] = float(sel[m.off].mean())
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problem", required=True)
    ap.add_argument("--n_target", type=int, default=50000, help="matrix size of the tuning set")
    ap.add_argument("--seed", type=int, default=2000, help="tuning-set seed")
    ap.add_argument("--n", type=int, default=8, help="number of tuning matrices")
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--base", default="theta", choices=["theta", "symm"])
    ap.add_argument("--gens", type=int, default=80)
    ap.add_argument("--pop", type=int, default=16)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--time_feat", type=int, default=1)
    ap.add_argument("--nb", default="", help="row features whose neighbour means are added, e.g. jump,s2,dom")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t_start = time.time()
    G["nb"] = tuple(x for x in a.nb.split(",") if x)
    G["names"] = LIN_NAMES + [k + "_nb" for k in G["nb"]]
    raw = load_set(a.problem, a.n_target, a.seed, a.n, coords=True)
    mats = [Mat(A, xy) for A, xy in raw]
    G["mats"] = mats
    G["phi"] = [phi_of(m) for m in mats]
    G["time_feat"] = a.time_feat
    with mp.get_context("fork").Pool(a.workers) as pool:
        # references: global sweeps of both classical rules, per matrix best (3 reps each, median)
        thetas = [0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        symms = [0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2]
        cands = [("theta", t) for t in thetas] + [("symm", t) for t in symms]
        tasks = [(i, k, p) for (k, p) in cands for i in range(a.n) for _ in range(3)]
        res = pool.map(ev, tasks, chunksize=1)
        T = np.array([d["total"] for d in res]).reshape(len(cands), a.n, 3)
        Tm = np.median(T, axis=2)
        ref = Tm.min(0)
        glob = int(np.argmin(np.log(Tm / ref).mean(1)))
        print("per-matrix best classical:", [cands[j] for j in Tm.argmin(0)], np.round(ref, 3))
        print("best single classical on tuning set:", cands[glob], "mean log ratio", np.log(Tm[glob] / ref).mean())
        jb = [j for j, c in enumerate(cands) if c[0] == a.base]
        j0 = min(jb, key=lambda j: np.log(Tm[j] / ref).mean())
        p0 = cands[j0][1]
        x0 = np.zeros(1 + len(G["names"]))
        x0[0] = logit(min(p0, 0.98) if a.base == "theta" else min(2 * p0, 0.98) if p0 > 0 else 0.01)
        es = SepCMA(x0, a.sigma, popsize=a.pop, seed=0)
        hist = []
        for g in range(a.gens):
            X = es.ask()
            tasks = [(i, "model", (a.base, x)) for x in X for i in range(a.n)]
            res = pool.map(ev, tasks, chunksize=1)
            Tt = np.array([d["total"] if d["iters"] < 500 else 99.0 for d in res]).reshape(len(X), a.n)
            f = np.log(Tt / ref).mean(1)
            es.tell(X, f)
            hist.append({"g": g, "fmin": float(f.min()), "fmed": float(np.median(f)), "sigma": float(es.sigma)})
            print(f"g{g} fmin={f.min():.4f} fmed={np.median(f):.4f} sigma={es.sigma:.3f} m={np.round(es.m, 2)} "
                  f"({time.time() - t_start:.0f}s)", flush=True)
        # final: mean vector, 5 reps
        tasks = [(i, "model", (a.base, es.m)) for i in range(a.n) for _ in range(5)]
        res = pool.map(ev, tasks, chunksize=1)
        Tf = np.median(np.array([d["total"] for d in res]).reshape(a.n, 5), 1)
        its = [res[5 * i]["iters"] for i in range(a.n)]
        print("final mean-vector model: log ratio to per-matrix best classical", np.log(Tf / ref).mean(),
              "to best single classical", np.log(Tf / Tm[glob]).mean(), "iters", its)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"problem": a.problem, "base": a.base, "names": G["names"], "nb": list(G["nb"]), "x": es.m.tolist(),
               "x0": x0.tolist(), "ref": ref.tolist(), "cands": cands, "Tm": Tm.tolist(), "glob": cands[glob],
               "final_total": Tf.tolist(), "final_iters": its, "hist": hist, "wall_s": time.time() - t_start,
               "workers": a.workers, "args": vars(a)}, open(a.out, "w"), indent=0)


if __name__ == "__main__":
    main()
