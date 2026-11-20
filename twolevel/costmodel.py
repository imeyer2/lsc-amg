"""Two-level cost model (sub-study "reward"): classical sweeps with full cost statistics, and the per-problem
least-squares fit used by the cost rewards of twolevel.train_node.

  python -m twolevel.costmodel sweep --problem anisotropy --n 50000 --seed 500 --count 8 --workers 8 \
         --out sweep/anisotropy_50k_s500.jsonl [--splu sym]
  python -m twolevel.costmodel fit --sweep_dir sweep --out cost_model.json

Fit (non-negative least squares, rows with n >= 50000):
  C + SA setup  ~ s_A nnz(A) + s_P nnz(P) + s_c nnz(A_c)
  LU factor     ~ f_flop flops(LU) + f_nnz nnz(LU)
  solve / iter  ~ i_A nnz(A) + i_P nnz(P) + i_lu nnz(LU) + i_n n
plus the size-scaling exponents of LU nnz and flops (a_nnz, a_flop) over sizes, and the rank
agreement (Spearman) of predicted vs measured time over the classical rules.
"""
import argparse
import collections
import glob
import json
import os
import time
import multiprocessing as mp

import numpy as np

from .common import load_set
from .nodepol import sweep_keep, two_level, cost_units
from .rules import C_from_keep, offdiag

RULES = [("theta", t) for t in (0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)] + \
        [("sym", t) for t in (0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5)] + [("topk", k) for k in (1, 2, 3, 4)]
G = {}


def task(a):
    i, rule, p = a
    A = G["As"][i]
    rows = []
    for rep in range(G["reps"]):
        t0 = time.perf_counter()
        keep = sweep_keep(A, rule, p)
        C = C_from_keep(A, keep)
        tc = time.perf_counter() - t0
        d = two_level(A, C, splu=G["splu"], want_lu_flops=(rep == 0))
        d["C_s"] = tc
        d["keep"] = float(keep[offdiag(A)[1]].mean())
        d["total_s"] = tc + d.get("setup_s", 0) + d.get("lu_s", 0) + d.get("solve_s", 0)
        rows.append(d)
    d = dict(rows[0])
    for key in ("C_s", "setup_s", "lu_s", "solve_s", "total_s"):
        d[key] = float(np.median([r.get(key, np.nan) for r in rows]))
    d.update(i=i, rule=rule, p=p)
    return d


def cmd_sweep(a):
    t0 = time.time()
    G["As"] = load_set(a.problem, a.n, a.seed, a.count)
    G["reps"] = a.reps
    G["splu"] = a.splu
    print(f"loaded {a.count} n={G['As'][0].shape[0]} in {time.time() - t0:.0f}s", flush=True)
    rules = RULES if a.rules == "all" else [(r.split(":")[0], float(r.split(":")[1])) for r in a.rules.split(",")]
    tasks = [(i, r, p) for (r, p) in rules for i in range(a.count)]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with mp.get_context("fork").Pool(a.workers) as pool, open(a.out + ".tmp", "w") as f:
        for d in pool.imap_unordered(task, tasks):
            d.update(problem=a.problem, size=a.n, splu=a.splu)
            f.write(json.dumps(d) + "\n")
            f.flush()
    os.replace(a.out + ".tmp", a.out)
    print(f"done {len(tasks)} in {time.time() - t0:.0f}s", flush=True)


def fit(sweep_dir, min_size=50000, problems=("anisotropy", "lognormal", "perforated_l"), limit=None):
    """limit: {file stem: k} uses only the first k rows of that sweep file (see results/twolevel_2026-10/README.md:
    the committed cost_model.json was fit while two 200k sweeps were still running)."""
    limit = limit or {}
    from scipy.optimize import nnls
    from scipy.stats import spearmanr
    res = {}
    for prob in problems:
        rows = []
        for f in sorted(glob.glob(os.path.join(sweep_dir, f"{prob}_*.jsonl"))):
            lines = open(f).read().splitlines()
            stem = os.path.basename(f)[:-len(".jsonl")]
            rows += [json.loads(line) for line in lines[:limit.get(stem, len(lines))]]
        rows = [r for r in rows if "lu_flops" in r and r["iters"] < 500]
        big = [r for r in rows if r["size"] >= min_size]
        if len(big) < 10:
            print(prob, "no data")
            continue

        def X(rs, cols):
            return np.array([[r[c] for c in cols] for r in rs], float)

        c1, _ = nnls(X(big, ["nnzA", "nnzP", "nnz_c"]), np.array([r["C_s"] + r["setup_s"] for r in big]))
        c2, _ = nnls(X(big, ["lu_flops", "lu_nnz"]), np.array([r["lu_s"] for r in big]))
        c3, _ = nnls(X(big, ["nnzA", "nnzP", "lu_nnz", "n"]), np.array([r["solve_s"] / max(r["iters"], 1) for r in big]))
        coef = dict(s_A=c1[0], s_P=c1[1], s_c=c1[2], f_flop=c2[0], f_nnz=c2[1], i_A=c3[0], i_P=c3[1], i_lu=c3[2], i_n=c3[3])
        coef = {k: float(v) for k, v in coef.items()}
        pred = np.array([cost_units(r, coef) for r in big])
        tot = np.array([r["total_s"] for r in big])
        rel = pred / tot - 1
        out = {"coef": coef, "rel_err_mean": float(np.mean(np.abs(rel))), "rel_err_max": float(np.max(np.abs(rel)))}
        for size in sorted(set(r["size"] for r in big)):
            g = collections.defaultdict(list)
            for r in big:
                if r["size"] == size:
                    g[(r["rule"], r["p"])].append((r["total_s"], cost_units(r, coef)))
            keys = list(g)
            mt = np.array([np.mean([x for x, _ in g[k]]) for k in keys])
            mp_ = np.array([np.mean([y for _, y in g[k]]) for k in keys])
            out[f"spearman_{size}"] = float(spearmanr(mt, mp_)[0])
            out[f"best_meas_{size}"] = str(keys[int(np.argmin(mt))])
            out[f"best_pred_{size}"] = str(keys[int(np.argmin(mp_))])
        agg = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in rows:
            agg[(r["rule"], r["p"])][r["size"]].append(r)
        an, af = [], []
        for k, d in agg.items():
            sizes = sorted(d)
            if len(sizes) < 2:
                continue
            n = np.array([np.mean([r["n"] for r in d[s]]) for s in sizes])
            ln = np.array([np.mean([r["lu_nnz"] for r in d[s]]) for s in sizes])
            lf = np.array([np.mean([r["lu_flops"] for r in d[s]]) for s in sizes])
            an.append(np.polyfit(np.log(n), np.log(ln), 1)[0])
            af.append(np.polyfit(np.log(n), np.log(lf), 1)[0])
        out["a_nnz"] = float(np.median(an)) if an else None
        out["a_flop"] = float(np.median(af)) if af else None
        out["a_nnz_range"] = [float(np.min(an)), float(np.max(an))] if an else None
        out["a_flop_range"] = [float(np.min(af)), float(np.max(af))] if af else None
        res[prob] = out
        print(prob, json.dumps(out, indent=1))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sweep")
    s.add_argument("--problem", required=True)
    s.add_argument("--n", type=int, required=True)
    s.add_argument("--seed", type=int, default=1000)
    s.add_argument("--count", type=int, default=16)
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--reps", type=int, default=1)
    s.add_argument("--rules", default="all")
    s.add_argument("--splu", default="default", choices=["default", "sym"])
    s.add_argument("--out", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--sweep_dir", required=True)
    f.add_argument("--min_size", type=int, default=50000)
    f.add_argument("--limit", action="append", default=[], metavar="STEM=K",
                   help="use only the first K rows of sweep file STEM.jsonl (repeatable)")
    f.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "sweep":
        cmd_sweep(a)
    else:
        res = fit(a.sweep_dir, a.min_size, limit={k: int(v) for k, v in (x.split("=") for x in a.limit)})
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
