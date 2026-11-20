"""Oracle / headroom search for two-level SA strength graphs (sub-study "coarse").

stage sweep : global classical (row-relative theta), pyamg-symmetric (negative couplings) and top-k
              sweeps per matrix (measured total time).
stage cma   : per-matrix search (sep-CMA on measured total time) over per-row threshold
              parameterisations: geo<K> (K x K spatial blocks), feat (8 x 8 bins of row features s2 x jump),
              lin (logit theta = w . phi). --base symm searches 0.5 * theta_i in the symmetric rule instead.
export      : the best per-matrix strength mask (over parameterisations x {mean, best-ever}) written as
              <masks>/<problem>_<feature_seed>.npy, for re-timing in the harness (method mask:<masks>/<problem>).
              Masks are a deterministic function of the stored parameters and the matrix.

  python -m twolevel.oracle search --problem perforated_l --n_target 50000 --seed 2000 --n 4 --workers 28 \
         --gens 60 --pop 28 --reps 5 --out oracle/perforated_l_theta
  python -m twolevel.oracle export --cma oracle/perforated_l_theta_cma.json --masks oracle/masks_theta
"""
import argparse
import json
import os
import time
import multiprocessing as mp

import numpy as np

from .common import load_set, set_paths, load_A
from .rowthr import Mat, solve, row_features, SepCMA, sig, logit, LIN_NAMES

G = {}


def init():
    os.environ["OMP_NUM_THREADS"] = "1"


def ev(args):
    """args = (mat index, kind, payload)"""
    i, kind, p = args
    m = G["mats"][i]
    t0 = time.perf_counter()
    if kind == "theta":
        sel = m.sel_theta(p)
    elif kind == "symm":
        sel = m.sel_symm(p)
    elif kind == "topk":
        sel = m.sel_topk(int(p))
    elif kind == "rowtheta":
        sel = m.sel_theta(np.asarray(p))
    elif kind == "param":   # (ptype, x)
        ptype, x = p
        sel = sel_param(m, i, ptype, x)
    C = m.C_from_sel(sel)
    t_rule = time.perf_counter() - t0
    try:
        d = solve(m.A, C, t_rule)
    except Exception as e:
        d = {"iters": 500, "setup": 99.0, "solve": 0.0, "rule": t_rule, "total": 99.0, "nc": 0, "nnzc": 0, "oc": 0.0,
             "agg": 0.0, "Pnnz": 0, "fail": str(e)[:60]}
    d["keep"] = float(sel[m.off].mean())
    return d


def theta_rows(m, i, ptype, x):
    if ptype.startswith("geo"):
        return sig(x[G["geo"][ptype][i]])
    if ptype == "feat":
        return sig(x[G["featbin"][i]])
    if ptype == "lin":
        return sig(x[0] + G["phi"][i] @ x[1:])
    raise ValueError(ptype)


def sel_param(m, i, ptype, x):
    th = theta_rows(m, i, ptype, x)
    if G.get("base") == "symm":
        return m.sel_symm_rows(0.5 * th)
    return m.sel_theta(th)


def prep_params(mats):
    G["geo"] = {}
    G["featbin"] = []
    G["phi"] = []
    for K in (4, 8, 16):
        lst = []
        for m in mats:
            xy = m.coords
            lo = xy.min(0)
            hi = xy.max(0)
            ij = np.clip(((xy - lo) / (hi - lo + 1e-12) * K).astype(int), 0, K - 1)
            lst.append(ij[:, 0] * K + ij[:, 1])
        G["geo"][f"geo{K}"] = lst
    for m in mats:
        F = row_features(m)

        def qbin(v, k=8):
            qs = np.quantile(v, np.linspace(0, 1, k + 1)[1:-1])
            return np.searchsorted(qs, v, side="right")
        G["featbin"].append(qbin(F["s2"]) * 8 + qbin(F["jump"]))
        Phi = np.column_stack([F[k] for k in LIN_NAMES])
        Phi = (Phi - Phi.mean(0)) / (Phi.std(0) + 1e-12)
        G["phi"].append(Phi)
    G["phi_names"] = LIN_NAMES


def stage_sweep(pool, nm, out):
    thetas = [0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85,
              0.9, 0.95, 1.0]
    symms = [0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    tasks = [(i, "theta", t) for t in thetas for i in range(nm)] + [(i, "symm", t) for t in symms for i in range(nm)] + \
            [(i, "topk", k) for k in (1, 2, 3, 4) for i in range(nm)]
    res = pool.map(ev, tasks, chunksize=1)
    rep = {}
    for (i, kind, p), d in zip(tasks, res):
        rep.setdefault(f"{kind}:{p}", [None] * nm)[i] = d
    for k, v in rep.items():
        print(f"{k:12s} it={np.mean([d['iters'] for d in v]):6.1f} total={np.mean([d['total'] for d in v]):.3f} "
              f"setup={np.mean([d['setup'] for d in v]):.3f} agg={np.mean([d['agg'] for d in v]):.2f} "
              f"keep={np.mean([d['keep'] for d in v]):.3f}", flush=True)
    json.dump(rep, open(out, "w"), indent=0, default=float)
    return rep


def stage_cma(pool, nm, out, sweep, ptypes, gens, pop, reps):
    rep = {"per_matrix": [], "base": G.get("base", "theta")}
    for i in range(nm):
        pre = "symm:" if G.get("base") == "symm" else "theta:"
        keys = [k for k in sweep if k.startswith(pre)]
        best_k = min(keys, key=lambda k: sweep[k][i]["total"])
        th0 = float(best_k.split(":")[1])
        if G.get("base") == "symm":
            th0 = max(2 * th0, 0.005)
        row = {"best_theta": th0, "best_theta_total_sweep": sweep[best_k][i]["total"]}
        for ptype in ptypes:
            if ptype == "lin":
                x0 = np.zeros(1 + len(G["phi_names"]))
                x0[0] = logit(min(th0, 0.98))
            else:
                K = {"geo4": 16, "geo8": 64, "geo16": 256, "feat": 64}[ptype]
                x0 = np.full(K, logit(min(th0, 0.98)))
            es = SepCMA(x0, 0.7, popsize=pop, seed=i)
            hist = []
            best = (np.inf, None)
            t0 = time.time()
            for g in range(gens):
                X = es.ask()
                res = pool.map(ev, [(i, "param", (ptype, x)) for x in X], chunksize=1)
                f = np.array([d["total"] if d["iters"] < 500 else 1e3 for d in res])
                es.tell(X, f)
                j = int(np.argmin(f))
                if f[j] < best[0]:
                    best = (f[j], X[j].copy(), res[j])
                hist.append({"g": g, "fmin": float(f.min()), "fmed": float(np.median(f)), "sigma": float(es.sigma),
                             "it_best": res[j]["iters"], "agg_best": res[j]["agg"]})
                if g % 10 == 0 or g == gens - 1:
                    print(f"m{i} {ptype} g{g} fmin={f.min():.3f} fmed={np.median(f):.3f} it={res[j]['iters']} "
                          f"agg={res[j]['agg']:.2f} sig={es.sigma:.3f} ({time.time() - t0:.0f}s)", flush=True)
            cand = {"mean": ("param", (ptype, es.m)), "best": ("param", (ptype, best[1])),
                    "classical": ("symm", th0 / 2) if G.get("base") == "symm" else ("theta", th0)}
            tasks = [(i, k2, p) for (k2, p) in cand.values() for _ in range(reps)]
            res = pool.map(ev, tasks, chunksize=1)
            summ = {}
            for ci, cname in enumerate(cand):
                rr = res[ci * reps:(ci + 1) * reps]
                summ[cname] = {"total_med": float(np.median([d["total"] for d in rr])), "iters": rr[0]["iters"],
                               "agg": rr[0]["agg"], "nc": rr[0]["nc"], "keep": rr[0]["keep"],
                               "setup_med": float(np.median([d["setup"] for d in rr])),
                               "solve_med": float(np.median([d["solve"] for d in rr]))}
            row[ptype] = {"summary": summ, "x_mean": es.m.tolist(), "x_best": best[1].tolist(), "hist": hist}
            print(f"m{i} {ptype} RESULT classical={summ['classical']['total_med']:.3f} ({summ['classical']['iters']} it) "
                  f"mean={summ['mean']['total_med']:.3f} ({summ['mean']['iters']} it) "
                  f"best={summ['best']['total_med']:.3f} ({summ['best']['iters']} it)", flush=True)
        rep["per_matrix"].append(row)
        json.dump(rep, open(out, "w"), indent=0, default=float)
    return rep


def cmd_search(a):
    t0 = time.time()
    raw = load_set(a.problem, a.n_target, a.seed, a.n, coords=True)
    mats = [Mat(A, xy) for A, xy in raw]
    print(f"{a.problem} {a.n_target} n={[m.n for m in mats]} nnz={[m.A.nnz for m in mats]} gen {time.time() - t0:.0f}s",
          flush=True)
    G["mats"] = mats
    prep_params(mats)
    G["base"] = a.base
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    meta = {"problem": a.problem, "n_target": a.n_target, "seed": a.seed, "n": a.n, "base": a.base}
    with mp.get_context("fork").Pool(a.workers, initializer=init) as pool:
        if "sweep" in a.stages:
            sweep = stage_sweep(pool, a.n, a.out + "_sweep.json")
        else:
            sweep = json.load(open(a.out + "_sweep.json"))
        if "cma" in a.stages:
            rep = stage_cma(pool, a.n, a.out + "_cma.json", sweep, a.ptypes.split(","), a.gens, a.pop, a.reps)
            rep["meta"] = meta
            json.dump(rep, open(a.out + "_cma.json", "w"), indent=0, default=float)


def compact(cma):
    """the parameters needed to rebuild the masks (no search history)."""
    out = {k: v for k, v in cma.items() if k != "per_matrix"}
    out["per_matrix"] = []
    for row in cma["per_matrix"]:
        r = {k: v for k, v in row.items() if not isinstance(v, dict)}
        for pt, v in row.items():
            if isinstance(v, dict):
                r[pt] = {"summary": v["summary"], "x_mean": v["x_mean"], "x_best": v["x_best"]}
        out["per_matrix"].append(r)
    return out


def cmd_export(a):
    J = json.load(open(a.cma))
    meta = J["meta"]
    p, nt, seed, base = meta["problem"], meta["n_target"], meta["seed"], meta["base"]
    raw = load_set(p, nt, seed, len(J["per_matrix"]), coords=True)
    mats = [Mat(A, xy) for A, xy in raw]
    G["mats"] = mats
    prep_params(mats)
    G["base"] = base
    os.makedirs(a.masks, exist_ok=True)
    summary = {}
    for i, row in enumerate(J["per_matrix"]):
        fseed = load_A(set_paths(p, nt, seed, i + 1)[i])[1]
        best = None
        for pt in ("geo4", "geo8", "geo16", "feat", "lin"):
            if pt not in row:
                continue
            for which in ("mean", "best"):
                t = row[pt]["summary"][which]["total_med"]
                if best is None or t < best[0]:
                    best = (t, pt, which)
        x = np.array(row[best[1]]["x_" + best[2]])
        sel = sel_param(mats[i], i, best[1], x)
        np.save(os.path.join(a.masks, f"{p}_{fseed}.npy"), sel)
        summary[f"{p}_{i}"] = {"ptype": best[1], "which": best[2], "pool_total": best[0], "fseed": fseed,
                               "classical_pool_total": row[best[1]]["summary"]["classical"]["total_med"],
                               "keep": float(sel[mats[i].off].mean())}
        print(p, i, summary[f"{p}_{i}"], flush=True)
    sp_ = os.path.join(a.masks, "summary.json")
    old = json.load(open(sp_)) if os.path.exists(sp_) else {}
    old.update(summary)
    json.dump(old, open(sp_, "w"), indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("--problem", required=True)
    s.add_argument("--n_target", type=int, default=50000)
    s.add_argument("--seed", type=int, default=2000)
    s.add_argument("--n", type=int, default=4)
    s.add_argument("--workers", type=int, default=28)
    s.add_argument("--stages", default="sweep,cma")
    s.add_argument("--ptypes", default="geo4,geo8,feat,lin")
    s.add_argument("--gens", type=int, default=60)
    s.add_argument("--pop", type=int, default=28)
    s.add_argument("--reps", type=int, default=5)
    s.add_argument("--base", default="theta", choices=["theta", "symm"])
    s.add_argument("--out", required=True, help="output prefix (<out>_sweep.json, <out>_cma.json)")
    e = sub.add_parser("export")
    e.add_argument("--cma", required=True)
    e.add_argument("--masks", required=True)
    a = ap.parse_args(argv)
    {"search": cmd_search, "export": cmd_export}[a.cmd](a)


if __name__ == "__main__":
    main()
