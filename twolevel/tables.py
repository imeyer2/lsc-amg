"""Tables, paired tests and fits from harness result JSONs.

  python -m twolevel.tables table   F.json [...]             per-method means, sorted by total (bar's format)
  python -m twolevel.tables large   F.json [...]             compact table (200k/500k)
  python -m twolevel.tables paired  F.json A B               per-matrix paired difference of methods A and B
  python -m twolevel.tables vsbest  F.json [...]             every method vs the best classical rule in the file(s):
                                                             paired mean ratio - 1, 95% CI, wins
  python -m twolevel.tables vsref   REF F.json [...]         every method vs method REF (paired mean ratio - 1, wins)
  python -m twolevel.tables winners F.json [...]             best rule per family (tuning)
  python -m twolevel.tables headroom F.json [...]            tuned single classical / per-matrix best classical /
                                                             per-matrix best oracle mask (oracle re-timing files)
  python -m twolevel.tables lufit   F.json [...]             t_LU ~ a nnz(LU), coarse solve per cycle ~ b nnz(LU),
                                                             nnz(LU) ~ coarse_n^e (tuning rows)
  python -m twolevel.tables compare NEW.json OLD.json [...]  regenerated vs committed: per method, matrices with
                                                             identical iterations / coarse size / LU nnz, mean totals
Method names are matched after canon(): paths are reduced to the model / policy name, so committed
results (results/twolevel_2026-10) and regenerated outputs compare directly.
A method argument matches a canonical name exactly or, failing that, as a substring (shortest wins).
"""
import re
import sys
import json
import argparse
import collections

import numpy as np

CLASSICAL = ("theta", "sym", "topk", "srowq")


def canon(spec):
    """canonical method name: model / policy files reduced to their names, options kept."""
    base, bar, opt = spec.partition("|")
    kind, _, arg = base.partition(":")
    if kind in ("lsc", "lscref"):
        mm = re.match(r".*?([a-z_]+_\d+k)/seed(\d)/model_best\.pt@(\w+)$", arg)
        if mm:
            arg = f"{mm.group(1)}/seed{mm.group(2)}@{mm.group(3)}"
    elif kind in ("lin", "node"):
        if arg.endswith("/train.json"):
            kind, arg = "node", arg[:-len("/train.json")].rsplit("/", 1)[-1]
        else:
            name = arg.rsplit("/", 1)[-1]
            name = name[:-5] if name.endswith(".json") else name
            if name.startswith("lin_"):
                kind, arg = "lin", name
            else:
                kind, arg = "node", name
    elif kind == "mask":
        d, _, p = arg.rpartition("/")
        tag = "symm" if ("sym" in d.rsplit("/", 1)[-1]) else "theta"
        arg = f"{tag}/{p}"
    return f"{kind}:{arg}" + (bar + opt if bar else "")


def label(m):
    """human-readable label (bar's naming for the paper models)."""
    c = canon(m)
    mm = re.match(r"(lscref|lsc):([a-z_]+)_(\d+k)/seed(\d)@(\w+)(\|.*)?", c)
    if mm:
        kind, prob, size, seed, dev, opt = mm.groups()
        s = f"LSC {size}-trained (s{seed}) fwd {'V100' if dev == 'cuda' else 'CPU'}"
        if kind == "lscref":
            s += " [reference path]"
        if opt:
            s += " " + opt
        return s
    return c


def is_classical(m):
    return canon(m).split(":")[0] in CLASSICAL


def load(files):
    rows = []
    for f in files:
        J = json.load(open(f))
        meta = J["meta"]
        for r in J["rows"]:
            r = dict(r)
            r["_p"] = meta["problem"]
            r["_n"] = meta["n_target"]
            r["_cpu"] = meta.get("cpu")
            r["method"] = canon(r["method"])
            rows.append(r)
    return rows


AGGKEYS = ("iters", "oc", "coarse_n", "lu_ratio", "t_feat", "t_fwd", "t_C", "t_sa", "t_lu", "t_setup", "t_solve",
           "total", "t_cycle", "t_cycle_coarse", "keep", "n")


def agg(rows):
    g = collections.OrderedDict()
    for r in rows:
        g.setdefault((r["_p"], r["_n"], r["method"]), []).append(r)
    out = collections.OrderedDict()
    for k, rs in g.items():
        v = {q: float(np.mean([r[q] for r in rs])) for q in AGGKEYS if q in rs[0]}
        v["sd"] = float(np.std([r["total"] for r in rs]))
        v["cnt"] = len(rs)
        v["fail"] = sum(r["iters"] >= 500 for r in rs)
        v["imax"] = max(r["iters"] for r in rs)
        out[k] = v
    return out


def table(files):
    A = agg(load(files))
    lines = []
    for p in sorted({k[0] for k in A}):
        for n in sorted({k[1] for k in A}):
            ks = sorted([k for k in A if k[0] == p and k[1] == n], key=lambda k: A[k]["total"])
            if not ks:
                continue
            v0 = A[ks[0]]
            lines.append(f"\n**{p}, n = {v0.get('n', n):.0f}, {v0['cnt']} matrices** (sorted by total)\n")
            lines.append("| method | iters | op.cx | coarse n | LU nnz/nnz(A) | keep | feat s | fwd s | C s | SA s | LU s | "
                         "setup s | solve s | ms/cycle (coarse) | **total s** (sd) |")
            lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            for k in ks:
                v = A[k]
                cyc = f"{v['t_cycle'] * 1e3:.1f} ({v['t_cycle_coarse'] * 1e3:.1f})" if "t_cycle" in v else "-"
                lines.append(f"| {label(k[2])} | {v['iters']:.1f}{' (max %d)' % v['imax'] if v['fail'] else ''} | "
                             f"{v['oc']:.2f} | {v['coarse_n']:.0f} | {v['lu_ratio']:.1f} | {v.get('keep', np.nan):.2f} | "
                             f"{v['t_feat']:.3f} | {v['t_fwd']:.3f} | {v['t_C']:.3f} | {v['t_sa']:.3f} | {v['t_lu']:.3f} | "
                             f"{v['t_setup']:.3f} | {v['t_solve']:.3f} | {cyc} | **{v['total']:.3f}** ({v['sd']:.3f}) |")
    return "\n".join(lines)


def large(files):
    A = agg(load(files))
    lines = []
    for p in sorted({k[0] for k in A}):
        for n in sorted({k[1] for k in A}):
            ks = sorted([k for k in A if k[0] == p and k[1] == n], key=lambda k: A[k]["total"])
            if not ks:
                continue
            v0 = A[ks[0]]
            lines.append(f"\n**{p}, n = {v0.get('n', n):.0f}, {v0['cnt']} matrices**\n")
            lines.append("| method | iters | op.cx | coarse n | LU nnz/nnz(A) | feat+fwd+C s | setup s (SA+LU) | solve s | **total s** |")
            lines.append("|---|---|---|---|---|---|---|---|---|")
            for k in ks:
                v = A[k]
                lines.append(f"| {label(k[2])} | {v['iters']:.1f} | {v['oc']:.2f} | {v['coarse_n']:.0f} | {v['lu_ratio']:.1f} | "
                             f"{v['t_feat'] + v['t_fwd'] + v['t_C']:.3f} | {v['t_setup']:.3f} | {v['t_solve']:.3f} | "
                             f"**{v['total']:.3f}** |")
    return "\n".join(lines)


def _by_method(rows):
    by = collections.defaultdict(dict)
    for r in rows:
        by[r["method"]][r["path"]] = r
    return by


def pick(by, s):
    s = canon(s)
    if s in by:
        return s
    ms = sorted([m for m in by if s in m], key=len)
    if not ms:
        raise KeyError(f"no method matches {s!r}; have {sorted(by)}")
    return ms[0]


def paired(files, a, b):
    by = _by_method(load(files))
    ma, mb = pick(by, a), pick(by, b)
    A, B = by[ma], by[mb]
    ks = sorted(set(A) & set(B))
    d = np.array([A[k]["total"] - B[k]["total"] for k in ks])
    ra = np.array([A[k]["total"] / B[k]["total"] for k in ks])
    res = {"A": ma, "B": mb, "n": len(ks), "mean_diff": float(d.mean()), "sd": float(d.std(ddof=1)) if len(d) > 1 else np.nan,
           "se": float(d.std(ddof=1) / np.sqrt(len(ks))) if len(d) > 1 else np.nan, "A_faster": int((d < 0).sum()),
           "mean_ratio": float(ra.mean()), "per_matrix": d.tolist()}
    txt = (f"A={ma}\nB={mb}\n n={len(ks)} mean(A-B)={res['mean_diff']:+.3f}s  sd={res['sd']:.3f}  se={res['se']:.3f}  "
           f"A faster on {res['A_faster']}/{len(ks)}  mean ratio A/B={res['mean_ratio']:.3f}\n"
           f" per-matrix A-B: " + " ".join(f"{x:+.3f}" for x in d))
    return res, txt


def vs(files, ref=None):
    """every method vs `ref` (default: the classical rule with the lowest mean total)."""
    rows = load(files)
    by = _by_method(rows)
    mean = {m: float(np.mean([r["total"] for r in v.values()])) for m, v in by.items()}
    if ref is None:
        ref = min([m for m in by if is_classical(m)], key=lambda m: mean[m])
    else:
        ref = pick(by, ref)
    out = []
    for m in sorted(by, key=lambda m: mean[m]):
        common = sorted(set(by[m]) & set(by[ref]))
        d = np.array([by[m][p]["total"] / by[ref][p]["total"] - 1 for p in common])
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
        rs = list(by[m].values())
        out.append({"method": m, "cnt": len(rs), "iters": float(np.mean([r["iters"] for r in rs])),
                    "coarse_n": float(np.mean([r["coarse_n"] for r in rs])),
                    "lu_ratio": float(np.mean([r["lu_ratio"] for r in rs])),
                    "pre": float(np.mean([r["t_feat"] + r["t_fwd"] + r["t_C"] for r in rs])),
                    "setup": float(np.mean([r["t_setup"] for r in rs])), "solve": float(np.mean([r["t_solve"] for r in rs])),
                    "total": mean[m], "vs_ref": float(d.mean()), "ci95": [float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)],
                    "wins": int((d < 0).sum()), "n_paired": len(d)})
    return ref, out


def vs_markdown(ref, out):
    lines = [f"reference: `{label(ref)}`\n",
             "| method | iters | coarse n | LU nnz/nnz(A) | feat+fwd+C s | SA+LU s | solve s | **total s** | vs ref (95% CI) | faster on |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for v in out:
        ci = "" if v["method"] == ref else f" [{100 * v['ci95'][0]:+.1f}, {100 * v['ci95'][1]:+.1f}]"
        lines.append(f"| {label(v['method'])} | {v['iters']:.1f} | {v['coarse_n']:.0f} | {v['lu_ratio']:.1f} | {v['pre']:.3f} | "
                     f"{v['setup']:.3f} | {v['solve']:.3f} | **{v['total']:.3f}** | {100 * v['vs_ref']:+.1f}%{ci} | "
                     f"{v['wins']}/{v['n_paired']} |")
    return "\n".join(lines)


def winners(files, families=CLASSICAL):
    """best (lowest mean total, failures = inf) rule per family, per (problem, n)."""
    rows = load(files)
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["_p"], r["_n"], r["method"])].append(r["total"] if r["iters"] < 500 else np.inf)
    fam = {}
    for (p, n, m), v in g.items():
        k = m.split(":")[0]
        if k not in families:
            continue
        t = float(np.mean(v))
        key = (p, n, k, "|" in m)
        if key not in fam or t < fam[key][1]:
            fam[key] = (m, t)
    return {f"{p}/{n}/{k}{'/opt' if o else ''}": {"method": m, "total": t} for (p, n, k, o), (m, t) in sorted(fam.items())}


def headroom(files):
    """per problem: best single classical rule (mean), per-matrix best classical, per-matrix best oracle mask."""
    rows = load(files)
    out = {}
    for p in sorted({r["_p"] for r in rows}):
        T = collections.defaultdict(dict)
        for r in rows:
            if r["_p"] == p:
                T[r["method"]][r["path"]] = r["total"]
        mats = sorted(set.intersection(*[set(v) for v in T.values()]))
        cls = [m for m in T if is_classical(m)]
        msk = [m for m in T if m.startswith("mask:")]
        mean = {m: float(np.mean([T[m][a] for a in mats])) for m in T}
        best = min(cls, key=mean.get)
        pmc = float(np.mean([min(T[m][a] for m in cls) for a in mats]))
        pmo = float(np.mean([min(T[m][a] for m in msk) for a in mats])) if msk else np.nan
        out[p] = {"tuned_single_classical": best, "tuned_single_total": mean[best], "per_matrix_best_classical": pmc,
                  "per_matrix_best_oracle": pmo, "headroom": 1 - pmo / mean[best], "masks": {m: mean[m] for m in msk},
                  "matrices": len(mats)}
    return out


def lufit(files):
    rows = [r for r in load(files) if "|" not in r["method"]]
    x = np.array([r["lu_nnz"] for r in rows], float) if "lu_nnz" in rows[0] else np.array([r["lu_ratio"] * r["nnz"] for r in rows])
    y = np.array([r["t_lu"] for r in rows])
    c = np.array([r["t_cycle_coarse"] for r in rows])
    cn = np.array([r["coarse_n"] for r in rows], float)
    a = float(x @ y / (x @ x))
    b = float(x @ c / (x @ x))
    r2 = lambda t, f: float(1 - ((t - f) ** 2).sum() / ((t - t.mean()) ** 2).sum())  # noqa: E731
    e = np.polyfit(np.log(cn), np.log(x), 1)
    return {"rows": len(rows), "t_lu_per_lu_nnz": a, "r2_lu": r2(y, a * x), "t_coarse_cycle_per_lu_nnz": b,
            "r2_cycle": r2(c, b * x), "lu_nnz_vs_coarse_n_exponent": float(e[0])}


def compare(new_file, old_files):
    """per method present in both: identical iterations / coarse size / LU nnz per matrix, and mean totals
    (over the common matrices) of the regenerated and the committed run."""
    new = _by_method(load([new_file]))
    old = _by_method(load(old_files))
    out = []
    for m in new:
        if m not in old:
            continue
        ks = sorted(set(new[m]) & set(old[m]))
        same = {q: sum(new[m][k][q] == old[m][k][q] for k in ks) for q in ("iters", "coarse_n", "lu_nnz")
                if all(q in new[m][k] and q in old[m][k] for k in ks)}
        tn = float(np.mean([new[m][k]["total"] for k in ks]))
        to = float(np.mean([old[m][k]["total"] for k in ks]))
        out.append({"method": m, "matrices": len(ks), "identical": same, "iters_new": float(np.mean([new[m][k]["iters"] for k in ks])),
                    "iters_old": float(np.mean([old[m][k]["iters"] for k in ks])), "total_new": tn, "total_old": to,
                    "ratio": tn / to})
    return out


def compare_text(out):
    lines = [f"{'method':60s} {'mats':>4s} {'same it/cn/lu':>14s} {'iters new/old':>14s} {'total new/old':>16s} {'ratio':>6s}"]
    for v in out:
        sm = "/".join(str(v["identical"].get(q, "-")) for q in ("iters", "coarse_n", "lu_nnz"))
        lines.append(f"{v['method'][-60:]:60s} {v['matrices']:4d} {sm:>14s} {v['iters_new']:6.1f}/{v['iters_old']:<6.1f} "
                     f"{v['total_new']:7.3f}/{v['total_old']:<7.3f} {v['ratio']:6.3f}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["table", "large", "paired", "vsbest", "vsref", "winners", "headroom", "lufit", "compare"])
    ap.add_argument("args", nargs="+")
    ap.add_argument("--json", action="store_true", help="print JSON instead of markdown/text")
    a = ap.parse_args(argv)
    if a.cmd == "table":
        print(table(a.args))
    elif a.cmd == "large":
        print(large(a.args))
    elif a.cmd == "paired":
        res, txt = paired([a.args[0]], a.args[1], a.args[2])
        print(json.dumps(res, indent=1) if a.json else txt)
    elif a.cmd == "compare":
        out = compare(a.args[0], a.args[1:])
        print(json.dumps(out, indent=1) if a.json else compare_text(out))
    elif a.cmd in ("vsbest", "vsref"):
        ref, files = (None, a.args) if a.cmd == "vsbest" else (a.args[0], a.args[1:])
        ref, out = vs(files, ref)
        print(json.dumps({"ref": ref, "rows": out}, indent=1) if a.json else vs_markdown(ref, out))
    else:
        fn = {"winners": winners, "headroom": headroom, "lufit": lufit}[a.cmd]
        print(json.dumps(fn(a.args), indent=1))


if __name__ == "__main__":
    sys.exit(main())
