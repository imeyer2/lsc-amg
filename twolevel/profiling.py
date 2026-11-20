"""Cost breakdowns on one matrix, 1 pinned core (sub-study "bar").

  python -m twolevel.profiling features --problem anisotropy --n 50000 --model <model_best.pt> --out f.json
      learned inference cost of the paper's GAT policy (FastPolicy): pattern maps, relaxed probes, row stats,
      node features, forward, selection, C; on the GPU also the relaxed probes computed on the device and the
      resulting change of the selection (max |diff|, flipped edges, iterations); forward cost of smaller
      untrained networks (timing only).
  python -m twolevel.profiling setup --problem anisotropy --n 50000 --methods theta:0.1 topk:1 --out s.json
      cProfile of the two-level SA setup + coarse splu + PCG per method (cumulative seconds per phase).
"""
import os
import time
import json
import argparse
import cProfile
import pstats
import io

from . import harness as H
from .common import set_paths, load_A
import numpy as np


def _pin():
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {sorted(os.sched_getaffinity(0))[0]})


def tmed(f, reps=5):
    import torch
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        r = f()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), r


def jacobi_probes_torch(A, K, iters, seed, dev, omega=2.0 / 3.0):
    """relaxed_smooth_vectors on the device: x <- x - omega D^-1 (A x) (b = 0), all K vectors at once."""
    import torch
    rng = np.random.default_rng(seed)
    X0 = rng.standard_normal((A.shape[0], K))
    Ad = torch.sparse_csr_tensor(torch.from_numpy(A.indptr.astype(np.int64)), torch.from_numpy(A.indices.astype(np.int64)),
                                 torch.from_numpy(A.data), size=A.shape).to(dev)
    dinv = torch.from_numpy(1.0 / A.diagonal()).to(dev)[:, None]
    X = torch.from_numpy(X0).to(dev)
    for _ in range(iters):
        X = X - omega * dinv * (Ad @ X)
    X = X / (X.abs().max() + 1e-6)
    return X.cpu().numpy()


def cmd_features(a):
    import torch
    _pin()
    torch.set_num_threads(1)
    from amg_core import relaxed_smooth_vectors, compute_row_stats, build_node_features
    from fast_policy import Graph, Prepared, FastPolicy
    from models import AMGEdgePolicy
    A, fs = load_A(set_paths(a.problem, a.n, a.seed, a.idx + 1)[a.idx])
    rep = {"n": A.shape[0], "nnz": A.nnz, "model": a.model}
    for dev in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        fp, m, fc = H.get_fast(a.model, dev)
        fp.threshold = float(m.threshold.item())
        d = {}
        d["graph"], g = tmed(lambda: Graph(A, dev))
        d["probes_pyamg"], R = tmed(lambda: relaxed_smooth_vectors(g.A, num_vecs=4, iters=5, seed=fs)[:, :4])
        d["rowstats"], rs = tmed(lambda: compute_row_stats(g.A))
        d["nodefeat"], x = tmed(lambda: build_node_features(g.A, None, fc, row_stats=rs, R=R))
        p = Prepared(g, x, R, rs)
        d["prepare_total"], _ = tmed(lambda: fp.prepare(A, fc, fs))
        d["logits"], _ = tmed(lambda: fp.logits(p))
        d["selection_incl_logits"], sel = tmed(lambda: fp.directed_selection(p))
        d["C"], C = tmed(lambda: fp.strength_from_selection(p, sel))
        if dev == "cuda":
            d["probes_gpu"], Rg = tmed(lambda: jacobi_probes_torch(g.A, 4, 5, fs, dev))
            d["probes_gpu_maxdiff"] = float(np.abs(Rg - R).max())
            xg = build_node_features(g.A, None, fc, row_stats=rs, R=Rg)
            selg = fp.directed_selection(Prepared(g, xg, Rg, rs))
            d["probes_gpu_sel_flips"] = int((selg != sel).sum())
            d["n_entries"] = int(len(sel))
            Cg = fp.strength_from_selection(p, selg)
            d["iters_cpu_probes"] = H.setup_solve(A, C)[1]["iters"]
            d["iters_gpu_probes"] = H.setup_solve(A, Cg)[1]["iters"]
        sm = {}
        for h in (8, 16, 32, 64):
            torch.manual_seed(0)
            mm = AMGEdgePolicy(in_channels=fc.node_dim, hidden=h, edge_raw_dim=fc.edge_raw_dim, adaptive_k=True,
                               edge_hidden=(2 * h, h)).eval()
            f2 = FastPolicy(mm, device=dev)
            t, _ = tmed(lambda: f2.directed_selection(p))
            sm[h] = {"params": int(sum(q.numel() for q in mm.parameters())), "fwd_s": t}
        d["small_nets"] = sm
        rep[dev] = d
        print(dev, json.dumps(d), flush=True)
    json.dump(rep, open(a.out, "w"), indent=1)


def cmd_setup(a):
    _pin()
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    A, fs = load_A(set_paths(a.problem, a.n, a.seed, a.idx + 1)[a.idx])
    rep = {}
    keys = ["smoothed_aggregation_solver", "_extend_hierarchy", "standard_aggregation", "fit_candidates",
            "approximate_spectral_radius", "jacobi_prolongation_smoother", "change_smoothers", "splu",
            "pcg_iteration_count", "cg", "matvec", "solve", "__matmul__", "_matmul_sparse", "tocsc", "gauss_seidel",
            "bsr_gauss_seidel", "block_gauss_seidel", "_mul_sparse_matrix", "symmetric_rescaling", "eliminate_zeros"]
    for m in a.methods:
        H.run_pipeline(A, fs, m)            # warm-up
        C, r = H.build_C(A, fs, m)
        pr = cProfile.Profile()
        pr.enable()
        ml, s = H.setup_solve(A, C, H.splu_opts(H.parse_method(m)[2]))
        pr.disable()
        st = pstats.Stats(pr)
        cum = {}
        for (fn, line, name), (cc, nc, tt, ct, callers) in st.stats.items():
            cum[name] = cum.get(name, 0.0) + ct
        d = {k: cum.get(k, 0.0) for k in keys}
        d.update(r)
        d.update(s)
        d.update(H.hier_stats(ml, A))
        rep[m] = d
        s_io = io.StringIO()
        pstats.Stats(pr, stream=s_io).sort_stats("cumulative").print_stats(28)
        print(f"===== {m}\n" + json.dumps({k: round(v, 4) for k, v in d.items() if isinstance(v, float)}) + "\n"
              + s_io.getvalue(), flush=True)
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("features", "setup"):
        s = sub.add_parser(name)
        s.add_argument("--problem", required=True)
        s.add_argument("--n", type=int, default=50000)
        s.add_argument("--seed", type=int, default=1000)
        s.add_argument("--idx", type=int, default=0)
        s.add_argument("--out", required=True)
        if name == "features":
            s.add_argument("--model", required=True)
        else:
            s.add_argument("--methods", nargs="+", required=True)
    a = ap.parse_args(argv)
    {"features": cmd_features, "setup": cmd_setup}[a.cmd](a)


if __name__ == "__main__":
    main()
