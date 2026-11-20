#!/usr/bin/env python3
"""
evaluate_preconditioners.py — Post-training preconditioner benchmark.

For each test sample, benchmark 6 preconditioner schemes:
  {none, jacobi, ilu, pyamg, ruge_stuben, learned}

For each scheme, collect:
  1. Per-step PCG residuals (for convergence plots)
  2. Approximate eigenvalues (Ritz values) of preconditioned operator
  3. Condition number estimates (raw + effective)
  4. Preconditioner setup time (wall clock)
  5. PCG solve time to convergence (wall clock)

Learned setup time: for threshold-mode policies the learned path runs through
fast_policy.FastPolicy and the timed setup covers everything from A to the
hierarchy -- relaxed probes and node features, the policy, edge selection, C,
and the SA setup. The reference path (used for other policies, or with
--fast_policy off) times policy + selection + SA setup on precomputed features.
The mode is recorded in preconditioner_benchmark/learned_setup.json.

Results saved as .pt (full arrays) and .json + .npz (scalars + arrays).

Usage (standalone):
  python evaluate_preconditioners.py \
      --checkpoint outputs/es_hpc_XXXXX/model_best.pt \
      --out_dir outputs/es_hpc_XXXXX \
      --test_samples 32 --seed 1000

Usage (from train_es_HPC.py):
  from evaluate_preconditioners import benchmark_preconditioners
  benchmark_preconditioners(model, test_set, args)
"""

import os
import json
import time
import argparse
from typing import List, Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

import pyamg

from amg_core import (
    FeatureConfig,
    sample_topk_without_replacement,
    select_edges_by_threshold,
    C_from_selected_edges,
    build_pyamg_solver,
    build_B_for_pyamg,
    fixed_np_rng,
    compute_device,
    build_row_groups,
    pcg_iterations,
    estimate_ritz_values,
    _pcg_ritz_values,
    spectral_metrics,
)
from fast_policy import FastPolicy

SCHEMES = ["none", "jacobi", "ilu", "pyamg", "ruge_stuben", "learned"]


# ============================================================
# Preconditioner builders
# ============================================================
def _build_preconditioner(
    A: sp.csr_matrix,
    scheme: str,
    model=None,
    sample=None,
    k_per_row: int = 4,
    selection_mode: str = "topk",
    theta_rs: float = 0.25,
) -> Tuple[Optional[spla.LinearOperator], Optional[object], float]:
    """Build a preconditioner and return (M, ml_or_None, setup_time).

    Args:
        A: System matrix (SPD, csr).
        scheme: One of SCHEMES.
        model: Trained AMGEdgePolicy (required for 'learned').
        sample: SpectralTrainSample (required for 'learned').
        k_per_row: Edges per row for learned top-k.
        selection_mode: 'topk' or 'threshold'.
        theta_rs: Strength threshold for Ruge-Stuben.

    Returns:
        M: LinearOperator or None (for unpreconditioned CG).
        ml: PyAMG multilevel solver (for AMG schemes) or None.
        setup_time: Wall-clock seconds for preconditioner construction.
    """
    n = A.shape[0]
    t0 = time.perf_counter()

    if scheme == "none":
        M, ml = None, None

    elif scheme == "jacobi":
        diag = A.diagonal().copy()
        diag[diag == 0] = 1.0  # guard against zero diagonal
        inv_diag = 1.0 / diag
        M = spla.LinearOperator(
            (n, n), matvec=lambda x: inv_diag * x, dtype=A.dtype
        )
        ml = None

    elif scheme == "ilu":
        ilu = spla.spilu(A.tocsc())
        M = spla.LinearOperator(
            (n, n), matvec=ilu.solve, dtype=A.dtype
        )
        ml = None

    elif scheme == "pyamg":
        # Same seeded spectral-radius estimate as the learned hierarchy
        # (build_pyamg_solver), so SA-AMG is deterministic too.
        with fixed_np_rng(0):
            ml = pyamg.smoothed_aggregation_solver(A)
        M = ml.aspreconditioner(cycle="V")

    elif scheme == "ruge_stuben":
        ml = pyamg.ruge_stuben_solver(
            A, strength=("classical", {"theta": theta_rs})
        )
        M = ml.aspreconditioner(cycle="V")

    elif scheme == "learned":
        if model is None or sample is None:
            raise ValueError("'learned' scheme requires model and sample")
        fp = getattr(model, "_fast_policy", None)
        if fp is not None and getattr(sample, "feature_seed", None) is not None:
            # A -> features -> policy -> C -> SA hierarchy, all timed.
            fast, feat_cfg = fp
            if fast.dev.type == "cuda":
                torch.cuda.synchronize()
            t0_learned = time.perf_counter()
            prep = fast.prepare(A, feat_cfg, sample.feature_seed)
            C = fast.strength(prep)
            ml = build_pyamg_solver(A, C, None, coarse_solver="splu")
            M = ml.aspreconditioner(cycle="V")
            return M, ml, time.perf_counter() - t0_learned
        # The benchmark loop already put the model on its device
        device = next(model.parameters()).device
        raw_edge_features = getattr(sample, "raw_edge_features", None)

        # Model should already be on device (kept there across samples)
        if next(model.parameters()).device != device:
            model.to(device)

        x_dev = sample.x.to(device)
        ei_dev = sample.edge_index.to(device)
        ew_dev = sample.edge_weight.to(device)
        ref_dev = (raw_edge_features.to(device)
                   if raw_edge_features is not None else None)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0_learned = time.perf_counter()

        with torch.no_grad():
            logits, B_extra = model(
                x_dev, ei_dev, ew_dev,
                raw_edge_features=ref_dev,
            )
        if selection_mode == "threshold":
            threshold = (model.threshold.item()
                         if hasattr(model, "threshold") else 0.0)
            sel_mask, _ = select_edges_by_threshold(
                logits=logits, edge_index=ei_dev,
                threshold=threshold,
            )
        else:
            row_groups = sample.row_groups
            sel_mask, _ = sample_topk_without_replacement(
                logits=logits, row_groups=row_groups,
                k=k_per_row, temperature=1.0, gumbel=False,
            )

        if device.type == "cuda":
            torch.cuda.synchronize()

        C = C_from_selected_edges(
            A, sample.edge_index, sel_mask.cpu().numpy()
        )
        B = build_B_for_pyamg(B_extra)
        ml = build_pyamg_solver(A, C, B, coarse_solver="splu")
        M = ml.aspreconditioner(cycle="V")

        setup_time = time.perf_counter() - t0_learned
        return M, ml, setup_time

    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    setup_time = time.perf_counter() - t0
    return M, ml, setup_time


# ============================================================
# Eigenvalue analysis for arbitrary preconditioner
# ============================================================
def _eigenvalue_analysis(
    A: sp.csr_matrix,
    M,
    ml,
    num_iters: int = 50,
    num_probes: int = 4,
) -> Dict:
    """Compute Ritz values and spectral metrics.

    For AMG schemes (ml is not None), uses estimate_ritz_values.
    For non-AMG schemes, uses _pcg_ritz_values directly with M.
    """
    if ml is not None:
        ritz = estimate_ritz_values(A, ml, num_iters=num_iters,
                                    num_probes=num_probes)
    elif M is not None:
        # M is a LinearOperator — use _pcg_ritz_values directly
        all_ritz = []
        for s in range(num_probes):
            ritz_s = _pcg_ritz_values(A, M, num_iters=num_iters, seed=s + 7)
            all_ritz.append(ritz_s)
        ritz = np.sort(np.concatenate(all_ritz))
    else:
        # Unpreconditioned: B = I
        I_op = spla.LinearOperator(
            (A.shape[0], A.shape[0]),
            matvec=lambda x: x,
            dtype=A.dtype,
        )
        all_ritz = []
        for s in range(num_probes):
            ritz_s = _pcg_ritz_values(A, I_op, num_iters=num_iters,
                                      seed=s + 7)
            all_ritz.append(ritz_s)
        ritz = np.sort(np.concatenate(all_ritz))

    metrics = spectral_metrics(ritz)
    metrics["ritz_values"] = ritz
    return metrics


# ============================================================
# V-cycle convergence factor rho(I - B A)
# ============================================================
def vcycle_convergence_factor(A: sp.csr_matrix, ml, iters: int = 40,
                              skip: int = 10, seed: int = 0) -> float:
    """Asymptotic convergence factor of the stationary V-cycle iteration.

    B is one V-cycle with zero initial guess (ml.aspreconditioner), so the
    error propagates as e <- (I - B A) e. Power iteration on that operator,
    renormalised every step in the A-norm; the factor is the geometric mean
    of the per-step A-norm reductions after `skip` warm-up steps. This is the
    quantity the paper calls rho_V; the figure previously plotted under that
    name was a PCG residual tail ratio (audit item I8).
    """
    rng = np.random.default_rng(seed)
    M = ml.aspreconditioner(cycle="V")
    e = rng.standard_normal(A.shape[0])
    e /= np.sqrt(e @ (A @ e))
    log_red = []
    for k in range(iters):
        e = e - M.matvec(A @ e)
        nrm = float(np.sqrt(max(e @ (A @ e), 0.0)))
        if nrm == 0.0 or not np.isfinite(nrm):
            return 0.0 if nrm == 0.0 else float("nan")
        if k >= skip:
            log_red.append(np.log(nrm))
        e /= nrm
    return float(np.exp(np.mean(log_red)))


# ============================================================
# Benchmark one sample across all schemes
# ============================================================
def _benchmark_single_sample(
    A: sp.csr_matrix,
    sample,
    model,
    schemes: List[str],
    k_per_row: int = 4,
    selection_mode: str = "topk",
    theta_rs: float = 0.25,
    pcg_tol: float = 1e-8,
    pcg_maxiter: int = 500,
    ritz_iters: int = 50,
    ritz_probes: int = 4,
) -> Dict[str, Dict]:
    """Benchmark all preconditioner schemes on a single sample.

    Returns dict mapping scheme name to result dict with keys:
        iters, residuals, ritz_values, kappa_raw, kappa_eff,
        setup_time, solve_time, op_complexity, grid_complexity
    """
    results = {}

    for scheme in schemes:
        # 1. Build preconditioner (timed)
        try:
            M, ml, setup_time = _build_preconditioner(
                A, scheme, model=model, sample=sample,
                k_per_row=k_per_row, selection_mode=selection_mode,
                theta_rs=theta_rs,
            )
        except Exception as e:
            print(f"    WARNING: {scheme} setup failed: {e}")
            results[scheme] = {
                "iters": pcg_maxiter,
                "residuals": [],
                "ritz_values": np.array([1.0]),
                "kappa_raw": float("inf"),
                "kappa_eff": float("inf"),
                "setup_time": 0.0,
                "solve_time": 0.0,
                "op_complexity": 0.0,
                "grid_complexity": 0.0,
                "rho_vcycle": float("nan"),
                "error": str(e),
            }
            continue

        # 2. PCG to convergence (timed)
        t0 = time.perf_counter()
        iters, residuals = pcg_iterations(
            A, M, tol=pcg_tol, maxiter=pcg_maxiter
        )
        solve_time = time.perf_counter() - t0

        # 3. Eigenvalue analysis
        try:
            eig_info = _eigenvalue_analysis(
                A, M, ml,
                num_iters=ritz_iters, num_probes=ritz_probes,
            )
        except Exception as e:
            print(f"    WARNING: {scheme} eigenvalue analysis failed: {e}")
            eig_info = {
                "ritz_values": np.array([1.0]),
                "kappa_raw": float("inf"),
                "kappa_eff": float("inf"),
            }

        # 4. Operator / grid complexity (AMG only)
        oc, gc = 0.0, 0.0
        if ml is not None:
            A0_nnz = ml.levels[0].A.nnz
            total_nnz = sum(lvl.A.nnz for lvl in ml.levels)
            oc = float(total_nnz / max(A0_nnz, 1))
            n0 = ml.levels[0].A.shape[0]
            total_n = sum(lvl.A.shape[0] for lvl in ml.levels)
            gc = float(total_n / max(n0, 1))

        # 5. V-cycle convergence factor (AMG only, untimed)
        rho_v = float("nan")
        if ml is not None:
            try:
                rho_v = vcycle_convergence_factor(A, ml)
            except Exception as e:
                print(f"    WARNING: {scheme} rho_V failed: {e}")

        results[scheme] = {
            "iters": iters,
            "residuals": [float(r) for r in residuals],
            "ritz_values": eig_info["ritz_values"],
            "kappa_raw": eig_info["kappa_raw"],
            "kappa_eff": eig_info["kappa_eff"],
            "setup_time": setup_time,
            "solve_time": solve_time,
            "op_complexity": oc,
            "grid_complexity": gc,
            "rho_vcycle": rho_v,
        }

    return results


# ============================================================
# Main benchmark entry point
# ============================================================
def benchmark_preconditioners(
    model,
    test_set: list,
    args,
    schemes: Optional[List[str]] = None,
    out_dir: Optional[str] = None,
) -> Dict:
    """Benchmark all preconditioner schemes on every test sample.

    Args:
        model: Trained AMGEdgePolicy (eval mode).
        test_set: List of SpectralTrainSample.
        args: Namespace with k_per_row, selection_mode, out_dir.
        schemes: List of schemes to benchmark (default: all 6).
        out_dir: Override output directory.

    Returns:
        Full results dict (also saved to disk).
    """
    if schemes is None:
        schemes = list(SCHEMES)

    save_dir = out_dir or getattr(args, "out_dir", ".")
    save_dir = os.path.join(save_dir, "preconditioner_benchmark")
    os.makedirs(save_dir, exist_ok=True)

    k_per_row = getattr(args, "k_per_row", 4)
    selection_mode = getattr(args, "selection_mode", "topk")
    theta_rs = getattr(args, "theta_rs", 0.25)

    model.eval()
    device = compute_device()
    model.to(device)
    if device.type == "cuda":
        print(f"  Model on GPU: {torch.cuda.get_device_name()}")

    # Fast exact inference for threshold-mode policies (see fast_policy.py).
    feat_cfg = FeatureConfig.from_preset(
        getattr(args, "feature_preset", "new_algebraic_plus_relaxed"))
    model._fast_policy = None
    use_fast = (selection_mode == "threshold"
                and getattr(args, "fast_policy", "auto") != "off"
                and FastPolicy.supported(model, feat_cfg)
                and all(getattr(s, "feature_seed", None) is not None for s in test_set))
    if use_fast:
        model._fast_policy = (FastPolicy(model, device), feat_cfg)
    learned_mode = ("fast_policy: features + policy + selection + C + SA setup"
                    if use_fast else
                    "reference: policy + selection + C + SA setup (features precomputed, untimed)")
    print(f"  Learned setup timing: {learned_mode}")
    with open(os.path.join(save_dir, "learned_setup.json"), "w") as f:
        json.dump({"fast_policy": use_fast, "timed": learned_mode,
                   "device": str(device)}, f, indent=2)

    # Untimed warm-up of the learned path. The first CUDA forward pays for
    # context/kernel initialisation (0.3-1.2 s measured on V100s); timing
    # it inflated the mean Learned-AMG setup time reported in the paper.
    if "learned" in schemes and len(test_set) > 0:
        for _ in range(2):
            _build_preconditioner(
                test_set[0].A, "learned", model=model, sample=test_set[0],
                k_per_row=getattr(args, "k_per_row", 4),
                selection_mode=getattr(args, "selection_mode", "topk"),
                theta_rs=getattr(args, "theta_rs", 0.25),
            )

    n_samples = len(test_set)
    print(f"\n{'='*70}")
    print(f"  Preconditioner Benchmark: {n_samples} samples × "
          f"{len(schemes)} schemes")
    print(f"  Schemes: {schemes}")
    print(f"  Output: {save_dir}")
    print(f"{'='*70}")

    all_results = {}
    # Summary table header
    header = (f"  {'sample':>6s}  {'n':>6s}  {'nnz':>8s}  "
              + "  ".join(f"{s:>12s}" for s in schemes))
    print(f"\n  PCG iterations to convergence (tol=1e-8):")
    print(header)
    print(f"  {'-'*len(header.strip())}")

    for idx, sample in enumerate(test_set):
        A = sample.A
        n = A.shape[0]
        nnz = A.nnz

        results = _benchmark_single_sample(
            A, sample, model,
            schemes=schemes,
            k_per_row=k_per_row,
            selection_mode=selection_mode,
            theta_rs=theta_rs,
        )

        all_results[idx] = {
            "matrix_size": n,
            "matrix_nnz": nnz,
            "schemes": results,
        }

        # Print summary row
        iter_strs = []
        for s in schemes:
            r = results[s]
            iters = r["iters"]
            t_total = r["setup_time"] + r["solve_time"]
            iter_strs.append(f"{iters:4d} ({t_total:5.3f}s)")
        row = (f"  {idx:6d}  {n:6d}  {nnz:8d}  "
               + "  ".join(f"{s:>12s}" for s in iter_strs))
        print(row)

    # ── Aggregate summary ──
    print(f"\n  {'':>6s}  {'':>6s}  {'':>8s}  "
          + "  ".join(f"{s:>12s}" for s in schemes))
    mean_iters = {}
    mean_setup = {}
    mean_solve = {}
    for s in schemes:
        iters_list = [all_results[i]["schemes"][s]["iters"]
                      for i in range(n_samples)]
        setup_list = [all_results[i]["schemes"][s]["setup_time"]
                      for i in range(n_samples)]
        solve_list = [all_results[i]["schemes"][s]["solve_time"]
                      for i in range(n_samples)]
        mean_iters[s] = np.mean(iters_list)
        mean_setup[s] = np.mean(setup_list)
        mean_solve[s] = np.mean(solve_list)

    print(f"  {'MEAN':>6s}  {'':>6s}  {'':>8s}  "
          + "  ".join(f"{mean_iters[s]:4.0f} ({mean_setup[s]+mean_solve[s]:5.3f}s)"
                      .rjust(12) for s in schemes))

    # Kappa summary
    print(f"\n  Condition number (kappa_eff):")
    print(f"  {'sample':>6s}  "
          + "  ".join(f"{s:>12s}" for s in schemes))
    for idx in range(n_samples):
        kappas = []
        for s in schemes:
            k = all_results[idx]["schemes"][s]["kappa_eff"]
            kappas.append(f"{k:12.1f}" if k < 1e12 else f"{'inf':>12s}")
        print(f"  {idx:6d}  " + "  ".join(kappas))

    # Timing summary
    print(f"\n  Timing breakdown (seconds):")
    print(f"  {'scheme':<14s}  {'setup':>8s}  {'solve':>8s}  {'total':>8s}")
    print(f"  {'-'*14}  {'-'*8}  {'-'*8}  {'-'*8}")
    for s in schemes:
        print(f"  {s:<14s}  {mean_setup[s]:8.4f}  {mean_solve[s]:8.4f}  "
              f"{mean_setup[s]+mean_solve[s]:8.4f}")

    # ── Save results ──
    _save_results(all_results, save_dir, schemes, n_samples)

    # Move model back to CPU after benchmark
    model._fast_policy = None
    model.to("cpu")

    print(f"\n  Results saved to: {save_dir}")
    return all_results


# ============================================================
# Save results in dual format (.pt + .json + .npz)
# ============================================================
def _save_results(
    all_results: Dict,
    save_dir: str,
    schemes: List[str],
    n_samples: int,
):
    # --- .pt (full dict, easy torch.load) ---
    # Convert numpy arrays for torch compatibility
    pt_results = {}
    for idx in range(n_samples):
        pt_results[idx] = {
            "matrix_size": all_results[idx]["matrix_size"],
            "matrix_nnz": all_results[idx]["matrix_nnz"],
            "schemes": {},
        }
        for s in schemes:
            r = all_results[idx]["schemes"][s]
            pt_results[idx]["schemes"][s] = {
                "iters": r["iters"],
                "residuals": np.array(r["residuals"], dtype=np.float64),
                "ritz_values": np.array(r["ritz_values"], dtype=np.float64),
                "kappa_raw": r["kappa_raw"],
                "kappa_eff": r["kappa_eff"],
                "setup_time": r["setup_time"],
                "solve_time": r["solve_time"],
                "op_complexity": r["op_complexity"],
                "grid_complexity": r["grid_complexity"],
                "rho_vcycle": r.get("rho_vcycle", float("nan")),
            }
    torch.save(pt_results, os.path.join(save_dir, "benchmark_results.pt"))

    # --- .json (scalar summaries) ---
    json_results = {}
    for idx in range(n_samples):
        json_results[str(idx)] = {
            "matrix_size": all_results[idx]["matrix_size"],
            "matrix_nnz": all_results[idx]["matrix_nnz"],
            "schemes": {},
        }
        for s in schemes:
            r = all_results[idx]["schemes"][s]
            json_results[str(idx)]["schemes"][s] = {
                "iters": r["iters"],
                "kappa_raw": float(r["kappa_raw"]),
                "kappa_eff": float(r["kappa_eff"]),
                "setup_time": float(r["setup_time"]),
                "solve_time": float(r["solve_time"]),
                "op_complexity": float(r["op_complexity"]),
                "grid_complexity": float(r["grid_complexity"]),
                "rho_vcycle": float(r.get("rho_vcycle", float("nan"))),
            }
    with open(os.path.join(save_dir, "benchmark_results.json"), "w") as f:
        json.dump(json_results, f, indent=2)

    # --- .npz (all arrays) ---
    arrays = {}
    for idx in range(n_samples):
        for s in schemes:
            r = all_results[idx]["schemes"][s]
            arrays[f"{idx}_{s}_residuals"] = np.array(
                r["residuals"], dtype=np.float64
            )
            arrays[f"{idx}_{s}_ritz"] = np.array(
                r["ritz_values"], dtype=np.float64
            )
    np.savez_compressed(
        os.path.join(save_dir, "benchmark_arrays.npz"), **arrays
    )


# ============================================================
# Standalone CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark preconditioners on test problems"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model_best.pt")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory (default: same as checkpoint)")
    parser.add_argument("--test_samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--grid_n", type=int, default=32)
    parser.add_argument("--n_edge", type=int, default=72)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--feature_preset", type=str,
                        default="new_algebraic_plus_relaxed")
    parser.add_argument("--k_per_row", type=int, default=4)
    parser.add_argument("--selection_mode", type=str, default="topk",
                        choices=["topk", "threshold"])
    parser.add_argument("--learn_B", action="store_true", default=False)
    parser.add_argument("--init_threshold", type=float, default=0.0)
    parser.add_argument("--theta_rs", type=float, default=0.25,
                        help="Strength threshold for Ruge-Stuben")
    parser.add_argument("--schemes", type=str, nargs="+", default=None,
                        help="Subset of schemes to run "
                             "(default: all 6)")
    parser.add_argument("--fast_policy", type=str, default="auto",
                        choices=["auto", "off"],
                        help="auto: exact fast inference (fast_policy.py) for "
                             "threshold-mode policies; off: reference path")
    parser.add_argument("--problem", type=str, default="lognormal",
                        choices=["lognormal", "anisotropy", "poisson",
                                 "perforated_l"],
                        help="Problem type for test set generation")
    args = parser.parse_args()

    # ── Resolve output directory ──
    if args.out_dir is None:
        args.out_dir = os.path.dirname(args.checkpoint)

    # ── Load config from training run if available ──
    config_path = os.path.join(args.out_dir, "config.json")
    if os.path.isfile(config_path):
        print(f"Loading config from: {config_path}")
        with open(config_path) as f:
            cfg = json.load(f)
        # Override defaults with training config (CLI takes precedence)
        for key in ["hidden", "feature_preset", "k_per_row",
                     "selection_mode", "learn_B", "grid_n", "n_edge",
                     "init_threshold", "problem"]:
            if key in cfg and key not in sys.argv:
                setattr(args, key, cfg[key])
        print(f"  hidden={args.hidden}, k_per_row={args.k_per_row}, "
              f"selection_mode={args.selection_mode}")

    # ── Feature config ──
    feat_cfg = FeatureConfig.from_preset(args.feature_preset)

    # ── Build model ──
    from models import AMGEdgePolicy
    model = AMGEdgePolicy(
        in_channels=feat_cfg.node_dim,
        hidden=args.hidden,
        learn_B=args.learn_B,
        edge_raw_dim=feat_cfg.edge_raw_dim,
        adaptive_k=(args.selection_mode == "threshold"),
        init_threshold=args.init_threshold,
    )

    # ── Load weights ──
    state_dict = torch.load(args.checkpoint, map_location="cpu",
                            weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # ── Build test set ──
    from problems import make_spectral_train_set

    print(f"Building {args.test_samples} test problems "
          f"(seed={args.seed}, n_edge={args.n_edge})...")
    t0 = time.time()
    test_set = make_spectral_train_set(
        num_samples=args.test_samples,
        n_grid=args.grid_n,
        seed=args.seed,
        feature_config=feat_cfg,
        n_edge=args.n_edge,
        problem=args.problem,
    )
    print(f"  Done in {time.time()-t0:.1f}s. "
          f"Matrix size: {test_set[0].A.shape[0]}")

    # ── Run benchmark ──
    schemes = args.schemes if args.schemes else list(SCHEMES)
    benchmark_preconditioners(model, test_set, args, schemes=schemes)


if __name__ == "__main__":
    import sys
    main()
