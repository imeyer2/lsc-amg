"""
amg_diagnostics.py — Theory-backed AMG quality diagnostics.

Four metrics, each with a direct literature citation proving it
determines AMG convergence:

  M1. Compatible Relaxation (CR) Rate
      Falgout & Vassilevski, SIAM J. Numer. Anal. 2004

  M2. Two-Grid Convergence Factor
      Xu & Zikatanov, Acta Numerica 2017 (arXiv:1611.01917)

  M3. Weak Approximation Property (WAP) Error
      MacLachlan & Olson, Numer. Linear Algebra Appl. 2014, Thm 3.1

  M4. Near-Nullspace Preservation
      Vanek, Mandel, Brezina, Computing 1996

All metrics are diagnostic-only — they never affect training loss.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import pyamg
from amg_core import energy_norm_sq, relaxed_smooth_vectors


# ============================================================
# M1. Compatible Relaxation (CR) Rate
# ============================================================
def compute_cr_rate(
    A: sp.csr_matrix,
    ml: pyamg.multilevel.MultilevelSolver,
    relax_iters: int = 20,
    num_probes: int = 4,
    omega: float = 2.0 / 3.0,
    seed: int = 42,
) -> float:
    """Estimate the Compatible Relaxation convergence rate.

    From the SA hierarchy, extract the C/F splitting via AggOp:
    coarse nodes are aggregate seeds (columns of AggOp), fine nodes
    are the rest. Run F-point-only weighted Jacobi on Ax=0 and
    measure how fast the F-point error decays.

    CR rate < 0.5  →  good coarse grid
    CR rate ≈ 1.0  →  coarse grid cannot represent smooth error

    Reference: Falgout & Vassilevski, "On Generalizing the AMG
    Framework", SIAM J. Numer. Anal. 42(4), 2004.
    """
    n = A.shape[0]

    # Derive C/F splitting from P or AggOp.
    # AggOp is only available with keep=True; fall back to P sparsity.
    if hasattr(ml.levels[0], 'AggOp') and ml.levels[0].AggOp is not None:
        AggOp = ml.levels[0].AggOp
    else:
        # Use P's sparsity pattern: binarize to get an aggregation-like operator
        AggOp = (abs(ml.levels[0].P) > 0).astype(float)

    # Each row of AggOp has nonzeros in one (or more) columns.
    # Coarse node = first node assigned to each aggregate.
    agg_membership = np.array(AggOp.argmax(axis=1)).ravel()

    coarse_mask = np.zeros(n, dtype=bool)
    seen = set()
    for i in range(n):
        ag = agg_membership[i]
        if ag not in seen:
            coarse_mask[i] = True
            seen.add(ag)
    fine_mask = ~coarse_mask
    fine_idx = np.where(fine_mask)[0]
    n_fine = len(fine_idx)

    if n_fine == 0:
        return 0.0  # degenerate: everything is coarse

    # Diagonal for Jacobi
    diag_A = np.array(A.diagonal()).ravel()
    diag_A[diag_A == 0] = 1.0  # safety

    rng = np.random.default_rng(seed)
    rates = []

    for _ in range(num_probes):
        # Random initial error, zero on coarse points
        e = rng.standard_normal(n)
        e[coarse_mask] = 0.0
        norm0 = np.linalg.norm(e[fine_idx])
        if norm0 < 1e-15:
            continue

        # F-point-only weighted Jacobi relaxation on Ax=0
        for _ in range(relax_iters):
            r = -(A @ e)  # residual for Ax=0
            e_new = e + omega * r / diag_A
            e_new[coarse_mask] = 0.0  # keep coarse points at zero
            e = e_new

        norm_k = np.linalg.norm(e[fine_idx])
        if norm0 > 1e-15:
            rate = (norm_k / norm0) ** (1.0 / relax_iters)
            rates.append(rate)

    return float(np.mean(rates)) if rates else 1.0


# ============================================================
# M2. V-Cycle Convergence Factor
# ============================================================
def compute_twogrid_convergence_factor(
    A: sp.csr_matrix,
    P: sp.csr_matrix,
    ml: "pyamg.multilevel.MultilevelSolver | None" = None,
    num_probes: int = 4,
    vcycle_iters: int = 30,
    seed: int = 42,
) -> float:
    """Estimate the ASYMPTOTIC V-cycle convergence factor ρ.

    Runs many V-cycle iterations from random initial error on Ax=0.
    After enough iterations the algebraically smooth error dominates
    and the per-step ratio ||e_{k+1}||_A / ||e_k||_A converges to
    the spectral radius of the error propagation operator.

    IMPORTANT: a single V-cycle on a random vector mostly measures the
    smoother (ρ ≈ 0.07) — NOT the coarse-grid correction quality.
    The asymptotic rate after 20-30 iterations reveals the true
    worst-case convergence behaviour.

    ρ < 0.5  →  good convergence
    ρ > 0.9  →  poor convergence (coarse grid cannot fix smooth error)

    Reference: Xu & Zikatanov, "Algebraic Multigrid Methods",
    Acta Numerica 26, pp. 591-721, 2017.
    """
    if ml is None:
        # Fallback: build a solver from A and P
        Ac = (P.T @ A @ P).tocsc()
        try:
            import pyamg
            ml_tmp = pyamg.multilevel.MultilevelSolver(
                [pyamg.multilevel.Level() for _ in range(2)])
            ml_tmp.levels[0].A = A
            ml_tmp.levels[0].P = P
            ml_tmp.levels[0].R = P.T
            ml_tmp.levels[1].A = Ac
            ml_tmp.coarse_solver = pyamg.coarse_grid_solver("splu")
            pyamg.relaxation.smoothing.change_smoothers(
                ml_tmp, presmoother="gauss_seidel", postsmoother="gauss_seidel")
            ml = ml_tmp
        except Exception:
            return 1.0

    n = A.shape[0]
    rng = np.random.default_rng(seed)
    b0 = np.zeros(n)

    # Use the LAST few iterations' ratios (asymptotic regime)
    tail = min(5, vcycle_iters // 2)
    max_asymptotic_rho = 0.0

    for _ in range(num_probes):
        e = rng.standard_normal(n)
        ratios = []

        for _ in range(vcycle_iters):
            en_before = energy_norm_sq(A, e)
            if en_before < 1e-30:
                break
            try:
                e_after = ml.solve(b0, x0=e, tol=1e-30, maxiter=1,
                                   accel=None, cycle="V")
            except TypeError:
                e_after = ml.solve(b0, x0=e, tol=0.0, maxiter=1,
                                   accel=None, cycle="V")
            en_after = energy_norm_sq(A, e_after)
            rho_k = np.sqrt(max(en_after / en_before, 0.0))
            ratios.append(rho_k)
            e = e_after

        if len(ratios) >= tail:
            asymptotic = np.mean(ratios[-tail:])
        elif ratios:
            asymptotic = np.mean(ratios)
        else:
            asymptotic = 1.0

        max_asymptotic_rho = max(max_asymptotic_rho, asymptotic)

    return float(max_asymptotic_rho)


# ============================================================
# M2b. Condition Number of Preconditioned Operator
# ============================================================
def compute_preconditioned_condition_number(
    A: sp.csr_matrix,
    ml: "pyamg.multilevel.MultilevelSolver",
    n_eigenvalues: int = 10,
) -> dict:
    """Estimate κ(M⁻¹A) — the condition number that determines PCG iters.

    Uses ARPACK to find the largest and smallest eigenvalues of the
    preconditioned operator M⁻¹A.  The PCG iteration count scales
    as O(√κ · ln(1/ε)).

    Also detects **negative eigenvalues** (non-SPD preconditioner),
    which can make PCG fail or converge very slowly.

    Returns dict with keys:
        lambda_max, lambda_min, kappa, is_spd
    """
    M_inv = ml.aspreconditioner(cycle="V")
    n = A.shape[0]

    def mv(x):
        return M_inv @ (A @ x)

    MA = spla.LinearOperator((n, n), matvec=mv)
    k = min(n_eigenvalues, n - 2)

    try:
        eigs_large, _ = spla.eigsh(MA, k=k, which="LM")
    except Exception:
        eigs_large = np.array([1.0])

    try:
        eigs_small, _ = spla.eigsh(MA, k=k, which="SM")
    except Exception:
        eigs_small = np.array([1.0])

    lam_max = float(np.max(eigs_large))
    lam_min = float(np.min(eigs_small))
    is_spd = bool(lam_min > 0)
    kappa = lam_max / max(lam_min, 1e-30) if is_spd else float("inf")

    return {
        "lambda_max": lam_max,
        "lambda_min": lam_min,
        "kappa": kappa,
        "is_spd": is_spd,
    }



# ============================================================
# M3. Weak Approximation Property (WAP) Error
# ============================================================
def compute_wap_error(
    A: sp.csr_matrix,
    P: sp.csr_matrix,
    num_probes: int = 8,
    relax_iters: int = 10,
    seed: int = 42,
) -> float:
    """Measure the weak approximation property error.

    For algebraically smooth test vectors v (produced by Jacobi
    relaxation on Ax=0), compute the A-orthogonal projection error:

        WAP(v) = ||v - π_P v||_A / ||v||_A

    where π_P = P (P^T A P)^{-1} P^T A is the A-orthogonal
    projector onto Range(P).

    Returns the mean WAP error over num_probes smooth test vectors.
    Small WAP error directly bounds two-grid convergence.

    Reference: MacLachlan & Olson, "Theoretical Bounds for
    Algebraic Multigrid Performance: Review and Analysis",
    Numer. Linear Algebra Appl. 21(2), 2014, Theorem 3.1.
    """
    # Generate smooth test vectors
    V = relaxed_smooth_vectors(A, num_probes, relax_iters, seed=seed)

    # Build and factor coarse-grid operator
    Ac = (P.T @ A @ P).tocsc()
    try:
        Ac_solve = spla.factorized(Ac)
    except Exception:
        return float("inf")

    errors = []
    for k in range(num_probes):
        v = V[:, k]
        denom = energy_norm_sq(A, v)
        if denom < 1e-30:
            continue

        # A-orthogonal projection: π_P v = P (P^T A P)^{-1} P^T A v
        Av = A @ v
        PtAv = P.T @ Av
        vc = Ac_solve(PtAv)
        proj = P @ vc

        diff = v - proj
        num = energy_norm_sq(A, diff)
        errors.append(np.sqrt(max(num / denom, 0.0)))

    return float(np.mean(errors)) if errors else float("inf")


# ============================================================
# M4. Near-Nullspace Preservation
# ============================================================
def compute_nullspace_preservation(
    A: sp.csr_matrix,
    P: sp.csr_matrix,
    B: np.ndarray | None = None,
) -> dict:
    """Measure how well P preserves the near-nullspace candidates.

    For each near-nullspace vector b_i, compute:

        err_i = ||b_i - P P^† b_i||_A / ||b_i||_A

    where P^† = (P^T A P)^{-1} P^T A is the A-orthogonal
    pseudoinverse (same as the coarse-grid correction).

    SA-AMG is designed so that P exactly interpolates B
    (Vanek et al. 1996). Nonzero error means the hierarchy
    is fundamentally broken.

    Reference: Vanek, Mandel, Brezina, "Algebraic Multigrid
    by Smoothed Aggregation for Second and Fourth Order Elliptic
    Problems", Computing 56(3), pp. 179-196, 1996.
    """
    n = A.shape[0]
    if B is None:
        # Default near-nullspace for scalar PDE: constant vector
        B = np.ones((n, 1)) / np.sqrt(n)

    if B.ndim == 1:
        B = B.reshape(-1, 1)

    # Build and factor coarse-grid operator
    Ac = (P.T @ A @ P).tocsc()
    try:
        Ac_solve = spla.factorized(Ac)
    except Exception:
        return {"mean": float("inf"), "max": float("inf")}

    errors = []
    for k in range(B.shape[1]):
        b = B[:, k]
        denom = energy_norm_sq(A, b)
        # For constant vector on Laplacian, energy norm is 0 — use 2-norm
        if denom < 1e-30:
            denom_2 = float(b @ b)
            if denom_2 < 1e-30:
                continue
            # Project via A-orthogonal projection
            Ab = A @ b
            PtAb = P.T @ Ab
            bc = Ac_solve(PtAb)
            proj = P @ bc
            diff = b - proj
            err = np.linalg.norm(diff) / np.sqrt(denom_2)
        else:
            Ab = A @ b
            PtAb = P.T @ Ab
            bc = Ac_solve(PtAb)
            proj = P @ bc
            diff = b - proj
            num = energy_norm_sq(A, diff)
            err = np.sqrt(max(num / denom, 0.0))
        errors.append(err)

    if not errors:
        return {"mean": 0.0, "max": 0.0}
    return {"mean": float(np.mean(errors)), "max": float(np.max(errors))}


# ============================================================
# Combined diagnostic snapshot
# ============================================================
def compute_diagnostic_snapshot(
    A: sp.csr_matrix,
    ml: pyamg.multilevel.MultilevelSolver,
    B: np.ndarray | None = None,
    seed: int = 42,
) -> dict:
    """Compute all theory-backed diagnostics for one sample.

    Args:
        A: System matrix (sparse CSR).
        ml: PyAMG multilevel solver (already built from learned C).
        B: Near-nullspace candidates (n, k) or None for default.
        seed: RNG seed for reproducible probe vectors.

    Returns:
        Dict with keys: cr_rate, vcycle_conv_factor, wap_error,
        nullspace_mean, nullspace_max, pcg_iters, kappa, is_spd,
        n_levels.
    """
    from amg_core import pcg_iterations

    P = ml.levels[0].P  # prolongation operator from level 0

    cr = compute_cr_rate(A, ml, seed=seed)
    tg = compute_twogrid_convergence_factor(A, P, ml=ml, seed=seed)
    wap = compute_wap_error(A, P, seed=seed)
    ns = compute_nullspace_preservation(A, P, B)
    cond = compute_preconditioned_condition_number(A, ml)

    # PCG iterations — the actual performance measure
    try:
        M = ml.aspreconditioner(cycle="V")
        iters, _ = pcg_iterations(A, M, tol=1e-8)
    except Exception:
        iters = 500

    return {
        "cr_rate": cr,
        "vcycle_conv_factor": tg,
        "wap_error": wap,
        "nullspace_mean": ns["mean"],
        "nullspace_max": ns["max"],
        "pcg_iters": iters,
        "kappa": cond["kappa"],
        "is_spd": cond["is_spd"],
        "n_levels": len(ml.levels),
    }


def compute_diagnostic_batch(
    samples,
    model,
    k_per_row: int,
    selection_mode: str = "topk",
    sample_indices: list | None = None,
    seed: int = 42,
) -> dict:
    """Run diagnostics on a batch of samples using current model weights.

    This is the main entry point for the training loop. It runs the model
    forward pass, builds the AMG hierarchy, and computes all diagnostics.

    Args:
        samples: List of SpectralTrainSample.
        model: AMGEdgePolicy (already has weights loaded).
        k_per_row: Number of edges per row (for topk mode).
        selection_mode: "topk" or "threshold".
        sample_indices: Which samples to evaluate (default: all).
        seed: RNG seed.

    Returns:
        Dict with per-metric lists and means.
    """
    import torch
    from amg_core import (
        sample_topk_without_replacement,
        select_edges_by_threshold,
        C_from_selected_edges,
        build_pyamg_solver,
        build_B_for_pyamg,
    )

    if sample_indices is None:
        sample_indices = list(range(len(samples)))

    results = {
        "cr_rate": [], "vcycle_conv_factor": [], "wap_error": [],
        "nullspace_mean": [], "nullspace_max": [], "pcg_iters": [],
        "kappa": [], "is_spd": [], "n_levels": [],
    }

    with torch.no_grad():
        for idx in sample_indices:
            sample = samples[idx]
            logits, B_extra = model(
                sample.x, sample.edge_index, sample.edge_weight,
                raw_edge_features=sample.raw_edge_features,
            )

            if selection_mode == "threshold":
                threshold = model.threshold.item() if hasattr(model, 'threshold') else 0.0
                sel_mask, _ = select_edges_by_threshold(
                    logits=logits, edge_index=sample.edge_index,
                    threshold=threshold,
                )
            else:
                sel_mask, _ = sample_topk_without_replacement(
                    logits=logits, row_groups=sample.row_groups,
                    k=k_per_row, temperature=1.0, gumbel=False,
                )

            C = C_from_selected_edges(sample.A, sample.edge_index,
                                       sel_mask.cpu().numpy())
            B = build_B_for_pyamg(B_extra) if B_extra is not None else None

            try:
                ml = build_pyamg_solver(sample.A, C, B)
                snap = compute_diagnostic_snapshot(sample.A, ml, B,
                                                    seed=seed + idx)
                for key in results:
                    results[key].append(snap[key])
            except Exception:
                # If solver build fails, record worst-case values
                for key in results:
                    if key == "pcg_iters":
                        results[key].append(500)
                    elif key == "kappa":
                        results[key].append(float("inf"))
                    elif key == "is_spd":
                        results[key].append(False)
                    elif key == "n_levels":
                        results[key].append(0)
                    elif "nullspace" in key or "wap" in key:
                        results[key].append(float("inf"))
                    else:
                        results[key].append(1.0)

    # Compute means
    summary = {}
    for key, vals in results.items():
        finite_vals = [v for v in vals if np.isfinite(v)]
        summary[key] = float(np.mean(finite_vals)) if finite_vals else float("inf")
        summary[f"{key}_per_sample"] = vals

    return summary
