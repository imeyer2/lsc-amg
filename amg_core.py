"""
amg_core.py

Shared AMG utility functions used by all training scripts:
  - Smooth vector generation
  - Node feature construction
  - Sampling (per-row top-k without replacement)
  - Strength matrix (C) construction
  - Near-nullspace (B) construction
  - PyAMG solver builder
  - Condition number estimation (CG-Lanczos)
  - Condition-number reward computation
  - PCG iteration evaluation
  - Deterministic C-from-model builder
  - C overlap statistics
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import contextlib
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import torch
import torch.nn.functional as F

from torch_geometric.utils import from_scipy_sparse_matrix

import pyamg

# cuBLAS needs this set before it is first used for deterministic matmuls.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def compute_device(prefer: str = "auto", deterministic: bool = True) -> torch.device:
    """GPU for GNN forwards if one is present and usable, else CPU.

    "Usable" means a kernel actually runs: some cluster GPUs (e.g. P100,
    compute capability 6.0) are too old for the pinned CUDA 12.8 wheels and
    fail with "no kernel image", in which case this falls back to the CPU.
    With deterministic=True, torch.use_deterministic_algorithms is enabled:
    PyG's GAT aggregation uses CUDA atomics, so otherwise repeated forwards
    differ by ~1e-7 and can flip edges that sit at the selection threshold
    (measured: 14.7 ms -> 40.5 ms per forward at n=50k on a V100).
    """
    if prefer == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        x = torch.ones(8, device="cuda")
        (x * 2).sum().item()
    except Exception as e:  # e.g. no kernel image for this architecture
        print(f"  WARNING: CUDA device unusable ({type(e).__name__}); using CPU", flush=True)
        if prefer == "cuda":
            raise
        return torch.device("cpu")
    if deterministic:
        torch.use_deterministic_algorithms(True)
    return torch.device("cuda")


# ============================================================
# Smooth error generation
# ============================================================
# (see also compute_smooth_error_targets below for self-supervised targets)
def relaxed_smooth_vectors(A: sp.csr_matrix, num_vecs: int, iters: int, omega: float = 2.0/3.0,
                          seed: Optional[int] = None) -> np.ndarray:
    """
    Generate smooth error samples by relaxing Ax = 0 from random initial errors.

    Vectors are relaxed in parallel via ThreadPoolExecutor (pyamg's Jacobi
    relaxation is a compiled C-extension that releases the GIL).
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((A.shape[0], num_vecs))
    else:
        X = np.random.randn(A.shape[0], num_vecs)

    b = np.zeros(A.shape[0])

    def _relax_one(k: int) -> np.ndarray:
        x = np.ascontiguousarray(X[:, k])
        b_k = np.ascontiguousarray(b.copy())
        pyamg.relaxation.relaxation.jacobi(A, x, b_k, iterations=iters, omega=omega)
        return x

    max_workers = min(num_vecs, os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(_relax_one, range(num_vecs)))

    for k, x in enumerate(results):
        X[:, k] = x

    X = X / (np.abs(X).max() + 1e-6)
    return X


# ============================================================
# Coordinate builders
# ============================================================
def build_node_coords(grid_n: int) -> np.ndarray:
    """Build 2D coordinates for an n×n structured grid."""
    N = grid_n * grid_n
    coords = np.zeros((N, 2), dtype=np.float64)
    coords[:, 0] = np.tile(np.linspace(0, 1, grid_n), grid_n)
    coords[:, 1] = np.repeat(np.linspace(0, 1, grid_n), grid_n)
    return coords


# ============================================================
# Node feature construction
# ============================================================
def node_features_for_policy(A: sp.csr_matrix, coords: np.ndarray,
                             num_vecs_in: int = 4, iters_in: int = 5,
                             dof_type: Optional[np.ndarray] = None) -> torch.Tensor:
    """
    Policy input features: relaxed vectors + coordinates + optional DOF-type indicator.

    Returns:
        (N, 6) for scalar PDEs: [4 relaxed | 2 coords]
        (N, 7) for elasticity:  [4 relaxed | 2 coords | 1 dof_type]

    Args:
        dof_type: Optional (N,1) array, e.g. 0=x-disp, 1=y-disp for elasticity.
    """
    X_in = relaxed_smooth_vectors(A, num_vecs=num_vecs_in, iters=iters_in)
    parts = [X_in, coords]
    if dof_type is not None:
        parts.append(dof_type)
    feats = np.hstack(parts)
    return torch.tensor(feats, dtype=torch.float32)


# ============================================================
# Feature configuration
# ============================================================
@dataclass
class FeatureConfig:
    """Toggle flags controlling which node/edge features are built."""
    # Node feature blocks
    use_node_relaxed_probes: bool = True      # r_1 .. r_4 (4 dims)
    use_node_operator_local: bool = True      # a_ii, m_i^-, s_i^-, rho_i, d_i (5 dims)
    use_node_geometry: bool = False            # x_i, y_i (2 dims)

    # Edge feature blocks
    use_edge_sign: bool = True                # 1[a_ij < 0] (1 dim)
    use_edge_row_relative_strength: bool = True  # s_ij^(i), s_ij^(j) (2 dims)
    use_edge_symm_normalized_mag: bool = True    # a_hat_ij (1 dim)
    use_edge_relaxed_diffs: bool = True       # delta_ij^k, delta_ij_rel^k k=1..4 (8 dims)
    use_edge_geometry: bool = False            # dx, dy, |dx|, |dy| (4 dims)

    # Relaxation parameters
    num_relaxed_probes: int = 4
    relax_iters: int = 5

    @property
    def node_dim(self) -> int:
        d = 0
        if self.use_node_relaxed_probes:
            d += self.num_relaxed_probes
        if self.use_node_operator_local:
            d += 5
        if self.use_node_geometry:
            d += 2
        return d

    @property
    def edge_raw_dim(self) -> int:
        d = 0
        if self.use_edge_sign:
            d += 1
        if self.use_edge_row_relative_strength:
            d += 2
        if self.use_edge_symm_normalized_mag:
            d += 1
        if self.use_edge_relaxed_diffs:
            d += 2 * self.num_relaxed_probes   # absolute + relative
        if self.use_edge_geometry:
            d += 4
        return d

    @classmethod
    def legacy(cls) -> "FeatureConfig":
        """Mimics old 6-dim node features + no raw edge features."""
        return cls(
            use_node_relaxed_probes=True, use_node_operator_local=False,
            use_node_geometry=True,
            use_edge_sign=False, use_edge_row_relative_strength=False,
            use_edge_symm_normalized_mag=False, use_edge_relaxed_diffs=False,
            use_edge_geometry=False,
        )

    @classmethod
    def new_algebraic_only(cls) -> "FeatureConfig":
        """Operator-local stats only, no relaxed probes or geometry."""
        return cls(
            use_node_relaxed_probes=False, use_node_operator_local=True,
            use_node_geometry=False,
            use_edge_sign=True, use_edge_row_relative_strength=True,
            use_edge_symm_normalized_mag=True, use_edge_relaxed_diffs=False,
            use_edge_geometry=False,
        )

    @classmethod
    def new_algebraic_plus_relaxed(cls) -> "FeatureConfig":
        """Full new default: relaxed probes + operator stats + algebraic edges."""
        return cls()  # all defaults

    @classmethod
    def new_full_with_geometry(cls) -> "FeatureConfig":
        """Everything on including geometry."""
        return cls(use_node_geometry=True, use_edge_geometry=True)

    @classmethod
    def probes_only(cls) -> "FeatureConfig":
        """Ablation: probe vectors only, no classical operator stats.
        Node: 4D (relaxed probes), Edge: 8D (relaxed diffs)."""
        return cls(
            use_node_relaxed_probes=True, use_node_operator_local=False,
            use_node_geometry=False,
            use_edge_sign=False, use_edge_row_relative_strength=False,
            use_edge_symm_normalized_mag=False, use_edge_relaxed_diffs=True,
            use_edge_geometry=False,
        )

    @classmethod
    def algebraic_no_edge_diffs(cls) -> "FeatureConfig":
        """Ablation: full node features but classical-only edge features
        (no probe diffs on edges).
        Node: 9D (probes + operator), Edge: 4D (sign + strength + mag)."""
        return cls(
            use_node_relaxed_probes=True, use_node_operator_local=True,
            use_node_geometry=False,
            use_edge_sign=True, use_edge_row_relative_strength=True,
            use_edge_symm_normalized_mag=True, use_edge_relaxed_diffs=False,
            use_edge_geometry=False,
        )

    @classmethod
    def from_preset(cls, name: str) -> "FeatureConfig":
        presets = {
            "legacy": cls.legacy,
            "new_algebraic_only": cls.new_algebraic_only,
            "new_algebraic_plus_relaxed": cls.new_algebraic_plus_relaxed,
            "new_full_with_geometry": cls.new_full_with_geometry,
            "probes_only": cls.probes_only,
            "algebraic_no_edge_diffs": cls.algebraic_no_edge_diffs,
        }
        if name not in presets:
            raise ValueError(f"Unknown feature preset '{name}'. Choose from {list(presets.keys())}")
        return presets[name]()


# ============================================================
# Row-level operator statistics (vectorized)
# ============================================================
def compute_row_stats(A: sp.csr_matrix) -> Dict[str, np.ndarray]:
    """
    Compute per-row statistics from A's CSR arrays.

    Returns dict with keys:
      diag      : (N,) diagonal entries a_ii
      rowmaxneg : (N,) max(-a_ij, 0) over off-diag j  (i.e. largest negative magnitude)
      rowsumneg : (N,) sum of -a_ij for a_ij < 0, off-diag
      rowsum_defect : (N,) |sum of row i|  (zero for ideal Laplacians)
      degree    : (N,) number of off-diagonal nonzeros
    """
    A_csr = sp.csr_matrix(A)
    N = A_csr.shape[0]
    indptr = A_csr.indptr
    indices = A_csr.indices
    data = A_csr.data

    diag = np.array(A_csr.diagonal(), dtype=np.float64)

    # Build per-entry off-diagonal mask
    row_of = np.repeat(np.arange(N), np.diff(indptr))
    off_diag_mask = indices != row_of

    # Negative off-diagonal values: -a_ij where a_ij < 0
    neg_offdiag = np.where(off_diag_mask & (data < 0), -data, 0.0)

    # Row max of negative off-diag (max is order-independent, so reduceat over
    # the CSR row slices equals np.maximum.at; empty rows stay 0)
    rowmaxneg = np.zeros(N, dtype=np.float64)
    nonempty = np.diff(indptr) > 0
    if nonempty.any():
        rowmaxneg[nonempty] = np.maximum(
            np.maximum.reduceat(neg_offdiag, indptr[:-1][nonempty]), 0.0)

    # Row sum of negative off-diag
    rowsumneg = np.zeros(N, dtype=np.float64)
    np.add.at(rowsumneg, row_of, neg_offdiag)

    # Row sum defect |sum(row i)|
    rowsum_defect = np.abs(np.array(A_csr.sum(axis=1)).ravel())

    # Off-diagonal degree (integer counts, exact)
    degree = np.bincount(row_of[off_diag_mask], minlength=N).astype(np.float64)

    return dict(
        diag=diag,
        rowmaxneg=rowmaxneg,
        rowsumneg=rowsumneg,
        rowsum_defect=rowsum_defect,
        degree=degree,
    )


# ============================================================
# New node feature builder
# ============================================================
def build_node_features(A: sp.csr_matrix, coords: np.ndarray,
                        feature_config: FeatureConfig,
                        row_stats: Optional[Dict[str, np.ndarray]] = None,
                        R: Optional[np.ndarray] = None) -> torch.Tensor:
    """
    Build node feature tensor according to FeatureConfig.

    Args:
        A: System matrix (N×N sparse).
        coords: Node coordinates (N, 2).
        feature_config: Which features to include.
        row_stats: Pre-computed row stats (optional, computed if None).
        R: Pre-computed relaxed probes (N, num_probes) (optional).

    Returns: (N, node_dim) float32 tensor.
    """
    parts = []

    if feature_config.use_node_relaxed_probes:
        if R is None:
            R = relaxed_smooth_vectors(A, num_vecs=feature_config.num_relaxed_probes,
                                       iters=feature_config.relax_iters)
        parts.append(R[:, :feature_config.num_relaxed_probes])

    if feature_config.use_node_operator_local:
        if row_stats is None:
            row_stats = compute_row_stats(A)
        eps = 1e-9
        diag = row_stats["diag"]
        rowmaxneg = row_stats["rowmaxneg"]
        rowsumneg = row_stats["rowsumneg"]
        rowsum_defect = row_stats["rowsum_defect"]
        degree = row_stats["degree"]
        # Normalize: divide by |diag| to make scale-free (abs handles indefinite matrices)
        safe_diag = np.abs(diag) + eps
        parts.append(np.column_stack([
            diag,
            rowmaxneg / safe_diag,
            rowsumneg / safe_diag,
            rowsum_defect / safe_diag,
            degree,
        ]))

    if feature_config.use_node_geometry:
        parts.append(coords[:, :2])

    feats = np.hstack(parts) if parts else np.zeros((A.shape[0], 0), dtype=np.float64)
    return torch.tensor(feats, dtype=torch.float32)


# ============================================================
# New edge feature builder
# ============================================================
def build_edge_features(A: sp.csr_matrix, edge_index: torch.Tensor,
                        feature_config: FeatureConfig,
                        row_stats: Optional[Dict[str, np.ndarray]] = None,
                        R: Optional[np.ndarray] = None,
                        coords: Optional[np.ndarray] = None) -> torch.Tensor:
    """
    Build raw edge feature tensor according to FeatureConfig.

    Args:
        A: System matrix (N×N sparse).
        edge_index: (2, E) edge index tensor.
        feature_config: Which features to include.
        row_stats: Pre-computed row stats (optional).
        R: Pre-computed relaxed probes (N, num_probes) (optional).
        coords: Node coordinates (N, 2), required if use_edge_geometry.

    Returns: (E, edge_raw_dim) float32 tensor.
    """
    eps = 1e-9
    row = edge_index[0].numpy()
    col = edge_index[1].numpy()
    E = len(row)

    # Get entry values a_ij from the sparse matrix: one binary search of the
    # edges' linear indices r*N + c in A's (sorted) row-major entry list.
    # Same result as searching each row's CSR slice: the first stored entry
    # equal to (r, c), or 0.0 if (r, c) is not stored.
    A_csr = sp.csr_matrix(A)
    A_csr.sort_indices()  # searchsorted requires sorted column indices
    N = A_csr.shape[0]
    entry_row = np.repeat(np.arange(N, dtype=np.int64), np.diff(A_csr.indptr))
    lin = entry_row * N + A_csr.indices.astype(np.int64)
    q = row.astype(np.int64) * N + col.astype(np.int64)
    pos = np.minimum(np.searchsorted(lin, q), max(len(lin) - 1, 0))
    hit = (lin[pos] == q) if len(lin) else np.zeros(E, dtype=bool)
    a_ij = np.where(hit, A_csr.data[pos] if len(lin) else 0.0, 0.0).astype(np.float64)

    if row_stats is None and (feature_config.use_edge_row_relative_strength
                              or feature_config.use_edge_symm_normalized_mag):
        row_stats = compute_row_stats(A)

    parts = []

    # 1[a_ij < 0]
    if feature_config.use_edge_sign:
        parts.append((a_ij < 0).astype(np.float64).reshape(-1, 1))

    # s_ij^(i) = max(-a_ij, 0) / m_i^-, s_ij^(j) = max(-a_ij, 0) / m_j^-  (m^- = row max of -a)
    if feature_config.use_edge_row_relative_strength:
        neg_a = np.maximum(-a_ij, 0.0)
        # Divide by the row maximum itself (1 where a row has no negative
        # entry, so 0/1 = 0): an absolute eps would make the strongest edge
        # of rows with a small maximum (lognormal coefficients span ~12
        # orders) come out below 1, e.g. 0.9998 at rowmax 5e-6.
        rmn = row_stats["rowmaxneg"]
        rmn_safe = np.where(rmn > 0, rmn, 1.0)
        s_ij_i = neg_a / rmn_safe[row]
        s_ij_j = neg_a / rmn_safe[col]
        parts.append(np.column_stack([s_ij_i, s_ij_j]))

    # a_hat_ij = |a_ij| / sqrt(a_ii * a_jj + eps)
    if feature_config.use_edge_symm_normalized_mag:
        diag = row_stats["diag"]
        a_hat = np.abs(a_ij) / (np.sqrt(np.abs(diag[row] * diag[col]) + eps))
        parts.append(a_hat.reshape(-1, 1))

    # delta_ij^k = |r_i^k - r_j^k|, delta_ij_rel^k = delta / (max(|r_i^k|, |r_j^k|) + eps)
    if feature_config.use_edge_relaxed_diffs:
        if R is None:
            R = relaxed_smooth_vectors(A, num_vecs=feature_config.num_relaxed_probes,
                                       iters=feature_config.relax_iters)
        R_sub = R[:, :feature_config.num_relaxed_probes]
        r_i = R_sub[row]   # (E, K)
        r_j = R_sub[col]   # (E, K)
        delta_abs = np.abs(r_i - r_j)
        denom = np.maximum(np.abs(r_i), np.abs(r_j)) + eps
        delta_rel = delta_abs / denom
        parts.append(delta_abs)
        parts.append(delta_rel)

    # Geometry: dx, dy, |dx|, |dy|
    if feature_config.use_edge_geometry:
        if coords is None:
            raise ValueError("coords required when use_edge_geometry=True")
        ci = coords[row, :2]
        cj = coords[col, :2]
        dxy = ci - cj
        parts.append(np.column_stack([dxy[:, 0], dxy[:, 1],
                                       np.abs(dxy[:, 0]), np.abs(dxy[:, 1])]))

    if parts:
        feats = np.hstack(parts)
    else:
        feats = np.zeros((E, 0), dtype=np.float64)
    return torch.tensor(feats, dtype=torch.float32)


# ============================================================
# Convenience: build all sample features at once
# ============================================================
def build_sample_features(A: sp.csr_matrix, coords: np.ndarray,
                          edge_index: torch.Tensor,
                          feature_config: FeatureConfig,
                          seed: Optional[int] = None,
                          ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Build both node and edge features for a sample, sharing computation.

    seed: seeds the random starting vectors of the relaxed probes. With
    seed=None they come from numpy's global RNG, so the same matrix gets
    different features in different processes (e.g. training vs a later
    re-benchmark), and the learned preconditioner changes with them.

    Returns:
        node_features: (N, node_dim) tensor
        raw_edge_features: (E, edge_raw_dim) tensor
    """
    # Compute shared intermediates only when needed
    needs_row_stats = (feature_config.use_node_operator_local
                       or feature_config.use_edge_row_relative_strength
                       or feature_config.use_edge_symm_normalized_mag)
    needs_R = (feature_config.use_node_relaxed_probes
               or feature_config.use_edge_relaxed_diffs)

    row_stats = compute_row_stats(A) if needs_row_stats else None
    R = (relaxed_smooth_vectors(A, num_vecs=feature_config.num_relaxed_probes,
                                iters=feature_config.relax_iters, seed=seed)
         if needs_R else None)

    node_feats = build_node_features(A, coords, feature_config,
                                     row_stats=row_stats, R=R)
    edge_feats = build_edge_features(A, edge_index, feature_config,
                                     row_stats=row_stats, R=R, coords=coords)
    # Return None instead of empty (E, 0) tensor to avoid truthy-but-empty pitfall
    if edge_feats.shape[1] == 0:
        edge_feats = None
    return node_feats, edge_feats


# ============================================================
# Smooth-error targets for self-supervised (continuous relaxation) training
# ============================================================
def compute_smooth_error_targets(A: sp.csr_matrix, edge_index: torch.Tensor,
                                  num_vecs: int = 12, relax_iters: int = 30,
                                  omega: float = 2.0/3.0) -> torch.Tensor:
    """
    Per-edge smooth-error similarity targets for self-supervised training.

    Classical AMG principle: *strong connections are those along which
    algebraically smooth error varies slowly*.  We measure this by relaxing
    random vectors on Ax = 0 and computing per-edge similarity.

    Targets are ROW-NORMALIZED: for each row i the edge with the smallest
    smooth-error difference gets target 1.0, the largest gets target 0.0.
    This mirrors the per-row relative thresholding used by θ-based strength
    (keep edge ij if |a_ij| >= θ * max_k |a_ik|).

    Returns:
        (E,) tensor in [0, 1]:  1 = strong (smooth errors most similar in row)
                                0 = weak   (smooth errors most different in row)
    """
    N = A.shape[0]
    E_vecs = relaxed_smooth_vectors(A, num_vecs=num_vecs, iters=relax_iters, omega=omega)

    row = edge_index[0].numpy()
    col = edge_index[1].numpy()

    # Per-edge mean squared difference across all smooth test vectors
    diff = E_vecs[row] - E_vecs[col]            # (num_edges, num_vecs)
    d_sq = np.mean(diff ** 2, axis=1)            # (num_edges,)

    off_diag = (row != col)

    # --- Row-relative normalization ---
    # For each row i, rescale d_sq to [0, 1] where:
    #   0 = smallest difference in row (strongest connection)
    #   1 = largest difference in row  (weakest connection)
    # Then target = 1 - normalized_d_sq (so strong → 1, weak → 0)
    targets = np.zeros_like(d_sq)

    for i in range(N):
        mask_i = (row == i) & off_diag
        if not np.any(mask_i):
            continue
        d_i = d_sq[mask_i]
        d_min = d_i.min()
        d_max = d_i.max()
        rng = d_max - d_min
        if rng < 1e-14:
            # All neighbors equally smooth → all strong
            targets[mask_i] = 1.0
        else:
            targets[mask_i] = 1.0 - (d_i - d_min) / rng

    targets[~off_diag] = 0.0  # never select diagonals

    return torch.tensor(targets, dtype=torch.float32)


# ============================================================
# Per-row edge grouping
# ============================================================
def build_row_groups(edge_index: torch.Tensor, num_nodes: int) -> List[torch.Tensor]:
    row = edge_index[0].cpu().numpy()
    buckets: List[List[int]] = [[] for _ in range(num_nodes)]
    for e, r in enumerate(row):
        buckets[int(r)].append(e)
    return [torch.tensor(b, dtype=torch.long) for b in buckets]


# ============================================================
# Sampling: per-row top-k WITHOUT replacement + logprob
# ============================================================
def sample_topk_without_replacement(
    logits: torch.Tensor,
    row_groups: List[torch.Tensor],
    k: int,
    temperature: float = 1.0,
    gumbel: bool = True,
    return_per_row_logps: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    For each node, sample exactly k of its outgoing edges to include in the
    AMG strength-of-connection matrix C — without replacement and with a
    differentiable log-probability for REINFORCE.

    Args:
        logits:      (E,) float tensor of unnormalised scores for every edge,
                     produced by the GNN policy. Higher logit → more likely
                     to be selected.
        row_groups:  List of length num_nodes. row_groups[i] is a 1-D LongTensor
                     containing the global edge indices (positions in the flat
                     edge_index array) of all outgoing edges from node i.
                     Built once by build_row_groups() and reused across rollouts.
                     Example: if node 5 has neighbours via edges 23,24,25,26 then
                     row_groups[5] = tensor([23, 24, 25, 26]).
        k:           Number of edges to select per node (cfg.k_per_row).
        temperature: Softmax temperature; higher → more uniform selection.
        gumbel:      If True, add Gumbel noise before argmax (Gumbel-top-k
                     trick) → stochastic sampling without replacement, used
                     during training rollouts.
                     If False, pure greedy argmax → deterministic top-k per
                     node, used at evaluation time.
        return_per_row_logps: If True, also return per-row log-probabilities
                     (num_rows,) for local credit assignment.

    Algorithm (k iterations of batched tensor ops over all nodes):
        1. Pad row_groups into a (num_nodes, max_edges_per_row) matrix so all
           nodes can be processed simultaneously.
        2. Repeat k times (one selection slot per iteration):
           a. Optionally perturb logits with Gumbel noise (-log(-log(U))).
              This converts argmax into a draw from the categorical without
              replacement (Gumbel-top-k).
           b. Argmax per row on the (perturbed) logits → one edge chosen per
              node this step.
           c. Accumulate log π(chosen) using the *unperturbed* log-softmax so
              the gradient flows correctly back through logits.
           d. Mask out the just-chosen edge so it cannot be re-selected.
        3. Return the boolean selection mask and the summed log-probability.

    **Vectorised across all rows**: the N×k Python loop is replaced by
    k iterations of batched tensor operations over all N rows, giving
    10-100× speedup on typical grids.

    Returns:
        selected_mask: (E,) bool tensor — True for each globally-indexed edge
                       that was selected (i.e. included in C).
        logprob_sum:   differentiable scalar — Σ_i Σ_{j=1}^{k} log π(choice_j |
                       remaining_i), used as the REINFORCE log-prob to compute
                       policy gradients.
        (optional) per_row_logps: (num_rows,) tensor of per-row log-probs,
                       returned only when return_per_row_logps=True.
    """
    device = logits.device
    E = logits.numel()
    tau = max(float(temperature), 1e-6)

    num_rows = len(row_groups)
    if num_rows == 0:
        empty_sel = torch.zeros(E, dtype=torch.bool, device=device)
        empty_logp = torch.zeros((), dtype=torch.float32, device=device)
        if return_per_row_logps:
            return empty_sel, empty_logp, torch.zeros(0, device=device)
        return empty_sel, empty_logp

    # ── Pad row_groups into (num_rows, max_edges_per_row) tensors ──
    lengths = [rg.numel() for rg in row_groups]
    max_len = max(lengths)
    if max_len == 0:
        empty_sel = torch.zeros(E, dtype=torch.bool, device=device)
        empty_logp = torch.zeros((), dtype=torch.float32, device=device)
        if return_per_row_logps:
            return empty_sel, empty_logp, torch.zeros(num_rows, device=device)
        return empty_sel, empty_logp

    edge_ids = torch.zeros(num_rows, max_len, dtype=torch.long, device=device)
    valid = torch.zeros(num_rows, max_len, dtype=torch.bool, device=device)

    for i, rg in enumerate(row_groups):
        L = rg.numel()
        if L > 0:
            edge_ids[i, :L] = rg.to(device)
            valid[i, :L] = True

    # Sanity check
    if valid.any() and int(edge_ids[valid].max()) >= E:
        raise RuntimeError(
            f"row_groups contain edge id {int(edge_ids[valid].max())} "
            f"but logits has E={E}")

    # ── Batched logits (invalid / padding positions masked to -1e18) ──
    NEGINF = torch.tensor(-1e18, device=device, dtype=logits.dtype)
    base_logits = torch.where(valid, logits[edge_ids] / tau, NEGINF)

    # How many to actually select per row (capped at row length and k)
    len_t = torch.tensor(lengths, dtype=torch.long, device=device)
    k_per = len_t.clamp(max=k)  # (num_rows,)

    selected = torch.zeros(E, dtype=torch.bool, device=device)
    logp_total = torch.zeros((), dtype=torch.float32, device=device)

    # Per-row log-prob accumulator (for local credit assignment)
    row_logps = torch.zeros(num_rows, dtype=torch.float32, device=device)

    active = valid.clone()          # positions not yet chosen
    remaining = k_per.clone()       # how many more to pick per row
    row_idx = torch.arange(num_rows, device=device)

    for _t in range(k):
        row_mask = remaining > 0    # which rows still need selections
        if not row_mask.any():
            break

        # Mask already-chosen positions
        cur_logits = torch.where(active, base_logits, NEGINF)

        # Gumbel perturbation (for stochastic sampling)
        if gumbel:
            u = torch.rand_like(cur_logits).clamp_(1e-10, 1.0 - 1e-10)
            perturbed = cur_logits - torch.log(-torch.log(u))
        else:
            perturbed = cur_logits

        # Log-softmax over remaining positions in each row
        log_probs = cur_logits - cur_logits.logsumexp(dim=1, keepdim=True)

        # Argmax per row (greedy on possibly-perturbed logits)
        chosen = perturbed.argmax(dim=1)                   # (num_rows,)

        # Gather log-probs and global edge ids at chosen positions
        chosen_logp  = log_probs[row_idx, chosen]          # (num_rows,)
        chosen_edges = edge_ids[row_idx, chosen]            # (num_rows,)

        # Accumulate log-prob (only for rows that still need selections)
        logp_total = logp_total + (chosen_logp * row_mask.float()).sum()
        row_logps = row_logps + chosen_logp * row_mask.float()

        # Mark selected edges
        selected[chosen_edges[row_mask]] = True

        # Deactivate chosen positions (functional update — avoids in-place
        # modification of a tensor referenced by torch.where's autograd)
        deactivate = torch.zeros_like(active)
        deactivate[row_idx, chosen] = True
        active = active & ~deactivate
        remaining = remaining - row_mask.long()

    if return_per_row_logps:
        return selected, logp_total, row_logps
    return selected, logp_total


def precompute_padded_row_groups(
    row_groups: List[torch.Tensor],
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pre-compute padded row-group tensors for reuse across K rollouts.

    Returns:
        edge_ids: (num_rows, max_len) padded edge indices
        valid:    (num_rows, max_len) bool mask
        len_t:    (num_rows,) lengths
        multi:    (num_rows,) bool — rows with >1 element (for entropy)
    """
    num_rows = len(row_groups)
    lengths = [rg.numel() for rg in row_groups]
    max_len = max(lengths) if lengths else 0
    if max_len == 0:
        z = torch.zeros(num_rows, 1, dtype=torch.long, device=device)
        f = torch.zeros(num_rows, 1, dtype=torch.bool, device=device)
        return z, f, torch.zeros(num_rows, dtype=torch.long, device=device), \
               torch.zeros(num_rows, dtype=torch.bool, device=device)

    edge_ids = torch.zeros(num_rows, max_len, dtype=torch.long, device=device)
    valid = torch.zeros(num_rows, max_len, dtype=torch.bool, device=device)

    for i, rg in enumerate(row_groups):
        L = rg.numel()
        if L > 0:
            edge_ids[i, :L] = rg.to(device)
            valid[i, :L] = True

    len_t = torch.tensor(lengths, dtype=torch.long, device=device)
    multi = len_t > 1
    return edge_ids, valid, len_t, multi


def sample_topk_cached(
    logits: torch.Tensor,
    edge_ids: torch.Tensor,
    valid: torch.Tensor,
    len_t: torch.Tensor,
    k: int,
    temperature: float = 1.0,
    gumbel: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Same as sample_topk_without_replacement but uses pre-computed padded tensors.

    Much faster when called multiple times for the same sample (K rollouts)
    because it skips the per-row Python padding loop.
    """
    device = logits.device
    E = logits.numel()
    tau = max(float(temperature), 1e-6)
    num_rows = edge_ids.shape[0]

    if num_rows == 0 or not valid.any():
        return (torch.zeros(E, dtype=torch.bool, device=device),
                torch.zeros((), dtype=torch.float32, device=device))

    NEGINF = torch.tensor(-1e18, device=device, dtype=logits.dtype)
    base_logits = torch.where(valid, logits[edge_ids] / tau, NEGINF)

    k_per = len_t.clamp(max=k)
    selected = torch.zeros(E, dtype=torch.bool, device=device)
    logp_total = torch.zeros((), dtype=torch.float32, device=device)
    active = valid.clone()
    remaining = k_per.clone()
    row_idx = torch.arange(num_rows, device=device)

    for _t in range(k):
        row_mask = remaining > 0
        if not row_mask.any():
            break
        cur_logits = torch.where(active, base_logits, NEGINF)
        if gumbel:
            u = torch.rand_like(cur_logits).clamp_(1e-10, 1.0 - 1e-10)
            perturbed = cur_logits - torch.log(-torch.log(u))
        else:
            perturbed = cur_logits
        log_probs = cur_logits - cur_logits.logsumexp(dim=1, keepdim=True)
        chosen = perturbed.argmax(dim=1)
        chosen_logp = log_probs[row_idx, chosen]
        chosen_edges = edge_ids[row_idx, chosen]
        logp_total = logp_total + (chosen_logp * row_mask.float()).sum()
        selected[chosen_edges[row_mask]] = True
        deactivate = torch.zeros_like(active)
        deactivate[row_idx, chosen] = True
        active = active & ~deactivate
        remaining = remaining - row_mask.long()

    return selected, logp_total


def sample_topk_cached_with_actions(
    logits: torch.Tensor,
    edge_ids: torch.Tensor,
    valid: torch.Tensor,
    len_t: torch.Tensor,
    k: int,
    temperature: float = 1.0,
    gumbel: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Like sample_topk_cached but also returns the chosen action sequence
    and per-row log-probabilities.

    The action sequence records which local position was chosen at each
    of the k selection steps, enabling log-probability recomputation
    under an updated policy (required for proper PPO).

    Returns:
        selected:        (E,) bool — selected edge mask
        logp_total:      scalar — sum of log-probs
        chosen_actions:  (num_rows, k) int64 — local position chosen at each step
        per_row_logps:   (num_rows,) float — log-prob per row (for per-row PPO)
    """
    device = logits.device
    E = logits.numel()
    tau = max(float(temperature), 1e-6)
    num_rows = edge_ids.shape[0]

    if num_rows == 0 or not valid.any():
        return (torch.zeros(E, dtype=torch.bool, device=device),
                torch.zeros((), dtype=torch.float32, device=device),
                torch.zeros(num_rows, k, dtype=torch.long, device=device),
                torch.zeros(num_rows, dtype=torch.float32, device=device))

    NEGINF = torch.tensor(-1e18, device=device, dtype=logits.dtype)
    base_logits = torch.where(valid, logits[edge_ids] / tau, NEGINF)

    k_per = len_t.clamp(max=k)
    selected = torch.zeros(E, dtype=torch.bool, device=device)
    logp_total = torch.zeros((), dtype=torch.float32, device=device)
    per_row_logps = torch.zeros(num_rows, dtype=torch.float32, device=device)
    active = valid.clone()
    remaining = k_per.clone()
    row_idx = torch.arange(num_rows, device=device)
    chosen_actions = torch.zeros(num_rows, k, dtype=torch.long, device=device)

    for _t in range(k):
        row_mask = remaining > 0
        if not row_mask.any():
            break
        cur_logits = torch.where(active, base_logits, NEGINF)
        if gumbel:
            u = torch.rand_like(cur_logits).clamp_(1e-10, 1.0 - 1e-10)
            perturbed = cur_logits - torch.log(-torch.log(u))
        else:
            perturbed = cur_logits
        log_probs = cur_logits - cur_logits.logsumexp(dim=1, keepdim=True)
        chosen = perturbed.argmax(dim=1)
        chosen_logp = log_probs[row_idx, chosen]
        chosen_edges = edge_ids[row_idx, chosen]
        logp_total = logp_total + (chosen_logp * row_mask.float()).sum()
        per_row_logps = per_row_logps + chosen_logp * row_mask.float()
        selected[chosen_edges[row_mask]] = True
        chosen_actions[:, _t] = chosen
        deactivate = torch.zeros_like(active)
        deactivate[row_idx, chosen] = True
        active = active & ~deactivate
        remaining = remaining - row_mask.long()

    return selected, logp_total, chosen_actions, per_row_logps


def logprob_of_topk_action_per_row(
    logits: torch.Tensor,
    edge_ids: torch.Tensor,
    valid: torch.Tensor,
    len_t: torch.Tensor,
    chosen_actions: torch.Tensor,
    k: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Recompute per-row log-probabilities of stored actions under current logits.

    Used in the PPO update phase with per-row ratio clipping: returns a
    (num_rows,) tensor so each row's importance-sampling ratio can be
    clipped independently, preventing the ratio explosion that occurs
    when the global log-prob sum (over ~8000 terms) shifts by even tiny
    per-row amounts.

    Differentiable w.r.t. ``logits``.

    Args:
        logits:         (E,) current policy edge logits (with grad)
        edge_ids:       (num_rows, max_len) padded edge indices
        valid:          (num_rows, max_len) validity mask
        len_t:          (num_rows,) row lengths
        chosen_actions: (num_rows, k) int64 — local positions chosen at each step
        k:              max selections per row
        temperature:    softmax temperature

    Returns:
        per_row_logps: (num_rows,) differentiable tensor
    """
    device = logits.device
    tau = max(float(temperature), 1e-6)
    num_rows = edge_ids.shape[0]

    if num_rows == 0 or not valid.any():
        return torch.zeros(num_rows, dtype=torch.float32, device=device)

    NEGINF = torch.tensor(-1e18, device=device, dtype=logits.dtype)
    base_logits = torch.where(valid, logits[edge_ids] / tau, NEGINF)

    k_per = len_t.clamp(max=k)
    per_row_logps = torch.zeros(num_rows, dtype=torch.float32, device=device)
    active = valid.clone()
    remaining = k_per.clone()
    row_idx = torch.arange(num_rows, device=device)

    for _t in range(k):
        row_mask = remaining > 0
        if not row_mask.any():
            break
        cur_logits = torch.where(active, base_logits, NEGINF)
        log_probs = cur_logits - cur_logits.logsumexp(dim=1, keepdim=True)
        chosen = chosen_actions[:, _t]
        chosen_logp = log_probs[row_idx, chosen]
        per_row_logps = per_row_logps + chosen_logp * row_mask.float()
        deactivate = torch.zeros_like(active)
        deactivate[row_idx, chosen] = True
        active = active & ~deactivate
        remaining = remaining - row_mask.long()

    return per_row_logps


def vectorized_entropy(
    logits: torch.Tensor,
    edge_ids: torch.Tensor,
    valid: torch.Tensor,
    multi: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """Compute entropy over row groups in one batched operation.

    Replaces the Python loop over ~4096 row_groups with a single
    padded softmax + log, giving ~100x speedup.
    """
    NEGINF = -1e18
    gathered = logits[edge_ids] / tau
    gathered = torch.where(valid, gathered, torch.tensor(NEGINF, device=logits.device))
    probs = torch.softmax(gathered, dim=1)
    log_probs = torch.log(probs + 1e-10)
    ent_per_row = -(probs * log_probs * valid.float()).sum(dim=1)
    return ent_per_row[multi].sum()


# ============================================================
# Build C from sampled edges
# ============================================================
def C_from_selected_edges(A: sp.csr_matrix, edge_index_cpu: torch.Tensor, selected_mask: np.ndarray) -> sp.csr_matrix:
    """
    Build symmetric binary strength matrix C with diagonal ones.
    """
    r = edge_index_cpu[0].cpu().numpy()
    c = edge_index_cpu[1].cpu().numpy()

    sel = selected_mask.astype(bool)
    keep = sel & (r != c)

    rr = r[keep]
    cc = c[keep]
    vals = np.ones(rr.shape[0], dtype=np.float64)

    n = A.shape[0]
    C = sp.csr_matrix((vals, (rr, cc)), shape=(n, n))
    C = C.maximum(C.T)
    C = C + sp.eye(n, format="csr")
    return C


def C_from_selected_edges_soft(
    A: sp.csr_matrix,
    edge_index_cpu: torch.Tensor,
    selected_mask: np.ndarray,
    logits: torch.Tensor,
    row_groups: list,
    temperature: float = 1.0,
) -> sp.csr_matrix:
    """Build symmetric strength matrix C with soft (probability-based) weights.

    For each row, the policy's softmax probabilities over candidate edges become
    the edge values in C. Unselected edges are absent (zero). The matrix is
    symmetrised by averaging: C_sym = (C + C^T) / 2, then the diagonal is set
    to 1 so that every node is "connected to itself" (required by PyAMG).

    To make these soft values actually influence the solver, pass
    ``smooth=('jacobi', {'filter_entries': True})`` to
    ``smoothed_aggregation_solver`` so that the prolongation smoother computes
    S = A ⊙ C (element-wise) instead of using raw A.
    """
    r = edge_index_cpu[0].cpu().numpy()
    c = edge_index_cpu[1].cpu().numpy()

    sel = selected_mask.astype(bool)
    keep = sel & (r != c)

    # Compute row-wise softmax probabilities for every candidate edge
    device = logits.device
    probs = torch.zeros_like(logits)
    tau = max(float(temperature), 1e-6)
    for rg in row_groups:
        rg_dev = rg.to(device) if rg.device != device else rg
        if rg_dev.numel() <= 1:
            probs[rg_dev] = 1.0
            continue
        row_logits = logits[rg_dev] / tau
        probs[rg_dev] = torch.softmax(row_logits, dim=0)

    probs_np = probs.detach().cpu().numpy().astype(np.float64)

    rr = r[keep]
    cc = c[keep]
    vals = probs_np[keep]

    n = A.shape[0]
    C = sp.csr_matrix((vals, (rr, cc)), shape=(n, n))
    # Symmetrise by averaging (both directed edges may have different probs)
    C = (C + C.T) / 2.0
    # Set diagonal to 1
    C = C + sp.eye(n, format="csr")
    C.eliminate_zeros()
    return C


# ============================================================
# Threshold-based edge selection (adaptive-K)
# ============================================================
def select_edges_by_threshold(
    logits: torch.Tensor,
    edge_index: torch.Tensor,
    threshold: float = 0.0,
) -> Tuple[torch.Tensor, np.ndarray]:
    """Select edges where logits exceed a learned threshold.

    Unlike sample_topk_without_replacement which picks exactly k edges
    per node, this allows the network to choose a variable number of
    connections per node — from 0 to degree.

    For ES training, this is fully deterministic (no Gumbel noise).
    The threshold is a model parameter that ES perturbs naturally.

    Args:
        logits:     (E,) edge scores from the GNN policy.
        edge_index: (2, E) edge index tensor.
        threshold:  scalar threshold; edges with logits > threshold are selected.

    Returns:
        selected_mask: (E,) bool tensor — True for selected edges.
        k_per_node:    (N,) int array — number of selected edges per node
                       (for diagnostics).
    """
    row = edge_index[0]
    col = edge_index[1]

    # Exclude self-loops from selection
    diag = (row == col)

    # Standardise logits so the threshold operates in z-score space.
    # Without this, raw logits can sit far from the threshold and no
    # ES perturbation flips any edge selection (spread ≈ 0).
    mu = logits[~diag].mean() if (~diag).any() else logits.mean()
    std = logits[~diag].std()  if (~diag).any() else logits.std()
    z = (logits - mu) / (std + 1e-6)

    selected = (z > threshold) & ~diag

    # Compute per-node selection count for diagnostics
    num_nodes = int(row.max().item()) + 1 if row.numel() > 0 else 0
    selected_np = selected.detach().cpu().numpy()
    row_np = row.detach().cpu().numpy()
    k_per_node = np.bincount(row_np[selected_np], minlength=num_nodes)

    return selected, k_per_node


# ============================================================
# Near-nullspace B construction (unified scalar + elasticity)
# ============================================================
def build_B_for_pyamg(B_extra: Optional[torch.Tensor],
                      B_base: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """
    Build near-nullspace candidates B for PyAMG smoothed aggregation.

    Args:
        B_extra: Learned additional columns from the GNN B-head, shape (N, p).
                 If None and B_base is None, returns None.
        B_base:  Known nullspace modes.
                 - For scalar PDEs: None (defaults to ones column).
                 - For elasticity: rigid body modes (N, 3).

    Returns:
        B as numpy array [B_base | normalized(B_extra)], or None.
    """
    parts = []

    if B_base is not None:
        parts.append(B_base.astype(np.float64))
    elif B_extra is not None:
        # Scalar PDE default: constant vector
        n = B_extra.shape[0]
        parts.append(np.ones((n, 1), dtype=np.float64))

    if B_extra is not None:
        Bx = B_extra.detach().cpu().numpy().astype(np.float64)
        n, p = Bx.shape
        for j in range(p):
            col = Bx[:, j]
            col = col - col.mean()
            denom = np.linalg.norm(col) + 1e-12
            Bx[:, j] = col / denom
        parts.append(Bx)

    if not parts:
        return None

    return np.hstack(parts)


# ============================================================
# PyAMG solver builder
# ============================================================
@contextlib.contextmanager
def fixed_np_rng(seed: int = 0):
    """Run a block with numpy's global RNG seeded, then restore its state.

    PyAMG's smoothed-aggregation setup estimates the spectral radius of
    D^-1 A for the Jacobi prolongation smoother from np.random.rand
    (pyamg.util.linalg._approximate_eigenvalues), so without this the same
    (A, C) gives a slightly different hierarchy -- and PCG iteration count --
    on every call, and ES rewards depend on which worker evaluated what.
    """
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def build_pyamg_solver(A: sp.csr_matrix, C: sp.csr_matrix, B: Optional[np.ndarray],
                      coarse_solver: str = "splu",
                      filter_entries: bool = False) -> pyamg.multilevel.MultilevelSolver:
    """Build SA solver using a predefined strength matrix C.

    Args:
        filter_entries: If True, pass smooth=('jacobi', {'filter_entries': True})
            so that the prolongation smoother uses S = A ⊙ C (element-wise)
            instead of raw A. This makes C's numerical values (soft weights)
            influence the prolongation operator.
    """
    strength = ("predefined", {"C": C})
    smooth = ('jacobi', {'filter_entries': True}) if filter_entries else 'jacobi'
    with fixed_np_rng(0):
        if B is None:
            ml = pyamg.smoothed_aggregation_solver(
                A, strength=strength, smooth=smooth, coarse_solver=coarse_solver,
                keep=True)
        else:
            ml = pyamg.smoothed_aggregation_solver(
                A, strength=strength, smooth=smooth, B=B, coarse_solver=coarse_solver,
                keep=True)
    return ml


def vcycle_preconditioner(ml: pyamg.multilevel.MultilevelSolver) -> spla.LinearOperator:
    """One V-cycle from a zero initial guess, as ml.aspreconditioner(cycle="V")
    computes it (MultilevelSolver.solve(b, maxiter=1, cycle="V", tol=1e-12)),
    but calling the cycle directly. solve() additionally evaluates
    ||b - A x|| before and after the cycle only for its own tolerance test,
    which with maxiter=1 never changes the returned x; those two SpMVs and
    norms are skipped. Returns the bit-identical vector."""
    shape = ml.levels[0].A.shape
    dtype = ml.levels[0].A.dtype
    if len(ml.levels) == 1:
        A0 = ml.levels[0].A

        def matvec(b):
            return ml.coarse_solver(A0, np.ravel(b))
    else:
        cycle = ml._MultilevelSolver__solve

        def matvec(b):
            b = np.ravel(b)
            x = np.zeros_like(b)
            cycle(0, x, b, "V", 1)
            return x
    return spla.LinearOperator(shape, matvec, dtype=dtype)


# ============================================================
# V-cycle helpers (kept for EDA / analysis)
# ============================================================
def energy_norm_sq(A: sp.csr_matrix, e: np.ndarray) -> float:
    Ae = A @ e
    return float(e @ Ae)

def one_vcycle_error_reduce_ratio(ml: pyamg.multilevel.MultilevelSolver, A: sp.csr_matrix,
                                  e0: np.ndarray, cycle: str = "V") -> float:
    """
    Apply one multigrid cycle to Ax=0 starting from x0=e0 and
    return ratio ||e1||_A^2 / ||e0||_A^2.
    """
    b0 = np.zeros_like(e0)
    try:
        e1 = ml.solve(b0, x0=e0, tol=0.0, maxiter=1, accel=None, cycle=cycle)
    except TypeError:
        e1 = ml.solve(b0, x0=e0, tol=1e-30, maxiter=1, accel=None, cycle=cycle)

    num = energy_norm_sq(A, e1)
    den = energy_norm_sq(A, e0) + 1e-30
    return num / den


# ============================================================
# Ritz-value extraction via PCG-Lanczos (single probe run)
# ============================================================
def _pcg_ritz_values(A: sp.csr_matrix,
                     B,  # preconditioner LinearOperator (one V-cycle)
                     num_iters: int = 30,
                     seed: Optional[int] = None) -> np.ndarray:
    """
    Run `num_iters` steps of preconditioned CG on Ax=b with
    preconditioner B (= one AMG V-cycle) and extract Ritz values
    of the preconditioned operator BA from the CG Lanczos connection.

    CG  on the preconditioned system generates scalars (α_j, β_j)
    from which we build the Lanczos tridiagonal T_k whose eigenvalues
    are Ritz approximations of σ(BA).  See Meurant, Chapter 3.

    Args:
        A:  SPD system matrix.
        B:  Preconditioner (z = B @ r ≈ A⁻¹r).  Typically
            ``ml.aspreconditioner()``.
        num_iters: Number of CG iterations (= Lanczos steps).
        seed: RNG seed for the random RHS.

    Returns:
        Sorted array of positive Ritz values (may be shorter than
        `num_iters` if CG converges early).
    """
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    b = rng.standard_normal(n)
    b_norm = np.linalg.norm(b)

    x = np.zeros(n)
    r = b.copy()
    z = B @ r
    p = z.copy()
    rz = np.dot(r, z)

    alphas = []
    betas_cg = []

    for _ in range(min(num_iters, n)):
        Ap = A @ p
        pAp = np.dot(p, Ap)
        if abs(pAp) < 1e-30:
            break

        alpha = rz / pAp
        alphas.append(alpha)

        x = x + alpha * p
        r = r - alpha * Ap
        z = B @ r
        rz_new = np.dot(r, z)

        if abs(rz) < 1e-30:
            break

        beta = rz_new / rz
        betas_cg.append(beta)

        p = z + beta * p
        rz = rz_new

        if np.linalg.norm(r) < 1e-12 * b_norm:
            break

    k = len(alphas)
    if k < 2:
        return np.array([1.0])

    # ── Build k×k Lanczos tridiagonal from CG recurrence ──
    # T_jj   = 1/α_j  + β_{j-1}/α_{j-1}   (β_{-1} ≡ 0)
    # T_{j,j+1} = √|β_j| / α_j
    diag = np.zeros(k)
    off  = np.zeros(k - 1)

    diag[0] = 1.0 / alphas[0]
    for j in range(1, k):
        diag[j] = 1.0 / alphas[j] + betas_cg[j - 1] / alphas[j - 1]
    for j in range(k - 1):
        off[j] = np.sqrt(abs(betas_cg[j])) / alphas[j]

    T = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
    ritz = np.linalg.eigvalsh(T)
    ritz = ritz[ritz > 1e-10]  # keep only positive (SPD)
    return ritz if len(ritz) > 0 else np.array([1.0])


# ============================================================
# Multi-probe Ritz estimation (average over random RHS vectors)
# ============================================================
def estimate_ritz_values(A: sp.csr_matrix,
                         ml: pyamg.multilevel.MultilevelSolver,
                         num_iters: int = 30,
                         num_probes: int = 4) -> np.ndarray:
    """
    Run `num_probes` independent PCG probe solves and concatenate
    all Ritz values.  Using multiple random RHS vectors dramatically
    reduces variance in the extremal eigenvalue estimates.

    Returns: sorted 1-D array of all Ritz values from all probes.
    """
    B = ml.aspreconditioner()
    all_ritz = []
    for s in range(num_probes):
        ritz_s = _pcg_ritz_values(A, B, num_iters=num_iters, seed=s + 7)
        all_ritz.append(ritz_s)
    return np.sort(np.concatenate(all_ritz))


# ============================================================
# Spectral metrics from Ritz values of the preconditioned op BA
# ============================================================
def spectral_metrics(ritz: np.ndarray, trim_pct: int = 5,
                     outlier_delta: float = 0.3) -> Dict[str, float]:
    """
    Compute spectral quality metrics from Ritz values of BA.

    For ideal preconditioning BA ≈ I, all Ritz values should cluster
    near 1.  We compute several complementary metrics:

    Args:
        ritz: Sorted positive Ritz values (from estimate_ritz_values).
        trim_pct: Percentile to trim for κ_eff (default 5 → trim
                  bottom/top 5% of Ritz values).
        outlier_delta: Band half-width for outlier metric.

    Returns dict with keys:
        kappa_raw:     θ_max / θ_min  (noisy, for logging only)
        kappa_eff:     κ after trimming bottom/top trim_pct% of values
        log_kappa_eff: -log(κ_eff)  (main reward component)
        log_spread:    -max_i |log(θ_i)|   (how far any Ritz value
                       is from 1 on a log scale)
        log_variance:  -Var(log(θ_i))  (tightness of clustering)
        outlier_mass:  -mean(max(0, |θ_i - 1| - δ))  (fraction of
                       eigenvalues outside the band [1-δ, 1+δ])
        ritz_median:   median(θ)  (diagnostic)
    """
    if len(ritz) < 2:
        return {"kappa_raw": 1.0, "kappa_eff": 1.0,
                "log_kappa_eff": 0.0, "log_spread": 0.0,
                "log_variance": 0.0, "outlier_mass": 0.0,
                "ritz_median": 1.0}

    # --- raw κ (for logging; NOT for reward) ---
    kappa_raw = float(ritz[-1] / ritz[0])

    # --- effective κ (trimmed extremes) ---
    lo = max(1, int(np.ceil(len(ritz) * trim_pct / 100.0)))
    hi = len(ritz) - lo
    if hi <= lo:
        lo, hi = 0, len(ritz)
    kappa_eff = float(ritz[hi - 1] / max(ritz[lo], 1e-30))
    log_kappa_eff = -float(np.log(max(kappa_eff, 1.0)))

    # --- log-scale metrics (ideal: all log(θ) = 0) ---
    log_ritz = np.log(ritz)
    log_spread = -float(np.max(np.abs(log_ritz)))
    log_variance = -float(np.var(log_ritz))

    # --- outlier mass outside [1-δ, 1+δ] ---
    deviations = np.maximum(0.0, np.abs(ritz - 1.0) - outlier_delta)
    outlier_mass = -float(np.mean(deviations))

    return {
        "kappa_raw": kappa_raw,
        "kappa_eff": kappa_eff,
        "log_kappa_eff": log_kappa_eff,
        "log_spread": log_spread,
        "log_variance": log_variance,
        "outlier_mass": outlier_mass,
        "ritz_median": float(np.median(ritz)),
    }


# ============================================================
# Condition-number + clustering reward for RL
# ============================================================
def compute_reward(A: sp.csr_matrix, ml: pyamg.multilevel.MultilevelSolver,
                   condnum_iters: int = 30,
                   num_probes: int = 4,
                   complexity_target: float = 2.5,
                   complexity_penalty: float = 0.5,
                   reward_mode: str = "combined",
                   **_kwargs) -> Tuple[float, Dict[str, float]]:
    """
    RL reward based on preconditioned-operator spectrum + complexity cost.

    The reward combines spectral quality (how well BA ≈ I) with
    practical cost (operator complexity).  Three reward modes:

      'kappa_eff'  :  R = -log(κ_eff) - α·log(OC)
      'clustering' :  R = w1·log_variance + w2·outlier_mass - α·log(OC)
      'combined'   :  R = -log(κ_eff) + 0.3·log_variance + 0.2·outlier_mass
                         - α·log(OC)

    Using -log(OC) rather than hinge penalty gives a smoother gradient
    signal (AMG literature convention).

    Args:
        A:  System matrix.
        ml: PyAMG multilevel solver.
        condnum_iters: CG/Lanczos iterations per probe.
        num_probes: Number of random-RHS probe solves (variance reduction).
        complexity_target: Target operator complexity (OC).
        complexity_penalty: Weight α on log(OC) penalty.
        reward_mode: 'kappa_eff', 'clustering', or 'combined'.

    Returns:
        (reward_scalar, metrics_dict) — metrics_dict contains all
        spectral metrics + complexity info for TensorBoard logging.
    """
    # ── Estimate Ritz values of BA ──
    ritz = estimate_ritz_values(A, ml, num_iters=condnum_iters,
                                 num_probes=num_probes)
    metrics = spectral_metrics(ritz)

    # ── Operator complexity ──
    A0_nnz = ml.levels[0].A.nnz
    total_A_nnz = sum(lvl.A.nnz for lvl in ml.levels)
    oc = float(total_A_nnz / max(A0_nnz, 1))
    metrics["op_complexity"] = oc

    # ── Grid complexity ──
    n0 = ml.levels[0].A.shape[0]
    total_n = sum(lvl.A.shape[0] for lvl in ml.levels)
    gc = float(total_n / max(n0, 1))
    metrics["grid_complexity"] = gc

    # ── Complexity cost: -α·log(OC) (smooth, always negative for OC>1) ──
    cost = complexity_penalty * float(np.log(max(oc, 1.0)))

    # ── Base reward (depends on mode) ──
    if reward_mode == "kappa_eff":
        base = metrics["log_kappa_eff"]
    elif reward_mode == "clustering":
        base = 0.6 * metrics["log_variance"] + 0.4 * metrics["outlier_mass"]
    else:  # "combined" (default)
        base = (metrics["log_kappa_eff"]
                + 0.3 * metrics["log_variance"]
                + 0.2 * metrics["outlier_mass"])

    reward = base - cost
    metrics["reward"] = reward
    metrics["reward_base"] = base
    metrics["reward_cost"] = cost

    return reward, metrics


def cache_reward_test_vecs(train_set, **_kwargs) -> None:
    """No-op. Condition-number reward does not need cached test vectors."""
    print(f"  [cache] Spectral reward (Ritz-value based) — "
          f"no test-vector caching needed ({len(train_set)} samples)")


# ============================================================
# Fast PCG iteration-count reward (for Evolution Strategies)
# ============================================================
def compute_reward_iters(
    A: sp.csr_matrix,
    ml,
    tol: float = 1e-8,
    maxiter: int = 200,
    num_probes: int = 3,
    complexity_target: float = 2.5,
    complexity_penalty: float = 0.5,
    **_kwargs,
) -> Tuple[float, Dict[str, float]]:
    """
    Fast reward based on average PCG iteration count.

    Runs ``num_probes`` PCG solves with different random RHS vectors and
    averages the iteration counts.  Much faster than Ritz-value extraction
    because it skips eigenvalue decomposition.

    Returns
    -------
    reward : float
        ``-mean_iters - alpha * log(OC)``
    metrics : dict
        ``mean_iters``, ``max_iters``, ``min_iters``, ``converged``,
        ``op_complexity``, ``grid_complexity``
    """
    B = ml.aspreconditioner()
    n = A.shape[0]
    iters_list = []

    for s in range(num_probes):
        rng = np.random.RandomState(s + 42)
        b = rng.randn(n)
        b_norm = float(np.linalg.norm(b))

        x = np.zeros(n)
        r = b.copy()
        z = B @ r
        p = z.copy()
        rz = float(np.dot(r, z))
        its = maxiter  # default: did not converge

        for it in range(1, maxiter + 1):
            Ap = A @ p
            pAp = float(np.dot(p, Ap))
            if abs(pAp) < 1e-30:
                break
            alpha_cg = rz / pAp
            x = x + alpha_cg * p
            r = r - alpha_cg * Ap
            if np.linalg.norm(r) / b_norm < tol:
                its = it
                break
            z = B @ r
            rz_new = float(np.dot(r, z))
            beta_cg = rz_new / (rz + 1e-30)
            p = z + beta_cg * p
            rz = rz_new

        iters_list.append(its)

    mean_its = float(np.mean(iters_list))
    max_its = int(np.max(iters_list))
    min_its = int(np.min(iters_list))
    all_converged = all(i < maxiter for i in iters_list)

    # Operator complexity penalty
    A0_nnz = ml.levels[0].A.nnz
    total_A_nnz = sum(lvl.A.nnz for lvl in ml.levels)
    oc = float(total_A_nnz / max(A0_nnz, 1))
    n0 = ml.levels[0].A.shape[0]
    total_n = sum(lvl.A.shape[0] for lvl in ml.levels)
    gc = float(total_n / max(n0, 1))

    cost = complexity_penalty * float(np.log(max(oc, 1.0)))
    reward = -mean_its - cost

    metrics = {
        "mean_iters": mean_its,
        "max_iters": max_its,
        "min_iters": min_its,
        "converged": float(all_converged),
        "op_complexity": oc,
        "grid_complexity": gc,
        "reward": reward,
        "reward_cost": cost,
    }
    return reward, metrics


# ============================================================
# PCG iteration count
# ============================================================
def pcg_iterations(A: sp.csr_matrix, M, tol: float = 1e-8,
                   maxiter: int = 500) -> Tuple[int, List[float]]:
    n = A.shape[0]
    b = np.ones(n)
    residuals: List[float] = []
    def cb(xk):
        r = b - A @ xk
        residuals.append(np.linalg.norm(r))
    spla.cg(A, b, M=M, rtol=tol, maxiter=maxiter, callback=cb)
    return len(residuals), residuals


def pcg_iteration_count(A: sp.csr_matrix, M, tol: float = 1e-8,
                        maxiter: int = 500) -> int:
    """Same PCG solve as pcg_iterations (b = ones, x0 = 0, rtol=tol, maxiter),
    returning only the iteration count. scipy's cg calls the callback exactly
    once per iteration, so counting calls gives the same number as
    len(residuals) without the extra SpMV + norm per iteration."""
    n = A.shape[0]
    b = np.ones(n)
    count = [0]

    def cb(xk):
        count[0] += 1
    spla.cg(A, b, M=M, rtol=tol, maxiter=maxiter, callback=cb)
    return count[0]


def pcg_iteration_count_stag(A: sp.csr_matrix, ml, tol: float = 1e-8,
                             maxiter: int = 500, stag_start: int = 100,
                             stag_window: int = 50, stag_step: int = 25,
                             stag_margin: float = 1.5) -> int:
    """PCG iteration count with a stagnation stop, for ES population members.

    The CG recurrence is scipy.sparse.linalg.cg's (b = ones, x0 = 0,
    rtol = tol, atol = 0) operation for operation, preconditioned by one
    V-cycle of ``ml`` (as vcycle_preconditioner), so without a stop it
    returns pcg_iteration_count's value (checked on 19,296 evaluations).
    From iteration stag_start on, every stag_step iterations, the geometric
    residual reduction rate over the last stag_window iterations predicts the
    total count k + (log||r_k|| - log(tol*||b||)) / rate; if that exceeds
    stag_margin * maxiter (or the residual did not decrease over the window)
    the solve stops and counts maxiter, the value it was heading for."""
    b = np.ones(A.shape[0])
    atol = tol * np.linalg.norm(b)
    log_atol = np.log(atol)
    hist = np.empty(maxiter + 1)
    if len(ml.levels) == 1:
        A0, coarse = ml.levels[0].A, ml.coarse_solver

        def psolve(r):
            return coarse(A0, r)
    else:
        cycle = ml._MultilevelSolver__solve

        def psolve(r):
            z = np.zeros_like(r)
            cycle(0, z, r, "V", 1)
            return z
    x = np.zeros_like(b)
    r = b.copy()
    rho_prev, p = None, None
    for iteration in range(maxiter):
        nr = np.linalg.norm(r)
        if nr < atol:
            return iteration
        hist[iteration] = nr
        if (iteration >= stag_start and iteration >= stag_window
                and (iteration - stag_start) % stag_step == 0):
            rate = (np.log(hist[iteration - stag_window]) - np.log(nr)) / stag_window
            if rate <= 0 or iteration + (np.log(nr) - log_atol) / rate > stag_margin * maxiter:
                return maxiter
        z = psolve(r)
        rho_cur = np.dot(r, z)
        if iteration > 0:
            beta = rho_cur / rho_prev
            p *= beta
            p += z
        else:
            p = np.empty_like(r)
            p[:] = z[:]
        q = A @ p
        alpha = rho_cur / np.dot(p, q)
        x += alpha * p
        r -= alpha * q
        rho_prev = rho_cur
    return maxiter


def pcg_fixed_budget(A: sp.csr_matrix, M, b: np.ndarray, K: int,
                     x0: Optional[np.ndarray] = None) -> List[float]:
    """
    Run up to K steps of preconditioned CG and return residual norms.

    Unlike pcg_iterations(), this accepts a custom RHS `b` and initial
    guess `x0`, and targets K iterations (tol ≈ 0).  CG may converge
    early on small/well-conditioned systems; callers should handle
    len(result) < K via padding.

    Returns:
        List of length ≤ K with ||b - A x_t||_2 after each iteration.
    """
    residuals: List[float] = []
    def cb(xk):
        r = b - A @ xk
        residuals.append(float(np.linalg.norm(r)))
    _, info = spla.cg(A, b, x0=x0, M=M, rtol=1e-30, maxiter=K, callback=cb)
    if info < 0:
        import warnings
        warnings.warn(f"pcg_fixed_budget: CG breakdown (info={info}), "
                      f"got {len(residuals)}/{K} residuals")
    return residuals


# ============================================================
# C overlap statistics
# ============================================================
def compute_C_overlap_stats(C1: sp.csr_matrix, C2: sp.csr_matrix) -> Dict[str, float]:
    """Compare two binary strength matrices. Returns IoU and related stats."""
    c1 = (np.abs(C1) > 1e-9).astype(int)
    c2 = (np.abs(C2) > 1e-9).astype(int)

    inter = c1.multiply(c2)
    nnz_inter = inter.nnz
    nnz_c1 = c1.nnz
    nnz_c2 = c2.nnz

    nnz_union = nnz_c1 + nnz_c2 - nnz_inter
    iou = nnz_inter / max(nnz_union, 1)

    print(f"--- C Overlap Stats ---")
    print(f"  C_std edges:    {nnz_c1}")
    print(f"  C_lrn edges:    {nnz_c2}")
    print(f"  Intersection:   {nnz_inter}")
    print(f"  IoU:            {iou:.4f}")
    print(f"  % Std in Lrn:   {nnz_inter/max(nnz_c1, 1):.4f}")
    print(f"  % Lrn in Std:   {nnz_inter/max(nnz_c2, 1):.4f}")
    print(f"-----------------------")

    return {
        "iou": iou,
        "n_std": nnz_c1,
        "n_lrn": nnz_c2,
        "n_inter": nnz_inter
    }


# ============================================================
# Deterministic top-k C builder (for evaluation)
# ============================================================
def deterministic_topk_C(A: sp.csr_matrix, model, x: torch.Tensor,
                         edge_index: torch.Tensor, edge_weight: torch.Tensor,
                         k_per_row: int, device: str = "cpu",
                         B_base: Optional[np.ndarray] = None) -> Tuple[sp.csr_matrix, Optional[np.ndarray]]:
    """
    Build C using deterministic top-k logit selection (no sampling).
    This is the shared core used by all build_C_from_model variants.

    Args:
        A: System matrix.
        model: AMGEdgePolicy model.
        x: Node features tensor (already built by caller).
        edge_index, edge_weight: Graph connectivity (already normalized by caller).
        k_per_row: Number of edges to select per row.
        device: Torch device.
        B_base: Optional base nullspace (e.g. rigid body modes for elasticity).

    Returns:
        (C, B) tuple.
    """
    ei = edge_index.to(device)
    ew = edge_weight.to(device)
    x = x.to(device)

    model.eval()
    with torch.no_grad():
        logits, B_extra = model(x, ei, ew)

    row_groups = build_row_groups(ei.cpu(), num_nodes=A.shape[0])
    selected = np.zeros(A.nnz, dtype=bool)
    logits_np = logits.cpu().numpy()

    for i, edges_i in enumerate(row_groups):
        if edges_i.numel() == 0:
            continue
        edges = edges_i.numpy()
        kk = min(k_per_row, edges.shape[0])
        if kk <= 0:
            continue

        vals = logits_np[edges]
        top_indices = np.argsort(vals)[::-1]
        top = top_indices[:kk]
        selected[edges[top]] = True

    C = C_from_selected_edges(A, ei.cpu(), selected)
    B = build_B_for_pyamg(B_extra, B_base=B_base)
    return C, B


# ============================================================
# Neighbor statistics
# ============================================================
def neighbor_stats(
    A: sp.csr_matrix,
    save_path: Optional[str] = None,
    title: str = "Neighbor Count Distribution",
) -> Dict:
    """
    Compute per-node neighbor counts from A (off-diagonal nonzeros per row)
    and print summary statistics.  Optionally save a histogram.

    Returns a dict with: min, max, mean, median, mode, p25, p75, histogram_path.
    Use this to make an informed choice for k_per_row.
    """
    A_csr = sp.csr_matrix(A)
    n = A_csr.shape[0]

    nnz_per_row = np.diff(A_csr.indptr)
    diag_mask = np.zeros(n, dtype=int)
    for i in range(n):
        start, end = A_csr.indptr[i], A_csr.indptr[i + 1]
        cols = A_csr.indices[start:end]
        if i in cols:
            diag_mask[i] = 1
    neighbors = nnz_per_row - diag_mask

    stats = {
        "min": int(neighbors.min()),
        "max": int(neighbors.max()),
        "mean": float(neighbors.mean()),
        "median": float(np.median(neighbors)),
        "p25": float(np.percentile(neighbors, 25)),
        "p75": float(np.percentile(neighbors, 75)),
        "std": float(neighbors.std()),
        "histogram_path": None,
    }

    vals, counts = np.unique(neighbors, return_counts=True)
    stats["mode"] = int(vals[np.argmax(counts)])

    print(f"\n  ┌─── Neighbor Count Stats (off-diag nnz per row) ───")
    print(f"  │  Nodes:   {n}")
    print(f"  │  Min:     {stats['min']}")
    print(f"  │  Max:     {stats['max']}")
    print(f"  │  Mean:    {stats['mean']:.2f}")
    print(f"  │  Median:  {stats['median']:.0f}")
    print(f"  │  Mode:    {stats['mode']}")
    print(f"  │  Std:     {stats['std']:.2f}")
    print(f"  │  P25/P75: {stats['p25']:.0f} / {stats['p75']:.0f}")
    print(f"  │")
    print(f"  │  Suggested k_per_row range: [{stats['min']}..{stats['mode']}]")
    print(f"  │  (k_per_row > mode has no effect for most nodes)")
    print(f"  └───────────────────────────────────────────────────")

    if save_path is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ensure_dir(os.path.dirname(save_path) or ".")
        fig, ax = plt.subplots(figsize=(7, 4))
        unique_vals = np.arange(stats["min"], stats["max"] + 1)
        counts_full = np.array([(neighbors == v).sum() for v in unique_vals])

        ax.bar(unique_vals, counts_full, color="steelblue", edgecolor="white", linewidth=0.5)
        ax.axvline(stats["mean"], color="red", linestyle="--", linewidth=1.5,
                   label=f"mean = {stats['mean']:.1f}")
        ax.axvline(stats["median"], color="orange", linestyle="-.", linewidth=1.5,
                   label=f"median = {stats['median']:.0f}")
        ax.set_xlabel("Number of Neighbors (off-diag nnz)")
        ax.set_ylabel("Node Count")
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.set_xticks(unique_vals)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(save_path, dpi=200)
        plt.close(fig)

        stats["histogram_path"] = save_path
        print(f"  Histogram saved to {save_path}")

    return stats


# ============================================================
# Misc utilities
# ============================================================
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
