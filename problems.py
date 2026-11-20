"""PDE test problems and dataset construction for LSC-AMG.

Finite-element stiffness matrices (scikit-fem, P1 on random triangular meshes)
for the problems in the paper, plus make_spectral_train_set(), which turns them
into graph samples (node/edge features, PyG edge_index) for the policy.

  poisson       -lap u = f on [0,1]^2
  anisotropy    -div(K grad u), K = R(a)^T diag(1, eps) R(a), eps~logU[1e-3,1e-1]
  lognormal     -div(a grad u), a = exp(8 Z), Z Gaussian field, corr. length 0.1
  perforated_l  L-domain with 3-8 holes, high-contrast rings (gamma up to 1e5)
                and radial anisotropy near the re-entrant corner

`n_grid` sets the target number of unknowns. For perforated_l the condensed
system has ~1.47 x n_grid unknowns (mesh max_area = 0.75 / (2 n_grid) on an
L-domain of area 0.75), so use n_grid = n / 1.4745 for a target size n.

Extracted verbatim from the legacy training script
seven_rl_eigen_copy_manually_edited_minibatch_track_P_upgradecopyLOG.py in the git history
(removed); tests/test_problems.py checks that it produces byte-identical
matrices and features.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import matplotlib.tri as mtri
import numpy as np
import scipy.sparse as sp
import torch
from scipy.spatial import Delaunay
from skfem import Basis, MeshTri, asm, condense
from skfem.element import ElementTriP1
from skfem.models.poisson import laplace
from torch_geometric.utils import from_scipy_sparse_matrix

from amg_core import (
    FeatureConfig,
    build_node_coords,
    build_row_groups,
    build_sample_features,
    node_features_for_policy,
)


@dataclass
class SpectralTrainSample:
    A: sp.csr_matrix
    grid_n: int
    coords: np.ndarray
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    x: torch.Tensor
    row_groups: List[torch.Tensor]
    tag: str = ""
    raw_edge_features: Optional[torch.Tensor] = None
    full_pts: Optional[np.ndarray] = None        # all mesh points (boundary+interior), (N_all, 2)
    full_simplices: Optional[np.ndarray] = None   # triangle connectivity, (T, 3)
    feature_seed: Optional[int] = None   # seed of the relaxed probes in x / raw_edge_features


COARSE_NUM_POINTS = 30_000


def build_poisson_sample(rng, n_edge=25, grid_n=64, n_grid=None):
    if n_grid is None:
        n_grid = COARSE_NUM_POINTS
    # -----------------------------
    # 1. Build strange irregular mesh
    # -----------------------------
    s = np.linspace(0, 1, n_edge)
    boundary = np.vstack([
        np.column_stack([s, np.zeros_like(s)]),
        np.column_stack([s, np.ones_like(s)]),
        np.column_stack([np.zeros_like(s), s]),
        np.column_stack([np.ones_like(s), s])
    ])

    coarse = rng.random((n_grid, 2))

    cloud1 = 0.12 * rng.standard_normal((120, 2)) + np.array([0.30, 0.75])
    cloud2 = 0.06 * rng.standard_normal((100, 2)) + np.array([0.78, 0.22])

    pts = np.vstack([boundary, coarse, cloud1, cloud2])
    pts = pts[
        (pts[:, 0] >= 0) & (pts[:, 0] <= 1) &
        (pts[:, 1] >= 0) & (pts[:, 1] <= 1)
    ]

    # remove duplicates
    pts = np.unique(np.round(pts, 12), axis=0)

    tri = Delaunay(pts)
    triang = mtri.Triangulation(pts[:, 0], pts[:, 1], tri.simplices)

    # -----------------------------
    # 2. Convert to scikit-fem mesh
    # -----------------------------
    # p: (2, npts), t: (3, ntri)
    mesh = MeshTri(pts.T, tri.simplices.T)

    # -----------------------------
    # 3. P1 basis and FEM stiffness
    # -----------------------------
    element = ElementTriP1()
    basis = Basis(mesh, element)

    # Full stiffness matrix (sparse CSR/CSC-like scipy sparse)
    A_full = asm(laplace, basis)

    # -----------------------------
    # 4. Zero Dirichlet BCs
    # -----------------------------
    boundary_dofs = basis.get_dofs().all()
    all_dofs = np.arange(basis.N)
    interior = np.setdiff1d(all_dofs, boundary_dofs)

    # Condensed interior matrix
    A_bc = condense(A_full, D=boundary_dofs)
    if isinstance(A_bc, tuple):
        A_bc = A_bc[0]

    # Coordinates aligned with A_bc
    coords = mesh.p.T[interior]   # shape (n_interior, 2)

    return A_bc.tocsr(), coords, triang


def build_lognormal_diffusion_sample(rng, n_edge=25, sigma_ln=8.0, corr_length=0.1, n_grid = 32):
    """
    Build an SPD matrix from  -div(kappa(x) grad u) = f  on an irregular P1 mesh,
    where kappa(x) = exp(sigma_ln * Z(x)) is a log-normal random field.

    Z is a smooth Gaussian field approximated on a coarse overlay grid and
    interpolated to quadrature points.  This models heterogeneous porous-media
    permeability and is extremely hard for standard SA-AMG (V-cycle CF ~ 0.99).

    Args:
        rng: numpy random generator
        n_edge: boundary discretization density
        sigma_ln: std-dev of log(kappa).  Higher = wider contrast.
                  sigma_ln=8 gives kappa range ~ 10^7.
        corr_length: spatial correlation length of the random field.
        n_grid: number of mesh points
    Returns:
        A_bc (csr_matrix): condensed interior stiffness matrix (SPD)
        coords (ndarray): interior node coordinates, shape (n_interior, 2)
        triang: matplotlib triangulation for visualization
    """
    from skfem import BilinearForm

    # ── 1. Build irregular mesh (same as Poisson) ──
    s = np.linspace(0, 1, n_edge)
    boundary = np.vstack([
        np.column_stack([s, np.zeros_like(s)]),
        np.column_stack([s, np.ones_like(s)]),
        np.column_stack([np.zeros_like(s), s]),
        np.column_stack([np.ones_like(s), s]),
    ])
    coarse = rng.random((n_grid, 2))
    cloud1 = 0.12 * rng.standard_normal((120, 2)) + np.array([0.30, 0.75])
    cloud2 = 0.06 * rng.standard_normal((100, 2)) + np.array([0.78, 0.22])

    pts = np.vstack([boundary, coarse, cloud1, cloud2])
    pts = pts[
        (pts[:, 0] >= 0) & (pts[:, 0] <= 1) &
        (pts[:, 1] >= 0) & (pts[:, 1] <= 1)
    ]
    pts = np.unique(np.round(pts, 12), axis=0)

    tri = Delaunay(pts)
    triang = mtri.Triangulation(pts[:, 0], pts[:, 1], tri.simplices)
    mesh = MeshTri(pts.T, tri.simplices.T)
    element = ElementTriP1()
    basis = Basis(mesh, element)

    # ── 2. Generate coarse Gaussian random field Z on overlay grid ──
    ng = max(4, int(1.0 / corr_length))  # overlay cells per side
    z_coarse = rng.standard_normal((ng + 1, ng + 1))

    # ── 3. Bilinear form with log-normal coefficient ──
    @BilinearForm
    def lognorm_diffusion(u, v, w):
        x, y = w.x[0], w.x[1]
        # Map quadrature points to overlay grid cells → interpolate Z
        ix = np.clip((x * ng).astype(int), 0, ng - 1)
        iy = np.clip((y * ng).astype(int), 0, ng - 1)
        kappa = np.exp(sigma_ln * z_coarse[ix, iy])
        return kappa * (u.grad[0] * v.grad[0] + u.grad[1] * v.grad[1])

    # ── 4. Assemble and apply Dirichlet BCs ──
    A_full = asm(lognorm_diffusion, basis)
    boundary_dofs = basis.get_dofs().all()
    all_dofs = np.arange(basis.N)
    interior = np.setdiff1d(all_dofs, boundary_dofs)

    A_bc = condense(A_full, D=boundary_dofs)
    if isinstance(A_bc, tuple):
        A_bc = A_bc[0]

    coords = mesh.p.T[interior]
    return A_bc.tocsr(), coords, triang


def build_anisotropic_diffusion_sample(rng, n_edge=25, eps=None, theta=None, n_grid=None):
    """
    Build an SPD matrix from -div(K grad u) = f on an irregular P1 mesh.

    The diffusion tensor is K = R(theta)^T diag(1, eps) R(theta), where
    theta is the rotation angle and eps << 1 controls anisotropy strength.
    This is much harder for standard SA-AMG than isotropic Poisson because
    the strong-connection direction is rotated away from mesh axes.

    Args:
        rng: numpy random generator
        n_edge: boundary discretization density
        eps: anisotropy ratio in (0, 1]. None => random in [0.001, 0.1].
        theta: rotation angle in radians. None => random in [0, pi).

    Returns:
        A_bc (csr_matrix): condensed interior stiffness matrix (SPD)
        coords (ndarray): interior node coordinates, shape (n_interior, 2)
        triang: matplotlib triangulation for visualization
    """
    from skfem import BilinearForm

    # Random anisotropy parameters if not specified
    if n_grid is None:
        n_grid = COARSE_NUM_POINTS
    if eps is None:
        eps = float(10 ** rng.uniform(-3, -1))   # log-uniform in [0.001, 0.1]
    if theta is None:
        theta = float(rng.uniform(0, np.pi))

    # ── 1. Build irregular mesh (same as Poisson) ──
    s = np.linspace(0, 1, n_edge)
    boundary = np.vstack([
        np.column_stack([s, np.zeros_like(s)]),
        np.column_stack([s, np.ones_like(s)]),
        np.column_stack([np.zeros_like(s), s]),
        np.column_stack([np.ones_like(s), s]),
    ])
    coarse = rng.random((n_grid, 2))
    cloud1 = 0.12 * rng.standard_normal((120, 2)) + np.array([0.30, 0.75])
    cloud2 = 0.06 * rng.standard_normal((100, 2)) + np.array([0.78, 0.22])

    pts = np.vstack([boundary, coarse, cloud1, cloud2])
    pts = pts[
        (pts[:, 0] >= 0) & (pts[:, 0] <= 1) &
        (pts[:, 1] >= 0) & (pts[:, 1] <= 1)
    ]
    pts = np.unique(np.round(pts, 12), axis=0)

    tri = Delaunay(pts)
    triang = mtri.Triangulation(pts[:, 0], pts[:, 1], tri.simplices)
    mesh = MeshTri(pts.T, tri.simplices.T)
    element = ElementTriP1()
    basis = Basis(mesh, element)

    # ── 2. Rotated anisotropic diffusion tensor ──
    c, sn = np.cos(theta), np.sin(theta)
    K00 = c**2 + eps * sn**2
    K01 = (1 - eps) * c * sn
    K11 = sn**2 + eps * c**2

    @BilinearForm
    def aniso_diffusion(u, v, w):
        return (K00 * u.grad[0] * v.grad[0] +
                K01 * u.grad[0] * v.grad[1] +
                K01 * u.grad[1] * v.grad[0] +
                K11 * u.grad[1] * v.grad[1])

    # ── 3. Assemble and apply Dirichlet BCs ──
    A_full = asm(aniso_diffusion, basis)
    boundary_dofs = basis.get_dofs().all()
    all_dofs = np.arange(basis.N)
    interior = np.setdiff1d(all_dofs, boundary_dofs)

    A_bc = condense(A_full, D=boundary_dofs)
    if isinstance(A_bc, tuple):
        A_bc = A_bc[0]

    coords = mesh.p.T[interior]
    return A_bc.tocsr(), coords, triang


def build_perforated_ldomain_sample(rng, n_edge=30, n_grid=None, **kwargs):
    """
    Build an SPD matrix from -div(K grad u) = f on a perforated L-domain.

    The domain is L-shaped with circular holes, high-contrast coefficient
    rings around holes, and anisotropy near the re-entrant corner at (0.5,0.5).

    Parameters are randomized per sample for training diversity:
      n_holes ~ Uniform(3, 8)
      contrast ~ LogUniform(10, 1e5)
      min_angle ~ Uniform(15, 25) degrees

    Args:
        n_grid: Controls mesh density.  Maps to max_area = 0.75 / (2*n_grid)
                so that n_grid points ≈ n_grid interior DOFs (same semantics
                as Poisson/anisotropy/lognormal builders).

    Returns:
        A_bc (csr_matrix): condensed interior stiffness matrix (SPD)
        coords (ndarray): interior node coordinates, shape (n_interior, 2)
        triang: matplotlib triangulation for visualization
    """
    from skfem import BilinearForm
    from perforated_l_mesh import build_perforated_l_mesh

    # Randomize parameters
    n_holes = int(rng.integers(3, 9))  # 3..8
    contrast = float(10 ** rng.uniform(1, 5))  # 10..1e5
    min_angle = float(rng.uniform(15, 25))

    # Convert n_grid to max_area (L-domain area ≈ 0.75)
    if n_grid is None:
        n_grid = COARSE_NUM_POINTS
    max_area = 0.75 / (2.0 * n_grid)

    points, triangles, boundary_nodes, hole_centers, hole_radii = \
        build_perforated_l_mesh(rng, n_holes=n_holes, min_angle=min_angle,
                                max_area=max_area)

    triang = mtri.Triangulation(points[:, 0], points[:, 1], triangles)
    mesh = MeshTri(points.T, triangles.T)
    element = ElementTriP1()
    basis = Basis(mesh, element)

    boundary_dofs = np.array(sorted(boundary_nodes))
    n_holes_actual = len(hole_centers)

    @BilinearForm
    def perforated_l_diff(u, v, w):
        x, y = w.x[0], w.x[1]
        kappa = np.ones_like(x)

        # High-contrast rings around holes
        for hi in range(n_holes_actual):
            cx, cy = hole_centers[hi]
            r_hole = hole_radii[hi]
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            mask = dist < 2.0 * r_hole
            kappa = np.where(mask, contrast, kappa)

        # Near re-entrant corner: anisotropic
        r_corner = np.sqrt((x - 0.5)**2 + (y - 0.5)**2)
        corner_mask = r_corner < 0.15
        theta = np.arctan2(y - 0.5, x - 0.5)
        eps_corner = 0.01
        c = np.cos(theta)
        s = np.sin(theta)
        K00_c = c**2 + eps_corner * s**2
        K01_c = (1 - eps_corner) * c * s
        K11_c = s**2 + eps_corner * c**2

        K00 = np.where(corner_mask, kappa * K00_c, kappa)
        K01 = np.where(corner_mask, kappa * K01_c, np.zeros_like(kappa))
        K11 = np.where(corner_mask, kappa * K11_c, kappa)

        return (K00 * u.grad[0] * v.grad[0] +
                K01 * u.grad[0] * v.grad[1] +
                K01 * u.grad[1] * v.grad[0] +
                K11 * u.grad[1] * v.grad[1])

    A_full = asm(perforated_l_diff, basis)
    A_bc = condense(A_full, D=boundary_dofs)
    if isinstance(A_bc, tuple):
        A_bc = A_bc[0]

    interior = np.setdiff1d(np.arange(basis.N), boundary_dofs)
    coords = mesh.p.T[interior]
    return A_bc.tocsr(), coords, triang


def make_spectral_train_set(num_samples: int = 30, n_grid: int = 32,
                             seed: int = 0,
                             feature_config: Optional[FeatureConfig] = None,
                             n_edge: int = 72,
                             problem: str = "lognormal") -> List[SpectralTrainSample]:
    """
    Build training set of diffusion problems with varying meshes.

    problem : str
        One of 'poisson', 'anisotropy', 'lognormal'.
    """
    builders = {
        "poisson": build_poisson_sample,
        "anisotropy": build_anisotropic_diffusion_sample,
        "lognormal": build_lognormal_diffusion_sample,
        "perforated_l": build_perforated_ldomain_sample,
    }
    if problem not in builders:
        raise ValueError(f"Unknown problem {problem!r}. "
                         f"Choose from {list(builders.keys())}")
    builder = builders[problem]

    # Seed the sample generator from the `seed` argument so the dataset is
    # reproducible: same seed -> same meshes/coefficients, different seeds ->
    # independent draws. (Previously this used
    # np.random.default_rng(np.random.randint(0, 1000)), which drew from
    # numpy's unseeded global RNG and silently ignored `seed`.)
    rng = np.random.default_rng(seed)

    samples: List[SpectralTrainSample] = []
    for i in range(num_samples):
        if i == 0:
            print(f"  Problem type: {problem} \u2192 {builder.__name__}()")

        A, coords, triang = builder(rng=rng, n_edge=n_edge, n_grid=n_grid)

        # -----------------------------
        # PyG conversion
        # -----------------------------
        ei, ew = from_scipy_sparse_matrix(A) # edge indicies and edge weights
        ew = ew.float()

        # Extract maximum weight for normalization (important for stable GNN training)
        max_w = ew.abs().max()
        if max_w > 0:
            ew = ew / max_w # normalize edge weights to [-1, 1] for stable GNN training. 
            # NOTE: If this is a global normalization, would it be better to do row-wise normalization? 
            # That way the GNN can learn a more consistent edge selection pattern across different samples, 
            # rather than having to adapt to varying weight scales in each sample. 
            # This is something to experiment with later. For now, we'll stick with global normalization for simplicity.

        # -----------------------------
        # Your downstream hooks
        # -----------------------------
        raw_ef = None
        feature_seed = None
        if feature_config is not None:
            # Per-sample feature seed derived from (seed, i), so a sample's
            # random relaxed probes are the same in every process.
            feature_seed = int(seed) * 100003 + i
            x, raw_ef = build_sample_features(A, coords, ei, feature_config,
                                              seed=feature_seed)
        else:
            x = node_features_for_policy(A, coords, num_vecs_in=4, iters_in=5)
        row_groups = build_row_groups(ei, num_nodes=A.shape[0])

        samples.append(SpectralTrainSample(
            A=A, grid_n=A.shape[0], coords=coords,
            edge_index=ei, edge_weight=ew, x=x,
            row_groups=row_groups, tag=f"{problem}_irregular",
            raw_edge_features=raw_ef,
            full_pts=np.column_stack([triang.x, triang.y]),
            full_simplices=triang.triangles,
            feature_seed=feature_seed,
        ))

    return samples
