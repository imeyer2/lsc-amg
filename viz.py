""" 
viz.py

Shared visualization functions used by all training/evaluation scripts:
  - Sparsity patterns (spy, density heatmap)
  - Strength matrix C comparison (overlay heatmap, spy zoom)
  - Iteration bar charts
  - Eigenvalue spectra and histograms
  - Weight-vs-logit scatter
  - Training curves (reward + probe)
"""

from __future__ import annotations

import os
from typing import List, Tuple, Optional, Dict

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from amg_core import ensure_dir


# ============================================================
# Mesh plotting
# ============================================================
def plot_mesh(coords: np.ndarray, save_path: str,
              triangles: Optional[np.ndarray] = None,
              grid_n: Optional[int] = None,
              title: Optional[str] = None,
              boundary_ids: Optional[np.ndarray] = None,
              figsize: Tuple[float, float] = (6, 6)):
    """
    Plot a 2D mesh: either a structured n×n grid or an unstructured triangulation.

    Args:
        coords:       (N, 2) node coordinates.
        save_path:    Where to save the figure.
        triangles:    (T, 3) triangle connectivity.  If given, uses triplot.
                      If None and grid_n is given, draws structured grid lines.
        grid_n:       Side length of a structured grid (only used when triangles is None).
        title:        Optional figure title.
        boundary_ids: Indices of boundary nodes (highlighted in red).
        figsize:      Figure size.
    """
    import matplotlib.pyplot as plt

    ensure_dir(os.path.dirname(save_path) or ".")
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if triangles is not None:
        # ---- Unstructured triangulation ----
        ax.triplot(coords[:, 0], coords[:, 1], triangles, linewidth=0.4, color="steelblue")
        ax.plot(coords[:, 0], coords[:, 1], 'k.', markersize=1.5)
        if boundary_ids is not None:
            ax.plot(coords[boundary_ids, 0], coords[boundary_ids, 1],
                    'r.', markersize=3, label="boundary")
            ax.legend(fontsize=8)
        mesh_label = f"Unstructured ({coords.shape[0]} nodes, {triangles.shape[0]} triangles)"
    elif grid_n is not None:
        # ---- Structured grid ----
        for i in range(grid_n):
            row_start = i * grid_n
            row_end = row_start + grid_n
            ax.plot(coords[row_start:row_end, 0], coords[row_start:row_end, 1],
                    'steelblue', linewidth=0.5)
        for j in range(grid_n):
            col_ids = np.arange(j, grid_n * grid_n, grid_n)
            ax.plot(coords[col_ids, 0], coords[col_ids, 1],
                    'steelblue', linewidth=0.5)
        ax.plot(coords[:, 0], coords[:, 1], 'k.', markersize=1.0)
        mesh_label = f"Structured {grid_n}×{grid_n} ({coords.shape[0]} nodes)"
    else:
        # ---- Fallback: just scatter the nodes ----
        ax.plot(coords[:, 0], coords[:, 1], 'k.', markersize=1.5)
        mesh_label = f"Nodes ({coords.shape[0]})"

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title or f"Mesh: {mesh_label}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Mesh plot saved to: {save_path}")


# ============================================================
# Training-set mesh visualization (for papers)
# ============================================================
def plot_mesh_triangulation(coords_full: np.ndarray,
                            simplices: np.ndarray,
                            save_path: str,
                            title: Optional[str] = None,
                            figsize: Tuple[float, float] = (6, 6)):
    """
    Plot the full FEM triangulation (boundary + interior nodes).

    Args:
        coords_full: (N_all, 2) all mesh node coordinates.
        simplices:   (T, 3)     triangle connectivity.
        save_path:   Output file path.
    """
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    ensure_dir(os.path.dirname(save_path) or ".")
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    triang = mtri.Triangulation(coords_full[:, 0], coords_full[:, 1], simplices)
    ax.triplot(triang, linewidth=0.3, color="steelblue")
    ax.plot(coords_full[:, 0], coords_full[:, 1], 'k.', markersize=0.8)

    n_nodes = coords_full.shape[0]
    n_tri = simplices.shape[0]
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title or f"FEM Mesh ({n_nodes} nodes, {n_tri} triangles)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_mesh_graph(coords: np.ndarray,
                    edge_index: "torch.Tensor",
                    save_path: str,
                    title: Optional[str] = None,
                    figsize: Tuple[float, float] = (6, 6)):
    """
    Plot the stiffness-graph connectivity (interior nodes only).

    Args:
        coords:     (N, 2) interior node coordinates.
        edge_index: (2, E) edge indices (PyG format).
        save_path:  Output file path.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    ensure_dir(os.path.dirname(save_path) or ".")
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    # Only draw each undirected edge once
    mask = src < dst
    segments = np.stack([coords[src[mask]], coords[dst[mask]]], axis=1)
    lc = LineCollection(segments, linewidths=0.25, colors="steelblue",
                        alpha=0.6)
    ax.add_collection(lc)
    ax.plot(coords[:, 0], coords[:, 1], 'k.', markersize=0.8)

    ax.set_aspect("equal")
    ax.set_xlim(coords[:, 0].min() - 0.02, coords[:, 0].max() + 0.02)
    ax.set_ylim(coords[:, 1].min() - 0.02, coords[:, 1].max() + 0.02)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    n_nodes = coords.shape[0]
    n_edges = int(mask.sum())
    ax.set_title(title or f"Stiffness Graph ({n_nodes} nodes, {n_edges} edges)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_training_meshes(train_set, out_dir: str,
                         num_samples: int = 4, seed: int = 0):
    """
    Save mesh visualizations for a few training samples at the start of a run.

    For each selected sample, saves:
      - mesh_{i}_triangulation.png  (full FEM mesh)
      - mesh_{i}_graph.png          (stiffness connectivity the GNN sees)

    Args:
        train_set:   List of SpectralTrainSample.
        out_dir:     Run output directory.
        num_samples: How many samples to plot.
        seed:        Used for selecting samples (evenly spaced).
    """
    mesh_dir = os.path.join(out_dir, "mesh_plots")
    ensure_dir(mesh_dir)

    n = len(train_set)
    num_samples = min(num_samples, n)
    # Evenly spaced indices through the training set
    indices = np.linspace(0, n - 1, num_samples, dtype=int)

    saved = []
    for idx in indices:
        s = train_set[idx]

        # Full triangulation plot
        if s.full_pts is not None and s.full_simplices is not None:
            path_tri = os.path.join(mesh_dir, f"mesh_{idx}_triangulation.png")
            plot_mesh_triangulation(
                s.full_pts, s.full_simplices, path_tri,
                title=f"Sample {idx}: FEM Mesh "
                      f"({s.full_pts.shape[0]} nodes, "
                      f"{s.full_simplices.shape[0]} triangles)",
            )
            saved.append(path_tri)

        # Stiffness graph plot
        path_graph = os.path.join(mesh_dir, f"mesh_{idx}_graph.png")
        plot_mesh_graph(
            s.coords, s.edge_index, path_graph,
            title=f"Sample {idx}: Stiffness Graph "
                  f"({s.coords.shape[0]} interior nodes)",
        )
        saved.append(path_graph)

    print(f"Saved {len(saved)} mesh plots to {mesh_dir}/")


# ============================================================
# C evolution GIF  (edge-strength on mesh)
# ============================================================
def save_c_evolution_gif(
    snapshots: List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]],
    save_path: str,
    fps: int = 3,
    title_prefix: str = "Edge Strengths",
    figsize: Tuple[float, float] = (6, 6),
):
    """
    Animated GIF showing predicted edge strengths on the mesh through training.

    Each frame draws the mesh edges colored by sigmoid(logit) from cool (weak)
    to hot (strong), so you can see the spatial pattern evolve.

    Args:
        snapshots: List of (epoch, coords, edge_index, strengths) tuples.
                   - coords:     (N, 2)  node positions
                   - edge_index: (2, E)  numpy arrays
                   - strengths:  (E,)    sigmoid(logit) values in [0, 1]
        save_path: Output .gif path.
        fps:       Frames per second.
        title_prefix: Title prefix for each frame.
        figsize:   Figure size.
    """
    from PIL import Image
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.collections import LineCollection
    import io

    if not snapshots:
        print("No strength snapshots to animate.")
        return

    ensure_dir(os.path.dirname(save_path) or ".")

    # Use a perceptually uniform colormap: blue (weak) -> red (strong)
    cmap = plt.cm.coolwarm

    frames: List[Image.Image] = []

    for epoch, coords, ei, strengths in snapshots:
        row, col = ei[0], ei[1]
        # Only off-diagonal, and only upper-triangle to avoid drawing each edge twice
        mask = row < col
        r, c, s = row[mask], col[mask], strengths[mask]

        # Build line segments
        segments = np.stack([coords[r], coords[c]], axis=1)  # (n_edges, 2, 2)

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        lc = LineCollection(segments, cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=1))
        lc.set_array(s)
        lc.set_linewidth(0.8)
        ax.add_collection(lc)

        # Nodes
        ax.plot(coords[:, 0], coords[:, 1], 'k.', markersize=0.8, zorder=2)

        ax.set_xlim(coords[:, 0].min() - 0.02, coords[:, 0].max() + 0.02)
        ax.set_ylim(coords[:, 1].min() - 0.02, coords[:, 1].max() + 0.02)
        ax.set_aspect("equal")
        ax.set_title(f"{title_prefix}  |  epoch {epoch}", fontsize=11)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        cb = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("σ(logit)  [0=weak, 1=strong]", fontsize=8)

        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120)
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()

    duration_ms = int(1000 / fps)
    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"C evolution GIF saved to: {save_path}  ({len(frames)} frames)")


# ============================================================
# Sparse pattern visualization
# ============================================================
def plot_sparse_pattern(M: sp.spmatrix, title: str, save_path: str, markersize: float = 0.5,
                        max_rows: Optional[int] = None, max_cols: Optional[int] = None):
    """Save a sparsity plot (spy) of a sparse matrix."""
    import matplotlib.pyplot as plt

    if not sp.isspmatrix(M):
        M = sp.csr_matrix(M)

    Mc = M.tocsr()

    if max_rows is not None or max_cols is not None:
        r = max_rows if max_rows is not None else Mc.shape[0]
        c = max_cols if max_cols is not None else Mc.shape[1]
        r = min(int(r), Mc.shape[0])
        c = min(int(c), Mc.shape[1])
        Mc = Mc[:r, :c]

    plt.figure()
    plt.spy(Mc, markersize=markersize)
    plt.title(f"{title}\nshape={M.shape}, nnz={M.nnz}, type={type(M).__name__}")
    plt.xlabel("column")
    plt.ylabel("row")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_sparse_density_heatmap(
    M: sp.spmatrix,
    title: str,
    save_path: str,
    bins: int = 512,
    log_scale: bool = True,
    crop: Optional[Tuple[int, int]] = None,
):
    """Heatmap-style visualization of sparsity (does NOT densify M)."""
    import matplotlib.pyplot as plt

    if not sp.isspmatrix(M):
        M = sp.csr_matrix(M)

    Mc = M.tocoo(copy=False)
    r = Mc.row.astype(np.int64, copy=False)
    c = Mc.col.astype(np.int64, copy=False)
    nrows, ncols = Mc.shape

    if crop is not None:
        max_r, max_c = crop
        max_r = min(int(max_r), nrows)
        max_c = min(int(max_c), ncols)
        keep = (r < max_r) & (c < max_c)
        r = r[keep]
        c = c[keep]
        nrows, ncols = max_r, max_c

    if r.size == 0:
        plt.figure()
        plt.title(f"{title}\n(empty)")
        plt.savefig(save_path, dpi=200)
        plt.close()
        return

    br = np.minimum((r * bins) // max(nrows, 1), bins - 1)
    bc = np.minimum((c * bins) // max(ncols, 1), bins - 1)

    H = np.zeros((bins, bins), dtype=np.int32)
    np.add.at(H, (br, bc), 1)

    if log_scale:
        H_plot = np.log1p(H.astype(np.float64))
        scale_note = "log1p(count)"
    else:
        H_plot = H.astype(np.float64)
        scale_note = "count"

    plt.figure()
    im = plt.imshow(H_plot, origin="upper", aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04, label=scale_note)
    plt.title(f"{title}\nshape={M.shape}, nnz={M.nnz}, bins={bins} ({scale_note})")
    plt.xlabel("column (binned)")
    plt.ylabel("row (binned)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# ============================================================
# Strength matrix C comparison
# ============================================================
def plot_c_comparison_heatmap(C1: sp.spmatrix, C2: sp.spmatrix, title: str, save_path: str, bins: int = 512):
    """Overlay density heatmap: C1=Red, C2=Blue. Purple indicates overlap."""
    import matplotlib.pyplot as plt

    if not sp.isspmatrix(C1): C1 = sp.csr_matrix(C1)
    if not sp.isspmatrix(C2): C2 = sp.csr_matrix(C2)

    def get_hist(M, shape, bins):
        Mc = M.tocoo()
        r = Mc.row
        c = Mc.col
        nrows, ncols = shape
        br = np.minimum((r * bins) // max(nrows, 1), bins - 1)
        bc = np.minimum((c * bins) // max(ncols, 1), bins - 1)
        H = np.zeros((bins, bins), dtype=np.float32)
        np.add.at(H, (br, bc), 1)
        return np.log1p(H)

    nrows = max(C1.shape[0], C2.shape[0])
    ncols = max(C1.shape[1], C2.shape[1])

    H1 = get_hist(C1, (nrows, ncols), bins)
    H2 = get_hist(C2, (nrows, ncols), bins)

    H1_norm = H1 / (H1.max() + 1e-9)
    H2_norm = H2 / (H2.max() + 1e-9)

    img = np.zeros((bins, bins, 3), dtype=np.float32)
    img[..., 0] = H1_norm
    img[..., 2] = H2_norm

    plt.figure(figsize=(6, 6))
    plt.imshow(img, origin="upper", interpolation="nearest")
    plt.title(f"{title}\nRed=Std, Blue=Lrn, Purple=Both")
    plt.xlabel("column (binned)")
    plt.ylabel("row (binned)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_c_comparison_spy(C1: sp.spmatrix, C2: sp.spmatrix, title: str, save_path: str,
                          center: Tuple[int, int], window: int = 50):
    """Direct spy overlay of a zoomed region. C1=Red, C2=Blue."""
    import matplotlib.pyplot as plt
    if not sp.isspmatrix(C1): C1 = sp.csr_matrix(C1)
    if not sp.isspmatrix(C2): C2 = sp.csr_matrix(C2)

    r_start = max(0, center[0] - window)
    r_end = min(C1.shape[0], center[0] + window)
    c_start = max(0, center[1] - window)
    c_end = min(C1.shape[1], center[1] + window)

    slice1 = C1[r_start:r_end, c_start:c_end].tocoo()
    slice2 = C2[r_start:r_end, c_start:c_end].tocoo()

    H = r_end - r_start
    W = c_end - c_start
    img = np.zeros((H, W, 3), dtype=np.float32)

    def fill_channel(Sl, ch_idx):
        mask = np.zeros((H, W), dtype=bool)
        mask[Sl.row, Sl.col] = True
        img[..., ch_idx][mask] = 1.0

    fill_channel(slice1, 0)  # R
    fill_channel(slice2, 2)  # B

    plt.figure(figsize=(6, 6))
    plt.imshow(img, origin="upper", interpolation="nearest")
    plt.xticks([0, W], [f"{c_start}", f"{c_end}"])
    plt.yticks([0, H], [f"{r_start}", f"{r_end}"])
    plt.title(f"{title}\nZoom [{r_start}:{r_end}, {c_start}:{c_end}]\nRed=Std, Blue=Lrn, Purple=Both")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# ============================================================
# Iteration bar chart
# ============================================================
def plot_iteration_bar(it_sa: int, it_ml: int, title: str, save_path: str):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.bar(["Std-AMG", "LearnedC"], [it_sa, it_ml])
    plt.ylabel("PCG Iterations")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# ============================================================
# Eigenvalue spectra + histogram
# ============================================================
def plot_eigenvalues(A: sp.csr_matrix, M_std: spla.LinearOperator, M_lrn: spla.LinearOperator,
                     title: str, save_path: str):
    """Compute and plot eigenvalues of M^-1 A for both preconditioners. Small grids only!"""
    import matplotlib.pyplot as plt

    if A.shape[0] > 2000:
        print(f"Skipping eigenvalue plot for N={A.shape[0]} (too large)")
        return

    print(f"Computing dense eigenvalues for N={A.shape[0]} (may take a moment)...")
    Ad = A.toarray()

    def get_dense_precond(M_op, n):
        I = np.eye(n)
        Minv = np.zeros((n, n))
        for i in range(n):
            Minv[:, i] = M_op.matvec(I[:, i])
        return Minv

    Minv_std = get_dense_precond(M_std, A.shape[0])
    Minv_lrn = get_dense_precond(M_lrn, A.shape[0])

    PA_std = Minv_std @ Ad
    PA_lrn = Minv_lrn @ Ad

    evals_std = np.linalg.eigvals(PA_std)
    evals_lrn = np.linalg.eigvals(PA_lrn)

    ev_std = np.sort(np.real(evals_std))
    ev_lrn = np.sort(np.real(evals_lrn))

    min_std = ev_std[ev_std > 1e-6].min() if (ev_std > 1e-6).any() else 1e-6
    max_std = ev_std.max()
    cond_std = max_std / min_std

    min_lrn = ev_lrn[ev_lrn > 1e-6].min() if (ev_lrn > 1e-6).any() else 1e-6
    max_lrn = ev_lrn.max()
    cond_lrn = max_lrn / min_lrn

    # Plot 1: Sorted eigenvalues
    plt.figure(figsize=(8, 6))
    plt.plot(ev_std, label=f"Std-AMG ($\\kappa={cond_std:.1f}$)", marker='.', linestyle='--', alpha=0.6)
    plt.plot(ev_lrn, label=f"Learned ($\\kappa={cond_lrn:.1f}$)", marker='x', linestyle='-', alpha=0.6)
    plt.axhline(1.0, color='k', linestyle=':', alpha=0.3)
    plt.title(f"Eigenvalue Spectra of $M^{{-1}}A$\n{title}")
    plt.ylabel("Eigenvalue $\\lambda$")
    plt.xlabel("Index")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    # Plot 2: Histogram
    plt.figure(figsize=(8, 6))
    plt.hist(ev_std, bins=50, alpha=0.5, label="Std-AMG", density=True, color='red')
    plt.hist(ev_lrn, bins=50, alpha=0.5, label="Learned", density=True, color='blue')
    plt.axvline(1.0, color='k', linestyle='--', alpha=0.5)
    plt.title(f"Eigenvalue Density Histogram\n{title}")
    plt.xlabel("Eigenvalue $\\lambda$")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    base, ext = os.path.splitext(save_path)
    plt.savefig(f"{base}_hist{ext}", dpi=200)
    plt.close()


# ============================================================
# Weight vs logit scatter
# ============================================================
def plot_weight_vs_logit(model, A: sp.csr_matrix, coords: np.ndarray,
                         save_path: str, device: str = "cpu",
                         dof_type: Optional[np.ndarray] = None):
    """Scatter plot of Edge Weight vs Model Logit."""
    import matplotlib.pyplot as plt
    import torch
    from amg_core import node_features_for_policy
    from torch_geometric.utils import from_scipy_sparse_matrix

    x = node_features_for_policy(A, coords, dof_type=dof_type).to(device)
    ei, ew = from_scipy_sparse_matrix(A)
    ew_raw = ew.float()
    ew_norm = ew_raw / (ew_raw.abs().max() + 1e-9)

    ei = ei.to(device)
    ew_norm = ew_norm.to(device)

    model.eval()
    with torch.no_grad():
        logits, _ = model(x, ei, ew_norm)

    row, col = ei.cpu().numpy()
    mask = (row != col)

    w_vals = ew_norm.cpu().numpy()[mask]
    l_vals = logits.cpu().numpy()[mask]

    plt.figure()
    plt.scatter(w_vals, l_vals, alpha=0.3, s=5)
    plt.xlabel(r"$a_{ij}\,/\,\max|a_{ij}|$")
    plt.ylabel("Edge MLP Output (Logit)")
    plt.title("Correlation: Is it just learning weights?")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# ============================================================
# Training curves
# ============================================================
def plot_training_curves(history: Dict[str, List[float]], save_dir: str,
                         probe_labels: Optional[Dict[str, str]] = None):
    """
    Plot training reward and probe metrics.

    Args:
        probe_labels: Optional dict with keys 'strong', 'weak', 'title' for
                      customizing probe subplot labels. Defaults to anisotropy labels.
    """
    import matplotlib.pyplot as plt
    ensure_dir(save_dir)

    if probe_labels is None:
        probe_labels = {
            "strong": "Strong Edge Logit",
            "weak": "Weak Edge Logit",
            "title": "Probe: Learning Physics Structure",
        }

    # 1. Rewards
    if "train_reward" in history and history["train_reward"]:
        plt.figure(figsize=(10, 5))
        r = history["train_reward"]
        plt.plot(r, alpha=0.3, label="Raw Reward")
        window = 50
        if len(r) > window:
            smooth = np.convolve(r, np.ones(window)/window, mode='valid')
            plt.plot(np.arange(window-1, len(r)), smooth, label=f"Smoothed (w={window})", linewidth=2)

        plt.xlabel("Step")
        plt.ylabel("Reward (Spectral Quality)")
        plt.title("Training Reward over Time")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "training_reward.png"), dpi=150)
        plt.close()

    # 2. Probe Metrics
    if "epochs" in history and history["epochs"]:
        epochs = history["epochs"]
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, history["probe_strong"], 'g-o', label=probe_labels["strong"])
        plt.plot(epochs, history["probe_weak"], 'r-x', label=probe_labels["weak"])
        plt.plot(epochs, history["probe_disc"], 'b--', linewidth=2, label="Discrimination (Strong - Weak)")

        plt.xlabel("Epoch")
        plt.ylabel("Average Logit Output")
        plt.title(probe_labels["title"])
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "training_probe_learning_curve.png"), dpi=150)
        plt.close()

    print(f"Training plots saved to: {save_dir}")
