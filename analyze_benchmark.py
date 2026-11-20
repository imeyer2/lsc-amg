#!/usr/bin/env python3
"""
analyze_benchmark.py — Visualize preconditioner benchmark results.

Reads benchmark_results.pt from evaluate_preconditioners.py and produces
a multi-panel diagnostic plot.

Panels:
  (a) PCG iteration count bar chart (per sample, grouped by scheme)
  (b) Convergence curves: residual norm vs PCG iteration (per scheme, aggregated)
  (c) Eigenvalue (Ritz value) distributions per scheme
  (d) Condition number comparison (kappa_raw bar chart)
  (e) Timing breakdown: setup vs solve (stacked bar)
  (f) Operator complexity (AMG schemes only)

Usage:
  python analyze_benchmark.py outputs/es_hpc_XXXXX/preconditioner_benchmark/
  python analyze_benchmark.py outputs/es_hpc_XXXXX/preconditioner_benchmark/benchmark_results.pt
"""
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# Consistent colors and display names for each scheme
SCHEME_STYLE = {
    "none":        {"color": "#888888", "label": "None (CG)"},
    "jacobi":      {"color": "#e6a817", "label": "Jacobi"},
    "ilu":         {"color": "#2ca02c", "label": "ILU"},
    "pyamg":       {"color": "#1f77b4", "label": "PyAMG SA"},
    "ruge_stuben": {"color": "#9467bd", "label": "Ruge–Stüben"},
    "learned":     {"color": "#d62728", "label": "Learned"},
}


def _label(scheme):
    return SCHEME_STYLE.get(scheme, {}).get("label", scheme)


def _color(scheme):
    return SCHEME_STYLE.get(scheme, {}).get("color", "black")


def analyze_benchmark(path):
    """Load benchmark results and generate diagnostic plots."""
    if os.path.isdir(path):
        pt_path = os.path.join(path, "benchmark_results.pt")
    else:
        pt_path = path
    if not os.path.isfile(pt_path):
        print(f"ERROR: {pt_path} not found.")
        sys.exit(1)

    save_dir = os.path.dirname(pt_path)
    data = torch.load(pt_path, map_location="cpu", weights_only=False)

    n_samples = len(data)
    schemes = list(data[0]["schemes"].keys())
    # Preserve a canonical order if possible
    order = ["none", "jacobi", "ilu", "pyamg", "ruge_stuben", "learned"]
    schemes = [s for s in order if s in schemes] + \
              [s for s in schemes if s not in order]

    print(f"\n{'='*70}")
    print(f"  Benchmark Visualization: {pt_path}")
    print(f"  Samples: {n_samples},  Schemes: {schemes}")
    print(f"{'='*70}")

    # ── Collect aggregate data ──
    all_iters = {s: [] for s in schemes}
    all_kappa = {s: [] for s in schemes}
    all_setup = {s: [] for s in schemes}
    all_solve = {s: [] for s in schemes}
    all_oc    = {s: [] for s in schemes}

    for idx in range(n_samples):
        for s in schemes:
            r = data[idx]["schemes"][s]
            all_iters[s].append(r["iters"])
            all_kappa[s].append(r["kappa_raw"])
            all_setup[s].append(r["setup_time"])
            all_solve[s].append(r["solve_time"])
            all_oc[s].append(r["op_complexity"])

    # ── Print summary table ──
    print(f"\n  {'scheme':<14s}  {'iters':>8s}  {'κ':>10s}  "
          f"{'setup(s)':>8s}  {'solve(s)':>8s}  {'OC':>6s}")
    print(f"  {'-'*14}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*6}")
    for s in schemes:
        k = np.mean(all_kappa[s])
        k_str = f"{k:.1f}" if k < 1e10 else "inf"
        print(f"  {_label(s):<14s}  {np.mean(all_iters[s]):8.1f}  "
              f"{k_str:>10s}  {np.mean(all_setup[s]):8.4f}  "
              f"{np.mean(all_solve[s]):8.4f}  "
              f"{np.mean(all_oc[s]):6.2f}")

    # ── Create figure ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Preconditioner Benchmark", fontsize=14, fontweight="bold")

    # ── (a) PCG iterations bar chart ──
    ax = axes[0, 0]
    x = np.arange(n_samples)
    n_schemes = len(schemes)
    width = 0.8 / n_schemes
    for i, s in enumerate(schemes):
        offset = (i - n_schemes / 2 + 0.5) * width
        ax.bar(x + offset, all_iters[s], width, color=_color(s),
               label=_label(s), alpha=0.85)
    ax.set_xlabel("Test sample")
    ax.set_ylabel("PCG iterations")
    ax.set_title("(a) Iterations to convergence")
    ax.set_xticks(x)
    if n_samples > 20:
        ax.set_xticks(x[::max(1, n_samples // 10)])
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    # ── (b) Convergence curves (median + band) ──
    ax = axes[0, 1]
    for s in schemes:
        # Collect all residual trajectories, pad to max length
        trajs = [np.array(data[idx]["schemes"][s]["residuals"])
                 for idx in range(n_samples)]
        max_len = max(len(t) for t in trajs)
        if max_len == 0:
            continue
        # Pad with last value
        padded = np.full((n_samples, max_len), np.nan)
        for j, t in enumerate(trajs):
            padded[j, :len(t)] = t
            if len(t) < max_len:
                padded[j, len(t):] = t[-1] if len(t) > 0 else np.nan

        median = np.nanmedian(padded, axis=0)
        p25 = np.nanpercentile(padded, 25, axis=0)
        p75 = np.nanpercentile(padded, 75, axis=0)
        iters = np.arange(1, max_len + 1)

        ax.semilogy(iters, median, color=_color(s), linewidth=1.5,
                    label=_label(s))
        ax.fill_between(iters, p25, p75, color=_color(s), alpha=0.1)

    ax.set_xlabel("PCG iteration")
    ax.set_ylabel("Residual norm")
    ax.set_title("(b) Convergence curves (median ± IQR)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # ── (c) Ritz value distributions (violin/box) ──
    ax = axes[0, 2]
    ritz_data = []
    ritz_labels = []
    ritz_colors = []
    for s in schemes:
        all_ritz = np.concatenate(
            [np.array(data[idx]["schemes"][s]["ritz_values"])
             for idx in range(n_samples)]
        )
        # Clip extreme outliers for visualization
        all_ritz = all_ritz[all_ritz > 0]
        if len(all_ritz) > 0:
            ritz_data.append(np.log10(all_ritz))
        else:
            ritz_data.append(np.array([0.0]))
        ritz_labels.append(_label(s))
        ritz_colors.append(_color(s))

    parts = ax.violinplot(ritz_data, positions=range(len(schemes)),
                          showmedians=True, showextrema=False)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(ritz_colors[i])
        pc.set_alpha(0.6)
    parts["cmedians"].set_color("black")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xticks(range(len(schemes)))
    ax.set_xticklabels(ritz_labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("log₁₀(Ritz value)")
    ax.set_title("(c) Eigenvalue distribution (Ritz)")
    ax.grid(True, alpha=0.3, axis="y")

    # ── (d) Condition number bar chart ──
    ax = axes[1, 0]
    means = [np.mean(all_kappa[s]) for s in schemes]
    # Cap for display
    cap = max(m for m in means if m < 1e10) * 1.2 if any(
        m < 1e10 for m in means) else 100
    display_means = [min(m, cap) for m in means]
    bars = ax.bar(range(len(schemes)), display_means,
                  color=[_color(s) for s in schemes], alpha=0.85)
    for i, (m, dm) in enumerate(zip(means, display_means)):
        label_text = f"{m:.0f}" if m < 1e6 else f"{m:.1e}"
        ax.text(i, dm, label_text, ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(schemes)))
    ax.set_xticklabels([_label(s) for s in schemes], rotation=30,
                       ha="right", fontsize=8)
    ax.set_ylabel("κ (condition number)")
    ax.set_title("(d) Condition number")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")

    # ── (e) Timing breakdown (stacked bar) ──
    ax = axes[1, 1]
    mean_setup = [np.mean(all_setup[s]) for s in schemes]
    mean_solve = [np.mean(all_solve[s]) for s in schemes]
    x_pos = np.arange(len(schemes))
    ax.bar(x_pos, mean_setup, color=[_color(s) for s in schemes],
           alpha=0.6, label="Setup")
    ax.bar(x_pos, mean_solve, bottom=mean_setup,
           color=[_color(s) for s in schemes], alpha=0.9, label="Solve")
    # Add total labels
    for i in range(len(schemes)):
        total = mean_setup[i] + mean_solve[i]
        ax.text(i, total, f"{total:.3f}s", ha="center", va="bottom",
                fontsize=7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([_label(s) for s in schemes], rotation=30,
                       ha="right", fontsize=8)
    ax.set_ylabel("Time (seconds)")
    ax.set_title("(e) Mean wall-clock time (setup + solve)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # ── (f) Operator complexity (AMG schemes only) ──
    ax = axes[1, 2]
    amg_schemes = [s for s in schemes if np.mean(all_oc[s]) > 0]
    if amg_schemes:
        oc_means = [np.mean(all_oc[s]) for s in amg_schemes]
        oc_stds = [np.std(all_oc[s]) for s in amg_schemes]
        x_pos = np.arange(len(amg_schemes))
        ax.bar(x_pos, oc_means, yerr=oc_stds,
               color=[_color(s) for s in amg_schemes], alpha=0.85,
               capsize=4)
        for i, m in enumerate(oc_means):
            ax.text(i, m, f"{m:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([_label(s) for s in amg_schemes], rotation=30,
                           ha="right", fontsize=8)
        ax.set_ylabel("Operator complexity")
        ax.set_title("(f) AMG operator complexity")
        ax.grid(True, alpha=0.3, axis="y")
    else:
        ax.text(0.5, 0.5, "No AMG schemes", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
        ax.set_title("(f) AMG operator complexity")

    plt.tight_layout()
    out_path = os.path.join(save_dir, "benchmark_visualization.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved: {out_path}")

    # ── Zoomed plot excluding outlier schemes ──
    _plot_zoomed(data, n_samples, schemes, all_iters, save_dir)

    # ── Per-sample convergence plots (individual) ──
    if n_samples <= 16:
        _plot_per_sample_convergence(data, n_samples, schemes, save_dir)


def _plot_zoomed(data, n_samples, schemes, all_iters, save_dir):
    """Zoomed iterations + convergence plot, auto-excluding outlier schemes.

    A scheme is an outlier if its median iteration count is >3× the median
    of the remaining (non-outlier) schemes.  Typically drops 'none' and
    'jacobi'.
    """
    # Determine which schemes to keep
    medians = {s: float(np.median(all_iters[s])) for s in schemes}
    sorted_schemes = sorted(schemes, key=lambda s: medians[s])
    keep = list(sorted_schemes)

    # Iteratively remove the worst scheme if it's >3× the median of the rest
    while len(keep) > 2:
        worst = keep[-1]
        rest_median = float(np.median([medians[s] for s in keep[:-1]]))
        if medians[worst] > 3 * rest_median:
            keep.pop()
        else:
            break

    excluded = [s for s in schemes if s not in keep]
    if not excluded:
        return  # nothing to zoom

    print(f"  Zoomed plot: excluding {[_label(s) for s in excluded]} "
          f"(outlier schemes)")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Preconditioner Comparison (zoomed, outliers excluded)",
                 fontsize=13, fontweight="bold")

    # ── (a) Iterations bar chart (zoomed) ──
    ax = axes[0]
    x = np.arange(n_samples)
    n_keep = len(keep)
    width = 0.8 / n_keep
    for i, s in enumerate(keep):
        offset = (i - n_keep / 2 + 0.5) * width
        ax.bar(x + offset, all_iters[s], width, color=_color(s),
               label=_label(s), alpha=0.85)
    ax.set_xlabel("Test sample")
    ax.set_ylabel("PCG iterations")
    ax.set_title("Iterations to convergence")
    ax.set_xticks(x)
    if n_samples > 20:
        ax.set_xticks(x[::max(1, n_samples // 10)])
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    # ── (b) Convergence curves (zoomed) ──
    ax = axes[1]
    for s in keep:
        trajs = [np.array(data[idx]["schemes"][s]["residuals"])
                 for idx in range(n_samples)]
        max_len = max(len(t) for t in trajs)
        if max_len == 0:
            continue
        padded = np.full((n_samples, max_len), np.nan)
        for j, t in enumerate(trajs):
            padded[j, :len(t)] = t
            if len(t) < max_len:
                padded[j, len(t):] = t[-1] if len(t) > 0 else np.nan
        median = np.nanmedian(padded, axis=0)
        p25 = np.nanpercentile(padded, 25, axis=0)
        p75 = np.nanpercentile(padded, 75, axis=0)
        iters_x = np.arange(1, max_len + 1)
        ax.semilogy(iters_x, median, color=_color(s), linewidth=1.5,
                    label=_label(s))
        ax.fill_between(iters_x, p25, p75, color=_color(s), alpha=0.1)

    ax.set_xlabel("PCG iteration")
    ax.set_ylabel("Residual norm")
    ax.set_title("Convergence curves (median ± IQR)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(save_dir, "benchmark_zoomed.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {out_path}")


def _plot_per_sample_convergence(data, n_samples, schemes, save_dir):
    """Individual convergence curve for each test sample."""
    ncols = min(4, n_samples)
    nrows = (n_samples + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows),
                             squeeze=False)
    fig.suptitle("Per-Sample Convergence Curves", fontsize=13,
                 fontweight="bold")

    for idx in range(n_samples):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        n = data[idx]["matrix_size"]
        for s in schemes:
            resid = np.array(data[idx]["schemes"][s]["residuals"])
            if len(resid) == 0:
                continue
            ax.semilogy(range(1, len(resid) + 1), resid, color=_color(s),
                        linewidth=1.2, label=_label(s))
        ax.set_title(f"Sample {idx} (n={n})", fontsize=9)
        ax.set_xlabel("Iteration", fontsize=8)
        ax.set_ylabel("‖r‖", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=6, ncol=2)

    # Hide unused axes
    for idx in range(n_samples, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(save_dir, "per_sample_convergence.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_benchmark.py "
              "<preconditioner_benchmark_dir_or_pt_file> [...]")
        sys.exit(1)
    for path in sys.argv[1:]:
        analyze_benchmark(path)
