#!/usr/bin/env python3
"""
thesis_chapter_analysis.py — Generate all figures and tables for the
learned AMG preconditioner thesis chapter.

Figures produced:
  1. TTS tables (LaTeX) — one per matrix size
  2. Eigenvalue box-and-whisker plot across matrix sizes
  3. Condition number box-and-whisker plot across matrix sizes
  4. V-cycle convergence factor ρ(I−BA): learned vs standard AMG
  5. ρ(I−BA) training evolution across sizes
  6. Coarsening ratio comparison (learned vs standard SA-AMG)
  7. Scalability versions of 1–3 (if data exists)

Usage:
  python thesis_chapter_analysis.py [--out_dir thesis_figures]
  python thesis_chapter_analysis.py --scalability  # also generate scalability plots
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Project imports ──────────────────────────────────────────
ROOT = Path(__file__).resolve().parent

# ============================================================
# Configuration: best run per matrix size, per problem type
# ============================================================
BEST_RUNS_BY_PROBLEM = {
    "lognormal": OrderedDict([
        ("5k",  {"grid_n": 5000,  "mat_size": 5000,  "out_dir": "outputs/train/lognormal_5k/seed0"}),
        ("10k", {"grid_n": 10000, "mat_size": 10000, "out_dir": "outputs/train/lognormal_10k/seed0"}),
        ("20k", {"grid_n": 20000, "mat_size": 20000, "out_dir": "outputs/train/lognormal_20k/seed2"}),
        ("30k", {"grid_n": 30000, "mat_size": 30000, "out_dir": "outputs/train/lognormal_30k/seed2"}),
        ("50k", {"grid_n": 50000, "mat_size": 50000, "out_dir": "outputs/train/lognormal_50k/seed0"}),
    ]),
    "anisotropy": OrderedDict([
        ("5k",  {"grid_n": 5000,  "out_dir": "outputs/train/anisotropy_5k/seed2"}),   # R=-2.8733
        ("10k", {"grid_n": 10000, "out_dir": "outputs/train/anisotropy_10k/seed3"}),   # R=-3.2089 (re-run, actual 10k matrices)
        ("20k", {"grid_n": 20000, "out_dir": "outputs/train/anisotropy_20k/seed1"}),   # R=-3.1800
        ("30k", {"grid_n": 30000, "out_dir": "outputs/train/anisotropy_30k/seed1"}),   # R=-3.2330
        ("50k", {"grid_n": 50000, "out_dir": "outputs/train/anisotropy_50k/seed0"}),   # R=-3.2678
    ]),
    "perforated_l": OrderedDict([
        ("5k",  {"grid_n": 3500,  "mat_size": 5000,  "out_dir": "outputs/train/perforated_l_5k/seed4"}),   # R=-2.6197  mat≈4910
        ("10k", {"grid_n": 7000,  "mat_size": 10000, "out_dir": "outputs/train/perforated_l_10k/seed4"}),   # R=-2.8272  mat≈10387
        ("20k", {"grid_n": 20000, "mat_size": 20000, "out_dir": "outputs/train/perforated_l_20k/seed0"}),   # R=-3.0111  mat≈29470
        ("30k", {"grid_n": 30000, "mat_size": 30000, "out_dir": "outputs/train/perforated_l_30k/seed2"}),   # R=-3.1130  mat≈43430
        ("50k", {"grid_n": 50000, "mat_size": 50000, "out_dir": "outputs/train/perforated_l_50k/seed0"}),   # R=-3.1623  mat≈73646
    ]),
    "poisson": OrderedDict([
        ("5k",  {"grid_n": 5000,  "out_dir": "outputs/train/poisson_5k/seed4"}),   # R=-2.4653
        ("10k", {"grid_n": 10000, "out_dir": "outputs/train/poisson_10k/seed3"}),   # R=-2.7139
        ("20k", {"grid_n": 20000, "out_dir": "outputs/train/poisson_20k/seed4"}),   # R=-2.6326
        ("30k", {"grid_n": 30000, "out_dir": "outputs/train/poisson_30k/seed0"}),   # R=-2.7264
        ("50k", {"grid_n": 50000, "out_dir": "outputs/train/poisson_50k/seed0"}),   # R=-2.7559
    ]),
}

# Default (lognormal) for backward compatibility
BEST_RUNS = BEST_RUNS_BY_PROBLEM["lognormal"]

SCALABILITY_SIZES = [
    ("100k", 100000),
    ("150k", 150000),
    ("200k", 200000),
    ("500k", 500000),
]

def _build_scalability_runs(problem="lognormal"):
    """Build SCALABILITY_RUNS dict for the given problem type."""
    runs = OrderedDict()
    for label, grid_n in SCALABILITY_SIZES:
        if problem == "lognormal":
            out_dir = f"outputs/scalability_{label}"
        else:
            out_dir = f"outputs/scalability_{problem}_{label}"
        runs[label] = {"grid_n": grid_n, "out_dir": out_dir}
    return runs

# Default (lognormal) for backward compatibility
SCALABILITY_RUNS = _build_scalability_runs("lognormal")

# All 6 schemes in the benchmark data
ALL_SCHEMES = ["none", "jacobi", "ilu", "pyamg", "ruge_stuben", "learned"]

# Display names for plots / tables.
#
# NOTE (M8): every entry here names a *preconditioner* M, never the plotted
# quantity.  The "none" entry used to read "CG (no precond.)", which a referee
# read as "the condition number of CG" -- not a quantity that exists.  The
# unpreconditioned arm is M = I, and that is what the label now says.
SCHEME_LABELS = {
    "none":        "No preconditioner ($M=I$)",
    "jacobi":      "Jacobi",
    "ilu":         "ILU(0)",
    "pyamg":       "SA-AMG",
    "ruge_stuben": "RS-AMG",
    "learned":     "Learned AMG",
}

SCHEME_COLORS = {
    "none":        "#999999",
    "jacobi":      "#e69f00",
    "ilu":         "#56b4e9",
    "pyamg":       "#009e73",
    "ruge_stuben": "#0072b2",
    "learned":     "#d55e00",
}


# ============================================================
# Helpers
# ============================================================
def _load_benchmark(out_dir: str):
    """Load benchmark_results.pt from a run directory."""
    path = ROOT / out_dir / "preconditioner_benchmark" / "benchmark_results.pt"
    if not path.exists():
        print(f"  WARNING: {path} not found, skipping.")
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def _load_diagnostics_evolution(out_dir: str):
    """Load diagnostics_evolution.json from a run directory."""
    path = ROOT / out_dir / "diagnostics_evolution.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _schemes_in_data(bench_data):
    """Return the list of schemes present in benchmark data."""
    sample0 = bench_data[list(bench_data.keys())[0]]
    return list(sample0["schemes"].keys())


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def _save_fig(fig, fpath_pdf):
    """Save figure as both PDF and PNG."""
    fig.savefig(fpath_pdf, dpi=150, bbox_inches="tight")
    fpath_png = fpath_pdf.replace(".pdf", ".png")
    fig.savefig(fpath_png, dpi=150, bbox_inches="tight")
    print(f"  Wrote {fpath_pdf}")
    print(f"  Wrote {fpath_png}")


# ============================================================
# 1. TTS Tables (LaTeX)
# ============================================================
def generate_tts_tables(runs, out_dir, prefix=""):
    """Generate one LaTeX table per matrix size with TTS breakdown."""
    _ensure_dir(out_dir)

    for label, info in runs.items():
        bench = _load_benchmark(info["out_dir"])
        if bench is None:
            continue

        schemes = [s for s in ALL_SCHEMES if s in _schemes_in_data(bench)]
        n_samples = len(bench)

        # Aggregate across samples
        agg = {}
        for s in schemes:
            iters_list, setup_list, solve_list = [], [], []
            for idx in bench:
                sd = bench[idx]["schemes"].get(s)
                if sd is None:
                    continue
                iters_list.append(sd["iters"])
                setup_list.append(sd["setup_time"])
                solve_list.append(sd["solve_time"])
            agg[s] = {
                "iters_mean": np.mean(iters_list),
                "iters_std":  np.std(iters_list),
                "setup_mean": np.mean(setup_list),
                "setup_std":  np.std(setup_list),
                "solve_mean": np.mean(solve_list),
                "solve_std":  np.std(solve_list),
                "tts_mean":   np.mean(np.array(setup_list) + np.array(solve_list)),
                "tts_std":    np.std(np.array(setup_list) + np.array(solve_list)),
            }

        mat_size = info.get("mat_size", info["grid_n"])
        fname = f"{prefix}tts_table_{label}.tex"
        fpath = os.path.join(out_dir, fname)

        with open(fpath, "w") as f:
            f.write(f"% TTS Table — {label} (n ≈ {mat_size:,}), {n_samples} test samples\n")
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write(f"\\caption{{Time-to-solution breakdown for $n = {mat_size:,}$.}}\n")
            f.write(f"\\label{{tab:tts_{prefix}{label}}}\n")
            f.write("\\begin{tabular}{lrrrr}\n")
            f.write("\\toprule\n")
            f.write("Method & PCG Iters & Setup (s) & Solve (s) & TTS (s) \\\\\n")
            f.write("\\midrule\n")
            for s in schemes:
                a = agg[s]
                lbl = SCHEME_LABELS.get(s, s)
                f.write(
                    f"{lbl} & "
                    f"${a['iters_mean']:.1f} \\pm {a['iters_std']:.1f}$ & "
                    f"${a['setup_mean']:.4f}$ & "
                    f"${a['solve_mean']:.4f}$ & "
                    f"${a['tts_mean']:.4f}$ \\\\\n"
                )
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")

        print(f"  Wrote {fpath}")

        # Also print to console
        print(f"\n  === TTS: {label} (n ≈ {mat_size:,}) ===")
        print(f"  {'Method':<20s} {'Iters':>10s} {'Setup(s)':>10s} "
              f"{'Solve(s)':>10s} {'TTS(s)':>10s}")
        print(f"  {'-'*60}")
        for s in schemes:
            a = agg[s]
            lbl = SCHEME_LABELS.get(s, s)
            print(f"  {lbl:<20s} {a['iters_mean']:>8.1f}±{a['iters_std']:<4.1f}"
                  f" {a['setup_mean']:>10.4f} {a['solve_mean']:>10.4f}"
                  f" {a['tts_mean']:>10.4f}")


# ============================================================
# 1b. Operator Complexity Table (LaTeX)
# ============================================================
def generate_operator_complexity_table(runs, out_dir, prefix=""):
    """Generate a single LaTeX table of operator complexity across all sizes."""
    _ensure_dir(out_dir)

    AMG_SCHEMES = ["pyamg", "ruge_stuben", "learned"]
    AMG_LABELS = {"pyamg": "SA-AMG", "ruge_stuben": "RS-AMG", "learned": "Learned AMG"}

    rows = []  # (label, mat_size, {scheme: (mean, std)})
    for label, info in runs.items():
        bench = _load_benchmark(info["out_dir"])
        if bench is None:
            continue
        mat_size = info.get("mat_size", info["grid_n"])
        oc_data = {}
        for s in AMG_SCHEMES:
            vals = []
            for idx in bench:
                sd = bench[idx]["schemes"].get(s)
                if sd is not None:
                    vals.append(sd.get("op_complexity", 0.0))
            if vals:
                oc_data[s] = (np.mean(vals), np.std(vals))
            else:
                oc_data[s] = (float("nan"), float("nan"))
        rows.append((label, mat_size, oc_data))

    if not rows:
        print("  WARNING: No operator complexity data, skipping.")
        return

    fname = f"{prefix}operator_complexity_table.tex"
    fpath = os.path.join(out_dir, fname)

    with open(fpath, "w") as f:
        f.write(f"% Operator Complexity Table — {prefix or 'all sizes'}\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Operator complexity (total nnz / fine-level nnz) across matrix sizes.}\n")
        f.write(f"\\label{{tab:{prefix}operator_complexity}}\n")
        f.write("\\begin{tabular}{lrrr}\n")
        f.write("\\toprule\n")
        f.write("Size ($n$) & SA-AMG & RS-AMG & Learned AMG \\\\\n")
        f.write("\\midrule\n")
        for label, mat_size, oc_data in rows:
            parts = [f"${mat_size:,}$"]
            for s in AMG_SCHEMES:
                m, sd = oc_data[s]
                if np.isnan(m):
                    parts.append("---")
                else:
                    parts.append(f"${m:.2f} \\pm {sd:.2f}$")
            f.write(" & ".join(parts) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"  Wrote {fpath}")

    # Console output
    print(f"\n  === Operator Complexity ===")
    print(f"  {'Size':>10s}  {'SA-AMG':>14s}  {'RS-AMG':>14s}  {'Learned':>14s}")
    print(f"  {'-'*56}")
    for label, mat_size, oc_data in rows:
        parts = [f"  {mat_size:>10,d}"]
        for s in AMG_SCHEMES:
            m, sd = oc_data[s]
            if np.isnan(m):
                parts.append(f"{'---':>14s}")
            else:
                parts.append(f"{m:>7.2f}±{sd:<5.2f}")
        print("  ".join(parts))


# ============================================================
# 2. Eigenvalue Box-and-Whisker Plot
# ============================================================
def plot_eigenvalue_boxplot(runs, out_dir, prefix="", skip_schemes=None):
    """Box-and-whisker: log(eigenvalue) for each scheme across matrix sizes."""
    _ensure_dir(out_dir)
    skip = set(skip_schemes or [])
    schemes_to_plot = [s for s in ALL_SCHEMES if s not in skip]

    fig, ax = plt.subplots(figsize=(14, 6))

    size_labels = []
    positions = []
    pos = 0
    width = 0.7

    for label, info in runs.items():
        bench = _load_benchmark(info["out_dir"])
        if bench is None:
            continue

        schemes_present = [s for s in schemes_to_plot if s in _schemes_in_data(bench)]
        group_start = pos

        for i, s in enumerate(schemes_present):
            all_ritz = []
            for idx in bench:
                sd = bench[idx]["schemes"].get(s)
                if sd is not None and "ritz_values" in sd:
                    rv = np.array(sd["ritz_values"])
                    rv = rv[rv > 0]  # filter out non-positive
                    all_ritz.extend(rv.tolist())
            if not all_ritz:
                pos += 1
                continue

            log_ritz = np.log10(np.array(all_ritz))
            bp = ax.boxplot(
                [log_ritz], positions=[pos], widths=width * 0.8,
                patch_artist=True, showfliers=False,
                medianprops=dict(color="black", linewidth=1.5),
            )
            color = SCHEME_COLORS.get(s, "#888888")
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            pos += 1

        center = (group_start + pos - 1) / 2
        size_labels.append(label)
        positions.append(center)
        pos += 1.5  # gap between groups

    ax.set_xticks(positions)
    ax.set_xticklabels(size_labels, fontsize=12)
    ax.set_ylabel(r"$\log_{10}(\lambda)$  (Ritz values of $M^{-1}A$)", fontsize=12)
    ax.set_xlabel("Matrix size", fontsize=12)
    ax.set_title("Preconditioned Eigenvalue Distribution", fontsize=14)

    # Legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=SCHEME_COLORS[s], alpha=0.7,
                     label=SCHEME_LABELS[s])
               for s in schemes_to_plot]
    ax.legend(handles=handles, loc="upper left", fontsize=9, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    suffix = "_no_cg" if skip else ""
    fpath = os.path.join(out_dir, f"{prefix}eigenvalue_boxplot{suffix}.pdf")
    _save_fig(fig, fpath)
    plt.close(fig)


# ============================================================
# 3. Condition Number Box-and-Whisker Plot
# ============================================================
def plot_condition_number_boxplot(runs, out_dir, prefix=""):
    """Box-and-whisker: log10 of the *estimated* condition number, per scheme.

    M8 -- what this plots, exactly, so no caption can misdescribe it again:

      * The field read is ``kappa_raw`` = theta_max / theta_min over the
        **untrimmed** Ritz values of M^{-1}A (``amg_core.spectral_metrics``,
        whose own docstring calls it "noisy, for logging only").  The trimmed
        ``kappa_eff`` computed alongside it is deliberately NOT plotted here;
        captions in the manuscript must therefore say "untrimmed".  We keep
        kappa_raw so the published numbers stay reproducible -- if a future
        revision prefers kappa_eff, that is a numerics change and must be
        announced as one, not slipped in under an unchanged caption.
      * Ritz values interlace the spectrum, so this is a LOWER BOUND:
        kappa_hat <= kappa(M^{-1}A).  The bound is loosest for arms whose CG
        recurrence terminates in the fewest steps (``_pcg_ritz_values`` breaks
        at ||r|| < 1e-12||b||), i.e. the bias favours the strongest
        preconditioner.  Never state kappa_hat ~ 1 as a fact.
      * Each box is one preconditioner M, not a quantity; see SCHEME_LABELS.
    """
    _ensure_dir(out_dir)

    fig, ax = plt.subplots(figsize=(14, 6))

    size_labels = []
    positions = []
    pos = 0

    for label, info in runs.items():
        bench = _load_benchmark(info["out_dir"])
        if bench is None:
            continue

        schemes_present = [s for s in ALL_SCHEMES if s in _schemes_in_data(bench)]
        group_start = pos

        for i, s in enumerate(schemes_present):
            kappas = []
            for idx in bench:
                sd = bench[idx]["schemes"].get(s)
                if sd is not None and "kappa_raw" in sd:
                    k = sd["kappa_raw"]
                    if k > 0 and np.isfinite(k):
                        kappas.append(k)
            if not kappas:
                pos += 1
                continue

            log_kappa = np.log10(np.array(kappas))
            bp = ax.boxplot(
                [log_kappa], positions=[pos], widths=0.7 * 0.8,
                patch_artist=True, showfliers=False,
                medianprops=dict(color="black", linewidth=1.5),
            )
            color = SCHEME_COLORS.get(s, "#888888")
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            pos += 1

        center = (group_start + pos - 1) / 2
        size_labels.append(label)
        positions.append(center)
        pos += 1.5

    ax.set_xticks(positions)
    ax.set_xticklabels(size_labels, fontsize=12)
    ax.set_ylabel(r"$\log_{10}(\kappa)$", fontsize=12)
    ax.set_xlabel("Matrix size", fontsize=12)
    ax.set_title("Condition Number Distribution", fontsize=14)

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=SCHEME_COLORS[s], alpha=0.7,
                     label=SCHEME_LABELS[s])
               for s in ALL_SCHEMES]
    ax.legend(handles=handles, loc="upper left", fontsize=9, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    fpath = os.path.join(out_dir, f"{prefix}condition_number_boxplot.pdf")
    _save_fig(fig, fpath)
    plt.close(fig)


def _per_sample_pcg_residual_tail_ratio(out_dir: str, scheme: str) -> list:
    """Per-sample mean of the last <=5 preconditioned-CG residual ratios.

    THIS IS NOT A CONVERGENCE FACTOR.  It was called
    ``_per_sample_rho_from_benchmark`` and its output was plotted and captioned
    as the asymptotic V-cycle convergence factor rho(I - BA); it is not that,
    hence the rename.  What it returns,
    per test sample, is

        mean over the last <= 5 iterations k of  ||r_k||_2 / ||r_{k-1}||_2

    where {r_k} is the residual history of a *preconditioned conjugate
    gradient* run written by ``evaluate_preconditioners._benchmark_single_sample``
    from ``amg_core.pcg_iterations``, with b = ones(n), x0 = 0, stopped at
    relative tolerance 1e-8.

    Three reasons it must never be relabelled as rho(I - BA):

      1. It is a Krylov quantity.  CG accelerates the underlying stationary
         iteration, so this ratio is not rho(I - BA), is not an upper bound on
         per-cycle error reduction, and cannot be compared with published
         V-cycle convergence factors.
      2. The averaging window is ARM-DEPENDENT.  Each arm's "tail" is the last
         five iterations of its own run to 1e-8 -- iterations 16-20 for a fast
         arm, 161-166 for a slow one.  Arms are not scored over a common
         window and CG residual ratios are not constant, so the ordering
         carries a systematic bias of unknown sign.
      3. It is a single deterministic RHS, unaveraged, whose exact solution
         x* = 1 is smoothed aggregation's own default near-nullspace vector.

    If you want an actual convergence factor, use
    ``amg_diagnostics.compute_asymptotic_rho_stats`` (renormalized power
    iteration, shared probe seeds, median, honesty gates) or
    ``amg_diagnostics.exact_error_propagation_spectrum`` for dense ground truth
    at small n.  Do not "fix" this function to return rho -- it reads residual
    histories that contain no spectral information at all.
    """
    bench = _load_benchmark(out_dir)
    if bench is None:
        return []
    tail_ratios = []
    for idx in bench:
        sd = bench[idx]["schemes"].get(scheme)
        if sd is None:
            tail_ratios.append(float("nan"))
            continue
        res = np.array(sd["residuals"], dtype=np.float64)
        if len(res) < 4:
            tail_ratios.append(float("nan"))
            continue
        tail = min(5, len(res) - 1)
        ratios = res[-tail:] / res[-(tail + 1):-1]
        ratios = ratios[ratios > 0]
        if len(ratios) > 0:
            tail_ratios.append(float(np.mean(ratios)))
        else:
            tail_ratios.append(float("nan"))
    return tail_ratios


# ============================================================
# 4. Preconditioned-CG residual tail ratio: scatter + table
#
#    Formerly titled "V-cycle Convergence Factor rho(I-BA)".  It never was
#    that; see _per_sample_pcg_residual_tail_ratio above (I8).
# ============================================================
def plot_pcg_residual_tail_ratio_comparison(runs, out_dir, prefix=""):
    """Scatter of the per-sample PCG residual tail ratio for the three AMG
    arms, plus an aggregate LaTeX table with mean +/- std.

    Renamed from ``plot_rho_comparison`` (I8).  Every label this function
    emits -- axis, title, table caption, table label, console header and
    output filename -- now names the plotted quantity, so a caption written
    against this figure cannot silently promote a Krylov residual ratio into a
    stationary convergence factor.

    Note that the emitted filename changed from ``{prefix}rho_comparison.pdf``
    to ``{prefix}pcg_residual_tail_ratio.pdf``.  Figures already included in
    the manuscript under the old name must be regenerated and their
    \\includegraphics paths updated; that is deliberate, so the stale
    mislabelled images cannot be reused unnoticed.
    """
    _ensure_dir(out_dir)

    AMG_SCHEMES = [
        ("pyamg",       "SA-AMG", SCHEME_COLORS["pyamg"],       "s"),
        ("ruge_stuben", "RS-AMG", SCHEME_COLORS["ruge_stuben"], "D"),
        ("learned",     "Learned AMG", SCHEME_COLORS["learned"], "o"),
    ]

    # Collect the per-sample tail ratio for every (size, scheme)
    # data[label] = {scheme_key: [tail_ratio_sample0, tail_ratio_sample1, ...]}
    labels_plot = []
    data = {}
    for label, info in runs.items():
        bench = _load_benchmark(info["out_dir"])
        if bench is None:
            continue
        size_data = {}
        for key, _, _, _ in AMG_SCHEMES:
            size_data[key] = _per_sample_pcg_residual_tail_ratio(
                info["out_dir"], key)
        if any(len(v) > 0 for v in size_data.values()):
            labels_plot.append(label)
            data[label] = size_data

    if not labels_plot:
        print("  WARNING: No PCG residual-tail-ratio data available, "
              "skipping plot.")
        return

    # ── Scatter plot ──
    fig, ax = plt.subplots(figsize=(10, 5))
    x_positions = np.arange(len(labels_plot))
    offset = 0.15

    for i, (key, lbl, color, marker) in enumerate(AMG_SCHEMES):
        x_off = x_positions + (i - 1) * offset
        for j, label in enumerate(labels_plot):
            vals = data[label].get(key, [])
            finite = [r for r in vals if np.isfinite(r)]
            if finite:
                # Jitter x slightly so overlapping dots are visible
                jitter = np.random.default_rng(42 + i + j).uniform(
                    -0.03, 0.03, len(finite))
                ax.scatter(x_off[j] + jitter, finite,
                           s=30, marker=marker, color=color, alpha=0.6,
                           edgecolors="none", zorder=4,
                           label=lbl if j == 0 else None)
                # Mean marker (larger, with edge)
                mean_val = np.mean(finite)
                ax.scatter(x_off[j], mean_val, s=120, marker=marker,
                           color=color, edgecolors="white", linewidths=1.2,
                           zorder=6)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels_plot, fontsize=12)
    ax.set_ylabel(r"$\hat\gamma$ (mean PCG residual tail ratio)", fontsize=12)
    ax.set_title("Preconditioned-CG Residual Tail Ratio per Sample "
                 "(not $\\rho(I-BA)$)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    fpath = os.path.join(out_dir, f"{prefix}pcg_residual_tail_ratio.pdf")
    _save_fig(fig, fpath)
    plt.close(fig)

    # ── Aggregate LaTeX table ──
    fpath_tex = os.path.join(
        out_dir, f"{prefix}pcg_residual_tail_ratio_table.tex")
    with open(fpath_tex, "w") as f:
        f.write("% Mean preconditioned-CG residual tail ratio "
                "(mean +/- std over test samples).\n")
        f.write("% NOT a convergence factor -- see "
                "evaluate_preconditioners.vcycle_convergence_factor for rho(I - BA).\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Mean preconditioned-CG residual tail ratio "
                "$\\hat\\gamma$ across matrix sizes (mean $\\pm$ std over "
                "test samples): the mean of the last five ratios "
                "$\\|\\vec{r}_k\\|_2/\\|\\vec{r}_{k-1}\\|_2$ of a PCG run on "
                "the single right-hand side $\\vec{b}=\\vec{1}$. "
                "\\emph{This is not the convergence factor $\\rho(I-BA)$}: it "
                "is a Krylov quantity, the averaging window differs from arm "
                "to arm, and it is measured on one deterministic "
                "right-hand side.}\n")
        f.write("\\label{tab:pcg_residual_tail_ratio}\n")
        f.write("\\begin{tabular}{lrrr}\n")
        f.write("\\toprule\n")
        f.write("Size & SA-AMG & RS-AMG & Learned AMG \\\\\n")
        f.write("\\midrule\n")
        for label in labels_plot:
            parts = [label]
            for key, _, _, _ in AMG_SCHEMES:
                vals = [r for r in data[label].get(key, []) if np.isfinite(r)]
                if vals:
                    parts.append(f"${np.mean(vals):.4f} \\pm {np.std(vals):.4f}$")
                else:
                    parts.append("---")
            f.write(" & ".join(parts) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"  Wrote {fpath_tex}")

    # Console summary
    print("\n  === Mean PCG residual tail ratio (NOT rho(I-BA)) ===")
    print(f"  {'Size':>6s}  {'SA-AMG':>14s}  {'RS-AMG':>14s}  {'Learned':>14s}")
    print(f"  {'-'*52}")
    for label in labels_plot:
        parts = [f"  {label:>6s}"]
        for key, _, _, _ in AMG_SCHEMES:
            vals = [r for r in data[label].get(key, []) if np.isfinite(r)]
            if vals:
                parts.append(f"{np.mean(vals):>7.4f}±{np.std(vals):<5.4f}")
            else:
                parts.append(f"{'---':>14s}")
        print("  ".join(parts))


# ============================================================
# 4b. True V-cycle convergence factor rho(I - BA)
# ============================================================
def plot_vcycle_rho_comparison(runs, out_dir, prefix=""):
    """Per-sample V-cycle convergence factor rho(I - BA) for the three AMG
    arms (field ``rho_vcycle`` written by evaluate_preconditioners.py: power
    iteration on the V-cycle error propagator in the A-norm), plus a LaTeX
    table. This is the quantity the manuscript's rho_V figures describe;
    benchmarks made before the field existed are skipped.
    """
    _ensure_dir(out_dir)
    AMG_SCHEMES = [
        ("pyamg",       "SA-AMG", SCHEME_COLORS["pyamg"],       "s"),
        ("ruge_stuben", "RS-AMG", SCHEME_COLORS["ruge_stuben"], "D"),
        ("learned",     "Learned AMG", SCHEME_COLORS["learned"], "o"),
    ]
    labels_plot, data = [], {}
    for label, info in runs.items():
        bench = _load_benchmark(info["out_dir"])
        if bench is None:
            continue
        size_data = {}
        for key, _, _, _ in AMG_SCHEMES:
            vals = [bench[i]["schemes"][key].get("rho_vcycle", float("nan"))
                    for i in bench if key in bench[i]["schemes"]]
            size_data[key] = [v for v in vals if np.isfinite(v)]
        if any(size_data.values()):
            labels_plot.append(label)
            data[label] = size_data
    if not labels_plot:
        print("  WARNING: no rho_vcycle in these benchmarks (re-run "
              "evaluate_preconditioners.py), skipping rho_V plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x_positions = np.arange(len(labels_plot))
    offset = 0.15
    for i, (key, lbl, color, marker) in enumerate(AMG_SCHEMES):
        x_off = x_positions + (i - 1) * offset
        for j, label in enumerate(labels_plot):
            vals = data[label].get(key, [])
            if not vals:
                continue
            jitter = np.random.default_rng(42 + i + j).uniform(-0.03, 0.03, len(vals))
            ax.scatter(x_off[j] + jitter, vals, s=30, marker=marker, color=color,
                       alpha=0.6, edgecolors="none", zorder=4,
                       label=lbl if j == 0 else None)
            ax.scatter(x_off[j], np.mean(vals), s=120, marker=marker, color=color,
                       edgecolors="white", linewidths=1.2, zorder=6)
    ax.axhline(0.5, color="#888888", linestyle="--", linewidth=1, zorder=1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels_plot, fontsize=12)
    ax.set_ylabel(r"$\rho_V = \rho(I - BA)$", fontsize=12)
    ax.set_title("V-cycle Convergence Factor per Sample", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, os.path.join(out_dir, f"{prefix}rho_comparison.pdf"))
    plt.close(fig)

    fpath_tex = os.path.join(out_dir, f"{prefix}rho_comparison_table.tex")
    with open(fpath_tex, "w") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{V-cycle convergence factor $\\rho_V=\\rho(I-BA)$ "
                "(mean $\\pm$ std over test samples), estimated by power iteration "
                "on the V-cycle error propagator in the $A$-norm.}\n")
        f.write(f"\\label{{tab:{prefix}rho_v}}\n")
        f.write("\\begin{tabular}{lrrr}\n\\toprule\n")
        f.write("Size & SA-AMG & RS-AMG & Learned AMG \\\\\n\\midrule\n")
        for label in labels_plot:
            parts = [label]
            for key, _, _, _ in AMG_SCHEMES:
                vals = data[label].get(key, [])
                parts.append(f"${np.mean(vals):.3f} \\pm {np.std(vals):.3f}$" if vals else "---")
            f.write(" & ".join(parts) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"  Wrote {fpath_tex}")



# ============================================================
# 5. ρ(I−BA) Training Evolution
# ============================================================
def plot_rho_training_evolution(runs, out_dir, prefix=""):
    """Plot the recorded per-epoch cycle convergence factor for each size.

    Distinct from ``plot_pcg_residual_tail_ratio_comparison`` above, and the
    distinction is the whole point of I8: this function reads the diagnostics
    field ``vcycle_conv_factor``, which really is an error-propagation factor
    (a stationary quantity), whereas that one reads PCG residual histories (a
    Krylov quantity).  Two different numbers; never plot them on shared axes
    or describe them with the same words.

    Caveat that must travel with any figure produced here: the recorded
    ``vcycle_conv_factor`` values were produced by the *old*, uncalibrated
    estimator (I2 -- no renormalization, arm-dependent underflow break, max
    rather than median over probes, effectively one probe from a fixed seed).
    Use ``amg_diagnostics.compute_asymptotic_rho_stats`` for anything quoted,
    and treat these curves as a training-progress trace only.
    """
    _ensure_dir(out_dir)

    fig, ax = plt.subplots(figsize=(10, 5))
    has_data = False

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(runs)))

    for i, (label, info) in enumerate(runs.items()):
        evol = _load_diagnostics_evolution(info["out_dir"])
        if evol is None:
            continue

        epochs = sorted(int(e) for e in evol.keys())
        rho_vals = []
        valid_epochs = []
        for e in epochs:
            v = evol[str(e)].get("vcycle_conv_factor")
            if v is not None and np.isfinite(v):
                valid_epochs.append(e)
                rho_vals.append(v)

        if valid_epochs:
            ax.plot(valid_epochs, rho_vals, "o-", label=label,
                    color=colors[i], markersize=3, linewidth=1.5)
            has_data = True

    if not has_data:
        print("  WARNING: No training evolution data, skipping plot.")
        plt.close(fig)
        return

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(r"$\rho_V$ (V-cycle convergence factor)", fontsize=12)
    ax.set_title(r"$\rho(I - BA)$ During Training", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    fpath = os.path.join(out_dir, f"{prefix}rho_training_evolution.pdf")
    _save_fig(fig, fpath)
    plt.close(fig)


# ============================================================
# 6. Coarsening Ratio (C/F) Comparison
# ============================================================
def plot_coarsening_ratio(runs, out_dir, prefix=""):
    """Compare coarsening ratios: learned SA-AMG vs standard SA-AMG vs RS-AMG.

    For SA-AMG: coarsening ratio = n_aggregates / n_fine_nodes
    For RS-AMG: coarsening ratio = n_C_points / n_total
    """
    _ensure_dir(out_dir)

    labels_plot = []
    cr_learned, cr_sa, cr_rs = [], [], []
    oc_learned, oc_sa, oc_rs = [], [], []

    for label, info in runs.items():
        bench = _load_benchmark(info["out_dir"])
        if bench is None:
            continue

        # Aggregate operator complexity across samples
        oc_l, oc_s, oc_r = [], [], []
        gc_l, gc_s, gc_r = [], [], []
        for idx in bench:
            schemes = bench[idx]["schemes"]
            if "learned" in schemes:
                oc_l.append(schemes["learned"].get("op_complexity", 0))
                gc_l.append(schemes["learned"].get("grid_complexity", 0))
            if "pyamg" in schemes:
                oc_s.append(schemes["pyamg"].get("op_complexity", 0))
                gc_s.append(schemes["pyamg"].get("grid_complexity", 0))
            if "ruge_stuben" in schemes:
                oc_r.append(schemes["ruge_stuben"].get("op_complexity", 0))
                gc_r.append(schemes["ruge_stuben"].get("grid_complexity", 0))

        if oc_l:
            labels_plot.append(label)
            cr_learned.append(np.mean(gc_l))
            cr_sa.append(np.mean(gc_s) if gc_s else float("nan"))
            cr_rs.append(np.mean(gc_r) if gc_r else float("nan"))
            oc_learned.append(np.mean(oc_l))
            oc_sa.append(np.mean(oc_s) if oc_s else float("nan"))
            oc_rs.append(np.mean(oc_r) if oc_r else float("nan"))

    if not labels_plot:
        print("  WARNING: No coarsening data available, skipping plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(labels_plot))
    w = 0.25

    # Panel (a): Grid complexity
    ax = axes[0]
    ax.bar(x - w, cr_sa, w, label="SA-AMG", color=SCHEME_COLORS["pyamg"], alpha=0.8)
    ax.bar(x, cr_rs, w, label="RS-AMG", color=SCHEME_COLORS["ruge_stuben"], alpha=0.8)
    ax.bar(x + w, cr_learned, w, label="Learned AMG", color=SCHEME_COLORS["learned"], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_plot, fontsize=11)
    ax.set_ylabel("Grid Complexity", fontsize=12)
    ax.set_title("(a) Grid Complexity", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Panel (b): Operator complexity
    ax = axes[1]
    ax.bar(x - w, oc_sa, w, label="SA-AMG", color=SCHEME_COLORS["pyamg"], alpha=0.8)
    ax.bar(x, oc_rs, w, label="RS-AMG", color=SCHEME_COLORS["ruge_stuben"], alpha=0.8)
    ax.bar(x + w, oc_learned, w, label="Learned AMG", color=SCHEME_COLORS["learned"], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_plot, fontsize=11)
    ax.set_ylabel("Operator Complexity", fontsize=12)
    ax.set_title("(b) Operator Complexity", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fpath = os.path.join(out_dir, f"{prefix}coarsening_ratio.pdf")
    _save_fig(fig, fpath)
    plt.close(fig)


# ============================================================
# 7. Iteration Count Bar Chart (summary across sizes)
# ============================================================
def plot_iteration_summary(runs, out_dir, prefix=""):
    """Grouped bar chart: mean PCG iterations per scheme per matrix size."""
    _ensure_dir(out_dir)

    fig, ax = plt.subplots(figsize=(12, 5))

    size_labels = []
    scheme_data = {s: [] for s in ALL_SCHEMES}

    for label, info in runs.items():
        bench = _load_benchmark(info["out_dir"])
        if bench is None:
            continue
        size_labels.append(label)

        for s in ALL_SCHEMES:
            iters_list = []
            for idx in bench:
                sd = bench[idx]["schemes"].get(s)
                if sd is not None:
                    iters_list.append(sd["iters"])
            scheme_data[s].append(np.mean(iters_list) if iters_list else 0)

    if not size_labels:
        print("  WARNING: No benchmark data, skipping iteration summary.")
        return

    x = np.arange(len(size_labels))
    n_schemes = len(ALL_SCHEMES)
    total_w = 0.8
    w = total_w / n_schemes

    for i, s in enumerate(ALL_SCHEMES):
        offset = (i - n_schemes / 2 + 0.5) * w
        vals = scheme_data[s]
        ax.bar(x + offset, vals, w, label=SCHEME_LABELS[s],
               color=SCHEME_COLORS[s], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(size_labels, fontsize=12)
    ax.set_ylabel("Mean PCG Iterations", fontsize=12)
    ax.set_xlabel("Matrix Size", fontsize=12)
    ax.set_title("PCG Iteration Count by Preconditioner", fontsize=14)
    ax.legend(fontsize=9, ncol=2)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(plt.LogLocator(base=10, numticks=15))
    ax.yaxis.set_minor_locator(plt.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=15))
    ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())
    ax.grid(axis="y", which="major", alpha=0.4)
    ax.grid(axis="y", which="minor", alpha=0.15, linestyle="--")
    fig.tight_layout()

    fpath = os.path.join(out_dir, f"{prefix}iteration_summary.pdf")
    _save_fig(fig, fpath)
    plt.close(fig)


# ============================================================
# 8. Timing Breakdown Bar Chart — one subplot per matrix size
# ============================================================
def plot_timing_breakdown(runs, out_dir, prefix=""):
    """Stacked bar chart: setup + solve time per scheme, one subplot per matrix size."""
    _ensure_dir(out_dir)

    # Collect data per size
    panels = []  # list of (label, setup_dict, solve_dict, iters_dict)
    for label, info in runs.items():
        bench = _load_benchmark(info["out_dir"])
        if bench is None:
            continue
        schemes_present = [s for s in ALL_SCHEMES if s in _schemes_in_data(bench)]
        setup_vals, solve_vals, iters_vals = {}, {}, {}
        for s in schemes_present:
            setup_list, solve_list, iters_list = [], [], []
            for idx in bench:
                sd = bench[idx]["schemes"].get(s)
                if sd is not None:
                    setup_list.append(sd["setup_time"])
                    solve_list.append(sd["solve_time"])
                    iters_list.append(sd["iters"])
            setup_vals[s] = np.mean(setup_list) if setup_list else 0
            solve_vals[s] = np.mean(solve_list) if solve_list else 0
            iters_vals[s] = np.mean(iters_list) if iters_list else 0
        panels.append((label, schemes_present, setup_vals, solve_vals, iters_vals))

    if not panels:
        return

    n_panels = len(panels)
    ncols = min(n_panels, 3)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows),
                             squeeze=False)

    for idx, (label, schemes_present, setup_vals, solve_vals, iters_vals) in enumerate(panels):
        ax = axes[idx // ncols][idx % ncols]
        x = np.arange(len(schemes_present))
        w = 0.6

        setup_arr = [setup_vals[s] for s in schemes_present]
        solve_arr = [solve_vals[s] for s in schemes_present]

        bars_setup = ax.bar(x, setup_arr, w,
                           label="Setup", color=[SCHEME_COLORS[s] for s in schemes_present],
                           alpha=0.45, edgecolor="white", linewidth=0.5)
        bars_solve = ax.bar(x, solve_arr, w, bottom=setup_arr,
                           label="Solve", color=[SCHEME_COLORS[s] for s in schemes_present],
                           alpha=0.9, edgecolor="white", linewidth=0.5)

        # Overlay diagonal red stripes on non-converged bars
        for i, s in enumerate(schemes_present):
            if iters_vals[s] >= 499.0:
                tts = setup_vals[s] + solve_vals[s]
                ax.bar(i, tts, w, color="none", edgecolor="red",
                       linewidth=0, hatch="//", zorder=5)

        # Add TTS value labels on top
        for i, s in enumerate(schemes_present):
            tts = setup_vals[s] + solve_vals[s]
            ax.text(i, tts, f"{tts:.3f}s", ha="center", va="bottom",
                    fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([SCHEME_LABELS[s] for s in schemes_present],
                           fontsize=9, rotation=30, ha="right")
        ax.set_ylabel("Time (s)", fontsize=11)
        mat_size = None
        for rl, ri in runs.items():
            if rl == label:
                mat_size = ri.get("mat_size", ri["grid_n"])
        ax.set_title(f"$n = {mat_size:,}$" if mat_size else label, fontsize=12)
        ax.grid(axis="y", alpha=0.3)

    # Hide unused subplots
    for idx in range(n_panels, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    # Shared legend (setup=lighter, solve=darker, stripes=non-converged)
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#888888", alpha=0.45, label="Setup time"),
        Patch(facecolor="#888888", alpha=0.9, label="Solve time"),
        Patch(facecolor="white", edgecolor="red", hatch="//",
              label="Did not converge"),
    ]
    fig.legend(handles=legend_handles, loc="upper right", fontsize=10)
    fig.suptitle("Time-to-Solution Breakdown", fontsize=14, y=1.01)
    fig.tight_layout()

    fpath = os.path.join(out_dir, f"{prefix}timing_breakdown.pdf")
    _save_fig(fig, fpath)
    plt.close(fig)


# ============================================================
# 9. Convergence Profiles — residual decay curves
# ============================================================
def plot_convergence_profiles(runs, out_dir, prefix=""):
    """Show residual history for each preconditioner on a representative sample,
    one subplot per matrix size."""
    _ensure_dir(out_dir)

    panels = []
    for label, info in runs.items():
        bench = _load_benchmark(info["out_dir"])
        if bench is None:
            continue
        # Pick median-iteration sample for 'learned' scheme
        learned_iters = {
            idx: bench[idx]["schemes"]["learned"]["iters"]
            for idx in bench if "learned" in bench[idx]["schemes"]
        }
        if not learned_iters:
            continue
        median_iters = np.median(list(learned_iters.values()))
        best_idx = min(learned_iters, key=lambda k: abs(learned_iters[k] - median_iters))
        mat_size = info.get("mat_size", info["grid_n"])
        panels.append((label, mat_size, bench[best_idx]["schemes"]))

    if not panels:
        return

    n = len(panels)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows),
                             squeeze=False)

    for idx, (label, mat_size, schemes) in enumerate(panels):
        ax = axes[idx // ncols][idx % ncols]
        for s in ALL_SCHEMES:
            if s not in schemes:
                continue
            res = np.array(schemes[s]["residuals"], dtype=float)
            if len(res) == 0:
                continue
            # Normalise to initial residual
            res_norm = res / res[0] if res[0] > 0 else res
            ax.semilogy(range(len(res_norm)), res_norm,
                        color=SCHEME_COLORS[s], label=SCHEME_LABELS[s],
                        linewidth=2 if s == "learned" else 1.3,
                        alpha=1.0 if s == "learned" else 0.7,
                        linestyle="-" if s in ("learned", "pyamg", "ruge_stuben") else "--")
        ax.set_xlabel("Iteration", fontsize=10)
        ax.set_ylabel("Relative Residual $\\|r_k\\|/\\|r_0\\|$", fontsize=10)
        ax.set_title(f"$n = {mat_size:,}$", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=1e-16)

    # Hide unused
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    # Single legend
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Convergence Profiles — Representative Samples", fontsize=14, y=1.06)
    fig.tight_layout()
    _save_fig(fig, os.path.join(out_dir, f"{prefix}convergence_profiles.pdf"))
    plt.close(fig)


# ============================================================
# 10. Radar / Spider Chart — multi-metric comparison
# ============================================================
def plot_radar_chart(runs, out_dir, prefix=""):
    """Radar chart comparing preconditioners across normalised metrics.
    Metrics: iterations, setup time, solve time, κ_eff, ρ(I-BA)."""
    _ensure_dir(out_dir)

    # Aggregate means across all sizes
    metric_names = ["Iterations", "Setup Time", "Solve Time", "κ"]
    raw = {s: {m: [] for m in metric_names} for s in ALL_SCHEMES}

    for label, info in runs.items():
        bench = _load_benchmark(info["out_dir"])
        if bench is None:
            continue
        for idx in bench:
            for s in ALL_SCHEMES:
                sd = bench[idx]["schemes"].get(s)
                if sd is None:
                    continue
                raw[s]["Iterations"].append(sd["iters"])
                raw[s]["Setup Time"].append(sd["setup_time"])
                raw[s]["Solve Time"].append(sd["solve_time"])
                raw[s]["κ"].append(sd["kappa_raw"])

    # Compute means
    means = {}
    for s in ALL_SCHEMES:
        means[s] = {}
        for m in metric_names:
            vals = raw[s][m]
            means[s][m] = np.mean(vals) if vals else 0

    # Normalise each metric to [0, 1] where 1 = worst
    maxvals = {m: max(means[s][m] for s in ALL_SCHEMES) for m in metric_names}
    normed = {}
    for s in ALL_SCHEMES:
        normed[s] = []
        for m in metric_names:
            normed[s].append(means[s][m] / maxvals[m] if maxvals[m] > 0 else 0)

    N = len(metric_names)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for s in ALL_SCHEMES:
        vals = normed[s] + normed[s][:1]
        ax.plot(angles, vals, color=SCHEME_COLORS[s], linewidth=2,
                label=SCHEME_LABELS[s])
        ax.fill(angles, vals, color=SCHEME_COLORS[s], alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8, color="grey")
    ax.set_title("Preconditioner Comparison\n(lower = better, normalised to worst)",
                 fontsize=13, pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    fig.tight_layout()
    _save_fig(fig, os.path.join(out_dir, f"{prefix}radar_chart.pdf"))
    plt.close(fig)


# ============================================================
# 11. Speedup Heatmap
# ============================================================
def plot_speedup_heatmap(runs, out_dir, prefix=""):
    """Heatmap: speedup of each preconditioner over unpreconditioned CG,
    measured by iteration count."""
    _ensure_dir(out_dir)

    size_labels = []
    # Skip CG itself; show speedup for preconditioned methods
    prec_schemes = [s for s in ALL_SCHEMES if s != "none"]
    speedup_matrix = []

    for label, info in runs.items():
        bench = _load_benchmark(info["out_dir"])
        if bench is None:
            continue
        size_labels.append(label)

        # Mean CG iterations
        cg_iters = []
        for idx in bench:
            sd = bench[idx]["schemes"].get("none")
            if sd is not None:
                cg_iters.append(sd["iters"])
        mean_cg = np.mean(cg_iters) if cg_iters else 1

        row = []
        for s in prec_schemes:
            iters_list = []
            for idx in bench:
                sd = bench[idx]["schemes"].get(s)
                if sd is not None:
                    iters_list.append(sd["iters"])
            mean_s = np.mean(iters_list) if iters_list else mean_cg
            row.append(mean_cg / mean_s)
        speedup_matrix.append(row)

    if not size_labels:
        return

    data = np.array(speedup_matrix)

    fig, ax = plt.subplots(figsize=(8, max(3, 0.8 * len(size_labels) + 1.5)))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd")

    ax.set_xticks(range(len(prec_schemes)))
    ax.set_xticklabels([SCHEME_LABELS[s] for s in prec_schemes], fontsize=10,
                       rotation=25, ha="right")
    ax.set_yticks(range(len(size_labels)))
    ax.set_yticklabels(size_labels, fontsize=11)
    ax.set_ylabel("Matrix Size", fontsize=12)
    ax.set_xlabel("Preconditioner", fontsize=12)

    # Annotate cells
    for i in range(len(size_labels)):
        for j in range(len(prec_schemes)):
            val = data[i, j]
            color = "white" if val > data.max() * 0.65 else "black"
            ax.text(j, i, f"{val:.1f}×", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Iteration Speedup over CG", fontsize=10)
    ax.set_title("Iteration Speedup Heatmap (vs. Unpreconditioned CG)", fontsize=13)
    fig.tight_layout()
    _save_fig(fig, os.path.join(out_dir, f"{prefix}speedup_heatmap.pdf"))
    plt.close(fig)


# ============================================================
# 12. Pareto Scatter — setup cost vs solve efficiency
# ============================================================
def plot_pareto_scatter(runs, out_dir, prefix=""):
    """Scatter plot: mean setup time vs mean iterations, bubble size = TTS.
    One marker style per matrix size, colour by scheme."""
    _ensure_dir(out_dir)

    SIZE_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "h"]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Legend bookkeeping
    scheme_handles = {}
    size_handles = {}

    for si, (label, info) in enumerate(runs.items()):
        bench = _load_benchmark(info["out_dir"])
        if bench is None:
            continue
        marker = SIZE_MARKERS[si % len(SIZE_MARKERS)]

        for s in ALL_SCHEMES:
            iters_list, setup_list, solve_list = [], [], []
            for idx in bench:
                sd = bench[idx]["schemes"].get(s)
                if sd is not None:
                    iters_list.append(sd["iters"])
                    setup_list.append(sd["setup_time"])
                    solve_list.append(sd["solve_time"])
            if not iters_list:
                continue
            mean_iters = np.mean(iters_list)
            mean_setup = np.mean(setup_list)
            mean_tts = mean_setup + np.mean(solve_list)

            # Bubble size proportional to TTS (scale for visibility)
            bsize = max(40, mean_tts * 300)

            sc = ax.scatter(mean_setup, mean_iters, s=bsize,
                           c=SCHEME_COLORS[s], marker=marker,
                           alpha=0.7, edgecolors="black", linewidths=0.5,
                           zorder=5)

            # Collect legend entries (once per scheme / size)
            if s not in scheme_handles:
                scheme_handles[s] = ax.scatter([], [], c=SCHEME_COLORS[s],
                                               s=80, marker="o",
                                               edgecolors="black", linewidths=0.5,
                                               label=SCHEME_LABELS[s])
            if label not in size_handles:
                size_handles[label] = ax.scatter([], [], c="grey", s=60,
                                                  marker=marker,
                                                  edgecolors="black", linewidths=0.5,
                                                  label=label)

    ax.set_xlabel("Mean Setup Time (s)", fontsize=12)
    ax.set_ylabel("Mean PCG Iterations", fontsize=12)
    ax.set_title("Setup Cost vs. Iteration Efficiency\n(bubble size ∝ TTS)", fontsize=13)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25, which="both")

    # Two-part legend
    leg1 = ax.legend(handles=list(scheme_handles.values()),
                     title="Preconditioner", loc="upper left",
                     fontsize=8, title_fontsize=9)
    ax.add_artist(leg1)
    ax.legend(handles=list(size_handles.values()),
              title="Matrix Size", loc="lower right",
              fontsize=8, title_fontsize=9)

    fig.tight_layout()
    _save_fig(fig, os.path.join(out_dir, f"{prefix}pareto_scatter.pdf"))
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Generate thesis chapter figures and tables"
    )
    parser.add_argument("--out_dir", type=str, default="thesis_figures",
                        help="Output directory for figures and tables")
    parser.add_argument("--scalability", action="store_true",
                        help="Also generate scalability plots")
    parser.add_argument("--problem", type=str, nargs="+",
                        default=["lognormal"],
                        choices=["lognormal", "anisotropy", "poisson",
                                 "perforated_l", "all"],
                        help="Problem type(s) for scalability plots "
                             "(default: lognormal). Use 'all' for all 4.")
    args = parser.parse_args()

    # Expand 'all' shortcut
    if "all" in args.problem:
        args.problem = ["lognormal", "anisotropy", "poisson", "perforated_l"]

    out = _ensure_dir(args.out_dir)

    print("=" * 70)
    print("  Thesis Chapter Analysis — Learned AMG Preconditioner")
    print("=" * 70)

    for problem in args.problem:
        best_runs = BEST_RUNS_BY_PROBLEM[problem]
        prefix = f"{problem}_" if problem != "lognormal" else ""

        print("\n" + "=" * 70)
        print(f"  Base Analysis — {problem}")
        print("=" * 70)

        # ── Check data availability ──
        print(f"\n[0] Checking data availability ({problem})...")
        for label, info in best_runs.items():
            bench_path = ROOT / info["out_dir"] / "preconditioner_benchmark" / "benchmark_results.pt"
            diag_path = ROOT / info["out_dir"] / "diagnostics_evolution.json"
            model_path = ROOT / info["out_dir"] / "model_best.pt"
            print(f"  {label:>5s}: bench={'OK' if bench_path.exists() else 'MISSING':>7s}  "
                  f"diag={'OK' if diag_path.exists() else 'MISSING':>7s}  "
                  f"model={'OK' if model_path.exists() else 'MISSING':>7s}  "
                  f"dir={info['out_dir']}")

        # ── 1. TTS Tables ──
        print(f"\n[1] Generating TTS tables ({problem})...")
        generate_tts_tables(best_runs, out, prefix=prefix)

        # ── 1b. Operator Complexity Table ──
        print(f"\n[1b] Generating operator complexity table ({problem})...")
        generate_operator_complexity_table(best_runs, out, prefix=prefix)

        # ── 2. Eigenvalue Box Plot ──
        print(f"\n[2] Generating eigenvalue box-and-whisker plot ({problem})...")
        plot_eigenvalue_boxplot(best_runs, out, prefix=prefix)

        print(f"\n[2b] Generating eigenvalue box plot without CG ({problem})...")
        plot_eigenvalue_boxplot(best_runs, out, skip_schemes=["none"], prefix=prefix)

        # ── 3. Condition Number Box Plot ──
        print(f"\n[3] Generating condition number box-and-whisker plot ({problem})...")
        plot_condition_number_boxplot(best_runs, out, prefix=prefix)

        # ── 4. PCG residual tail ratio comparison (NOT ρ(I−BA); see I8) ──
        print(f"\n[4] Generating PCG residual tail ratio plot ({problem})...")
        plot_pcg_residual_tail_ratio_comparison(best_runs, out, prefix=prefix)
        plot_vcycle_rho_comparison(best_runs, out, prefix=prefix)

        # ── 5. ρ(I−BA) Training Evolution ──
        print(f"\n[5] Generating ρ(I−BA) training evolution plot ({problem})...")
        plot_rho_training_evolution(best_runs, out, prefix=prefix)

        # ── 6. Coarsening Ratio ──
        print(f"\n[6] Generating coarsening ratio comparison ({problem})...")
        plot_coarsening_ratio(best_runs, out, prefix=prefix)

        # ── 7. Iteration Summary ──
        print(f"\n[7] Generating iteration count summary ({problem})...")
        plot_iteration_summary(best_runs, out, prefix=prefix)

        # ── 8. Timing Breakdown ──
        print(f"\n[8] Generating timing breakdown ({problem})...")
        plot_timing_breakdown(best_runs, out, prefix=prefix)

        # ── 9. Convergence Profiles ──
        print(f"\n[9] Generating convergence profiles ({problem})...")
        plot_convergence_profiles(best_runs, out, prefix=prefix)

        # ── 10. Radar Chart ──
        print(f"\n[10] Generating radar chart ({problem})...")
        plot_radar_chart(best_runs, out, prefix=prefix)

        # ── 11. Speedup Heatmap ──
        print(f"\n[11] Generating speedup heatmap ({problem})...")
        plot_speedup_heatmap(best_runs, out, prefix=prefix)

        # ── 12. Pareto Scatter ──
        print(f"\n[12] Generating Pareto scatter plot ({problem})...")
        plot_pareto_scatter(best_runs, out, prefix=prefix)

    # ── Scalability ──
    if args.scalability:
      for problem in args.problem:
        scale_runs_all = _build_scalability_runs(problem)
        prefix = f"scale_{problem}_" if problem != "lognormal" else "scale_"

        print("\n" + "=" * 70)
        print(f"  Scalability Analysis — {problem}")
        print("=" * 70)

        print("\n[S0] Checking scalability data...")
        avail = {}
        for label, info in scale_runs_all.items():
            bench_path = ROOT / info["out_dir"] / "preconditioner_benchmark" / "benchmark_results.pt"
            exists = bench_path.exists()
            avail[label] = exists
            print(f"  {label}: {'OK' if exists else 'MISSING'}")

        scale_runs = OrderedDict(
            (k, v) for k, v in scale_runs_all.items() if avail.get(k, False)
        )
        if scale_runs:
            print(f"\n[S1] Scalability TTS tables ({problem})...")
            generate_tts_tables(scale_runs, out, prefix=prefix)

            print(f"\n[S2] Scalability eigenvalue box plot ({problem})...")
            plot_eigenvalue_boxplot(scale_runs, out, prefix=prefix)

            print(f"\n[S3] Scalability condition number box plot ({problem})...")
            plot_condition_number_boxplot(scale_runs, out, prefix=prefix)

            print(f"\n[S4] Scalability iteration summary ({problem})...")
            plot_iteration_summary(scale_runs, out, prefix=prefix)

            print(f"\n[S5] Scalability timing breakdown ({problem})...")
            scale_runs_no150 = OrderedDict(
                (k, v) for k, v in scale_runs.items() if k != "150k"
            )
            plot_timing_breakdown(scale_runs_no150, out, prefix=prefix)

            print(f"\n[S6] Scalability operator complexity table ({problem})...")
            generate_operator_complexity_table(scale_runs, out, prefix=prefix)

            # Combined table: all sizes (5k–500k)
            print(f"\n[S7] Combined operator complexity table ({problem}, all sizes)...")
            best_runs = BEST_RUNS_BY_PROBLEM[problem]
            all_runs = OrderedDict(list(best_runs.items()) + list(scale_runs.items()))
            generate_operator_complexity_table(all_runs, out, prefix=f"all_{problem}_" if problem != "lognormal" else "all_")
        else:
            print(f"\n  No scalability data available for {problem}.")
            print(f"  Run: sbatch job_scalability.sh {problem} <model_path>")

    print("\n" + "=" * 70)
    print(f"  Done. Figures saved to: {out}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
