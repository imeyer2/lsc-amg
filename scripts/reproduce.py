#!/usr/bin/env python3
"""Reproduce every experiment of the LSC-AMG manuscript (journal submission).

Steps, run from the repository root, in order:

  python scripts/reproduce.py train      # 15 configs x 5 seeds (configs/experiments.csv)
  python scripts/reproduce.py select     # best seed per config -> outputs/best_runs.json
  python scripts/reproduce.py benchmark  # 6 preconditioners on 32 test matrices per config
  python scripts/reproduce.py scale      # 50k models on 100k-500k (configs/scalability.csv)
  python scripts/reproduce.py compare    # tables vs. the paper -> outputs/comparison.md
  python scripts/reproduce.py figures    # all paper figures/tables -> outputs/figures/
  python scripts/reproduce.py status     # progress at any time

By default train/benchmark/scale submit SLURM jobs through scripts/slurm/run.sh;
pass site options with --sbatch, e.g.

  python scripts/reproduce.py train --sbatch "--account=ACCT --partition=gpu --gres=gpu:1"

Later --sbatch options override the defaults in RESOURCES. Extra training
options go through --train-args, e.g. --train-args "--fast_policy all --device cuda".

Use --local to run the commands one after another on this machine instead
(fine for the smoke test in tests/, far too slow for the full paper), and
--dry-run to only print them. Every step skips work that is already done, so
rerunning a step after a failure or preemption only redoes what is missing.

Layout of outputs/ (git-ignored):
  train/<problem>_<label>/seed<k>/   training run (model_best.pt, history, ...)
  scalability_<problem>_<label>/     zero-shot runs (lognormal: scalability_<label>,
                                     the names thesis_chapter_analysis.py expects)
  best_runs.json, comparison.md, figures/, logs/, wall_times.tsv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import shutil
import statistics as st
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / os.environ.get("LSC_OUTPUTS", "outputs")
PAPER = ROOT / "results" / "paper_original"
SEEDS = range(5)
SHORT = {"lognormal": "logn", "anisotropy": "aniso", "perforated_l": "perfl"}

# Per-step SLURM resources. 50k runs need ~175 GB with 46 workers.
RESOURCES = {
    "train": "--cpus-per-task=48 --mem=240G --time=48:00:00",
    "benchmark": "--cpus-per-task=16 --mem=128G --time=08:00:00",
    "scale": "--cpus-per-task=16 --mem=180G --time=12:00:00",
}


# ----------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------
def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def train_dir(row, seed):
    return OUT / "train" / f"{row['problem']}_{row['label']}" / f"seed{seed}"


def scale_dir(problem, label):
    return OUT / (f"scalability_{label}" if problem == "lognormal" else f"scalability_{problem}_{label}")


def train_cmd(row, seed, out):
    return [
        "python", "-u", "train_es_HPC.py",
        "--problem", row["problem"], "--grid_n", row["grid_n"],
        "--es_pop_size", row["pop"], "--es_mini_batch", row["mini_batch"],
        "--rl_epochs", row["epochs"], "--seed", str(seed),
        "--train_samples", "64", "--test_samples", "32",
        "--es_sigma", "0.02", "--rl_lr", "0.01", "--sigma_decay", "0.999",
        "--sigma_min", "0.005", "--weight_decay", "0.01",
        "--k_per_row", "4", "--hidden", "64", "--n_edge", "72",
        "--sigma_scale_threshold", "2.5", "--selection_mode", "threshold",
        "--init_threshold", "0.0", "--reward_lambda_cx", "2.0",
        "--full_eval_every", "1", "--checkpoint_every", "25",
        "--diagnostic_every", "5", "--soc_track_every", "5",
        "--out_dir", str(out.relative_to(ROOT)), "--resume",
    ]


def history(run):
    p = run / "training_history.json"
    return json.loads(p.read_text()) if p.exists() else None


def is_trained(row, run):
    h = history(run)
    return (h is not None and len(h["epochs"]) >= int(row["epochs"])
            and (run / "model_best.pt").exists())


# ----------------------------------------------------------------------------
# job submission
# ----------------------------------------------------------------------------
def launch(args, name, step, cmd, nodes=1):
    if args.local:
        print(f"[local] {name}: {shlex.join(cmd)}", flush=True)
        if not args.dry_run:
            env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
            subprocess.run(cmd, cwd=ROOT, env=env, check=True)
        return
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    wrapper, extra = "run.sh", []
    if nodes > 1:
        # population evaluation spread over `nodes` nodes (multinode.py)
        wrapper, extra = "run_multinode.sh", ["-N", str(nodes), "--ntasks-per-node=1"]
    sb = ["sbatch", "--parsable", "-J", name,
          "-o", str(OUT / "logs" / f"{name}_%j.out"), "-e", str(OUT / "logs" / f"{name}_%j.err"),
          *shlex.split(RESOURCES[step]), *extra, *shlex.split(args.sbatch),
          str(ROOT / "scripts" / "slurm" / wrapper), *cmd]
    if args.dry_run:
        print(shlex.join(sb))
        return
    jid = subprocess.run(sb, cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    with open(OUT / "submitted.tsv", "a") as f:
        f.write(f"{jid}\t{name}\t{shlex.join(cmd)}\n")
    print(f"{jid}  {name}")


# ----------------------------------------------------------------------------
# steps
# ----------------------------------------------------------------------------
def step_train(args):
    for row in read_csv(args.experiments):
        if args.only and f"{row['problem']}_{row['label']}" not in args.only:
            continue
        for seed in SEEDS if args.seeds is None else args.seeds:
            run = train_dir(row, seed)
            if is_trained(row, run) and not args.force:
                continue
            run.mkdir(parents=True, exist_ok=True)
            nodes = 1 if args.local or args.single_node else int(row.get("nodes") or 1)
            launch(args, f"train_{SHORT.get(row['problem'], row['problem'])}_{row['label']}_s{seed}",
                   "train", train_cmd(row, seed, run) + shlex.split(args.train_args), nodes=nodes)


def step_select(args):
    best = {}
    for row in read_csv(args.experiments):
        rs = {}
        for seed in SEEDS:
            run = train_dir(row, seed)
            h = history(run)
            rs[seed] = {"R": h["best_reward_so_far"][-1] if h else None,
                        "epochs": len(h["epochs"]) if h else 0,
                        "complete": is_trained(row, run)}
        done = {s: v for s, v in rs.items() if v["complete"]}
        e = {"grid_n": int(row["grid_n"]), "epochs": int(row["epochs"]),
             "complete": len(done) == len(SEEDS), "seeds": rs}
        if done:
            s = max(done, key=lambda k: done[k]["R"])
            e.update(best_seed=s, R=done[s]["R"],
                     out_dir=str(train_dir(row, s).relative_to(ROOT)))
        best.setdefault(row["problem"], {})[row["label"]] = e
        print(f"{row['problem']:13s} {row['label']:6s} seeds done {len(done)}/5"
              + (f"  best seed {e['best_seed']}  R={e['R']:.4f}" if done else ""))
    OUT.mkdir(exist_ok=True)
    (OUT / "best_runs.json").write_text(json.dumps(best, indent=2))
    return best


def load_best():
    p = OUT / "best_runs.json"
    if not p.exists():
        sys.exit("run `reproduce.py select` first")
    return json.loads(p.read_text())


def step_benchmark(args):
    best = load_best()
    for problem, runs in best.items():
        for label, e in runs.items():
            if not e["complete"]:
                print(f"skip {problem} {label}: not all seeds finished")
                continue
            run = ROOT / e["out_dir"]
            bench = run / "preconditioner_benchmark" / "benchmark_results.json"
            marker = run / "preconditioner_benchmark" / "GPU_REBENCHMARK"
            if marker.exists() and not args.force:
                continue
            # keep the training job's own benchmark (made on a CPU node when
            # the run did not have a GPU) next to the reference one
            if bench.exists() and not (run / "preconditioner_benchmark_trainnode").exists():
                shutil.copytree(run / "preconditioner_benchmark", run / "preconditioner_benchmark_trainnode")
            seed = json.loads((run / "config.json").read_text())["seed"]
            cmd = ["bash", "-c",
                   shlex.join(["python", "-u", "evaluate_preconditioners.py",
                               "--checkpoint", f"{e['out_dir']}/model_best.pt", "--out_dir", e["out_dir"],
                               "--test_samples", "32", "--seed", str(seed + 1000)])
                   + f" && touch {shlex.quote(str(marker.relative_to(ROOT)))}"]
            launch(args, f"bench_{SHORT.get(problem, problem)}_{label}", "benchmark", cmd)


def step_scale(args):
    best = load_best()
    for row in read_csv(args.scalability):
        src = best.get(row["problem"], {})
        big = max(src, key=lambda lab: src[lab]["grid_n"]) if src else None
        if big is None or not src[big]["complete"]:
            print(f"skip {row['problem']} {row['label']}: largest training size not finished")
            continue
        out = scale_dir(row["problem"], row["label"])
        if (out / "preconditioner_benchmark" / "benchmark_results.json").exists() and not args.force:
            continue
        out.mkdir(parents=True, exist_ok=True)
        cmd = ["python", "-u", "evaluate_preconditioners.py",
               "--checkpoint", f"{src[big]['out_dir']}/model_best.pt",
               "--out_dir", str(out.relative_to(ROOT)), "--grid_n", row["grid_n"],
               "--test_samples", "16", "--seed", "1000", "--problem", row["problem"],
               "--selection_mode", "threshold", "--hidden", "64",
               "--feature_preset", "new_algebraic_plus_relaxed"]
        launch(args, f"scale_{SHORT.get(row['problem'], row['problem'])}_{row['label']}", "scale", cmd)


def step_status(args):
    print(f"{'config':20s} {'seeds done':>10s}  epochs per seed")
    for row in read_csv(args.experiments):
        eps = []
        for seed in SEEDS:
            h = history(train_dir(row, seed))
            eps.append(len(h["epochs"]) if h else 0)
        done = sum(is_trained(row, train_dir(row, s)) for s in SEEDS)
        print(f"{row['problem'] + '_' + row['label']:20s} {done:>7d}/5  {eps} / {row['epochs']}")
    if (OUT / "best_runs.json").exists():
        for problem, runs in load_best().items():
            for label, e in runs.items():
                if "out_dir" in e:
                    b = ROOT / e["out_dir"] / "preconditioner_benchmark" / "GPU_REBENCHMARK"
                    print(f"benchmark {problem}_{label}: {'done' if b.exists() else 'missing'}")
    for row in read_csv(args.scalability):
        d = scale_dir(row["problem"], row["label"]) / "preconditioner_benchmark" / "benchmark_results.json"
        print(f"scale {row['problem']}_{row['label']}: {'done' if d.exists() else 'missing'}")


# ----------------------------------------------------------------------------
# comparison with the paper
# ----------------------------------------------------------------------------
SCHEMES = ["ilu", "pyamg", "ruge_stuben", "learned"]
NAMES = {"ilu": "ILU(0)", "pyamg": "SA-AMG", "ruge_stuben": "RS-AMG", "learned": "Learned AMG"}


def summarize(path):
    if not Path(path).exists():
        return None
    b = json.loads(Path(path).read_text())
    out = {"n": st.mean(b[k]["matrix_size"] for k in b)}
    for s in SCHEMES:
        rows = [b[k]["schemes"][s] for k in b if s in b[k]["schemes"]]
        if rows:
            rho = [r.get("rho_vcycle") for r in rows]
            rho = [x for x in rho if x is not None and x == x]
            out[s] = {"iters": st.mean(r["iters"] for r in rows),
                      "tts": st.mean(r["setup_time"] + r["solve_time"] for r in rows),
                      "cop": st.mean(r["op_complexity"] for r in rows),
                      "rho": st.mean(rho) if rho else None}
    return out


def f(d, s, k, fmt):
    return "-" if not d or s not in d or d[s].get(k) is None else fmt % d[s][k]


def step_compare(args):
    best = load_best()
    md = ["# Reproduction vs. paper\n",
          "paper = the manuscript's numbers (results/paper_original: the April run "
          "directories, which regenerate the paper's tables exactly). rerun = best "
          "seed of this reproduction. rho_V is the true V-cycle convergence factor "
          "(the paper's rho_V figures plotted a PCG residual tail ratio). Paper TTS "
          "for Learned AMG includes a one-off CUDA warm-up; the rerun's does not.\n"]
    for problem, runs in best.items():
        md.append(f"\n## {problem}\n")
        md.append("| Size | Method | n paper / rerun | Iters paper / rerun | TTS (s) paper / rerun | "
                  "C_op paper / rerun | rho_V rerun |")
        md.append("|---|---|---|---|---|---|---|")
        rows = [(lab, PAPER / f"{problem}_{lab}" / "benchmark_results.json",
                 ROOT / e["out_dir"] / "preconditioner_benchmark" / "benchmark_results.json" if "out_dir" in e else None)
                for lab, e in runs.items()]
        rows += [(lab, PAPER / scale_dir(problem, lab).name / "benchmark_results.json",
                  scale_dir(problem, lab) / "preconditioner_benchmark" / "benchmark_results.json")
                 for lab in ["100k", "150k", "200k", "500k"]]
        for lab, pp, rp in rows:
            p, r = summarize(pp), summarize(rp) if rp else None
            if p is None and r is None:
                continue
            n = " / ".join("-" if x is None else f"{x['n']:,.0f}" for x in (p, r))
            for s in SCHEMES:
                md.append(f"| {lab} | {NAMES[s]} | {n} | {f(p, s, 'iters', '%.1f')} / {f(r, s, 'iters', '%.1f')} | "
                          f"{f(p, s, 'tts', '%.4f')} / {f(r, s, 'tts', '%.4f')} | "
                          f"{f(p, s, 'cop', '%.2f')} / {f(r, s, 'cop', '%.2f')} | {f(r, s, 'rho', '%.3f')} |")
    (OUT / "comparison.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote {OUT / 'comparison.md'}")


def step_figures(args):
    sys.path.insert(0, str(ROOT))
    import thesis_chapter_analysis as tca
    from collections import OrderedDict
    tca.ROOT = ROOT
    _scal = tca._build_scalability_runs

    def _scal_here(problem="lognormal"):   # analysis hard-codes outputs/
        runs = _scal(problem)
        for v in runs.values():
            v["out_dir"] = v["out_dir"].replace("outputs/", f"{OUT.relative_to(ROOT)}/", 1)
        return runs
    tca._build_scalability_runs = _scal_here
    for problem, runs in load_best().items():
        tca.BEST_RUNS_BY_PROBLEM[problem] = OrderedDict(
            (lab, {"grid_n": e["grid_n"],
                   "mat_size": int(lab[:-1]) * 1000 if lab[:-1].isdigit() else e["grid_n"],
                   "out_dir": e["out_dir"]})
            for lab, e in runs.items() if "out_dir" in e)
    tca.BEST_RUNS = tca.BEST_RUNS_BY_PROBLEM.get("lognormal", tca.BEST_RUNS)
    sys.argv = ["thesis_chapter_analysis.py", "--out_dir", str(OUT / "figures"), "--scalability",
                "--problem", *load_best().keys()]
    tca.main()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["train", "select", "benchmark", "scale", "status", "compare", "figures"])
    ap.add_argument("--experiments", default=str(ROOT / "configs" / "experiments.csv"))
    ap.add_argument("--scalability", default=str(ROOT / "configs" / "scalability.csv"))
    ap.add_argument("--sbatch", default="", help="extra sbatch options (account, partition, gres, ...)")
    ap.add_argument("--train-args", default="", help="extra train_es_HPC.py options, e.g. '--fast_policy all'")
    ap.add_argument("--single-node", action="store_true",
                    help="train: ignore the nodes column and run every config on one node")
    ap.add_argument("--local", action="store_true", help="run here instead of submitting to SLURM")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="redo work that already exists")
    ap.add_argument("--only", nargs="*", help="train: only these configs, e.g. lognormal_50k")
    ap.add_argument("--seeds", type=int, nargs="*", help="train: only these seeds")
    args = ap.parse_args()
    {"train": step_train, "select": step_select, "benchmark": step_benchmark, "scale": step_scale,
     "status": step_status, "compare": step_compare, "figures": step_figures}[args.step](args)


if __name__ == "__main__":
    main()
