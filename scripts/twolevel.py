#!/usr/bin/env python3
"""Two-level comparison: learned vs time-tuned classical strength graphs for two-level SA-AMG.

Steps, run from the repository root, in order (each step submits SLURM jobs; rerun a step after its
jobs finish to run its aggregation tail, e.g. the cost-model fit; finished work is skipped):

  python scripts/twolevel.py data       # test (seed 1000) / tuning (2000) / training (0, 1) / calibration (500) matrices
  python scripts/twolevel.py tune       # classical rules tuned by total time (50k, 500k; default and improved splu)
  python scripts/twolevel.py bar        # the bar: tuned classical + paper GAT models at 50k / 200k / 500k, both splu
  python scripts/twolevel.py paper      # paper-model variants: threshold re-tuned by time, GPU probes, cost profiles
  python scripts/twolevel.py costmodel  # classical sweeps + cost-model fit (reward of the node policies)
  python scripts/twolevel.py train      # small learned policies: node (ES on the cost model) and lin (CMA on time)
  python scripts/twolevel.py select     # node-policy variants on the tuning set (selection)
  python scripts/twolevel.py final      # final paired comparisons on the test set, 50k / 200k / 500k, both splu
  python scripts/twolevel.py headroom   # oracle search for the best strength graph per matrix + re-timing
  python scripts/twolevel.py report     # tables from whatever exists -> outputs/twolevel/report.md

By default `final` and `select` use the policies committed in results/twolevel_2026-10/policies, `train`
uses the committed cost models and `headroom` re-times masks rebuilt from the committed oracle parameters,
so every step can be rerun on its own; pass --policies new / --cost-model new / --oracle new to chain
on freshly regenerated artifacts instead. The paper's GAT models are read from --models (default:
results/rerun_2026-10b/runs if present -- those checkpoints are not part of this release -- else
$LSC_OUTPUTS/train from scripts/reproduce.py); the best seed per configuration comes from best_runs.json
next to that directory. The `headroom` step's default needs results/twolevel_2026-10/headroom/oracle_params,
which is not in this release either: pass --oracle new to rerun the search.

SLURM: timing jobs need whole nodes of ONE node type (the 2026-10 numbers: V100 nodes with Xeon Gold
6248 CPUs); compute jobs (data, sweeps, training, oracle search) can run anywhere. Site options go
through --sbatch (all jobs), --sbatch-gpu (timing jobs) and --sbatch-cpu. Example (placeholders; use your
site's account, partitions and constraints):

  python scripts/twolevel.py final --sbatch=--account=ACCOUNT \
      --sbatch-gpu="--partition=GPU_PARTITION --constraint=GPU_CONSTRAINT" --sbatch-cpu="--partition=CPU_PARTITION"

--local runs the commands here one after another; --small shrinks everything (n ~ 3k, 2 matrices, 1 rep,
tiny searches; with --local the whole pipeline runs in a few minutes: tests/test_twolevel.py);
--dry-run prints the commands. --problems / --sizes restrict a step.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import twolevel.common  # noqa: E402,F401  (single-threaded BLAS/OpenMP before numpy/torch load)
OUT = ROOT / os.environ.get("LSC_OUTPUTS", "outputs")
TL = Path(os.environ["TWOLEVEL_OUT"]) if os.environ.get("TWOLEVEL_OUT") else OUT / "twolevel"
COMMITTED = ROOT / "results" / "twolevel_2026-10"
PY = [sys.executable, "-u", "-m"]

PROBLEMS = ["anisotropy", "lognormal", "perforated_l"]
SIZES = [50000, 200000, 500000]
TEST_SEED, TUNE_SEED = 1000, 2000
TEST_COUNT = {50000: 16, 200000: 8, 500000: 8}
TUNE_COUNT = {50000: 8, 200000: 4, 500000: 4}
S = "perm=MMD_AT_PLUS_A,dpt=0,sym=1"           # improved splu (both methods)

RESOURCES = {
    # timing: a whole node, 1 pinned thread per worker (W=2 for definitive tables, W=8 for tuning)
    "gpu": "--gres=gpu:1 --exclusive --mem=0 --time=08:00:00",
    "cpu": "--cpus-per-task=48 --mem=200G --time=12:00:00",
    "cpu_small": "--cpus-per-task=16 --mem=96G --time=06:00:00",
}

# ----------------------------------------------------------------------------------------------------
# What the 2026-10 study found / chose. `tune` and `select` regenerate the evidence for these choices
# and print what they find; the later steps use these constants so that they reproduce the reported runs.
# ----------------------------------------------------------------------------------------------------
# tuned classical rules (best of each family by mean total time on the 8 tuning matrices, default splu, 50k)
CL50 = {"anisotropy": ["theta:0.1", "topk:4", "sym:0.02", "srowq:0.7"],
        "lognormal": ["theta:0.02", "topk:5", "sym:0.02", "srowq:0.8"],
        "perforated_l": ["theta:0.2", "topk:6", "sym:0.08", "srowq:0.7"]}
# the same, re-tuned at 500k (4 tuning matrices, 1 rep)
CL500 = {"anisotropy": ["theta:0.05", "topk:6", "sym:0.01", "srowq:0.8"],
         "lognormal": ["theta:0.02", "topk:6", "sym:0.01", "srowq:0.8"],
         "perforated_l": ["theta:0.1", "topk:6", "sym:0.08", "srowq:0.8"]}
# improved splu, 50k
CLS50 = {"anisotropy": ["theta:0.1", "topk:4", "sym:0.08", "srowq:0.6"],
         "lognormal": ["theta:0.02", "topk:6", "sym:0.04", "srowq:0.8"],
         "perforated_l": ["theta:0.4", "topk:6", "sym:0.1", "srowq:0.7"]}
REF_RULES = ["topk:1", "topk:2", "srowq:0.3"]       # the rules quoted in the brief (not tuned)
# paper GAT models used (best seed of the rerun_2026-10b configs): 50k- and 5k-trained
PAPER = {"anisotropy": ("50k", "5k"), "lognormal": ("50k", "5k"), "perforated_l": ("50k", "5k")}
# learned threshold re-chosen by total time on the tuning set (paper/thr_def_*): z-threshold per model
THR_GRID = {"anisotropy": [-1.5, -1.0, -0.75, -0.5, -0.25, 0.0], "lognormal": [-2.25, -2.0, -1.75, -1.5, -1.25, -0.75],
            "perforated_l": [-2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5]}
THR_GRID_S = {"anisotropy": [-1.0, -0.5, 0.0, 0.25, 0.5], "lognormal": [-2.0, -1.5, -1.25, -0.75, -0.5, 0.0],
              "perforated_l": [-1.75, -1.25, -0.75, -0.5, 0.0]}
# improved splu: learned variant selected on the tuning set (paper/thr_sym_*), run with GPU probes
LV_SYM = {"anisotropy": ("50k", "thr=-1.0,"), "lognormal": ("5k", ""), "perforated_l": ("5k", "")}

# small learned policies (configs of the reported runs). node: twolevel.train_node, lin: twolevel.train_lin
NODE_POLICIES = {}
for _p, _row, _abs in (("anisotropy", "theta:0.1", "sym:0.02"), ("lognormal", "theta:0.1", "sym:0.02"),
                       ("perforated_l", "theta:0.3", "sym:0.02")):
    for _b, _init in (("noderow", _row), ("nodeabs", _abs)):
        NODE_POLICIES[f"{_p}_{_b}_cost50k"] = (_p, ["--init", _init, "--reward", "cost", "--n", "50000", "--n_train", "16",
                                                     "--n_val", "8", "--mb", "4", "--epochs", "100"])
        NODE_POLICIES[f"{_p}_{_b}_costtgt"] = (_p, ["--init", _init, "--reward", "cost_tgt", "--pop", "32", "--mb", "12",
                                                     "--epochs", "150", "--n_val", "16"])
for _p, _row, _abs in (("anisotropy", "theta:0.2", "sym:0.05"), ("lognormal", "theta:0.1", "sym:0.02"),
                       ("perforated_l", "theta:0.4", "sym:0.05")):
    for _b, _init in (("noderow", _row), ("nodeabs", _abs)):
        NODE_POLICIES[f"S_{_p}_{_b}_cost50k"] = (_p, ["--init", _init, "--reward", "cost", "--n", "50000", "--n_train", "16",
                                                       "--n_val", "8", "--mb", "4", "--epochs", "100", "--splu", "sym"])
LIN_POLICIES = {"lin_anisotropy_50k_theta": ("anisotropy", ["--base", "theta"]),
                "lin_lognormal_50k_theta_W8ext": ("lognormal", ["--base", "theta", "--workers", "8", "--nb", "jump,s2,dom"]),
                "lin_perforated_l_50k_theta": ("perforated_l", ["--base", "theta"])}
LIN_OF = {"anisotropy": "lin_anisotropy_50k_theta", "lognormal": "lin_lognormal_50k_theta_W8ext",
          "perforated_l": "lin_perforated_l_50k_theta"}
TWO = "two:1,0.17,1,0.05"     # hand rule distilled from the perforated-L lin model

# final comparisons: methods per (solver, problem, size); "node:X" / "lin:X" are policy names
FINAL = {
    ("def", "anisotropy", 50000): ["theta:0.1", "srowq:0.7", "sym:0.02", "node:anisotropy_noderow_cost50k",
                                   "node:anisotropy_nodeabs_cost50k", "lin:" + LIN_OF["anisotropy"]],
    ("def", "anisotropy", 200000): ["theta:0.1", "sym:0.02", "sym:0.04", "sym:0.06", "lin:" + LIN_OF["anisotropy"]],
    ("def", "anisotropy", 500000): ["theta:0.1", "sym:0.02", "sym:0.04", "sym:0.06", "lin:" + LIN_OF["anisotropy"]],
    ("def", "lognormal", 50000): ["sym:0.02", "sym:0.01", "theta:0.02", "node:lognormal_nodeabs_costtgt",
                                  "node:lognormal_nodeabs_cost50k", "lin:" + LIN_OF["lognormal"]],
    ("def", "lognormal", 200000): ["sym:0.01", "sym:0.02", "lin:" + LIN_OF["lognormal"]],
    ("def", "lognormal", 500000): ["sym:0.01", "sym:0.02", "lin:" + LIN_OF["lognormal"]],
    ("def", "perforated_l", 50000): ["theta:0.1", "theta:0.2", "srowq:0.7", "node:perforated_l_noderow_costtgt",
                                     "node:perforated_l_noderow_cost50k", "node:perforated_l_nodeabs_costtgt",
                                     "lin:" + LIN_OF["perforated_l"], TWO],
    ("sym", "anisotropy", 50000): ["theta:0.1", "theta:0.05", "node:S_anisotropy_noderow_cost50k",
                                   "node:anisotropy_noderow_cost50k", "lin:" + LIN_OF["anisotropy"]],
    ("sym", "lognormal", 50000): ["sym:0.04", "sym:0.02", "node:lognormal_nodeabs_costtgt",
                                  "node:S_lognormal_nodeabs_cost50k", "lin:" + LIN_OF["lognormal"]],
    ("sym", "lognormal", 200000): ["sym:0.02", "sym:0.04", "lin:" + LIN_OF["lognormal"]],
    ("sym", "lognormal", 500000): ["sym:0.02", "sym:0.04", "lin:" + LIN_OF["lognormal"]],
    ("sym", "perforated_l", 50000): ["theta:0.2", "theta:0.3", "theta:0.4", "theta:0.5",
                                     "node:perforated_l_noderow_costtgt", "node:S_perforated_l_noderow_cost50k",
                                     "lin:" + LIN_OF["perforated_l"], TWO],
}
for _n in (200000, 500000):
    FINAL[("def", "perforated_l", _n)] = ["theta:0.1", "theta:0.2", "srowq:0.7", "srowq:0.8",
                                          "node:perforated_l_noderow_costtgt", "node:perforated_l_noderow_cost50k",
                                          "lin:" + LIN_OF["perforated_l"], TWO]
    FINAL[("sym", "perforated_l", _n)] = ["theta:0.2", "theta:0.3", "theta:0.4", "lin:" + LIN_OF["perforated_l"], TWO]
# selection of node-policy variants on the tuning set (seed 2000, 8 matrices)
SELECT = {
    ("def", "anisotropy"): ["theta:0.1", "srowq:0.7", "topk:4", "sym:0.02"] + [f"node:anisotropy_{b}_{r}" for b in ("nodeabs", "noderow") for r in ("cost50k", "costtgt")],
    ("def", "lognormal"): ["sym:0.02", "theta:0.02", "srowq:0.8"] + [f"node:lognormal_{b}_{r}" for b in ("nodeabs", "noderow") for r in ("cost50k", "costtgt")],
    ("def", "perforated_l"): ["theta:0.2", "srowq:0.7", "topk:6", "sym:0.08", "theta:0.3"] + [f"node:perforated_l_{b}_{r}" for b in ("nodeabs", "noderow") for r in ("cost50k", "costtgt")],
    ("sym", "anisotropy"): ["theta:0.1", "theta:0.05", "topk:4", "sym:0.04", "node:S_anisotropy_nodeabs_cost50k",
                            "node:S_anisotropy_noderow_cost50k", "node:anisotropy_noderow_cost50k"],
    ("sym", "lognormal"): ["sym:0.04", "sym:0.02", "node:S_lognormal_nodeabs_cost50k", "node:S_lognormal_noderow_cost50k",
                           "node:lognormal_nodeabs_costtgt"],
    ("sym", "perforated_l"): ["theta:0.4", "theta:0.3", "theta:0.5", "node:S_perforated_l_nodeabs_cost50k",
                              "node:S_perforated_l_noderow_cost50k", "node:perforated_l_noderow_costtgt"],
}
# oracle searches: base -> (workers, pop); 'symm' only where it helped (anisotropy, lognormal)
ORACLE = {"anisotropy": {"theta": (28, 28), "symm": (8, 16)}, "lognormal": {"theta": (28, 28), "symm": (8, 16)},
          "perforated_l": {"theta": (28, 28)}}


# ----------------------------------------------------------------------------------------------------
# configuration (full / --small)
# ----------------------------------------------------------------------------------------------------
class Cfg:
    def __init__(self, args):
        self.a = args
        self.small = args.small
        self.problems = args.problems or PROBLEMS
        self.sizes = args.sizes or SIZES
        self.cuda = _has_cuda() if args.small else True

    def n(self, n):
        return {50000: 3000, 200000: 3500, 500000: 4000, 5000: 1500}.get(n, n) if self.small else n

    def count(self, which, n):
        if self.small:
            return 2
        return (TEST_COUNT if which == "test" else TUNE_COUNT)[n]

    def reps(self, r):
        return 1 if self.small else r

    def w(self, w, harness=False):
        if self.small:
            return 0 if harness else min(w, 2)     # tiny runs: the harness runs in-process
        return w

    def grid(self, lst, k=3):
        return lst[:k] if self.small else lst


def model_dir(args):
    if args.models:
        return Path(args.models)
    cand = ROOT / "results" / "rerun_2026-10b" / "runs"
    return cand if cand.exists() else OUT / "train"


def paper_model(cfg, problem, label):
    """model_best.pt of the paper's policy for (problem, label), best seed from best_runs.json."""
    if cfg.small:
        p = ROOT / "results" / "paper_original" / f"{problem}_5k" / "model_best.pt"
        return str(p) if p.exists() else None
    d = model_dir(cfg.a)
    for br in (d.parent / "best_runs.json", d / "best_runs.json"):
        if br.exists():
            seed = json.loads(br.read_text()).get(problem, {}).get(label, {}).get("best_seed")
            if seed is not None:
                p = d / f"{problem}_{label}" / f"seed{seed}" / "model_best.pt"
                return str(p) if p.exists() else None
    return None


def policy_path(cfg, kind, name):
    """committed or regenerated small policy file."""
    if cfg.a.policies == "committed" and not cfg.small:
        return rel(COMMITTED / "policies" / kind / f"{name}.json")
    return rel(TL / "policies" / kind / (f"{name}.json" if kind == "lin" else f"{name}/train.json"))


def resolve(cfg, spec, suffix=""):
    """method string of the harness: node:/lin: policy names -> files; optional '|<suffix>'."""
    kind, _, arg = spec.partition(":")
    if kind in ("node", "lin") and "/" not in arg:
        spec = f"{kind}:{policy_path(cfg, kind, arg)}"
    return spec + (f"|{suffix}" if suffix else "")


# ----------------------------------------------------------------------------------------------------
# jobs
# ----------------------------------------------------------------------------------------------------
def rel(p):
    p = Path(p)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def methods_for(cfg, methods):
    """--small on a machine without a GPU: the @cuda rows run on the CPU (same code path otherwise)."""
    if cfg.small and not cfg.cuda:
        methods = [m.replace("@cuda", "@cpu") for m in methods]
    return list(dict.fromkeys(methods))


def harness_run(cfg, problem, n, which, methods, out, reps=3, workers=2, count=None):
    """(command, expected number of result rows) of one harness run."""
    seed = TEST_SEED if which == "test" else TUNE_SEED
    methods = methods_for(cfg, methods)
    count = cfg.count(which, n) if (count is None or cfg.small) else count
    cmd = PY + ["twolevel.harness", "run", "--problem", problem, "--n", str(cfg.n(n)), "--seed", str(seed),
                "--count", str(count), "--reps", str(cfg.reps(reps)), "--workers", str(cfg.w(workers, harness=True)),
                *([] if cfg.a.force else ["--resume"]), "--methods", *methods, "--out", rel(out)]
    return cmd, count * len(methods)


def complete(path, nrows=None):
    path = Path(path)
    if not path.exists():
        return False
    if nrows is None or path.suffix != ".json":
        return True
    try:
        return len(json.loads(path.read_text())["rows"]) >= nrows
    except (ValueError, KeyError):
        return False


class Job:
    def __init__(self, name, kind, cmds, outputs):
        self.name, self.kind, self.cmds, self.outputs = name, kind, cmds, outputs   # outputs: [(path, nrows|None)]

    def done(self):
        return all(complete(p, n) for p, n in self.outputs)


def run_local(cmd):
    """python -m twolevel.X ... runs in this process (no interpreter start-up per command)."""
    if cmd[:3] == PY and cmd[3].startswith("twolevel."):
        import importlib
        cwd = os.getcwd()
        os.chdir(ROOT)
        try:
            importlib.import_module(cmd[3]).main(cmd[4:])
        finally:
            os.chdir(cwd)
    else:
        subprocess.run(cmd, cwd=ROOT, check=True)


def submit(args, jobs):
    todo = [j for j in jobs if args.force or not j.done()]
    for j in jobs:
        if j not in todo:
            print(f"done   {j.name}")
    for j in todo:
        if args.local:
            for c in j.cmds:
                print(f"[local] {j.name}: {shlex.join(c)}", flush=True)
                if not args.dry_run:
                    run_local(c)
            continue
        (TL / "logs").mkdir(parents=True, exist_ok=True)
        script = " && ".join(shlex.join(c) for c in j.cmds)
        site = args.sbatch_gpu if j.kind == "gpu" else args.sbatch_cpu
        sb = ["sbatch", "--parsable", "-J", f"tl_{j.name}", "-o", str(TL / "logs" / f"{j.name}_%j.out"),
              "-e", str(TL / "logs" / f"{j.name}_%j.err"), *shlex.split(RESOURCES[j.kind]),
              *shlex.split(args.sbatch), *shlex.split(site), str(ROOT / "scripts" / "slurm" / "run.sh"), "bash", "-c", script]
        if args.dry_run:
            print(shlex.join(sb))
            continue
        jid = subprocess.run(sb, cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        with open(TL / "submitted.tsv", "a") as f:
            f.write(f"{jid}\t{j.name}\t{script}\n")
        print(f"{jid}  {j.name}")
    return todo


# ---------------------------------------------------------------------------------------------- data
def step_data(cfg):
    sets = []
    for n in (50000, 200000, 500000):
        sets.append((n, TEST_SEED, cfg.count("test", n)))
        sets.append((n, TUNE_SEED, cfg.count("tune", n)))
    small = cfg.small
    # node-policy training/validation sets and cost-model calibration sets
    sets += [(5000, 0, 2 if small else 64), (5000, 1, 2 if small else 16), (50000, 0, 2 if small else 16),
             (50000, 1, 2 if small else 8), (50000, 500, 2 if small else 8)]
    jobs = []
    for p in cfg.problems:
        cmds = [PY + ["twolevel.harness", "gen", "--problem", p, "--sets",
                      *[f"{cfg.n(n)}:{seed}:{count}" for n, seed, count in sets]]]
        from twolevel.common import set_paths
        outs = [(set_paths(p, cfg.n(n), seed, count)[-1], None) for n, seed, count in sets]
        jobs.append(Job(f"data_{p}", "cpu_small", cmds, outs))
    return jobs


# ---------------------------------------------------------------------------------------------- tune
TUNE_GRID = (["topk:1", "topk:2", "topk:3", "topk:4", "topk:5", "topk:6"]
             + [f"theta:{t}" for t in (0.0, 0.02, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)]
             + [f"sym:{t}" for t in (0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)]
             + [f"srowq:{q}" for q in (0.17, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8)])
TUNE500_GRID = ["theta:0.02", "theta:0.05", "theta:0.1", "theta:0.2", "theta:0.4", "sym:0.01", "sym:0.02", "sym:0.04",
                "sym:0.08", "topk:2", "topk:4", "topk:6", "srowq:0.5", "srowq:0.7", "srowq:0.8"]
TUNE_GRID_S = (["topk:1", "topk:2", "topk:3", "topk:4", "topk:6"]
               + [f"theta:{t}" for t in (0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]
               + [f"sym:{t}" for t in (0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3)]
               + [f"srowq:{q}" for q in (0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8)])
TUNE500_GRID_S = ["theta:0.05", "theta:0.1", "theta:0.2", "theta:0.4", "sym:0.02", "sym:0.04", "sym:0.08", "sym:0.1",
                  "topk:3", "topk:4", "topk:6", "srowq:0.6", "srowq:0.7", "srowq:0.8"]
SPLU_PROBE = ["", "dpt=0", "perm=MMD_AT_PLUS_A", "perm=MMD_AT_PLUS_A,dpt=0", S, "perm=MMD_ATA", "perm=NATURAL,dpt=0"]


def step_tune(cfg):
    jobs = []
    for p in cfg.problems:
        cmds, outs = [], []

        def add(n, methods, out, reps, workers, which="tune", count=None):
            c, k = harness_run(cfg, p, n, which, methods, out, reps, workers, count)
            cmds.append(c)
            outs.append((out, k))
        add(50000, cfg.grid(TUNE_GRID, 6), TL / "tune" / f"def_50000_{p}.json", 3, 8)
        add(50000, cfg.grid([f"{m}|{S}" for m in TUNE_GRID_S], 4), TL / "tune" / f"sym_50000_{p}.json", 3, 8)
        add(500000, cfg.grid(TUNE500_GRID, 3), TL / "tune" / f"def_500000_{p}.json", 1, 2)
        add(500000, cfg.grid([f"{m}|{S}" for m in TUNE500_GRID_S], 3), TL / "tune" / f"sym_500000_{p}.json", 1, 4)
        probe = [m + (f"|{o}" if o else "") for m in ("sym:0.02", "theta:0.1", "topk:1") for o in SPLU_PROBE]
        add(50000, cfg.grid(probe, 3), TL / "tune" / f"splu_options_{p}.json", 1, 2, count=2)
        if p == "anisotropy":
            mdl = paper_model(cfg, p, "5k")
            meth = ["topk:1", "theta:0.5", "sym:0.08", "srowq:0.3"] + ([f"lsc:{mdl}@cpu", f"lsc:{mdl}@cuda"] if mdl else [])
            for W in (1, 8, 16):
                add(50000, cfg.grid(meth, 2), TL / "tune" / f"contention_W{W}.json", 3, W, count=4)
        jobs.append(Job(f"tune_{p}", "gpu", cmds, outs))
    return jobs


def tune_tail(cfg):
    from twolevel import tables
    found = {}
    for sol in ("def", "sym"):
        for n in (50000, 500000):
            files = [str(TL / "tune" / f"{sol}_{n}_{p}.json") for p in cfg.problems]
            files = [f for f in files if Path(f).exists()]
            if files:
                found[f"{sol}_{n}"] = tables.winners(files)
    if found:
        (TL / "tune" / "winners.json").write_text(json.dumps(found, indent=1))
        print("tuned winners per family (compare with CL50 / CL500 / CLS50 in scripts/twolevel.py):")
        for k, v in found.items():
            for kk, vv in v.items():
                print(f"  {k:10s} {kk:28s} {vv['method']:40s} {vv['total']:.4f}")


# ---------------------------------------------------------------------------------------------- bar
def _need_models(cfg, p):
    m50, m5 = (paper_model(cfg, p, lab) for lab in PAPER[p])
    if not (m50 and m5):
        print(f"[{p}] paper GAT models not found under {model_dir(cfg.a)}: learned rows skipped "
              "(pass --models, see the module docstring)")
    return m50, m5


def step_bar(cfg):
    jobs = []
    for p in cfg.problems:
        a, b = _need_models(cfg, p)
        cmds, outs = [], []

        def add(n, methods, out):
            c, k = harness_run(cfg, p, n, "test", methods, out, 3, 2)
            cmds.append(c)
            outs.append((out, k))
        if 50000 in cfg.sizes:
            m = CL50[p] + REF_RULES
            if a:
                m += [f"lsc:{a}@cpu", f"lsc:{a}@cuda", f"lsc:{b}@cpu", f"lsc:{b}@cuda", f"lscref:{a}@cpu", f"lscref:{a}@cuda"]
            add(50000, m, TL / "bar" / f"def_50000_{p}.json")
        for n in (200000, 500000):
            if n in cfg.sizes:
                m = list(dict.fromkeys(CL50[p] + CL500[p])) + ([f"lsc:{a}@cpu", f"lsc:{a}@cuda"] if a else [])
                add(n, m, TL / "bar" / f"def_{n}_{p}.json")
        lab, opt = LV_SYM[p]
        lv = paper_model(cfg, p, lab)
        for n in cfg.sizes:
            m = [f"{r}|{S}" for r in CLS50[p]]
            if a:
                m += [f"lsc:{a}@cuda|{S}", f"lsc:{a}@cpu|{S}", f"lsc:{a}@cuda|gprobe=1,{S}", f"lsc:{lv}@cuda|{opt}gprobe=1,{S}"]
            add(n, m, TL / "bar" / f"sym_{n}_{p}.json")
        jobs.append(Job(f"bar_{p}", "gpu", cmds, outs))
    return jobs


# ---------------------------------------------------------------------------------------------- paper
def step_paper(cfg):
    jobs = []
    for p in cfg.problems:
        a, b = _need_models(cfg, p)
        if not a:
            continue
        cmds, outs = [], []

        def add(n, which, methods, out, reps=3, workers=2):
            c, k = harness_run(cfg, p, n, which, methods, out, reps, workers)
            cmds.append(c)
            outs.append((out, k))
        # threshold of the trained networks re-chosen by total time on the tuning set (V100 forward)
        m = []
        for mdl in (a, b):
            m += [f"lsc:{mdl}@cuda", f"lsc:{mdl}@cuda|gprobe=1"] + [f"lsc:{mdl}@cuda|thr={t}" for t in cfg.grid(THR_GRID[p], 2)]
        add(50000, "tune", m, TL / "paper" / f"thr_def_{p}.json", 3, 8)
        m = []
        for mdl in (a, b):
            m += [f"lsc:{mdl}@cuda|{S}"] + [f"lsc:{mdl}@cuda|thr={t},{S}" for t in cfg.grid(THR_GRID_S[p], 2)]
        add(50000, "tune", m, TL / "paper" / f"thr_sym_{p}.json", 3, 8)
        # 50k head-to-head with the time-tuned threshold and GPU probes
        v2 = {"anisotropy": ["theta:0.1", "sym:0.02", "srowq:0.7", f"lsc:{b}@cuda", f"lsc:{b}@cuda|thr=-1.0",
                             f"lsc:{b}@cuda|thr=-1.0,gprobe=1", f"lsc:{a}@cuda|thr=-1.0,gprobe=1", f"lsc:{b}@cpu|thr=-1.0"],
              "lognormal": ["sym:0.02", "theta:0.02", "srowq:0.8", f"lsc:{b}@cuda", f"lsc:{b}@cuda|gprobe=1",
                            f"lsc:{b}@cuda|thr=-1.5,gprobe=1", f"lsc:{a}@cuda|gprobe=1", f"lsc:{b}@cpu"],
              "perforated_l": ["theta:0.2", "srowq:0.7", "topk:6", f"lsc:{b}@cuda", f"lsc:{b}@cuda|gprobe=1",
                               f"lsc:{a}@cuda|gprobe=1", f"lsc:{b}@cuda|thr=-0.75,gprobe=1", f"lsc:{b}@cpu"]}[p]
        add(50000, "test", v2, TL / "paper" / f"v2_50000_{p}.json")
        if p == "perforated_l":
            for n in (500000, 200000):
                if n in cfg.sizes:
                    add(n, "test", ["theta:0.1", "theta:0.2", "srowq:0.8", f"lsc:{a}@cuda", f"lsc:{a}@cuda|gprobe=1",
                                    f"lsc:{b}@cuda|gprobe=1"], TL / "paper" / f"gprobe_{n}_{p}.json")
        # cost breakdowns on test matrix 0
        fp = TL / "paper" / f"featprof_{p}.json"
        cmds.append(PY + ["twolevel.profiling", "features", "--problem", p, "--n", str(cfg.n(50000)), "--model", a,
                          "--out", rel(fp)])
        outs.append((fp, None))
        cl = {"anisotropy": "theta:0.1", "lognormal": "sym:0.02", "perforated_l": "theta:0.2"}[p]
        ps = TL / "paper" / f"profsetup_{p}.json"
        cmds.append(PY + ["twolevel.profiling", "setup", "--problem", p, "--n", str(cfg.n(50000)), "--methods",
                          *methods_for(cfg, [cl, f"{cl}|{S}", "topk:1", f"lsc:{a}@cuda"]), "--out", rel(ps)])
        outs.append((ps, None))
        jobs.append(Job(f"paper_{p}", "gpu", cmds, outs))
    return jobs


# ---------------------------------------------------------------------------------------------- cost model
def _sweeps(cfg, p):
    """(splu, n, seed, count, workers, file tag)"""
    s = []
    for n, seed, count, w, tag in ((5000, 0, 16, 16, "5k_s0"), (50000, 500, 8, 8, "50k_s500"),
                                   (50000, 1000, 16, 8, "50k_s1000"), (200000, 1000, 8, 8, "200k_s1000")):
        s.append(("default", n, seed, count, w, tag))
    for n, seed, count, w, tag in ((5000, 0, 16, 16, "5k_s0"), (50000, 500, 8, 8, "50k_s500"),
                                   (200000, 1000, 4, 8, "200k_s1000")):
        s.append(("sym", n, seed, count, w, tag))
    return s


def step_costmodel(cfg):
    jobs = []
    for p in cfg.problems:
        cmds, outs = [], []
        for splu, n, seed, count, w, tag in _sweeps(cfg, p):
            d = TL / "costmodel" / ("sweep" if splu == "default" else "sweep_sym")
            out = d / f"{p}_{tag}.jsonl"
            cmd = PY + ["twolevel.costmodel", "sweep", "--problem", p, "--n", str(cfg.n(n)), "--seed", str(seed),
                        "--count", str(2 if cfg.small else count), "--workers", str(cfg.w(w)), "--splu", splu,
                        "--out", rel(out)]
            if cfg.small:
                cmd[cmd.index("--workers"):cmd.index("--workers")] = ["--rules", "theta:0.1,theta:0.3,sym:0.02,topk:2"]
            cmds.append(cmd)
            outs.append((out, None))
        jobs.append(Job(f"costmodel_{p}", "cpu", cmds, outs))
    return jobs


def costmodel_tail(cfg, jobs):
    if any(not j.done() for j in jobs):
        return
    from twolevel.costmodel import fit
    for d, name in (("sweep", "cost_model.json"), ("sweep_sym", "cost_model_sym.json")):
        out = TL / "costmodel" / name
        if out.exists() and not cfg.a.force:
            continue
        res = fit(str(TL / "costmodel" / d), min_size=cfg.n(50000))
        out.write_text(json.dumps(res, indent=1))
        print(f"wrote {rel(out)}")


# ---------------------------------------------------------------------------------------------- train
def cost_model_path(cfg, sym):
    name = "cost_model_sym.json" if sym else "cost_model.json"
    if cfg.a.cost_model == "committed" and not cfg.small:
        return rel(COMMITTED / name)
    return rel(TL / "costmodel" / name)


def step_train(cfg):
    jobs = []
    for p in cfg.problems:
        cmds, outs = [], []
        for name, (pp, extra) in NODE_POLICIES.items():
            if pp != p:
                continue
            extra = list(extra)
            sym = "--splu" in extra
            out = TL / "policies" / "node" / name
            if cfg.small:   # tiny: few matrices, epochs, population
                for k, v in (("--n_train", "4"), ("--n_val", "2"), ("--mb", "2"), ("--epochs", "2")):
                    if k in extra:
                        extra[extra.index(k) + 1] = v
                    else:
                        extra += [k, v]
                if "--n" in extra:
                    extra[extra.index("--n") + 1] = str(cfg.n(50000))
                else:
                    extra += ["--n", str(cfg.n(5000))]
                extra += ["--pop", "4", "--target_n", str(cfg.n(50000))]
            cmds.append(PY + ["twolevel.train_node", "--problem", p, "--workers", str(cfg.w(46)),
                              "--cm", cost_model_path(cfg, sym), *extra, "--out", rel(out)])
            outs.append((out / "train.json", None))
        for name, (pp, extra) in LIN_POLICIES.items():
            if pp != p:
                continue
            extra = list(extra)
            if "--workers" not in extra:
                extra += ["--workers", "28"]
            extra[extra.index("--workers") + 1] = str(cfg.w(int(extra[extra.index("--workers") + 1])))
            if cfg.small:
                extra += ["--n_target", str(cfg.n(50000)), "--n", "2", "--gens", "2", "--pop", "4"]
            out = TL / "policies" / "lin" / f"{name}.json"
            cmds.append(PY + ["twolevel.train_lin", "--problem", p, *extra, "--out", rel(out)])
            outs.append((out, None))
        jobs.append(Job(f"train_{p}", "cpu", cmds, outs))
    return jobs


# ---------------------------------------------------------------------------------------------- select / final
def step_select(cfg):
    jobs = []
    for p in cfg.problems:
        cmds, outs = [], []
        for sol in ("def", "sym"):
            m = [resolve(cfg, x, S if sol == "sym" else "") for x in SELECT[(sol, p)]]
            out = TL / "select" / f"{sol}_50000_{p}.json"
            c, k = harness_run(cfg, p, 50000, "tune", m, out, 3, 2)
            cmds.append(c)
            outs.append((out, k))
        jobs.append(Job(f"select_{p}", "gpu", cmds, outs))
    return jobs


def step_final(cfg):
    jobs = []
    for p in cfg.problems:
        cmds, outs = [], []
        for n in cfg.sizes:
            for sol in ("def", "sym"):
                key = (sol, p, n)
                if key not in FINAL:
                    continue
                m = [resolve(cfg, x, S if sol == "sym" else "") for x in FINAL[key]]
                out = TL / "final" / f"{sol}_{n}_{p}.json"
                c, k = harness_run(cfg, p, n, "test", m, out, 3, 2)
                cmds.append(c)
                outs.append((out, k))
        jobs.append(Job(f"final_{p}", "gpu", cmds, outs))
    return jobs


# ---------------------------------------------------------------------------------------------- headroom
RETIME_GRID = (["topk:1", "topk:2", "topk:3", "topk:4"]
               + [f"theta:{t}" for t in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8)]
               + [f"sym:{t}" for t in (0.0, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.15)])


def oracle_params(cfg, p, base):
    if cfg.a.oracle == "committed" and not cfg.small:
        return COMMITTED / "headroom" / "oracle_params" / f"{p}_{base}.json"
    return TL / "headroom" / "oracle" / f"{p}_{base}_cma.json"


def step_headroom(cfg):
    jobs = []
    for p in cfg.problems:
        if cfg.a.oracle == "new" or cfg.small:
            cmds, outs = [], []
            for base, (w, pop) in ORACLE[p].items():
                pre = TL / "headroom" / "oracle" / f"{p}_{base}"
                cmd = PY + ["twolevel.oracle", "search", "--problem", p, "--n_target", str(cfg.n(50000)), "--seed",
                            str(TUNE_SEED), "--n", "1" if cfg.small else "4", "--workers", str(cfg.w(w)),
                            "--gens", "2" if cfg.small else "60", "--pop", "4" if cfg.small else str(pop),
                            "--reps", "1" if cfg.small else "5", "--base", base, "--out", rel(pre)]
                if cfg.small:
                    cmd += ["--ptypes", "geo4,lin"]
                cmds.append(cmd)
                outs.append((Path(str(pre) + "_cma.json"), None))
            jobs.append(Job(f"oracle_{p}", "cpu", cmds, outs))
        # re-time: masks rebuilt from the oracle parameters, the classical grid and the lin model (V100, W=2)
        cmds, outs, masks = [], [], []
        for base in ORACLE[p]:
            md = TL / "headroom" / f"masks_{base}"
            cmds.append(PY + ["twolevel.oracle", "export", "--cma", rel(oracle_params(cfg, p, base)), "--masks", rel(md)])
            masks.append(f"mask:{rel(md)}/{p}")
        m = cfg.grid(RETIME_GRID, 4) + masks + [resolve(cfg, "lin:" + LIN_OF[p])]
        out = TL / "headroom" / f"retime_{p}.json"
        c, k = harness_run(cfg, p, 50000, "tune", m, out, 3, 2, count=4)
        if cfg.small:
            c[c.index("--count") + 1] = "1"
            k = len(methods_for(cfg, m))
        cmds.append(c)
        outs.append((out, k))
        jobs.append(Job(f"retime_{p}", "gpu", cmds, outs))
    return jobs


# ---------------------------------------------------------------------------------------------- report
def step_report(cfg):
    from twolevel import tables
    md = ["# Two-level comparison: regenerated tables\n", f"outputs: {rel(TL)}\n"]

    def files(d, pat):
        return sorted(str(f) for f in (TL / d).glob(pat))
    sections = [("Bar: tuned classical vs paper GAT models, default splu", "bar", "def_*.json", tables.table),
                ("Bar: improved splu (both methods)", "bar", "sym_*.json", tables.table),
                ("Paper models: time-tuned threshold / GPU probes", "paper", "v2_*.json", tables.table),
                ("Paper models: perforated-L 200k/500k with GPU probes", "paper", "gprobe_*.json", tables.table)]
    for title, d, pat, fn in sections:
        fs = files(d, pat)
        if fs:
            md += [f"\n## {title}\n", fn(fs)]
    fs = files("final", "*.json")
    if fs:
        md.append("\n## Final paired comparisons (vs the best classical rule in the same run)\n")
        for f in fs:
            ref, out = tables.vs([f])
            md += [f"\n**{Path(f).stem}**\n", tables.vs_markdown(ref, out)]
    for d in ("select",):
        for f in files(d, "*.json"):
            ref, out = tables.vs([f])
            md += [f"\n**selection {Path(f).stem}**\n", tables.vs_markdown(ref, out)]
    fs = files("headroom", "retime_*.json")
    if fs:
        md += ["\n## Headroom (oracle masks re-timed on the tuning matrices)\n", "```", json.dumps(tables.headroom(fs), indent=1), "```"]
    if (TL / "tune" / "winners.json").exists():
        md += ["\n## Tuned classical winners\n", "```", (TL / "tune" / "winners.json").read_text(), "```"]
    fs = files("tune", "def_50000_*.json")
    if fs:
        md += ["\n## Coarse LU cost fit (default splu, tuning rows)\n", "```", json.dumps(tables.lufit(fs), indent=1), "```"]
    out = TL / "report.md"
    TL.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n")
    print(f"wrote {rel(out)}")


# ----------------------------------------------------------------------------------------------------
STEPS = ["data", "tune", "bar", "paper", "costmodel", "train", "select", "final", "headroom", "report"]


def run_step(name, cfg, args):
    if name == "report":
        step_report(cfg)
        return
    jobs = {"data": step_data, "tune": step_tune, "bar": step_bar, "paper": step_paper, "costmodel": step_costmodel,
            "train": step_train, "select": step_select, "final": step_final, "headroom": step_headroom}[name](cfg)
    submit(args, jobs)
    if args.dry_run:
        return
    if name == "tune":
        tune_tail(cfg)
    if name == "costmodel":
        costmodel_tail(cfg, jobs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=STEPS + ["all"])
    ap.add_argument("--problems", nargs="*", choices=PROBLEMS)
    ap.add_argument("--sizes", nargs="*", type=int, choices=SIZES)
    ap.add_argument("--models", help="directory with <problem>_<label>/seed<k>/model_best.pt (paper GAT models)")
    ap.add_argument("--policies", choices=["committed", "new"], default="committed")
    ap.add_argument("--cost-model", choices=["committed", "new"], default="committed")
    ap.add_argument("--oracle", choices=["committed", "new"], default="committed")
    ap.add_argument("--sbatch", default="", help="extra sbatch options for every job (account, ...)")
    ap.add_argument("--sbatch-gpu", default="", help="extra sbatch options for timing jobs (partition, constraint)")
    ap.add_argument("--sbatch-cpu", default="", help="extra sbatch options for compute jobs")
    ap.add_argument("--local", action="store_true", help="run here instead of submitting to SLURM")
    ap.add_argument("--small", action="store_true", help="tiny sizes and searches (tests)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="redo work that already exists")
    args = ap.parse_args()
    cfg = Cfg(args)
    if args.small:
        args.policies = args.cost_model = args.oracle = "new"
    if args.step == "all" and not (args.local or args.dry_run):
        sys.exit("`all` needs --local (on SLURM every step must wait for the previous step's jobs)")
    for name in (STEPS if args.step == "all" else [args.step]):
        run_step(name, cfg, args)


if __name__ == "__main__":
    main()
