# Reproducing LSC-AMG

There are three levels, from seconds to cluster-weeks. Run every command from the repository root.

```bash
git clone https://github.com/imeyer2/lsc-amg.git
cd lsc-amg
```

## What "reproduce" means here

| kind | what you get | how |
|---|---|---|
| **Replay stored results** | The exact tables behind the README, recomputed from the per-matrix JSONs. | Level A |
| **Rerun on corrected inputs** | New matrices from the seeded generators, at the corrected sizes and with the corrected features, plus new training and timing. On the reference platform (Linux x86_64, pinned packages) the matrices, iteration counts and coarse sizes are identical. On other machines they are statistically equivalent: the last bits differ, and some perforated-L meshes have a few nodes more or fewer. Seconds always depend on the hardware. | Levels B and C |
| **Reproduce the original random draws** | Not possible. The manuscript's April runs used unseeded data generation and an unseeded AMG setup, so their exact matrices cannot be regenerated. Seeded reruns reproduce their iteration counts within seed noise (`results/fixed_eval_2026-10/comparison_fixed.md`). | — |

## Level A: stored results (NumPy only)

```bash
python -m twolevel.tables vsbest results/twolevel_2026-10/bar/def_50000_anisotropy.json
python -m twolevel.tables vsbest results/twolevel_2026-10/final/reward_def_50000_perforated_l.json
```

These need only Python 3 and NumPy, with no training, CUDA or SLURM. Other views of the same files:
- `python -m twolevel.tables table F.json` gives the per-phase columns (features, forward, C, SA, LU, solve).
- `python -m twolevel.tables vsref REF F.json` gives paired comparisons against a chosen reference.

[`results/twolevel_2026-10/README.md`](../results/twolevel_2026-10/README.md) lists the files behind
each number. `python scripts/reproduce.py compare` (level C) puts the manuscript's numbers in
`results/paper_original/` next to a rerun.

## Level B: CPU install and smoke test

Python 3.11 or newer is required (NumPy 2.4 and SciPy 1.17).
[`requirements-cpu.txt`](../requirements-cpu.txt) is a subset of `requirements.txt` with the same pins:
CUDA, triton, `nvidia-*` and tensorboard are left out.

```bash
uv venv -p 3.11 .venv
# macOS arm64 (the PyPI torch wheel is CPU-only already):
uv pip install -p .venv/bin/python -r requirements-cpu.txt
# Linux x86_64 (CPU-only torch wheels, avoids about 3 GB of CUDA packages):
uv pip install -p .venv/bin/python --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match -r requirements-cpu.txt
# pip alternative: python3.11 -m venv .venv && .venv/bin/pip install \
#     --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-cpu.txt

bash tests/smoke.sh
```

`tests/smoke.sh` activates `.venv` and recreates `outputs_smoke/`. It then runs these steps:
1. `tests/test_problems.py`: dataset hashes. It warns off the reference platform.
2. `tests/test_fast_policy.py`: `FastPolicy` against the reference forward.
3. `tests/test_twolevel.py --identity-only`: fingerprints of the two-level harness. It warns off the
   reference platform.
4. `tests/test_twolevel.py --pipeline-only`: the whole two-level pipeline at n ≈ 3k.
5. `scripts/reproduce.py` `train` (5 seeds × 2 epochs of perforated-L, n ≈ 1.5k), then `select`,
   `benchmark`, `scale`, `compare` and `figures`.

It ends with `SMOKE TEST PASSED: best seed <k> R <reward>`.

Observed results:

| platform | Python | result | wall time |
|---|---|---|---|
| macOS 26 arm64 (Apple M4, 10 cores) | 3.11.15 | passed, with 2 hash warnings | about 185 s |
| Linux x86_64, `python:3.11-slim` container (emulated on Apple silicon, 4 vCPU), `--shm-size=2g` | 3.11.18 | passed, with 2 hash warnings | about 227 s |

Notes:
- **Hash checks.** These are `tests/test_problems.py` and the identity part of `tests/test_twolevel.py`.
  They compare SHA-1/SHA-256 hashes recorded on the reference Linux x86_64 cluster. On other CPUs the
  float last bits differ, and the perforated-L Triangle mesh can differ by a few nodes. So off the
  reference platform they print `WARNING ... differs from the reference-platform hashes`, and
  `SMOKE_STRICT=1 bash tests/smoke.sh` makes them fatal. On the Linux x86_64 container, all method
  fingerprints were identical and only the lognormal and perforated-L matrix hashes differed.
- **Docker.** Training's shared-memory dataset cache needs more than Docker's default 64 MB `/dev/shm`.
  Run containers with `--shm-size=2g` (or `--ipc=host`). Otherwise training dies with SIGBUS.
- **Linux aarch64.** pyamg 5.3.0 and meshpy 2025.1.1 publish no wheels for this platform, so they build
  from source and need a C++ compiler. This setup was not tested.
- **pytest.** The files in `tests/` are scripts, not pytest test functions, so run them as
  `python tests/<file>.py`.

## Level C: the full study

**Environment.** [`requirements.txt`](../requirements.txt) has the historical pins of the original runs:
Linux x86_64, PyTorch 2.10 with CUDA 12.8 wheels, PyG 2.7, PyAMG 5.3, SciPy 1.17, NumPy 2.4,
scikit-fem 12.0.1 and meshpy 2025.1.1.

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

A GPU is optional. Benchmarks used V100s, and the CPU path gives the same results up to float ties
(METHODS §4).

### Original benchmark: `scripts/reproduce.py`

| step | command | jobs | what it does |
|---|---|---|---|
| train | `reproduce.py train` | 75 | 15 configurations × seeds 0–4 (`configs/experiments.csv`) |
| select | `reproduce.py select` | – | best seed per configuration → `outputs/best_runs.json` |
| benchmark | `reproduce.py benchmark` | 15 | 6 preconditioners on the 32 test matrices of each chosen run |
| scale | `reproduce.py scale` | 12 | each 50k model on 100k–500k unknowns (`configs/scalability.csv`) |
| compare | `reproduce.py compare` | – | side by side with `results/paper_original` → `outputs/comparison.md` |
| figures | `reproduce.py figures` | – | figures and LaTeX tables (`thesis_chapter_analysis.py`) → `outputs/figures/` |
| status | `reproduce.py status` | – | progress of all of the above |

On a SLURM cluster, pass your site's options. The account, partition and constraint values below are
placeholders:

```bash
S="--account=ACCOUNT --partition=PARTITION"
python scripts/reproduce.py train --sbatch "$S"
python scripts/reproduce.py status                     # until every config shows 5/5
python scripts/reproduce.py select
python scripts/reproduce.py benchmark --sbatch "$S --gres=gpu:1"
python scripts/reproduce.py scale --sbatch "$S --gres=gpu:1"
python scripts/reproduce.py compare
python scripts/reproduce.py figures
```

Without SLURM, add `--local` to run each command in sequence on this machine. That is how the smoke test
works; the full study would take weeks on one node.

Other options:
- `--dry-run` prints the commands without running them.
- `--only lognormal_50k anisotropy_30k` and `--seeds 0 3` restrict `train`.
- `--force` redoes finished work.
- `--experiments FILE` and `--scalability FILE` select other configuration tables.
- `LSC_OUTPUTS=dir` writes somewhere other than `outputs/`.

Every step is idempotent. Training checkpoints every 25 epochs and on SIGTERM, which
`scripts/slurm/run.sh` forwards before a time limit or preemption. After a failure, run the same step
again.

**Settings to know before comparing with the manuscript.**
- `reproduce.py` passes the *current* defaults (`--sigma_scale_threshold 2.5 --reward_lambda_cx 2.0`).
  For the manuscript's reward weight and threshold noise, append them with `--train-args`, which comes
  after the defaults and therefore overrides them:
  ```bash
  python scripts/reproduce.py train --sbatch "$S" --train-args "--sigma_scale_threshold 50 --reward_lambda_cx 0.1"
  ```
- `configs/experiments.csv` uses **half the manuscript's mini-batch** (METHODS §3, item 7). The
  `--train-args` above therefore give the manuscript's λ and threshold noise with half the mini-batch.
  That is the configuration of the GAT in the README headline. For the full mini-batch, edit the
  `mini_batch` column.
- The near-exact **PCG stagnation stop** (`stag`) for population members is on by default. To disable it,
  pass `LSC_SPEEDUPS=pcg,sched,soc,diag,overlap,nopickle,preload` (METHODS §4).
- The 30k and 50k configurations use 3 nodes. `--single-node` turns that off.

**Cost.** Average wall time per seed, CPU forwards, 32–48 cores per run:

| size | lognormal | anisotropy | perforated-L |
|---|---|---|---|
| 5k | 2.8 h | 1.2 h | 1.3 h |
| 10k | 6.0 h | 1.3 h (100 epochs) | 2.2 h |
| 20k | 12.3 h | 7.0 h | 4.4 h |
| 30k | 24.8 h (450 epochs) | 14.7 h | 7.2 h |
| 50k | 27.1 h | 14.3 h | 11.3 h |

That is about 25,000 core-hours for all 75 runs. Memory peaks at about 175 GB at 50k with 46 workers
(about 3.7 GB per worker).

### Two-level study: `scripts/twolevel.py`

The steps are `data`, `tune`, `bar`, `paper`, `costmodel`, `train`, `select`, `final`, `headroom` and
`report`, with the same `--local`, `--dry-run` and `--sbatch*` conventions. Timing jobs need whole nodes of
one node type. `python tests/test_twolevel.py --pipeline-only` runs every step at n ≈ 3k
(`scripts/twolevel.py all --local --small`).

The four steps in the first table run from what this release includes. The second table lists what they
need that is not bundled.

| step | runs from |
|---|---|
| `final` | the committed small policies (`results/twolevel_2026-10/policies/`) |
| `train` | the committed cost models (cost-reward training is deterministic, so it reproduces the committed node policies) |
| `select` | the committed small policies |
| `costmodel` | none (regenerates its own sweeps) |

| step | needs | how to supply it |
|---|---|---|
| `bar`, `paper` | the GAT checkpoints | `scripts/reproduce.py train` / `select` with the manuscript `--train-args` above, then `--models outputs/train` |
| `headroom` | the oracle parameters | `--oracle new` reruns the oracle search |

## Outputs

```
outputs/
├── train/<problem>_<label>/seed<k>/   config.json, training_history.json, model_best.pt,
│                                     checkpoint_resume.pt, diagnostics_*.json, preconditioner_benchmark/
├── scalability_<label>/              lognormal zero-shot runs
├── scalability_<problem>_<label>/    anisotropy / perforated-L zero-shot runs
├── best_runs.json, comparison.md, figures/, logs/, submitted.tsv, wall_times.tsv
└── twolevel/                         scripts/twolevel.py (same file names as results/twolevel_2026-10/)
```

**Original benchmark files.**
- `preconditioner_benchmark/benchmark_results.json` has one entry per test matrix (`matrix_size`,
  `matrix_nnz`).
- Each entry has per-scheme metrics for `none`, `jacobi`, `ilu`, `pyamg`, `ruge_stuben` and `learned`:
  - `iters`: PCG iterations to 1e-8, capped at 500;
  - `setup_time` and `solve_time`;
  - `kappa_raw` and `kappa_eff`;
  - `op_complexity` and `grid_complexity`;
  - `rho_vcycle`: the true V-cycle factor, AMG schemes only.

**Two-level files.** Each JSON has a `meta` block (CPU and GPU model, workers, problem, size, seed, matrix
count, repeats, methods). Its `rows` hold one entry per (matrix, method), with the iterations, coarse size,
LU fill and phase times.

Files written by `twolevel/harness.py` and `train_es_HPC.py` record the host name and SLURM job id. The
stored files in `results/` have had those fields removed.

## Troubleshooting

| symptom | cause and fix |
|---|---|
| `tests/test_problems.py` MISMATCH | Different CPU or BLAS from the reference platform. Expected off Linux x86_64. Check package versions first. |
| SIGBUS while "Creating shared-memory cache" | Docker's 64 MB `/dev/shm`. Use `--shm-size=2g`. |
| Job state `OUT_OF_MEMORY` | About 3.7 GB per worker at 50k. Give more memory or fewer cores (workers = cpus − 2). |
| `CUDA error: no kernel image is available` | GPU too old for the CUDA 12.8 wheels. The code falls back to the CPU. |
| `PREEMPTED` / `TIMEOUT` | Rerun the same step. Training resumes from its checkpoint. |
| Benchmark times differ from the tables | GPU type, CPU type and load all change timings. Compare iteration counts first, and timings only on the same hardware. |

## Extending

- **New PDE.** Add `build_<name>_sample(rng, n_edge, n_grid)` returning `(A_bc, coords, triang)` to
  `problems.py`, register it in `make_spectral_train_set`, add it to the `--problem` choices in
  `train_es_HPC.py` and `evaluate_preconditioners.py`, and add rows to `configs/`.
- **New baseline.** Add a branch to `_build_preconditioner` in `evaluate_preconditioners.py`.
- **New features.** Add a `FeatureConfig` preset in `amg_core.py`.
