# Two-level learned vs. time-tuned classical strength graphs (October 2026)

This is a follow-up to the thesis and manuscript work. It asks whether a learned strength-of-connection
graph can make **two-level** smoothed-aggregation AMG faster in *total* time than a **two-level classical
rule tuned for total time**.

LSC-AMG is always two-level: `strength=("predefined", {"C": C})` gives PyAMG a two-level hierarchy whose
coarse grid is solved by `splu`. A two-level classical rule tuned for time is therefore the fair baseline.

The study has three parts, referred to in the code and in method names:
- **"bar"** sets the baseline: tuned classical rules and the GAT models.
- **"coarse"** studies the `lin` per-row threshold model, the hand rule and the oracle headroom.
- **"reward"** covers the node-modulated policies trained on a fitted cost model.

`twolevel/` and `scripts/twolevel.py` regenerate the numbers. This directory holds a compact subset of
the results: the files behind the main README and the policies and cost models that the code needs.

## Findings

**"Learned dominates" (lower mean total time at 50k on all three problems): not met.**

### GAT policies vs. the best tuned classical rule

Files: `bar/def_50000_*.json`. All rows of a file come from one job on a Xeon Gold 6248 + V100 node,
with 16 test matrices. The GAT is the best seed of an October 2026 retrain in the manuscript
configuration (λ = 0.1, threshold noise 50) on corrected code.

| problem (50k) | best classical | GAT, V100 forward | GAT, 1-thread CPU forward | GAT 5k-trained, V100 |
|---|---|---|---|---|
| anisotropy | `theta:0.1` 0.535 s (41.6 it, coarse n 8.6k, LU 5.4 × nnz(A)) | 0.779 s, +46.8%, 0/16 (27.9 it, coarse n 13.9k, LU 12.5×) | 1.117 s, +110.9% | 0.772 s |
| lognormal | `sym:0.02` 0.376 s (28.4 it) | 0.449 s, +19.5%, 0/16 (32.4 it) | 0.805 s | 0.451 s |
| perforated-L | `theta:0.2` 0.315 s (28.6 it) | 0.338 s, +7.6%, 0/16 (23.3 it) | 0.683 s | 0.346 s |

Learned total = features + forward + C + SA setup + coarse LU + PCG. Classical total = C + SA setup +
coarse LU + PCG. Each worker runs one pinned thread. Percentages are paired per-matrix means of
total/total_ref − 1.

**Why the GAT loses.**
- Every classical family's time optimum keeps 63–94% of the edges. That gives a coarse n of about n/6 to
  n/8 and a coarse LU of 2–5 × nnz(A).
- The training reward counts iterations and operator complexity, but not the coarse-LU fill. It pushes the
  GAT towards keeping fewer strong edges.
- On anisotropy this gives a coarse n of about n/3.6 and an LU of 12.5 × nnz(A). The factorisation alone
  takes 0.31 s, and each V-cycle costs about 50% more.
- On lognormal and perforated-L the GAT is near the classical coarse size:
  - lognormal needs more iterations than `sym:0.02` (32.4 vs 28.4);
  - perforated-L needs fewer (23.3 vs 28.6), but 0.06 s of features + forward + C cancels the gain.
- A 1-thread CPU forward of the 52.8k-parameter GAT costs 0.35–0.39 s at 50k. That is more than the whole
  classical solve.

### Small learned policies vs. the tuned classical reference of the same job

Files: `final/*`. 16 test matrices at 50k, 8 at 200k and 500k. 95% CI in brackets; "faster on" counts
the matrices where the learned policy wins.

| problem | n | reference | node policy (11 params, ES on cost model) | lin policy (8 params, CMA on measured time) | verdict |
|---|---|---|---|---|---|
| perforated-L | 50k | `theta:0.2` 0.338 s | noderow 0.290 s, **−14.0%** [−15.4, −12.7], 16/16 | 0.277 s, **−13.2%**, 16/16 | learned wins |
| perforated-L | 200k | `theta:0.2` 1.715 s | **−19.5%**, 8/8 | (not bundled) | wins |
| perforated-L | 500k | `theta:0.2` 5.566 s | **−22.8%**, 8/8 | (not bundled) | wins |
| anisotropy | 50k | `theta:0.1` 0.578 s | noderow +1.3% [−0.1, +2.8], 5/16 | +2.2%, 4/16 | tie / no gain |
| lognormal | 50k | `sym:0.02` 0.406 s | nodeabs +2.4% [−0.4, +5.2], 4/16; cost50k variant +4.0% [+1.7, +6.3] | +4.3%, 2/16 | loses |

- The node files are `final/reward_def_*.json`. The lin and hand-rule files are `final/coarse_50000_*.json`;
  their reference times come from a separate job (`theta:0.2` 0.320 s, `theta:0.1` 0.542 s,
  `sym:0.02` 0.378 s).
- The policy learned a simple rule. It keeps every coupling at coefficient interfaces and in the
  anisotropic corner (θ ≈ 1) and coarsens aggressively in the isotropic bulk (θ ≈ 0.17).
- The hand rule distilled from that, `two:1,0.17,1,0.05`, gets −8.9% at 50k.

**Other results of the study** (data not in this compact release; regenerate with `scripts/twolevel.py`):
- A per-matrix oracle search (`headroom`) bounds what any strength graph could gain over the tuned single
  rule: anisotropy 3.6%, lognormal 9.9%, perforated-L 21.7%. On anisotropy the headroom is smaller than
  the cost of any learned features, which explains why nothing learned wins there.
- A tuned `splu` setting (`permc_spec="MMD_AT_PLUS_A"`, no pivoting, `SymmetricMode`; method suffix
  `|perm=MMD_AT_PLUS_A,dpt=0,sym=1`) speeds up both methods by 10–25%. It leaves the conclusions
  unchanged. The 50k lin files (`final/coarse_50000_*`) include these rows.

## Files in this directory

All paths are relative to this directory. Tables are printed with `python -m twolevel.tables <cmd> <files>`
from the repository root, with only NumPy needed. "Regenerate" commands write to `outputs/twolevel/`.

| numbers | file(s) | table command | regenerate |
|---|---|---|---|
| bar 50k: tuned classical + GAT (CPU/V100, fast and reference forward) | `bar/def_50000_{anisotropy,lognormal,perforated_l}.json` | `table`, `vsbest` | `twolevel.py bar --sizes 50000 --models DIR` (needs GAT checkpoints, see [docs/REPRODUCE.md](../../docs/REPRODUCE.md)) |
| node-policy finals, 50k (all problems) and perforated-L 200k / 500k | `final/reward_def_*.json` | `vsbest` | `twolevel.py final` (`final/def_<n>_<problem>.json`) |
| lin and hand-rule finals, 50k | `final/coarse_50000_*.json` | `vsref theta:0.2 F` and similar | `twolevel.py final` |
| node policies (18) and lin policies (3) | `policies/node/*.json`, `policies/lin/*.json` | | `twolevel.py train` |
| cost models (reward of the node policies) | `cost_model.json`, `cost_model_sym.json` | | `twolevel.py costmodel` |

## Protocol

- **Problems and sizes.** `anisotropy`, `lognormal` and `perforated_l` from `problems.py`, at true sizes
  n ≈ 50k / 200k / 500k. Perforated-L uses n_grid = n / 1.4745.
- **Seeds.**

  | role | seed | matrices |
  |---|---|---|
  | test | 1000 | 16 at 50k, 8 at 200k / 500k |
  | tuning and selection | 2000 | 8 / 4 / 4 |
  | node-policy training / validation | 0 / 1 | |
  | cost-model calibration | 0 (5k), 500 (50k), 1000 (50k, 200k) | |

- **Solver.** Both methods are two-level:
  `pyamg.smoothed_aggregation_solver(A, strength=("predefined", {"C": C}), smooth="jacobi", coarse_solver="splu")`
  with default SA smoothers, the `splu` factorisation forced and timed inside setup, and PCG (b = ones,
  x0 = 0, rtol 1e-8, maxiter 500) with one V-cycle per iteration.
- **Classical rules** are tuned by mean total time on the tuning set.
- **Timing.**
  - Whole V100 nodes, exclusive, with one pinned thread per worker.
  - W = 2 workers (one per socket) for every test-set table; W = 8 for tuning, which is used only for
    ranking.
  - Median of 3 repeats per (matrix, method), with one untimed warm-up per method.
  - Paired comparisons are always within one job.

## Notes and caveats

- **Hardware.** The node-policy finals at 50k (`final/reward_def_50000_*`) ran on a Xeon Gold 6154 + V100
  node, not a Gold 6248 one. All rows of a comparison are in the same job, so the paired percentages are
  valid, but the absolute times are about 8% higher than the bar's. On the Gold 6248 bar, the same
  `theta:0.2` reference took 0.315 s.
- **Choice of reference.** The "best time-tuned classical" reference at 50k is the best of the four
  per-family tuning winners *as measured on the test set*. For example, anisotropy uses θ = 0.1 although
  `sym:0.02` won the tuning set by 3%. This favours classical slightly.
- **Cost-model calibration.** The cost-model sweeps include the 50k and 200k test matrices (seed 1000), so
  timing coefficients of classical rules on test matrices informed the node policies' reward. The
  committed `cost_model.json` was fitted while two 200k sweeps were still running. It uses the first 124
  of 208 rows of `lognormal_200k_s1000` and none of `perforated_l_200k_s1000`, which `--limit` in
  `twolevel.costmodel fit` reproduces. A fresh `twolevel.py costmodel` fits all rows.
- **Screening.** The "reward" part's early policy screening used test seed 1000, before switching to
  selection on seed 2000.
- **Determinism.** Node-policy training with a cost reward is deterministic, because it depends only on
  iteration counts and LU sizes. `twolevel.py train` with the committed cost models reproduced all six
  perforated-L node policies bit-exactly. The lin policies are trained on measured time, so a rerun gives
  a statistically similar model, not an identical one.
- **Rerun through the driver.** Running the 50k finals through `twolevel.py final` (one Gold 6248 + V100
  node per problem) reproduced the iterations, coarse sizes and coarse LU nnz exactly for every
  (matrix, method) pair. The paired results agreed:

  | problem | node policy | lin policy |
  |---|---|---|
  | perforated-L | −14.7% [−16.2, −13.3] | −13.3% |
  | anisotropy | +0.9% [−0.5, +2.4] | +1.3% |
  | lognormal | +2.2% [−0.6, +5.1] | +4.1% |

- **Identity tests.** `tests/twolevel_reference.json` (used by `tests/test_twolevel.py`) holds fingerprints
  computed with the original prototype scripts of the three parts: the sha1 of C, iterations, coarse size
  and coarse LU nnz for 90 settings at n ≈ 3k. The consolidated harness reproduces all 90 on the reference
  Linux x86_64 platform. On other CPUs the regenerated matrices can differ in the last bits.
