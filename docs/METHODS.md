# Methods, provenance and differences from the manuscript

This page has the method details behind the [README](../README.md): how the policy is trained, what the
stored results are, how they differ from the manuscript text, and the determinism and timing conventions.
[REPRODUCE.md](REPRODUCE.md) explains how to rerun the code.

## 1. Method

**Goal.** Precondition CG on large sparse SPD systems `A x = b` from finite-element discretisations of
elliptic PDEs. The preconditioner is a smoothed-aggregation AMG (SA-AMG) V-cycle whose *strength of
connection* is learned rather than hand-tuned.

**Pipeline for one matrix `A`.**

1. **Graph and features** (`amg_core.build_sample_features`). The sparsity graph of `A` is the input graph.
   The `new_algebraic_plus_relaxed` preset gives each node 9 features: 4 smooth "relaxed probe" vectors
   (random vectors Jacobi-relaxed on `Ax = 0`) and 5 row statistics of `A`. Each edge gets 12 features: the
   sign, row-relative strength and symmetrically normalised magnitude of `a_ij`, and differences of the
   relaxed probes across the edge.
2. **Policy** (`models.AMGEdgePolicy`, 52,802 parameters).
   - Two GAT layers (2 heads, then 1 head, hidden size 64) with LayerNorm produce node embeddings.
   - An edge MLP scores each edge from the two endpoint embeddings and the raw edge features.
   - A learned scalar threshold is part of the model.
3. **Strength graph** (`amg_core.select_edges_by_threshold`).
   - The logits are z-scored over the off-diagonal edges.
   - An edge is strong if its z-score exceeds the learned threshold, so the number of strong connections
     per row adapts.
   - The selected edges, symmetrised and with a unit diagonal, form the binary matrix `C`.
4. **AMG hierarchy** (`amg_core.build_pyamg_solver`).
   - It calls PyAMG `smoothed_aggregation_solver(A, strength=('predefined', {'C': C}))` with Jacobi
     prolongation smoothing and a `splu` coarse solver.
   - With a predefined strength graph, PyAMG builds a **two-level** hierarchy.
5. Reward: −log(PCG iterations) − λ·log(operator complexity) (train_es_HPC.py::amg_reward; manuscript runs used λ = 0.1, --sigma_scale_threshold 50)
   `--reward residual --reward_k K` instead trains on the K-step PCG residual log-ratio against a default SA-AMG baseline, minus λ·log(operator complexity).

**Training** (`train_es_HPC.py`). The map from parameters to `C`, to the hierarchy and to the reward is not
differentiable, so the policy is trained with OpenAI-style evolution strategies:
- antithetic Gaussian perturbations (population 150 or 200) with rank-based fitness shaping;
- per-parameter-group perturbation scales: σ = 0.02, multiplied by `--sigma_scale_threshold` for the
  threshold;
- σ decays by 0.999 per epoch down to 0.005, with a reset after 30 epochs without improvement;
- the ES gradient goes to Adam with weight decay (lr 0.01, wd 0.01).

Each epoch evaluates every perturbation on a random mini-batch of the 64 training matrices. The model kept
(`model_best.pt`) has the best reward on a fixed 10-matrix evaluation subset. 32 separate test matrices are
used only for reporting. Each configuration is trained with seeds 0–4, and the seed with the highest
`best_reward_so_far` is reported.

**Problems** (`problems.py`): P1 finite elements on random triangular meshes, assembled with scikit-fem,
with the Dirichlet boundary condensed.

| name | PDE | coefficient |
|---|---|---|
| `anisotropy` | `-div(K grad u) = f`, unit square | `K = R(a)^T diag(1, eps) R(a)`, `eps ~ logU[1e-3, 1e-1]`, `a ~ U[0, pi)` |
| `lognormal` | `-div(a grad u) = f`, unit square | `a = exp(8 Z)`, `Z` Gaussian field, correlation length 0.1 |
| `perforated_l` | `-div(c K grad u) = f`, L-shaped domain with 3–8 circular holes | high-contrast rings `c ∈ [10, 1e5]` around the holes, radial anisotropy (`eps = 0.01`) near the re-entrant corner |
| `poisson` | `-lap u = f` | not used in the study; kept for tests |

The perforated-L meshes come from Triangle through `meshpy` ([`perforated_l_mesh.py`](../perforated_l_mesh.py)).
Triangle has its own non-free licence; it is a dependency here and is not redistributed.

**Baselines in the original benchmark** (`evaluate_preconditioners.py`): no preconditioner, Jacobi, ILU(0)
(`scipy spilu`), SA-AMG (PyAMG defaults, multilevel) and Ruge–Stüben AMG (`theta = 0.25`), all used as CG
preconditioners on the same matrices.

## 2. What the stored results are

| directory | contents | provenance |
|---|---|---|
| [`results/paper_original/`](../results/paper_original) | `config.json` and `benchmark_results.json` for the 15 manuscript configurations and the 12 zero-shot runs, plus the three 5k checkpoints used by the tests | the manuscript runs (April 2026). Run metadata is reduced to the run settings. |
| [`results/fixed_eval_2026-10/`](../results/fixed_eval_2026-10) | `comparison_fixed.md/json`: paper numbers, original models re-measured with the fixed evaluator, and a first October 2026 rerun | October 2026. The rerun's run directories are not part of this release. |
| [`results/twolevel_2026-10/`](../results/twolevel_2026-10) | two-level comparison: per-matrix JSONs, the small learned policies, the cost models | October 2026 follow-up. See its [README](../results/twolevel_2026-10/README.md). |

**The GAT in the README headline** is `lsc:anisotropy_50k/seed4` (lognormal: seed 4; perforated-L: seed 3).
It is the best seed of an October 2026 retrain in the manuscript configuration (λ = 0.1, threshold noise
50) on the corrected code of this repository. That retrain differs from the manuscript runs in four ways:
- half the mini-batch (item 7 in §3);
- the corrected s_ij feature (item 11);
- the PCG stagnation stop (§4);
- 3-node training for 30k and 50k.

Its checkpoints are not included. The stored JSONs and table commands do not need them. Regenerating the
GAT rows of the two-level comparison requires training them with `scripts/reproduce.py` (see
[REPRODUCE.md](REPRODUCE.md)).

## 3. Differences from the manuscript

1. **Matrix sizes.** The perforated-L mesh has about 1.47 × `grid_n` unknowns. The manuscript's
   perforated-L "20k/30k/50k" rows were therefore really about 29.5k/44.7k/73.8k unknowns, and its
   zero-shot "100k–500k" were 149k–747k. Its lognormal "5k" row came from older code with `grid_n = 32`
   and n ≈ 250. `configs/` uses `grid_n` values that give the stated sizes.
2. **Hyperparameters.** The text says population 200 and mini-batch 16 for all problems. The anisotropy
   and perforated-L runs actually used 150 / 8 (except anisotropy 30k). Anisotropy 10k ran 100 epochs and
   lognormal 30k ran 450. `configs/experiments.csv` follows the runs (but see item 7).
3. **ρ_V.** The manuscript's "V-cycle convergence factor ρ(I − BA)" figures and values (learned 0.18–0.44,
   "below 0.5") are a PCG residual tail ratio. The true factor (`rho_vcycle`, §4) is 0.58–0.86 for the
   learned hierarchy and 0.81–0.94 for SA-AMG. The learned hierarchy is still consistently better, but not
   below 0.5. See `results/fixed_eval_2026-10/comparison_fixed.md`.
4. **Learned-AMG timing.** The manuscript's learned setup and TTS means include a one-off CUDA
   initialisation on the first test matrix (0.3–1.2 s). The benchmark now warms up first.
5. **Training time.** The manuscript's "9 minutes to 2.5 hours" does not match the cluster accounting of
   the original runs: 19–34 h at 50k on 32 CPUs.
6. **Repeatability.** The original runs used unseeded data generation and an unseeded AMG setup, so they
   cannot be repeated exactly (§4).
7. **Mini-batch halved.** `configs/experiments.csv` uses half the manuscript's mini-batch, with the
   population unchanged:
   - lognormal 16 → 8;
   - anisotropy and perforated-L 8 → 4;
   - anisotropy 30k 16 → 8.

   This halves training cost. In paired tests the final models were within seed noise of the full
   mini-batch:

   | test | best reward, halved | best reward, full | p (every metric) |
   |---|---|---|---|
   | lognormal 5k, 5 seeds | −3.177 ± 0.132 | −3.162 ± 0.082 | 0.76–0.94 |
   | anisotropy 5k, 5 seeds | −2.988 ± 0.040 | −2.982 ± 0.025 | 0.57–0.93 |
   | lognormal 20k, 3 seeds | −3.476 ± 0.050 | −3.473 ± 0.112 | 0.67–0.99 |

   Perforated-L 8 → 4 was not tested separately. Halving the population instead was significantly worse
   (p 0.02).
8. **Multi-node training.** The 30k and 50k configurations run on 3 nodes (`nodes` column,
   `scripts/slurm/run_multinode.sh`, `multinode.py`).
   - The history is bit-identical to a single-node run when all nodes have the same CPU model.
   - Mixed CPU models differ at the level of float rounding.
   - `reproduce.py train --single-node` turns multi-node training off.
9. **Smaller threshold noise (later default).** `--sigma_scale_threshold` now defaults to 2.5 (0.05
   z-units), down from the manuscript's 50 (1.0 z-unit). At 1.0, one perturbation sweeps the kept-edge
   fraction across roughly 16–84%. The threshold then accounts for almost all reward differences, and the
   weights' ES gradient is not detectable. Workstation test (anisotropy, about 2.2k unknowns, 5 seeds):

   | | threshold noise 2.5 | threshold noise 50 |
   |---|---|---|
   | final best reward | −2.643 ± 0.044 | −2.844 ± 0.045 |
   | test PCG iterations | 12.9 ± 0.3 | 16.7 ± 0.8 |

   The iteration difference has p < 0.001. With the smaller noise, however, the hierarchies became heavier
   (item 10).
10. **Larger complexity weight (later default).** `--reward_lambda_cx` now defaults to 2.0 (manuscript:
    0.1), together with item 9. With the smaller noise alone, the learned hierarchies get heavier and time
    to solution rises by about 10–15%.

    | λ | 0.1 | 0.5 | 1 | 2 | manuscript settings |
    |---|---|---|---|---|---|
    | PCG solve time (ms) | 6.74 | 6.44 | 5.56 | 5.24 | 5.87 |

    The first four columns use the smaller noise, in a workstation screen with 5 seeds per arm.
11. **Row-relative strength feature fixed.** The edge feature `s_ij = max(-a_ij, 0) / m_i` used to divide
    by `m_i + 1e-9`.
    - On lognormal matrices this made 19.7% of the values wrong, by up to 0.99.
    - It now divides by `m_i`. The other problems change only in the last float32 bit.
    - The manuscript's lognormal policies were trained with the old feature.

## 4. Determinism and timing

| source of randomness | control |
|---|---|
| meshes and coefficients | `make_spectral_train_set(seed=...)`; test sets use seed + 1000 |
| relaxed-probe features | per-sample seed `seed * 100003 + i` |
| PyAMG SA setup (random start vector of its spectral-radius estimate) | `amg_core.fixed_np_rng(0)` around every SA setup |
| ES noise, mini-batches, model initialisation | `np.random.seed(seed)`, `torch.manual_seed(seed)` |
| CUDA atomics in PyG's GAT aggregation | `torch.use_deterministic_algorithms(True)` when a GPU is used |

- **Repeat runs.** A run repeated on the same kind of device is bit-identical and does not depend on the
  number of workers.
- **CPU and GPU forwards** agree to about 1e-7 but are not bit-identical: an edge whose z-score lies within
  float rounding of the threshold can flip. Compare runs made on the same device type.
- **Across machines.** The generated matrices are not bit-identical across CPU architectures. The
  perforated-L Triangle mesh in particular can differ by a few nodes. A rerun on another machine therefore
  uses statistically equivalent inputs, not identical ones.

**Fast inference** (`fast_policy.py`). An exact reformulation of features → policy → threshold → `C` for
threshold-mode policies.
- It avoids the E × 12 edge-feature tensor and runs the GAT layers as sparse × dense products on `A`'s own
  pattern.
- It selects the same edges as the reference forward, except for ties within float rounding
  (`tests/test_fast_policy.py`).
- Time for A → C at 50k, features included:

  | | 1 CPU thread | V100 |
  |---|---|---|
  | reference forward | 1.93 s | 1.05 s |
  | `FastPolicy` | 0.41 s | 0.058 s |

**Training speedups** (`LSC_SPEEDUPS`, on by default). The options `pcg`, `sched`, `soc`, `diag`,
`overlap`, `nopickle` and `preload` are exact: they give a bit-identical `training_history.json`.

`stag` is near-exact. It applies to population members on the CPU path:
- A member's PCG is stopped, and counted as 500 iterations, once it has run max(100, 3 × the centre's
  iterations) and its recent residual decay predicts more than 750 iterations.
- The population ranking is unchanged to Spearman ≥ 0.9997.
- The final models are within seed noise of the exact code.
- To restore exact behaviour, set `LSC_SPEEDUPS` without `stag`, for example
  `LSC_SPEEDUPS=pcg,sched,soc,diag,overlap,nopickle,preload`.

**Benchmark setup time.**
- With `FastPolicy`, the learned method's timed setup covers everything from `A` to the hierarchy: probes,
  features, policy, selection, `C` and the SA setup.
- The manuscript's reference path timed only policy + selection + SA setup.
- `preconditioner_benchmark/learned_setup.json` records which was used.

**ρ_V** (`evaluate_preconditioners.vcycle_convergence_factor`).
- Power iteration on the V-cycle error propagator `I − BA`, renormalised in the A-norm.
- It returns the geometric mean of the per-step reductions after 10 warm-up steps (40 in total).
- On 2D Poisson with PyAMG defaults it gives 0.251, which matches PyAMG's own residual factor.
- This is a different quantity from the PCG residual decay or iteration count.

**Two-level timing protocol** (follow-up study).
- Both methods are two-level, with the `splu` factorisation forced and timed inside setup.
- 1 pinned thread per worker, 2 workers per node for test-set tables, median of 3 repeats, and one untimed
  warm-up per method.
- Comparisons are paired within one job.

## 5. Caveats of the two-level study

- The classical reference is the best of the per-family tuning winners as measured on the test matrices.
  This slightly favours classical.
- The cost model that defines the small policies' reward was calibrated partly on test-seed matrices.
  Early policy screening also used the test seed. Final selection used a separate tuning seed (2000).
- The node-policy finals at 50k ran on a Xeon Gold 6154 node; the headline bar ran on a Gold 6248 node.
  The absolute times are about 8% apart. Only the paired percentages within a job should be compared.
- The GAT checkpoints behind the bar are not bundled, so their rows can be replayed from the JSON but not
  regenerated without retraining.
