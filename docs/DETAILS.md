# LSC-AMG: full results and details

<sub>Code for the paper (under review), based on Ian Meyer's mathematics master's thesis, University of Nebraska at Omaha.</sub>

Ian Meyer, Mahboub Baccouch, Lochana Palayangoda

This repository contains the code for LSC-AMG, which learns the strength-of-connection graph of algebraic
multigrid (AMG). We train a graph attention network (GAT) with evolution strategies to choose which matrix
entries count as strong connections. Smoothed aggregation then builds a two-level V-cycle from that graph,
and the V-cycle preconditions conjugate gradients (CG). On 50k-unknown anisotropic diffusion the learned
graph cuts PCG iterations by 33% compared with a time-tuned classical rule, but the total solve takes 46.8%
longer, and the results below show why.

## Method

| Step | What happens | Code |
|---|---|---|
| 1. Features | The sparsity graph of `A` is the input: 9 node features (relaxed probe vectors and row statistics) and 12 edge features. | [`amg_core.py`](../amg_core.py) `build_sample_features` |
| 2. Score edges | A 2-layer GAT (52,802 parameters) scores each edge. A learned threshold on the z-scored logits selects the strong ones. | [`models.py`](../models.py) `AMGEdgePolicy`, [`fast_policy.py`](../fast_policy.py) |
| 3. Strength graph | The selected edges are symmetrised and given a unit diagonal, giving the binary matrix `C`. | [`amg_core.py`](../amg_core.py) `C_from_selected_edges` |
| 4. Smoothed aggregation | PyAMG builds a two-level hierarchy from `C` and solves the coarse problem with `splu`. | [`amg_core.py`](../amg_core.py) `build_pyamg_solver` |
| 5. Precondition CG | One V-cycle per PCG iteration, run to relative residual 1e-8. | [`evaluate_preconditioners.py`](../evaluate_preconditioners.py) |

We train the policy with evolution strategies ([`train_es_HPC.py`](../train_es_HPC.py)): antithetic Gaussian
perturbations, rank-based fitness shaping, and Adam on the ES gradient. The reward is
−log(PCG iterations) − λ·log(operator complexity), with λ = 0.1 and `--sigma_scale_threshold 50` in the
manuscript runs. This default produced every reported result. `--reward residual --reward_k K` instead
trains on the K-step PCG residual log-ratio against a default SA-AMG baseline.

The test problems ([`problems.py`](../problems.py), [`perforated_l_mesh.py`](../perforated_l_mesh.py)) are P1
finite elements on random meshes: rotated anisotropic diffusion, a log-normal coefficient `exp(8Z)`, and a
perforated L-shaped domain with high-contrast rings around the holes.

## Results

On 16 test matrices of 50k-unknown anisotropic diffusion, the learned hierarchy needs fewer iterations but
more time than the fastest classical two-level rule (`theta:0.1`):

| Mean over 16 matrices (seconds unless noted) | GAT (V100 forward) | Classical `theta:0.1` |
|---|---|---|
| PCG iterations | **27.9** | **41.6** |
| Features + GNN forward + build `C` | 0.063 | 0.011 |
| Smoothed-aggregation setup | 0.098 | 0.089 |
| Coarse LU factorisation | 0.305 | 0.114 |
| PCG solve | 0.311 | 0.321 |
| **Total** | **0.779** | **0.535** |
| Coarse unknowns · LU fill ÷ nnz(A) · ms per V-cycle | 13,937 · 12.5 · 10.6 | 8,639 · 5.4 · 7.0 |

The learned graph keeps 35% of the edges (classical: 63%), so the coarse grid is 60% larger. Its LU factor
is 2.3× larger, and every V-cycle costs about 50% more. The reward counts iterations and operator
complexity but not coarse-LU fill, so it does not see this cost. Paired, the GAT is +46.8% (95% CI +41.8 to
+51.8) slower and faster on 0 of 16 matrices.

The same holds on all three problem families:

| Problem (50k) | GAT iters / s | Best classical iters / s | Paired difference |
|---|---|---|---|
| Anisotropy | 27.9 / 0.779 | 41.6 / 0.535 (`theta:0.1`) | +46.8% |
| Log-normal | 32.4 / 0.449 | 28.4 / 0.376 (`sym:0.02`) | +19.5% |
| Perforated-L | 23.3 / 0.338 | 28.6 / 0.315 (`theta:0.2`) | +7.6% |

In a later, smaller follow-up study (not part of the thesis), an 11-parameter per-row threshold policy that
modulates the classical rule does beat it on perforated-L, but not on the other two families:

| Problem | n | Policy s | Classical s | Paired difference (95% CI) | Faster on |
|---|---|---|---|---|---|
| Perforated-L | 50k | 0.290 | 0.338 | −14.0% (−15.4, −12.7) | 16/16 |
| Perforated-L | 200k | 1.379 | 1.715 | −19.5% (−21.1, −17.9) | 8/8 |
| Perforated-L | 500k | 4.324 | 5.566 | −22.3% (−23.7, −20.9) | 8/8 |
| Anisotropy | 50k | 0.585 | 0.578 | +1.3% (−0.1, +2.8) | 5/16 |
| Log-normal | 50k | 0.415 | 0.406 | +2.4% (−0.4, +5.2) | 4/16 |

The policy was trained against a fitted cost model rather than measured time, and some test-seed matrices
were used to calibrate that model. Details, data files and all variants are in
[`results/twolevel_2026-10/README.md`](../results/twolevel_2026-10/README.md).

All timings are single-thread measurements on cluster nodes: Xeon Gold 6248 with a V100 for the first two
tables and the 200k/500k rows, and Gold 6154 for the 50k follow-up rows. Each JSON file records its node.
Paired percentages transfer across machines better than seconds.

## Installation

The stored results need only NumPy. To train and evaluate on a CPU, install Python 3.11+ and the CPU
requirements (tested on macOS arm64 and Linux x86_64):

```bash
git clone https://github.com/imeyer2/lsc-amg.git && cd lsc-amg
uv venv -p 3.11 .venv
uv pip install -p .venv/bin/python -r requirements-cpu.txt
```

On Linux x86_64, add `--index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match`
to the install to skip about 3 GB of CUDA packages. [`requirements.txt`](../requirements.txt) keeps the
historical Linux/CUDA pins used for the full study.

## Reproducing the results

**Stored results (seconds, NumPy only).** To print the headline and follow-up comparisons, run:

```bash
python -m twolevel.tables vsbest results/twolevel_2026-10/bar/def_50000_anisotropy.json            # Results, first table
python -m twolevel.tables vsbest results/twolevel_2026-10/final/reward_def_50000_perforated_l.json  # Results, follow-up, 50k
```

Compare the `LSC 50k-trained (s4) fwd V100` row with `theta:0.1`, and
`node:perforated_l_noderow_costtgt` with `theta:0.2`. Use `table` instead of `vsbest` for the per-phase
columns.

**Small end-to-end run (about 3 minutes on a CPU).** To train 5 seeds × 2 epochs on a tiny perforated-L
problem, then select, benchmark, scale, compare and plot, run:

```bash
bash tests/smoke.sh        # ends with "SMOKE TEST PASSED"; writes only to outputs_smoke/
```

It took 185 s on an Apple M4 and 227 s on emulated Linux x86_64. Inside Docker, pass `--shm-size=2g`.
Regenerated matrices are not bit-identical across machines, so the hash checks only warn unless
`SMOKE_STRICT=1`.

**Full study (about 25,000 core-hours).** [`scripts/reproduce.py`](../scripts/reproduce.py) runs
`train` → `select` → `benchmark` → `scale` → `compare` → `figures` on SLURM or with `--local`, and
[`scripts/twolevel.py`](../scripts/twolevel.py) runs the two-level study. For the manuscript settings, run:

```bash
python scripts/reproduce.py train --train-args "--sigma_scale_threshold 50 --reward_lambda_cx 0.1"
```

[docs/REPRODUCE.md](REPRODUCE.md) covers the remaining options and the difference between replaying
stored results, rerunning on corrected inputs, and reproducing the original random draws.

## Repository structure

```
train_es_HPC.py            ES training and the reward
models.py, fast_policy.py  GAT edge policy and fast exact inference
amg_core.py                features, strength graph, PyAMG hierarchy, PCG
problems.py                PDE problem generators (with perforated_l_mesh.py)
evaluate_preconditioners.py  benchmarks against Jacobi, ILU, SA-AMG, RS-AMG
twolevel/                  two-level comparison, small node policies, reporting (tables.py)
scripts/                   reproduction drivers and SLURM examples
configs/                   experiment and scalability settings
results/                   stored measurements used above
tests/                     smoke test and numerical regression checks
docs/                      methods and full reproduction notes
```

## Limitations

1. The learned hierarchy is always two-level, so its coarse direct solve dominates as problems grow.
2. The GAT's reward does not price coarse-LU fill. Against a classical two-level rule tuned for total time,
   the GAT loses at 50k on all three families, even with a V100 forward.
3. On a CPU, the GAT forward alone (0.36 s) costs more than the classical PCG phase (0.32 s).
4. The small follow-up policy wins only on perforated-L.
5. "Iterations" are PCG iterations. The independently estimated V-cycle convergence factor ρ(I − BA) is
   0.58–0.86 for the learned hierarchy and 0.81–0.94 for SA-AMG; see
   [`results/fixed_eval_2026-10/comparison_fixed.md`](../results/fixed_eval_2026-10/comparison_fixed.md).
6. The original manuscript runs used unseeded data generation, so their exact matrices cannot be
   regenerated. Seeded reruns reproduce them within seed noise.

More detail is in [docs/METHODS.md](METHODS.md).

## Citation

If you use this code, please cite:

```bibtex
@unpublished{meyer_lsc_amg,
  title  = {Preconditioning Large Sparse Linear Systems via Learned Algebraic Multigrid},
  author = {Meyer, Ian and Baccouch, Mahboub and Palayangoda, Lochana},
  note   = {Manuscript submitted for publication},
  year   = {2026}
}
```

## License

MIT; see [LICENSE](../LICENSE).

## Acknowledgements

The code is by Ian Meyer. The paper is by Ian Meyer, Mahboub Baccouch and Lochana Palayangoda. The
two-level comparison and the small row-threshold policies are later follow-up work. The implementation
builds on [PyAMG](https://github.com/pyamg/pyamg) and [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric).
An earlier prototype is at <https://github.com/imeyer2/Learning_SoC_Preconditioning>.
