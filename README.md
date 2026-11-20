# LSC-AMG · Learning strength of connection for algebraic multigrid

**Question.** Algebraic multigrid (AMG) decides which matrix entries are "strong" connections with a fixed
rule of thumb (keep `a_ij` if it is large compared with the rest of its row). Can a neural network learn
that choice for a family of PDE matrices, so that AMG becomes a better preconditioner for conjugate
gradients (CG)? And does the gain survive when the cost of the whole solve, not just the iteration count,
is measured?

**Contribution.** A graph-attention network (GAT) scores every off-diagonal entry of a sparse SPD matrix.
The selected entries form the strength graph of classical smoothed aggregation, and the resulting
two-level V-cycle preconditions CG. The policy is trained with evolution strategies, because the path from
network weights to solver iterations is not differentiable. This repository contains the implementation,
the PDE problem generators, the training and benchmarking pipeline, stored results, and a later follow-up
study. That study compares the learned hierarchy with a classical rule tuned for total time and explains
when fewer iterations still take longer.

## Pipeline

| Step | What happens | Code |
|---|---|---|
| 1. Features | The sparsity graph of `A` is the input graph: 9 node features (relaxed probe vectors and row statistics) and 12 edge features. | [`amg_core.py`](amg_core.py) `build_sample_features` |
| 2. Score edges | A GAT (2 layers, 52,802 parameters) scores each edge, and a learned threshold on the z-scored logits selects the strong ones. | [`models.py`](models.py) `AMGEdgePolicy`; fast exact inference in [`fast_policy.py`](fast_policy.py) |
| 3. Strength graph | The selected edges are symmetrised and given a unit diagonal, which gives the binary matrix `C`. | [`amg_core.py`](amg_core.py) `select_edges_by_threshold`, `C_from_selected_edges` |
| 4. Smoothed aggregation | `pyamg.smoothed_aggregation_solver(A, strength=("predefined", {"C": C}))`. A predefined strength graph gives a **two-level** hierarchy, and its coarse problem is solved with `splu`. | [`amg_core.py`](amg_core.py) `build_pyamg_solver` |
| 5. Precondition CG | One V-cycle per PCG iteration, run to relative residual 1e-8. | [`amg_core.py`](amg_core.py), [`evaluate_preconditioners.py`](evaluate_preconditioners.py) |

The test problems ([`problems.py`](problems.py), [`perforated_l_mesh.py`](perforated_l_mesh.py)) are P1
finite elements on random meshes: rotated anisotropic diffusion, a log-normal coefficient `exp(8Z)` with
`Z` a Gaussian random field, and a perforated L-shaped domain with high-contrast rings around the holes.

**Training** ([`train_es_HPC.py`](train_es_HPC.py)). The policy is trained with OpenAI-style evolution
strategies: antithetic Gaussian perturbations, rank-based fitness shaping, and Adam on the ES gradient.

Reward: −log(PCG iterations) − λ·log(operator complexity) (`train_es_HPC.py::amg_reward`; manuscript runs used λ = 0.1, `--sigma_scale_threshold 50`). This default produced every reported result.
`--reward residual --reward_k K` instead trains on the K-step PCG residual log-ratio against a default SA-AMG baseline, minus λ·log(operator complexity).

## Headline result: fewer iterations, more time

Both methods use two levels on 16 test matrices of 50k-unknown anisotropic diffusion. The comparison runs
on one Xeon Gold 6248 node with a V100: one pinned thread per worker, median of 3 repeats, and paired
within one job. The GAT is the manuscript configuration (λ = 0.1, threshold noise 50), retrained in
October 2026 on corrected code (best of 5 seeds). The classical reference `theta:0.1` is the fastest of the
classical two-level rules tuned for total time, as measured on these test matrices.

| mean over 16 matrices (seconds unless noted) | GAT (50k-trained, V100 forward) | classical `theta:0.1` |
|---|---|---|
| PCG iterations | **27.9** | **41.6** |
| features (1 CPU thread) + GNN forward (V100) + build `C` | 0.043 + 0.015 + 0.005 | 0 + 0 + 0.011 |
| smoothed-aggregation setup | 0.098 | 0.089 |
| coarse LU factorisation | 0.305 | 0.114 |
| PCG solve | 0.311 | 0.321 |
| **total** | **0.779** | **0.535** |
| coarse unknowns · LU fill ÷ nnz(A) · ms per V-cycle | 13,937 · 12.5 · 10.6 | 8,639 · 5.4 · 7.0 |

The learned graph keeps only 35% of the edges (the classical rule keeps 63%). The aggregates are therefore
smaller, and the coarse grid is 60% larger. Its direct LU factor is 2.3× larger, so factorisation costs
0.19 s more and every V-cycle costs about 50% more. The 33% fewer iterations save only 0.010 s of solve
time. The reward counts iterations and operator complexity but not coarse-LU fill, so it does not see
this cost. Paired, the GAT is **+46.8% [95% CI +41.8, +51.8]** slower and is faster on 0 of 16 matrices.

Data: [`results/twolevel_2026-10/bar/def_50000_anisotropy.json`](results/twolevel_2026-10/bar/def_50000_anisotropy.json).
Reproduce the table (NumPy only): `python -m twolevel.tables vsbest results/twolevel_2026-10/bar/def_50000_anisotropy.json`
for the paired summary, or `... table ...` for the per-phase columns.

All three families, same setting: anisotropy 0.779 vs 0.535 s (+46.8%), lognormal 0.449 vs 0.376 s
(`sym:0.02`, +19.5%), perforated-L 0.338 vs 0.315 s (`theta:0.2`, +7.6%). The GAT is never faster on any
matrix. See [`results/twolevel_2026-10/README.md`](results/twolevel_2026-10/README.md) for the complete
table and the files behind it.

## Follow-up: a small learned policy that does win (later study)

In a later, smaller study (October 2026, not part of the thesis), an **11-parameter per-row threshold
policy** modulates the classical rule with cheap node features. On 50k perforated-L it takes **0.290 s
versus 0.338 s** for `theta:0.2` (18.5 vs 28.6 iterations). The mean paired reduction is **−14.0%
[−15.4, −12.7]**, and it is faster on all 16 test matrices. Applied zero-shot at 200k and 500k unknowns,
the gain is −19.5% and −22.3% (8 of 8 matrices each). The other two families do not show a gain.
Anisotropy is a tie (+1.3% [−0.1, +2.8]). On lognormal the policy loses (+2.4% [−0.4, +5.2] and +4.0%
[+1.7, +6.3] for the two variants).

Read these qualifications together with the result:
- The policy was trained by ES against a **fitted cost model**, not measured time. Some test-seed matrices
  were used to calibrate that model, and early policy screening also used the test seed. Final selection
  used a separate tuning seed.
- The classical reference is the best of the per-family tuning winners as measured on the test matrices,
  which slightly favours classical.
- The seconds are observations on the recorded hardware. This comparison ran on a **Xeon Gold 6154** node;
  the headline comparison above ran on a Gold 6248 node, where the same `theta:0.2` reference took 0.315 s.
  Only the paired, within-job percentage is meant to transfer. A later rerun through the driver gave
  −14.7% [−16.2, −13.3].

Data: [`results/twolevel_2026-10/final/reward_def_50000_perforated_l.json`](results/twolevel_2026-10/final/reward_def_50000_perforated_l.json)
(ties and losses: [`..._anisotropy.json`](results/twolevel_2026-10/final/reward_def_50000_anisotropy.json),
[`..._lognormal.json`](results/twolevel_2026-10/final/reward_def_50000_lognormal.json)). Reproduce with
`python -m twolevel.tables vsbest results/twolevel_2026-10/final/reward_def_50000_perforated_l.json`.

**Two different convergence measures.** "Iterations" above are PCG iterations. The earlier manuscript's
"V-cycle convergence factor" figures actually plotted a PCG residual tail ratio (0.18–0.44). The
independently estimated V-cycle factor ρ(I − BA), from power iteration in the A-norm, is 0.58–0.86 for the
learned hierarchy and 0.81–0.94 for SA-AMG. The learned hierarchy is still consistently better, but its
factor is not below 0.5. See [`results/fixed_eval_2026-10/comparison_fixed.md`](results/fixed_eval_2026-10/comparison_fixed.md).

## Reproduce

```bash
git clone https://github.com/imeyer2/lsc-amg.git && cd lsc-amg
```

**A. Read stored results (seconds, NumPy only).**

```bash
python -m pip install numpy   # any Python 3 with NumPy
python -m twolevel.tables vsbest results/twolevel_2026-10/bar/def_50000_anisotropy.json
python -m twolevel.tables vsbest results/twolevel_2026-10/final/reward_def_50000_perforated_l.json
```

Compare the `LSC 50k-trained (s4) fwd V100` row with `theta:0.1` (first command), and
`node:perforated_l_noderow_costtgt` with `theta:0.2` (second command).

**B. Run a small end-to-end experiment on a CPU (about 3–4 minutes).** This needs Python 3.11 or newer.

```bash
uv venv -p 3.11 .venv
uv pip install -p .venv/bin/python -r requirements-cpu.txt            # macOS (arm64) and others
# Linux x86_64: use the CPU-only torch wheels to avoid about 3 GB of CUDA packages
uv pip install -p .venv/bin/python --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match -r requirements-cpu.txt
bash tests/smoke.sh        # ends with "SMOKE TEST PASSED"
```

`tests/smoke.sh` trains 5 seeds × 2 epochs on a tiny perforated-L problem. It then selects the best seed,
benchmarks it, runs the scaling step, compares and draws figures, writing only to `outputs_smoke/`.
Observed run times were 185 s on macOS arm64 (Apple M4) and 227 s on Linux x86_64 (emulated). Inside Docker,
pass `--shm-size=2g`. Some checks compare hashes recorded on the reference Linux x86_64 cluster, and
regenerated matrices, especially the perforated-L meshes, are not bit-identical across machines. Those checks
therefore only print a warning unless `SMOKE_STRICT=1` is set.

**C. Full study.** [`scripts/reproduce.py`](scripts/reproduce.py) runs `train` → `select` → `benchmark` →
`scale` → `compare` → `figures` (SLURM or `--local`), and [`scripts/twolevel.py`](scripts/twolevel.py) runs
the two-level study. The full study takes about 25,000 core-hours. See
[docs/REPRODUCE.md](docs/REPRODUCE.md) for this route. It also covers the manuscript settings
(`--train-args "--sigma_scale_threshold 50 --reward_lambda_cx 0.1"`), the halved mini-batch in
`configs/experiments.csv`, and the default PCG early stop. Finally, it explains the difference between
replaying stored results, rerunning on corrected inputs, and reproducing the original random draws.
[`requirements.txt`](requirements.txt) keeps the historical Linux/CUDA pins.

## Limitations

The learned hierarchy is always two-level, so its coarse direct solve dominates at larger sizes. The
GAT's reward does not price coarse-LU fill. Against a classical two-level rule tuned for total time, the
GAT loses at 50k on all three families even with a V100 forward, and on a CPU its forward alone (0.36 s)
costs more than the classical PCG phase (0.32 s). The small follow-up policy wins only on perforated-L. All timings
are single-thread observations on specific cluster nodes. Iteration counts and paired percentages
transfer better than seconds. The original manuscript runs used unseeded data generation, so their exact
matrices cannot be regenerated. Seeded reruns reproduce them within seed noise. Further details are in
[docs/METHODS.md](docs/METHODS.md).

## Credits and citation

The code is by Ian Meyer. The paper is by Ian Meyer, Mahboub Baccouch and Lochana Palayangoda (University
of Nebraska at Omaha). The method was developed for Ian Meyer's mathematics master's thesis. The two-level
comparison and the small row-threshold policies are later follow-up work. An earlier prototype trained
with REINFORCE is at <https://github.com/imeyer2/Learning_SoC_Preconditioning>.

```bibtex
@unpublished{meyer_lsc_amg,
  title  = {Preconditioning Large Sparse Linear Systems via Learned Algebraic Multigrid},
  author = {Meyer, Ian and Baccouch, Mahboub and Palayangoda, Lochana},
  note   = {Manuscript submitted for publication},
  year   = {2026}
}
```

License: [MIT](LICENSE).
