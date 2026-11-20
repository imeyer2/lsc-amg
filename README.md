# LSC-AMG: Learning Strength of Connection for Algebraic Multigrid

A graph neural network, trained **without gradients** through a non-differentiable sparse solver, learns
which matrix connections algebraic multigrid should keep, and in doing so preconditions conjugate gradients.

Ian Meyer · Master's thesis in mathematics, University of Nebraska at Omaha · Paper under review

## Highlights

- **33% fewer CG iterations** than the best tuned classical AMG rule on 50k-unknown anisotropic diffusion
  (27.9 vs 41.6, 16 test matrices).
- **Trained small, wins big:** an 11-parameter learned policy trained in **49 s (one 46-core node) on 5k-unknown matrices** is
  **22% faster end to end at 500k unknowns** (100× larger than training), on 8 of 8 test matrices.
- **Better V-cycles:** the learned hierarchy's measured convergence factor is 0.58–0.86, versus 0.81–0.94 for
  standard smoothed-aggregation AMG.
- **Gradient-free training:** evolution strategies optimize a 52.8k-parameter graph attention network
  directly on solver iterations and operator complexity.
- **Costs measured, not assumed:** every result is paired, per-matrix, with 95% CIs and a per-phase cost
  breakdown. That breakdown showed fewer iterations can still mean a slower solve (coarse-LU fill), which
  led to the cost-aware policy above.

## Results

| Wall-clock, perforated-L diffusion | Learned policy | Best classical rule | Speed-up | Faster on |
|---|---|---|---|---|
| 50k unknowns | 0.290 s | 0.338 s | **−14.0%** | 16/16 |
| 200k unknowns | 1.379 s | 1.715 s | **−19.5%** | 8/8 |
| 500k unknowns | 4.324 s | 5.566 s | **−22.3%** | 8/8 |

The speed-up grows with problem size. Full results, including problem families where classical rules still
win, are in [docs/DETAILS.md](docs/DETAILS.md).

## Try it

Replay the stored results (NumPy only, seconds):

```bash
git clone https://github.com/imeyer2/lsc-amg.git && cd lsc-amg
python -m twolevel.tables vsbest results/twolevel_2026-10/final/reward_def_50000_perforated_l.json
```

Train, select, benchmark and plot end to end on a laptop CPU (about 3 minutes):

```bash
uv venv -p 3.11 .venv && uv pip install -p .venv/bin/python -r requirements-cpu.txt
bash tests/smoke.sh        # ends with "SMOKE TEST PASSED"
```

## How it works

1. A graph attention network scores every off-diagonal entry of a sparse SPD matrix ([`models.py`](models.py)).
2. The selected entries become the strength graph for smoothed aggregation in PyAMG ([`amg_core.py`](amg_core.py)).
3. The resulting V-cycle preconditions CG; the reward is −log(iterations) − λ·log(operator complexity).
4. Evolution strategies train the network on that reward ([`train_es_HPC.py`](train_es_HPC.py)).

Method details, the full study and reproduction on a cluster: [docs/DETAILS.md](docs/DETAILS.md),
[docs/METHODS.md](docs/METHODS.md), [docs/REPRODUCE.md](docs/REPRODUCE.md).

## Citation

```bibtex
@unpublished{meyer_lsc_amg,
  title  = {Preconditioning Large Sparse Linear Systems via Learned Algebraic Multigrid},
  author = {Meyer, Ian and Baccouch, Mahboub and Palayangoda, Lochana},
  note   = {Manuscript submitted for publication},
  year   = {2026}
}
```

Code by Ian Meyer; paper by Ian Meyer, Mahboub Baccouch and Lochana Palayangoda. MIT [license](LICENSE).
