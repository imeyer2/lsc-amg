"""Shared paths, matrix sets and small helpers for the two-level study.

Paths (all relative to the repository root unless absolute):
  LSC_OUTPUTS        output root shared with scripts/reproduce.py (default outputs/)
  TWOLEVEL_OUT       two-level outputs (default $LSC_OUTPUTS/twolevel)
  TWOLEVEL_DATA      matrix cache (default $TWOLEVEL_OUT/data); a set is generated on first use

Matrix sets: matrix i of set (problem, n, seed) is the i-th sample drawn from
np.random.default_rng(seed) by the problem builder with n_edge=72 -- the same sequence as
problems.make_spectral_train_set(seed=seed) -- with feature_seed = seed * 100003 + i as in the
dataset. Seeds used: 1000 = held-out test set, 2000 = tuning/selection set, 0/1 = training/validation
of the node policies, 500 = cost-model calibration (see results/twolevel_2026-10/README.md).
"""
import os
import sys
import time
import platform
from pathlib import Path

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS")
for _v in THREAD_VARS:              # every measurement in this study is single-threaded; set before numpy loads
    os.environ[_v] = "1"

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROBLEMS = ("anisotropy", "lognormal", "perforated_l")
PERF_L_FACTOR = 1.4745          # perforated_l: n ~ 1.4745 * n_grid (mesh with holes)


def _rel(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def out_root():
    if os.environ.get("TWOLEVEL_OUT"):
        return _rel(os.environ["TWOLEVEL_OUT"])
    return _rel(os.environ.get("LSC_OUTPUTS", "outputs")) / "twolevel"


def data_root():
    if os.environ.get("TWOLEVEL_DATA"):
        return _rel(os.environ["TWOLEVEL_DATA"])
    return out_root() / "data"


def single_thread():
    for v in THREAD_VARS:
        os.environ[v] = "1"


def n_grid_for(problem, n):
    """problems.py grid parameter for a target matrix size n."""
    return int(round(n / PERF_L_FACTOR)) if problem == "perforated_l" else int(n)


def set_paths(problem, n, seed, count):
    d = data_root() / problem
    return [str(d / f"n{n}_s{seed}_{i}.npz") for i in range(count)]


def gen_set(problem, n, seed, count, verbose=True):
    """Generate (once) and return the paths of matrices 0..count-1 of set (problem, n, seed)."""
    paths = set_paths(problem, n, seed, count)
    if all(os.path.exists(p) for p in paths):
        return paths
    from problems import (build_anisotropic_diffusion_sample, build_lognormal_diffusion_sample,
                          build_perforated_ldomain_sample, build_poisson_sample)
    B = {"anisotropy": build_anisotropic_diffusion_sample, "lognormal": build_lognormal_diffusion_sample,
         "perforated_l": build_perforated_ldomain_sample, "poisson": build_poisson_sample}[problem]
    os.makedirs(os.path.dirname(paths[0]), exist_ok=True)
    rng = np.random.default_rng(seed)
    for i, p in enumerate(paths):
        t0 = time.time()
        A, coords, _ = B(rng=rng, n_edge=72, n_grid=n_grid_for(problem, n))
        A = sp.csr_matrix(A)
        A.sum_duplicates()
        A.sort_indices()
        if not os.path.exists(p):
            tmp = f"{p}.{os.getpid()}.tmp.npz"
            np.savez(tmp, data=A.data, indices=A.indices, indptr=A.indptr, shape=np.array(A.shape),
                     coords=np.asarray(coords), feature_seed=seed * 100003 + i)
            os.replace(tmp, p)
        if verbose:
            print(f"gen {problem} n={A.shape[0]} nnz={A.nnz} seed={seed} [{i}] {time.time() - t0:.1f}s", flush=True)
    return paths


def load_A(path, coords=False):
    """(A, feature_seed) or (A, feature_seed, coords)."""
    z = np.load(path)
    A = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    if coords:
        return A, int(z["feature_seed"]), z["coords"]
    return A, int(z["feature_seed"])


def load_set(problem, n, seed, count, coords=False):
    """list of A (or (A, coords)) for set (problem, n, seed), generated on first use."""
    out = []
    for p in gen_set(problem, n, seed, count, verbose=False):
        r = load_A(p, coords=coords)
        out.append((r[0], r[2]) if coords else r[0])
    return out


def cpu_model():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()
