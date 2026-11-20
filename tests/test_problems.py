#!/usr/bin/env python3
"""Regression test for the dataset generator (problems.py).

The hashes cover matrices and node/edge features for 3 samples of each
problem at seed 7, recorded on the reference cluster (Linux x86_64, the pinned
requirements.txt). They were re-recorded when the row-relative strength
feature s_ij was changed to divide by the row maximum itself instead of
(row maximum + 1e-9); before that they matched the legacy training script. Other platforms or BLAS
builds may differ in the last bits; the full reproduction ran on the reference
platform.

Run from the repository root:  python tests/test_problems.py
"""
import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from amg_core import FeatureConfig  # noqa: E402
from problems import make_spectral_train_set  # noqa: E402

REFERENCE = {
    ("lognormal", 2000): "10577dbd5d4723e26f2fb331d2f75df17bcaca8614504ad2e3289c423a538c42",
    ("anisotropy", 2000): "c3421b4f9aa1d150a0a62680efddd9d2ff04621176ac3352ae5fb874ee9cd02d",
    ("perforated_l", 1500): "f0b16a189eeaf9638c58a1e3836cba08c0b66bbc57e65e8d5d85d8763c500c39",
    ("poisson", 2000): "81983a2661ad5f28f8a8757505803d9f3b7c59de0a479a3e3329f698ce7bf118",
}


def digest(samples):
    h = hashlib.sha256()
    for s in samples:
        A = s.A.tocsr()
        A.sort_indices()
        ref = s.raw_edge_features.numpy() if s.raw_edge_features is not None else np.zeros(0)
        for arr in (A.indptr, A.indices, A.data, s.x.numpy(), s.edge_index.numpy(), s.edge_weight.numpy(), ref):
            h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def main():
    cfg = FeatureConfig.from_preset("new_algebraic_plus_relaxed")
    ok = True
    for (problem, n), want in REFERENCE.items():
        got = digest(make_spectral_train_set(num_samples=3, n_grid=n, seed=7, feature_config=cfg, problem=problem))
        status = "ok" if got == want else "MISMATCH"
        ok &= got == want
        print(f"{problem:13s} {status}  {got[:16]}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
