#!/usr/bin/env python3
"""Exactness test for fast_policy.FastPolicy against the reference path.

For every problem type, several untrained policies (different weights and
thresholds) and every available device it checks that

  * the node features FastPolicy rebuilds from A equal the dataset's bit for bit;
  * the selected directed edges equal the reference selection
    (AMGEdgePolicy.forward + select_edges_by_threshold), except edges whose
    z-score lies within 1e-4 of the threshold, where any change in float
    summation order can flip the comparison (the reference CPU and GPU forwards
    differ on such ties too); at most 1 in 10^4 edges may be such ties;
  * FastPolicy.strength equals C_from_selected_edges on its own selection.

Run from the repository root:  python tests/test_fast_policy.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from amg_core import (FeatureConfig, C_from_selected_edges,  # noqa: E402
                      select_edges_by_threshold)
from fast_policy import FastPolicy  # noqa: E402
from models import AMGEdgePolicy  # noqa: E402
from problems import make_spectral_train_set  # noqa: E402

PROBLEMS = [("lognormal", 2000), ("anisotropy", 2000), ("perforated_l", 1500)]
THRESHOLDS = [-1.0, 0.0, 0.7]
TIE = 1e-4


def reference(model, s, device):
    with torch.no_grad():
        lg, _ = model(s.x.to(device), s.edge_index.to(device), s.edge_weight.to(device),
                      raw_edge_features=s.raw_edge_features.to(device))
        sel, _ = select_edges_by_threshold(logits=lg, edge_index=s.edge_index.to(device),
                                           threshold=model.threshold.item())
        off = s.edge_index[0] != s.edge_index[1]
        v = lg[off.to(device)]
        z = ((lg - v.mean()) / (v.std() + 1e-6)).cpu().numpy()
    return sel.cpu().numpy(), z, off.numpy()


def main():
    cfg = FeatureConfig.from_preset("new_algebraic_plus_relaxed")
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    ok = True
    for problem, n in PROBLEMS:
        samples = make_spectral_train_set(num_samples=3, n_grid=n, seed=7,
                                          feature_config=cfg, problem=problem)
        for dev in devices:
            flips = ties = edges = 0
            for wseed in (0, 1):
                torch.manual_seed(wseed)
                m = AMGEdgePolicy(in_channels=cfg.node_dim, hidden=64,
                                  edge_raw_dim=cfg.edge_raw_dim, adaptive_k=True)
                m.eval().to(dev)
                for t in THRESHOLDS:
                    with torch.no_grad():
                        m.threshold.fill_(t)
                    fast = FastPolicy(m, dev)
                    for s in samples:
                        p = fast.prepare(s.A, cfg, s.feature_seed)
                        if not torch.equal(p.x, s.x):
                            print(f"  node features differ: {problem} {dev}")
                            ok = False
                        sel_ref, z, off = reference(m, s, dev)
                        sel = fast.edge_mask(p, s.edge_index)
                        bad = sel != sel_ref
                        near = np.abs(z - t) < TIE
                        flips += int((bad & ~near).sum())
                        ties += int((bad & near).sum())
                        edges += int(off.sum())
                        C_fast = fast.strength(p)
                        C_sel = C_from_selected_edges(s.A, s.edge_index, sel)
                        if abs(C_fast - C_sel).nnz:
                            print(f"  C assembly differs: {problem} {dev}")
                            ok = False
            good = flips == 0 and ties <= edges // 10000
            ok &= good
            print(f"{problem:13s} {dev:4s} {'ok' if good else 'MISMATCH'}  "
                  f"edges={edges} flips={flips} ties={ties}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
