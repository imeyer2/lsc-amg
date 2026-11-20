"""
models.py

Shared GNN policy network for AMG strength-of-connection learning.
Used by all training scripts (anisotropic, biharmonic, elasticity, etc.).
"""

from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv




# ============================================================
# SmallEdgePolicy - tiny model for Evolution Strategies
# ============================================================
class SmallEdgePolicy(nn.Module):
    """
    Lightweight edge-scoring model (~178 params) designed for ES optimisation.
    Skips the GNN entirely. Uses 9 hand-crafted edge features.
    Smart init: logit ~ |a_ij| (classical strength).
    """
    def __init__(self, in_channels: int = 6, hidden: int = 16):
        super().__init__()
        self.in_channels = in_channels
        n_feat = 9
        self.mlp = nn.Sequential(
            nn.Linear(n_feat, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.alpha = nn.Parameter(torch.tensor(1.0))
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-3)
                nn.init.zeros_(m.bias)

    def forward(self, x, edge_index, edge_weight):
        row, col = edge_index
        w = edge_weight.abs().unsqueeze(1)
        phys_i = x[row, :4]
        phys_j = x[col, :4]
        sim = F.cosine_similarity(phys_i, phys_j, dim=1).unsqueeze(1)
        ci = x[row, 4:6]
        cj = x[col, 4:6]
        dxy = ci - cj
        dx = dxy[:, [0]]
        dy = dxy[:, [1]]
        adx = dx.abs()
        ady = dy.abs()
        dn = torch.sqrt(dx * dx + dy * dy + 1e-12)
        udx = dx / dn
        udy = dy / dn
        feat = torch.cat([w, sim, dx, dy, adx, ady, dn, udx, udy], dim=1)
        logits = self.mlp(feat).squeeze(-1) + self.alpha * edge_weight.abs()
        diag = (row == col)
        logits = logits.masked_fill(diag, -1e9)
        return logits, None


class AMGEdgePolicy(nn.Module):
    """
    Outputs edge logits for selecting strong connections.

    Architecture:
    - 2-layer GATConv for node embeddings
    - Direction-aware edge features (coords, similarity, weight)
    - Edge MLP producing logits for stochastic per-row top-k selection
    - Optional learned B-head for near-nullspace candidates

    Parameters:
        in_channels: Number of input node features.
            - 6 for scalar PDEs (4 relaxed + 2 coords)
            - 7 for elasticity (4 relaxed + 2 coords + 1 dof_type)
            - Or FeatureConfig.node_dim for new features
        hidden: Hidden dimension for GATConv layers.
        learn_B: Whether to learn additional near-nullspace candidates.
        B_extra: Number of extra learned B columns.
            - 2 for scalar PDEs
            - 3 for elasticity (augmenting 3 rigid body modes)
        edge_hidden: Tuple of hidden dims for edge MLP.
            Default (128, 64) gives ~25K params (needs pre-training).
            Use (16,) for ES-friendly ~400-param edge MLP.
        skip_alpha: If True, add learnable skip connection
            logits += alpha * |a_ij| (classical strength baseline).
            Combined with small_init, lets model start at classical
            strength and learn corrections via ES.
        small_init: If True, initialise edge_mlp weights near zero
            so initial output is dominated by the skip connection.
        edge_raw_dim: Dimension of pre-computed raw edge features.
            When > 0, edge MLP input = [h_i, h_j, |h_i-h_j|, h_i*h_j, raw_edge_features].
            When 0, falls back to legacy edge features (cosine sim + direction + |w|).
    """
    def __init__(self, in_channels: int = 6, hidden: int = 64,
                 learn_B: bool = False, B_extra: int = 2,
                 edge_hidden: tuple = (128, 64),
                 skip_alpha: bool = False,
                 small_init: bool = False,
                 edge_raw_dim: int = 0,
                 adaptive_k: bool = False,
                 init_threshold: float = 0.0):
        super().__init__()
        self.learn_B = bool(learn_B)
        self.B_extra = int(B_extra)
        self.skip_alpha_enabled = skip_alpha
        self.edge_raw_dim = edge_raw_dim
        self.adaptive_k = adaptive_k

        if adaptive_k:
            self.threshold = nn.Parameter(torch.tensor(init_threshold))
            print(f"\033[94mAMGEdgePolicy: adaptive-K enabled, "
                  f"init_threshold={init_threshold:.2f}\033[0m")

        # Throw warning in red if learn_B is False.
        if self.learn_B:
            print(f"\033[92mAMGEdgePolicy initialized with learn_B=True, B_extra={self.B_extra}\033[0m")
        else:
            print("\033[91mWARNING: learn_B is False, no near-nullspace candidates will be learned!\033[0m")

        self.conv1 = GATConv(in_channels, hidden, heads=2, concat=True)
        self.conv2 = GATConv(hidden * 2, hidden, heads=1, concat=False)

        if edge_raw_dim > 0:
            # New: h_i, h_j, |h_i-h_j|, h_i*h_j, raw_edge_features
            edge_in = 4 * hidden + edge_raw_dim
            # LayerNorm prevents h_i*h_j product from amplifying unbounded
            # embeddings quadratically, which causes gradient explosion.
            self.emb_norm = nn.LayerNorm(hidden)
        else:
            # Legacy: node_i, node_j, |a_ij|, sim(relaxed), direction(6)
            edge_in = 2 * hidden + 1 + 1 + 6

        # Build edge MLP from edge_hidden tuple
        layers = []
        prev = edge_in
        for h in edge_hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.edge_mlp = nn.Sequential(*layers)

        # Optional skip connection: logits += alpha * |a_ij|
        if skip_alpha:
            self.alpha = nn.Parameter(torch.tensor(1.0))

        # Optional near-zero init for ES (so initial logits ≈ skip)
        if small_init:
            for m in self.edge_mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=1e-3)
                    nn.init.zeros_(m.bias)

        if self.learn_B:
            self.B_head = nn.Sequential(
                nn.Linear(hidden, 64),
                nn.ReLU(),
                nn.Linear(64, self.B_extra)
            )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor,
                return_node_emb: bool = False,
                raw_edge_features: Optional[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
          edge_logits: (E,)
          B_extra: (N, B_extra) if learn_B else None
          If return_node_emb=True, returns (logits, B_extra, h) where h is (N, hidden).
        """
        h = F.elu(self.conv1(x, edge_index))
        h = self.conv2(h, edge_index)

        row, col = edge_index

        if self.edge_raw_dim > 0 and raw_edge_features is not None:
            # New feature path: h_i, h_j, |h_i - h_j|, h_i * h_j, raw_edge_features
            h = self.emb_norm(h)
            node_i = h[row]
            node_j = h[col]
            diff = (node_i - node_j).abs()
            prod = node_i * node_j
            edge_feat = torch.cat([node_i, node_j, diff, prod, raw_edge_features], dim=1)
        else:
            # Legacy feature path: cosine sim + direction + |w|
            phys_i = x[row, :4]
            phys_j = x[col, :4]
            sim = F.cosine_similarity(phys_i, phys_j, dim=1).unsqueeze(1)

            ci = x[row, 4:6]
            cj = x[col, 4:6]
            dxy = ci - cj
            dx = dxy[:, [0]]
            dy = dxy[:, [1]]
            adx = dx.abs()
            ady = dy.abs()
            dn = torch.sqrt(dx * dx + dy * dy + 1e-12)
            udx = dx / dn
            udy = dy / dn
            dir_feat = torch.cat([dx, dy, adx, ady, udx, udy], dim=1)

            node_i = h[row]
            node_j = h[col]
            w = edge_weight.unsqueeze(1).abs()
            edge_feat = torch.cat([node_i, node_j, w, sim, dir_feat], dim=1)

        logits = self.edge_mlp(edge_feat).squeeze(-1)

        # Optional skip connection: classical strength baseline
        if self.skip_alpha_enabled:
            logits = logits + self.alpha * edge_weight.abs()

        # mask diagonals heavily (if present)
        diag = (row == col)
        logits = logits.masked_fill(diag, -1e9)

        B_extra = None
        if self.learn_B:
            B_extra = self.B_head(h)  # (N, B_extra)

        if return_node_emb:
            return logits, B_extra, h
        return logits, B_extra
