"""Fast inference for the LSC-AMG policy: system matrix A -> strength matrix C.

FastPolicy computes the same strength matrix as the reference path

    build_sample_features -> AMGEdgePolicy.forward -> select_edges_by_threshold
    -> C_from_selected_edges

for threshold-mode policies (adaptive_k, no learn_B, no skip_alpha, the default
`new_algebraic_plus_relaxed` features) on matrices whose sparsity pattern is
symmetric with every diagonal entry stored -- the finite-element matrices used
here. Only floating-point summation order differs; tests/test_fast_policy.py
checks that C and the selected edges are identical.

Measured on the reference cluster (Xeon Gold CPU, V100 GPU) at n = 50k (A -> C, features included):
    reference: 1.93 s (1 CPU thread), 1.05 s (V100, deterministic kernels)
    FastPolicy: 0.41 s (1 CPU thread), 0.058 s (V100)

Where the time goes away:
  * edge features are never materialised as an E x 12 tensor;
  * the two GATConv layers run as CSR sparse x dense products over A's own
    pattern (A's stored diagonal is exactly the self-loop set GATConv adds);
  * the first edge-MLP layer W1 [h_i, h_j, |h_i-h_j|, h_i*h_j, raw] is split
    into per-node terms (h W_a^T)[i] + (h W_b^T)[j], one GEMM per unordered
    pair for the i<->j-symmetric |h_i-h_j|, h_i*h_j block (shared by (i,j) and
    (j,i)), and a 12-column GEMM for each direction's raw features;
  * diagonal entries, whose logits the reference masks out, are skipped;
  * C is assembled directly on A's pattern from a symmetric mask.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from amg_core import (FeatureConfig, build_node_features, compute_row_stats,
                      relaxed_smooth_vectors)

EPS = 1e-9   # same eps as build_edge_features / build_node_features


def _default_layout(cfg: FeatureConfig) -> bool:
    """FastPolicy hard-codes the edge-feature layout of the default preset:
    [sign, s_ij^(i), s_ij^(j), a_hat, |dr| x K, |dr|_rel x K]."""
    return (cfg.use_node_relaxed_probes and cfg.use_node_operator_local
            and not cfg.use_node_geometry and cfg.use_edge_sign
            and cfg.use_edge_row_relative_strength
            and cfg.use_edge_symm_normalized_mag and cfg.use_edge_relaxed_diffs
            and not cfg.use_edge_geometry)


class Graph:
    """A's CSR pattern and the index maps FastPolicy needs. Raises ValueError
    if the pattern is not symmetric or a diagonal entry is missing."""

    def __init__(self, A: sp.csr_matrix, device="cpu"):
        A = sp.csr_matrix(A)
        if not A.has_canonical_format:
            A = A.copy()
            A.sum_duplicates()
            A.sort_indices()
        n = A.shape[0]
        indptr = A.indptr.astype(np.int64)
        indices = A.indices.astype(np.int64)
        row = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
        diag = indices == row
        if int(diag.sum()) != n:
            raise ValueError("FastPolicy needs every diagonal entry of A stored")
        # rev[e] = position of (j,i) for the stored entry e = (i,j)
        T = sp.csr_matrix((np.arange(len(indices), dtype=np.float64), indices, indptr),
                          shape=A.shape).T.tocsr()
        T.sort_indices()
        if not (np.array_equal(T.indptr, indptr) and np.array_equal(T.indices, indices)):
            raise ValueError("FastPolicy needs a structurally symmetric A")
        rev = T.data.astype(np.int64)
        self.A, self.n = A, n
        self.np_indptr, self.np_indices, self.np_row = indptr, indices, row
        self.np_off = ~diag
        self.np_upper = np.nonzero(self.np_off & (indices > row))[0]   # one entry per pair
        self.np_lower = rev[self.np_upper]                              # its (j,i) twin
        self.lin = row * n + indices                                    # sorted, for lookups
        self.set_device(device)

    def set_device(self, device):
        d = torch.device(device)
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(d)
        self.device = d
        self.indptr, self.indices, self.row = t(self.np_indptr), t(self.np_indices), t(self.np_row)
        self.lengths = torch.diff(self.indptr)
        self.up, self.lo = t(self.np_upper), t(self.np_lower)
        self.ui, self.uj = self.row[self.up], self.indices[self.up]
        return self

    def positions(self, edge_index: torch.Tensor) -> np.ndarray:
        """Position in A's (canonical) entry order of each edge in edge_index."""
        ei = edge_index.cpu().numpy().astype(np.int64)
        pos = np.searchsorted(self.lin, ei[0] * self.n + ei[1])
        if not np.array_equal(self.lin[np.minimum(pos, len(self.lin) - 1)], ei[0] * self.n + ei[1]):
            raise ValueError("edge_index has entries outside A's pattern")
        return pos


@dataclass
class Prepared:
    """Per-matrix inputs: pattern, node features and the per-node quantities
    the raw edge features are formed from."""
    g: Graph
    x: torch.Tensor          # (N, 9) node features, identical to build_node_features
    R: np.ndarray            # relaxed probes (float64), as build_sample_features
    rs: dict                 # compute_row_stats(A)


class FastPolicy:
    """Exact fast inference for a threshold-mode AMGEdgePolicy.

    Defaults (measured on the reference cluster, all exact): on CPU the fused path with
    2048-pair chunks keeps the temporaries in cache; on GPU one large chunk.
    """

    def __init__(self, model, device=None, chunk: Optional[int] = None,
                 fused: Optional[bool] = None):
        self.dev = torch.device(device) if device is not None else next(model.parameters()).device
        on_gpu = self.dev.type == "cuda"
        self.fused = (not on_gpu) if fused is None else fused
        self.chunk = chunk or ((1 << 20) if on_gpu else 2048)
        self.load(model)

    @staticmethod
    def supported(model, feature_config: FeatureConfig) -> bool:
        return (getattr(model, "adaptive_k", False) and not model.learn_B
                and not model.skip_alpha_enabled and model.edge_raw_dim > 0
                and model.edge_raw_dim == feature_config.edge_raw_dim
                and _default_layout(feature_config))

    @torch.no_grad()
    def load(self, model):
        """(Re)read the policy weights, e.g. after vector_to_parameters."""
        sd = {k: v.detach().float().to(self.dev) for k, v in model.state_dict().items()}
        H = model.conv2.out_channels
        self.H = H
        self.c1 = (sd["conv1.lin.weight"], sd["conv1.att_src"], sd["conv1.att_dst"],
                   sd["conv1.bias"], model.conv1.heads, model.conv1.concat,
                   model.conv1.negative_slope)
        self.c2 = (sd["conv2.lin.weight"], sd["conv2.att_src"], sd["conv2.att_dst"],
                   sd["conv2.bias"], model.conv2.heads, model.conv2.concat,
                   model.conv2.negative_slope)
        self.ln = (sd["emb_norm.weight"], sd["emb_norm.bias"], model.emb_norm.eps)
        W1 = sd["edge_mlp.0.weight"]
        self.b1 = sd["edge_mlp.0.bias"]
        self.Wa = W1[:, :H].T.contiguous()             # h_i
        self.Wb = W1[:, H:2 * H].T.contiguous()        # h_j
        self.Wh = W1[:, 2 * H:4 * H].T.contiguous()    # |h_i-h_j|, h_i*h_j
        self.Wr = W1[:, 4 * H:].T.contiguous()         # raw edge features
        lin = sorted({k.rsplit(".", 1)[0] for k in sd if k.startswith("edge_mlp.")},
                     key=lambda k: int(k.split(".")[1]))
        self.rest = [(sd[k + ".weight"].T.contiguous(), sd[k + ".bias"]) for k in lin[1:]]
        self.threshold = float(sd["threshold"])
        return self

    # ------------------------------------------------------------ inputs
    def prepare(self, A: sp.csr_matrix, feature_config: FeatureConfig,
                feature_seed: Optional[int]) -> Prepared:
        """Pattern + node features from A alone. With the sample's feature seed
        the relaxed probes, and hence every feature, equal the dataset's."""
        g = Graph(A, self.dev)
        K = feature_config.num_relaxed_probes
        R = relaxed_smooth_vectors(g.A, num_vecs=K, iters=feature_config.relax_iters,
                                   seed=feature_seed)[:, :K]
        rs = compute_row_stats(g.A)
        x = build_node_features(g.A, None, feature_config, row_stats=rs, R=R)
        return Prepared(g, x, R, rs)

    # ------------------------------------------------------------ network
    @staticmethod
    def _gat(x, g: Graph, W, att_s, att_d, bias, heads, concat, slope):
        """GATConv (eval mode, add_self_loops) with targets as CSR rows."""
        N = x.shape[0]
        xw = (x @ W.T).view(N, heads, -1)
        a_s = (xw * att_s).sum(-1)
        a_d = (xw * att_d).sum(-1)
        e = F.leaky_relu(a_s[g.indices] + a_d[g.row], slope)
        mx = torch.segment_reduce(e, "max", lengths=g.lengths, axis=0)
        ex = (e - mx[g.row]).exp()
        den = torch.segment_reduce(ex, "sum", lengths=g.lengths, axis=0)
        alpha = ex / (den[g.row] + 1e-16)          # torch_geometric.utils.softmax
        outs = []
        for k in range(heads):
            S = torch.sparse_csr_tensor(g.indptr, g.indices, alpha[:, k].contiguous(), size=(N, N))
            outs.append(torch.sparse.mm(S, xw[:, k, :].contiguous()))
        out = torch.cat(outs, 1) if concat else torch.stack(outs, 0).mean(0)
        return out + bias

    def _raw_np(self, p: Prepared, ent, r, c):
        """The 12 raw edge features of entries ent = (r, c), computed in float64
        exactly as build_edge_features does, then cast to float32."""
        a = p.g.A.data[ent]
        neg = np.maximum(-a, 0.0)
        rmn, d = p.rs["rowmaxneg"], p.rs["diag"]
        rmn = np.where(rmn > 0, rmn, 1.0)          # as build_edge_features
        ri, rj = p.R[r], p.R[c]
        dab = np.abs(ri - rj)
        f = np.hstack([(a < 0).astype(np.float64)[:, None],
                       np.column_stack([neg / rmn[r], neg / rmn[c]]),
                       (np.abs(a) / np.sqrt(np.abs(d[r] * d[c]) + EPS))[:, None],
                       dab, dab / (np.maximum(np.abs(ri), np.abs(rj)) + EPS)])
        return torch.from_numpy(f.astype(np.float32))

    def _raw_t(self, p: Prepared, ent, r, c, cache):
        """Same features computed on the device (float64 then float32)."""
        if "Ad" not in cache:
            dev = self.dev
            cache["Ad"] = torch.from_numpy(p.g.A.data).to(dev)
            cache["d"] = torch.from_numpy(p.rs["diag"]).to(dev)
            rmn = p.rs["rowmaxneg"]
            cache["rmn"] = torch.from_numpy(np.where(rmn > 0, rmn, 1.0)).to(dev)
            cache["R"] = torch.from_numpy(np.ascontiguousarray(p.R)).to(dev)
        a = cache["Ad"][ent]
        neg = torch.clamp(-a, min=0.0)
        rr, rc = cache["R"][r], cache["R"][c]
        dab = (rr - rc).abs()
        f = torch.cat([(a < 0).double()[:, None],
                       (neg / cache["rmn"][r])[:, None],
                       (neg / cache["rmn"][c])[:, None],
                       (a.abs() / torch.sqrt((cache["d"][r] * cache["d"][c]).abs() + EPS))[:, None],
                       dab, dab / (torch.maximum(rr.abs(), rc.abs()) + EPS)], 1)
        return f.float()

    @torch.no_grad()
    def logits(self, p: Prepared):
        """Edge logits for every pair, as (upper (i,j), lower (j,i)) halves."""
        g, dev = p.g, self.dev
        h = F.elu(self._gat(p.x.to(dev), g, *self.c1))
        h = self._gat(h, g, *self.c2)
        h = F.layer_norm(h, (h.shape[1],), *self.ln)
        A_, B_ = h @ self.Wa, h @ self.Wb
        H = h.shape[1]
        n, ch = len(g.np_upper), self.chunk
        out_u = torch.empty(n, device=dev)
        out_l = torch.empty(n, device=dev)
        if self.fused:
            RAW = self._raw_np(p, slice(None), g.np_row, g.np_indices).to(dev)
        cache = {}
        X = torch.empty(min(ch, n), 2 * H, device=dev)
        for s in range(0, n, ch):
            e = min(s + ch, n)
            m = e - s
            i, j = g.ui[s:e], g.uj[s:e]
            hi, hj = h.index_select(0, i), h.index_select(0, j)
            Xm = X[:m]
            torch.sub(hi, hj, out=Xm[:, :H]).abs_()
            torch.mul(hi, hj, out=Xm[:, H:])
            S = torch.addmm(self.b1, Xm, self.Wh)          # shared by (i,j) and (j,i)
            for p_, q_, ent, out in ((i, j, g.up[s:e], out_u), (j, i, g.lo[s:e], out_l)):
                z = S + A_.index_select(0, p_)
                z += B_.index_select(0, q_)
                raw = RAW.index_select(0, ent) if self.fused else self._raw_t(p, ent, p_, q_, cache)
                z.addmm_(raw, self.Wr)
                y = z.relu_()
                for k, (W, b) in enumerate(self.rest):
                    y = torch.addmm(b, y, W)
                    if k < len(self.rest) - 1:
                        y.relu_()
                out[s:e] = y.squeeze(-1)
        return out_u, out_l

    # ------------------------------------------------------------ selection
    @torch.no_grad()
    def directed_selection(self, p: Prepared) -> np.ndarray:
        """Selected directed edges, as a bool mask over A's canonical entries
        (same rule as select_edges_by_threshold: z-score over off-diagonal
        logits, z > threshold, diagonal never selected)."""
        lu, ll = self.logits(p)
        allv = torch.cat([lu, ll])
        mu, sd = allv.mean(), allv.std()
        su = ((lu - mu) / (sd + 1e-6) > self.threshold).cpu().numpy()
        sl = ((ll - mu) / (sd + 1e-6) > self.threshold).cpu().numpy()
        sel = np.zeros(len(p.g.np_indices), dtype=bool)
        sel[p.g.np_upper] = su
        sel[p.g.np_lower] = sl
        return sel

    def edge_mask(self, p: Prepared, edge_index: torch.Tensor) -> np.ndarray:
        """Directed selection in edge_index order (what workers pass to
        C_from_selected_edges)."""
        return self.directed_selection(p)[p.g.positions(edge_index)]

    def strength_from_selection(self, p: Prepared, sel: np.ndarray) -> sp.csr_matrix:
        """C = max(S, S^T) + I on A's pattern, as C_from_selected_edges."""
        g = p.g
        keep = sel.copy()
        keep[g.np_upper] |= sel[g.np_lower]
        keep[g.np_lower] |= sel[g.np_upper]
        keep |= ~g.np_off
        counts = np.add.reduceat(keep.astype(np.int64), g.np_indptr[:-1])
        indptr = np.concatenate([[0], np.cumsum(counts)])
        return sp.csr_matrix((np.ones(int(counts.sum())), g.np_indices[keep], indptr),
                             shape=g.A.shape)

    def strength(self, p: Prepared) -> sp.csr_matrix:
        return self.strength_from_selection(p, self.directed_selection(p))
