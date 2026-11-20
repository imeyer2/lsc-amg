"""Node-modulated learned strength policies (sub-study "reward") and the two-level cost model.

NodePolicy: a learned per-node scaling of the classical strength, with 5 cheap node features
phi = (rel, an, c50, dom, ldeg) standardised with training-set statistics (no per-edge features, cost
~ a classical rule):
  row base ('noderow'): keep (i,j) iff -a_ij H_j >= exp(-tau_i) max_k(-a_ik H_k),
                        H = exp(v . z(phi)), tau_i = tau exp(u . z(phi_i))     (v=u=0: classical theta rule)
  abs base ('nodeabs'): keep (i,j) iff |a_ij| G_i G_j >= exp(-tau),
                        G = exp(-0.5 log a_ii + v . z(phi))                    (v=0: pyamg symmetric rule)
theta = [v (m), u (m, row base only), log tau]; C = max(S, S^T) + I.

Cost model (fit by twolevel.costmodel on classical sweeps):
  seconds = setup + LU factorisation + iters * per-iteration
  setup = s_A nnz(A) + s_P nnz(P) + s_c nnz(A_c);  LU = f_flop flops(LU) + f_nnz nnz(LU)
  per-iteration = i_A nnz(A) + i_P nnz(P) + i_lu nnz(LU) + i_n n
"""
import math
import time
import json

import numpy as np
import scipy.sparse as sp

from . import common  # noqa: F401
from .rules import C_from_keep, classical_keep, offdiag
from amg_core import build_pyamg_solver, vcycle_preconditioner, pcg_iteration_count

TINY = 1e-300
NODE_NAMES = ["rel", "an", "c50", "dom", "ldeg"]
SPLU_SYM = ("splu", {"permc_spec": "MMD_AT_PLUS_A", "diag_pivot_thresh": 0.0, "options": {"SymmetricMode": True}})


class CSRG:
    __slots__ = ("A", "n", "ip", "c", "a", "r", "off", "starts")

    def __init__(self, A):
        self.A = A
        self.n = A.shape[0]
        self.ip = A.indptr
        self.c = A.indices
        self.a = A.data
        self.r = np.repeat(np.arange(self.n, dtype=np.int32), np.diff(self.ip))
        self.off = self.c != self.r
        self.starts = self.ip[:-1]


def node_features(g, names):
    """per-node features (n-sized; one pass over the entries for an/c50/dom)."""
    a, r, off = g.a, g.r, g.off
    out = {}
    d = np.abs(g.A.diagonal()) + TINY
    neg = np.where(off, -a, 0.0)
    np.maximum(neg, 0.0, out=neg)
    rmax = np.maximum.reduceat(neg, g.starts)
    out["_d"] = d
    out["_neg"] = neg
    out["_rmax"] = rmax
    need = set(names)
    if "rel" in need:
        out["rel"] = np.log(rmax + TINY) - np.log(d)                                  # row max / diagonal
    if "an" in need:
        sec = np.maximum.reduceat(np.where(neg < rmax[r], neg, 0.0), g.starts)
        out["an"] = np.maximum(np.log(sec + TINY) - np.log(rmax + TINY), -30.0)       # 2nd / max
    if "c50" in need:
        out["c50"] = np.log1p(np.add.reduceat(neg >= 0.5 * rmax[r], g.starts).astype(np.float64) - 0.0)
    if "dom" in need:
        out["dom"] = np.log(d) - np.log(np.add.reduceat(np.abs(np.where(off, a, 0.0)), g.starts) + TINY)
    if "ldeg" in need:
        out["ldeg"] = np.log(np.diff(g.ip).astype(np.float64))
    return out


class NodePolicy:
    kind = "node"

    def __init__(self, names, mu, sd, base="row"):
        self.names = list(names)
        self.mu = np.asarray(mu, float)
        self.sd = np.asarray(sd, float)
        self.base = base

    @property
    def n_params(self):
        return len(self.names) * (2 if self.base == "row" else 1) + 1

    def init_theta(self, tau):
        return np.concatenate([np.zeros(self.n_params - 1), [math.log(max(tau, 1e-6))]])

    def needed(self, th):
        m = len(self.names)
        v = th[:m]
        u = th[m:2 * m] if self.base == "row" else np.zeros(m)
        return [self.names[k] for k in range(m) if v[k] != 0 or u[k] != 0]

    def mask(self, th, g, feats=None):
        m = len(self.names)
        v = th[:m]
        u = th[m:2 * m] if self.base == "row" else np.zeros(m)
        tau = math.exp(th[-1])
        N = feats if feats is not None else node_features(g, self.needed(th))
        lin_v = None
        lin_u = None
        for k in range(m):
            if v[k] != 0:
                t = (v[k] / self.sd[k]) * (N[self.names[k]] - self.mu[k])
                lin_v = t if lin_v is None else lin_v + t
            if u[k] != 0:
                t = (u[k] / self.sd[k]) * (N[self.names[k]] - self.mu[k])
                lin_u = t if lin_u is None else lin_u + t
        r, c, off = g.r, g.c, g.off
        if self.base == "abs":
            lg = -0.5 * np.log(N["_d"]) + (lin_v if lin_v is not None else 0.0)
            G = np.exp(lg)
            s = np.abs(g.a) * G[r]
            s *= G[c]
            return (s >= math.exp(-tau)) & off & (g.a != 0)
        neg = N["_neg"]
        if lin_v is not None:
            mm = neg * np.exp(lin_v)[c]
            mx = np.maximum.reduceat(mm, g.starts)
        else:
            mm = neg
            mx = N["_rmax"]
        thr = np.exp(-tau * np.exp(lin_u)) if lin_u is not None else math.exp(-tau)
        thr_e = (mx * thr)[r] if lin_u is not None else mx[r] * thr
        return (mm >= thr_e) & (neg > 0)

    def to_json(self):
        return {"kind": "node", "names": self.names, "mu": self.mu.tolist(), "sd": self.sd.tolist(), "base": self.base}

    @staticmethod
    def from_json(d):
        return NodePolicy(d["names"], d["mu"], d["sd"], d.get("base", "row"))


def node_stats(As, names):
    cols = {k: [] for k in names}
    for A in As:
        N = node_features(CSRG(A), names)
        for k in names:
            cols[k].append(N[k])
    return ([float(np.mean(np.concatenate(cols[k]))) for k in names],
            [float(np.std(np.concatenate(cols[k]))) + 1e-6 for k in names])


def load_node(path):
    """(policy, theta) from a train.json written by twolevel.train_node (best validation theta)."""
    J = json.load(open(path))
    if J["policy"].get("kind") != "node":
        raise ValueError(f"{path}: only node policies are supported (kind={J['policy'].get('kind')})")
    return NodePolicy.from_json(J["policy"]), np.array(J["best"]["theta"])


def node_C(A, pol, th):
    """learned node-policy C with the harness' timing split (feat / fwd / C)."""
    t0 = time.perf_counter()
    g = CSRG(A)
    feats = node_features(g, pol.needed(th))
    t1 = time.perf_counter()
    keep = pol.mask(th, g, feats)
    t2 = time.perf_counter()
    C = C_from_keep(A, keep)
    t3 = time.perf_counter()
    return C, {"t_feat": t1 - t0, "t_fwd": t2 - t1, "t_C": t3 - t2, "keep": float(keep[g.off].mean())}


# ----------------------------------------------------------------------------- cost-model evaluation
def sweep_keep(A, rule, p):
    """classical rules as used for the cost-model sweeps (sym restricted to negative couplings)."""
    if rule == "sym":
        row, off = offdiag(A)
        neg = np.where(off, np.maximum(-A.data, 0.0), 0.0)
        d = np.abs(A.diagonal())
        return off & (neg >= p * np.sqrt(d[row] * d[A.indices])) & (neg > 0)
    return classical_keep(A, rule, p)


def lu_stats(LU):
    L, U = sp.csc_matrix(LU.L), sp.csc_matrix(LU.U)
    lc = np.diff(L.indptr)                               # column counts of L (unit diag included)
    ur = np.bincount(U.indices, minlength=U.shape[0])    # row counts of U
    flops = float(np.sum(lc.astype(np.float64) * ur))
    return int(L.nnz + U.nnz), flops


def two_level(A, C, splu="default", want_lu_flops=True, maxiter=500):
    """build 2L SA from predefined C, factor the coarse LU, PCG. Returns statistics + times (s)."""
    t0 = time.perf_counter()
    try:
        if splu == "sym":
            ml = build_pyamg_solver(A, C, None, coarse_solver=SPLU_SYM)
        else:
            ml = build_pyamg_solver(A, C, None)
    except Exception as e:
        return {"iters": maxiter, "fail": str(e)[:80]}
    t1 = time.perf_counter()
    Ac = ml.levels[-1].A
    ml.coarse_solver(Ac, np.zeros(Ac.shape[0]))          # triggers the (lazy) splu factorisation
    t2 = time.perf_counter()
    M = vcycle_preconditioner(ml)
    it = pcg_iteration_count(A, M, tol=1e-8, maxiter=maxiter)
    t3 = time.perf_counter()
    lv = ml.levels
    d = {"iters": int(it), "n": int(A.shape[0]), "nnzA": int(A.nnz), "n_c": int(Ac.shape[0]), "nnz_c": int(Ac.nnz),
         "nnzP": int(lv[0].P.nnz), "oc": (A.nnz + Ac.nnz) / A.nnz, "n_levels": len(lv),
         "setup_s": t1 - t0, "lu_s": t2 - t1, "solve_s": t3 - t2}
    LU = getattr(ml.coarse_solver, "LU", None)
    if LU is not None:
        d["lu_nnz"] = int(LU.nnz)
        if want_lu_flops:
            try:
                d["lu_nnz"], d["lu_flops"] = lu_stats(LU)
            except Exception:
                pass
    return d


def cost_units(d, cm):
    """cost model: predicted seconds = setup + lu_factor + iters * per_iter (coefficients cm)."""
    setup = cm["s_A"] * d["nnzA"] + cm["s_P"] * d["nnzP"] + cm["s_c"] * d["nnz_c"]
    fac = cm["f_flop"] * d.get("lu_flops", 0.0) + cm["f_nnz"] * d.get("lu_nnz", 0)
    per = cm["i_A"] * d["nnzA"] + cm["i_P"] * d["nnzP"] + cm["i_lu"] * d.get("lu_nnz", 0) + cm["i_n"] * d["n"]
    return setup + fac + d["iters"] * per
