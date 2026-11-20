"""Per-row threshold strength graphs (sub-study "coarse"): edge arrays, row features, the learned linear
per-row threshold model ('lin'), the distilled two-class hand rule ('two'), separable CMA-ES, and the
measured-time evaluator used by the oracle search and by train_lin.

Learned model (lin):  theta_i = sigmoid(w0 + w . z(phi_i)),  phi = per-row features standardised per
matrix; base 'theta': keep (i,j) if s_ij = -a_ij / max_k(-a_ik) >= theta_i; base 'symm': keep if
-a_ij >= (theta_i / 2) sqrt(a_ii a_jj). C = max(S, S^T) + I.
"""
import time
import json

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from . import common  # noqa: F401  (puts the repository root on sys.path)
from amg_core import build_pyamg_solver, vcycle_preconditioner, pcg_iteration_count

LIN_NAMES = ["s2", "dom", "deg", "nstrong5", "logdiag", "jump", "posfrac"]


class Mat:
    """Edge arrays of A in CSR order, for building strength graphs fast."""
    def __init__(self, A, coords=None):
        A = sp.csr_matrix(A)
        A.sum_duplicates()
        A.sort_indices()
        self.A = A
        self.n = A.shape[0]
        self.coords = coords
        self.r = np.repeat(np.arange(self.n), np.diff(A.indptr))
        self.c = A.indices
        self.off = self.r != self.c
        self.neg = np.where(self.off, np.maximum(-A.data, 0.0), 0.0)
        self.rowmax = np.maximum.reduceat(self.neg, A.indptr[:-1])   # every row has its diagonal entry
        self.s = self.neg / np.where(self.rowmax[self.r] > 0, self.rowmax[self.r], 1.0)  # row-relative, float64
        self.diag = A.diagonal()

    def C_from_sel(self, sel):
        sel = sel & self.off
        S = sp.csr_matrix((sel.astype(np.float64), self.A.indices.copy(), self.A.indptr.copy()), shape=self.A.shape)
        S.eliminate_zeros()
        C = S + S.T
        C.data[:] = 1.0
        C = C + sp.eye(self.n, format="csr")
        C.data[:] = 1.0
        return C.tocsr()

    def sel_theta(self, theta_row):
        """classical row-relative rule with per-row theta: -a_ij >= theta_i * max_k(-a_ik)."""
        th = theta_row if np.ndim(theta_row) == 0 else theta_row[self.r]
        return self.off & (self.neg > 0) & (self.s >= th - 1e-12)

    def sel_symm(self, theta):
        """pyamg symmetric rule |a_ij| >= theta sqrt(|a_ii a_jj|) restricted to negative couplings."""
        d = np.abs(self.diag)
        return self.off & (self.neg > 0) & (self.neg >= theta * np.sqrt(d[self.r] * d[self.c]))

    def sel_symm_rows(self, th):
        """per-row symmetric threshold th_i (array over rows)."""
        d = np.abs(self.diag)
        return self.off & (self.neg > 0) & (self.neg >= th[self.r] * np.sqrt(d[self.r] * d[self.c]))

    def row_rank(self):
        if not hasattr(self, "_rank"):
            key = np.where(self.off, -self.neg, np.inf)
            order = np.lexsort((key, self.r))
            rank = np.empty(len(order), dtype=np.int64)
            pos = np.arange(len(order)) - np.repeat(self.A.indptr[:-1], np.diff(self.A.indptr))
            rank[order] = pos + 1
            self._rank = rank
        return self._rank

    def sel_topk(self, k):
        return self.off & (self.neg > 0) & (self.row_rank() <= k)


def solve(A, C, t_rule=0.0, maxiter=500, extra=False):
    """two-level SA from predefined C (amg_core.build_pyamg_solver; splu factorised lazily inside the
    solve), V-cycle PCG to 1e-8. Returns iterations, times and coarse statistics."""
    t0 = time.perf_counter()
    ml = build_pyamg_solver(A, C, None)
    t1 = time.perf_counter()
    M = vcycle_preconditioner(ml)
    it = pcg_iteration_count(A, M, tol=1e-8, maxiter=maxiter)
    t2 = time.perf_counter()
    Ac = ml.levels[-1].A
    d = {"iters": int(it), "setup": t1 - t0, "solve": t2 - t1, "rule": t_rule, "total": t_rule + (t1 - t0) + (t2 - t1),
         "nc": int(Ac.shape[0]), "nnzc": int(Ac.nnz), "oc": (A.nnz + Ac.nnz) / A.nnz,
         "agg": A.shape[0] / Ac.shape[0], "Pnnz": int(ml.levels[0].P.nnz)}
    if extra:
        try:
            lu = spla.splu(sp.csc_matrix(Ac))
            d["lu_nnz"] = int(lu.L.nnz + lu.U.nnz)
            d["lu_ratio"] = d["lu_nnz"] / A.nnz
        except Exception:
            d["lu_nnz"] = -1
    return d


def row_features(m):
    """per-row features for the oracle parameterisations (sort-based reference implementation)."""
    A = m.A
    n = m.n
    r, c, off = m.r, m.c, m.off
    s = m.s
    s2 = np.zeros(n)
    s_off = np.where(off, s, -1.0)
    order = np.lexsort((-s_off, r))
    pos = np.arange(len(order)) - np.repeat(A.indptr[:-1], np.diff(A.indptr))
    sec = order[pos == 1]
    s2[r[sec]] = s_off[sec]
    deg = np.diff(A.indptr) - 1
    nstrong5 = np.bincount(r, weights=(off & (s >= 0.5)).astype(float), minlength=n)
    ld = np.log(np.abs(m.diag) + 1e-300)
    jump = np.zeros(n)
    np.maximum.at(jump, r, np.where(off, np.abs(ld[r] - ld[c]), 0.0))
    rowsum = np.bincount(r, weights=m.neg, minlength=n)
    dom = m.rowmax / np.maximum(rowsum, 1e-300)
    pos_frac = np.bincount(r, weights=np.where(off, np.maximum(A.data, 0), 0), minlength=n) / np.maximum(rowsum, 1e-300)
    F = {"s2": s2, "dom": dom, "deg": deg.astype(float), "nstrong5": nstrong5, "logdiag": ld, "jump": jump,
         "posfrac": pos_frac}
    if m.coords is not None:
        F["x"] = m.coords[:, 0]
        F["y"] = m.coords[:, 1]
    return F


def row_features_fast(m, nb=()):
    """same values as row_features (except coords), with segment reductions; nb adds neighbour means."""
    A = m.A
    r, c, off = m.r, m.c, m.off
    ip = A.indptr[:-1]
    s_off = np.where(off, m.s, -1.0)
    m1 = np.maximum.reduceat(s_off, ip)
    ism = s_off == m1[r]
    cnt = np.add.reduceat(ism.astype(np.int64), ip)
    s2a = np.maximum.reduceat(np.where(ism, -1.0, s_off), ip)
    s2 = np.where(cnt >= 2, m1, s2a)
    s2 = np.where(np.diff(A.indptr) - 1 >= 1, s2, 0.0)
    deg = (np.diff(A.indptr) - 1).astype(float)
    nstrong5 = np.add.reduceat((off & (m.s >= 0.5)).astype(float), ip)
    ld = np.log(np.abs(m.diag) + 1e-300)
    jump = np.maximum.reduceat(np.where(off, np.abs(ld[r] - ld[c]), 0.0), ip)
    rowsum = np.add.reduceat(m.neg, ip)
    dom = m.rowmax / np.maximum(rowsum, 1e-300)
    pos_frac = np.add.reduceat(np.where(off, np.maximum(A.data, 0), 0), ip) / np.maximum(rowsum, 1e-300)
    F = {"s2": s2, "dom": dom, "deg": deg, "nstrong5": nstrong5, "logdiag": ld, "jump": jump, "posfrac": pos_frac}
    if nb:
        dg = np.maximum(deg, 1.0)
        for k in nb:
            F[k + "_nb"] = np.add.reduceat(np.where(off, F[k][c], 0.0), ip) / dg
    return F


def standardised_phi(m, names, nb=()):
    F = row_features_fast(m, tuple(nb))
    phi = np.column_stack([F[k] for k in names])
    return (phi - phi.mean(0)) / (phi.std(0) + 1e-12)


def sel_lin(m, phi, base, x):
    """directed strong-edge mask of the linear per-row threshold model."""
    z = x[0] + phi @ x[1:]
    if base == "theta":
        return m.sel_theta(sig(z))
    return m.sel_symm_rows(0.5 * sig(z))


def load_lin(path):
    J = json.load(open(path))
    return {"x": np.array(J["x"]), "base": J["base"], "names": J["names"], "nb": tuple(J.get("nb", ()))}


def lin_C(A, model):
    """learned per-row threshold C with the harness' timing split (feat / fwd / C)."""
    t0 = time.perf_counter()
    m = Mat(A)
    phi = standardised_phi(m, model["names"], model["nb"])
    t1 = time.perf_counter()
    sel = sel_lin(m, phi, model["base"], model["x"])
    t2 = time.perf_counter()
    C = m.C_from_sel(sel)
    t3 = time.perf_counter()
    return C, {"t_feat": t1 - t0, "t_fwd": t2 - t1, "t_C": t3 - t2, "keep": float(sel[m.off].mean())}


def two_C(A, th_hard, th_bulk, jt, pt):
    """hand rule distilled from the learned perforated-L model: theta_hard where jump > jt or
    posfrac > pt (coefficient interfaces, anisotropic corner), theta_bulk elsewhere."""
    t0 = time.perf_counter()
    m = Mat(A)
    F = row_features_fast(m)
    hard = (F["jump"] > jt) | (F["posfrac"] > pt)
    sel = m.sel_theta(np.where(hard, th_hard, th_bulk))
    t1 = time.perf_counter()
    C = m.C_from_sel(sel)
    t2 = time.perf_counter()
    return C, {"t_feat": t1 - t0, "t_fwd": 0.0, "t_C": t2 - t1, "keep": float(sel[m.off].mean())}


def mask_C(A, sel):
    """precomputed (oracle) directed strong-edge mask."""
    t0 = time.perf_counter()
    m = Mat(A)
    C = m.C_from_sel(sel)
    t1 = time.perf_counter()
    return C, {"t_feat": 0.0, "t_fwd": 0.0, "t_C": t1 - t0, "keep": float(sel[m.off].mean())}


class SepCMA:
    """minimal separable CMA-ES (Ros & Hansen 2008), minimisation."""
    def __init__(self, x0, sigma0, popsize=None, seed=0):
        self.N = N = len(x0)
        self.m = np.array(x0, float)
        self.sigma = sigma0
        self.lam = popsize or 4 + int(3 * np.log(N))
        self.mu = self.lam // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1 / np.sum(self.w ** 2)
        self.cc = 4 / (N + 4)
        self.cs = (self.mueff + 2) / (N + self.mueff + 3)
        self.c1 = 2 / ((N + 1.3) ** 2 + self.mueff) * (N + 2) / 3
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((N + 2) ** 2 + self.mueff) * (N + 2) / 3)
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (N + 1)) - 1) + self.cs
        self.pc = np.zeros(N)
        self.ps = np.zeros(N)
        self.D = np.ones(N)
        self.chiN = np.sqrt(N) * (1 - 1 / (4 * N) + 1 / (21 * N ** 2))
        self.rng = np.random.default_rng(seed)
        self.g = 0

    def ask(self):
        self.z = self.rng.standard_normal((self.lam, self.N))
        return self.m + self.sigma * self.z * self.D

    def tell(self, X, f):
        idx = np.argsort(f)[: self.mu]
        zs = self.z[idx]
        xs = X[idx]
        old = self.m.copy()
        self.m = self.w @ xs
        zw = self.w @ zs
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * zw
        self.g += 1
        hs = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * self.g)) < (1.4 + 2 / (self.N + 1)) * self.chiN
        y = (self.m - old) / self.sigma
        self.pc = (1 - self.cc) * self.pc + hs * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * y
        Dz2 = (self.D * zs) ** 2
        C = self.D ** 2
        C = (1 - self.c1 - self.cmu) * C + self.c1 * self.pc ** 2 + self.cmu * (self.w @ Dz2)
        self.D = np.sqrt(np.maximum(C, 1e-20))
        self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))


def sig(x):
    return 1 / (1 + np.exp(-x))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))
