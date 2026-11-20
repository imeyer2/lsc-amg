"""Classical strength-of-connection rules for the predefined two-level C.

  theta:<t>   classical row-relative   -a_ij >= t * max_k(-a_ik)
  sym:<t>     pyamg 'symmetric'        |a_ij| >= t * sqrt(|a_ii a_jj|)
  topk:<k>    k strongest negative couplings per row
  srowq:<q>   global top-q fraction of s_ij = -a_ij / max_k(-a_ik)

Every rule returns a directed mask over A's stored entries; C = max(S, S^T) + I (binary),
as amg_core.C_from_selected_edges.
"""
import numpy as np
import scipy.sparse as sp


def offdiag(A):
    n = A.shape[0]
    row = np.repeat(np.arange(n), np.diff(A.indptr))
    return row, A.indices != row


def classical_keep(A, kind, p):
    """bool mask over A's stored entries: directed strong connections (not yet symmetrised)."""
    row, off = offdiag(A)
    a = A.data
    if kind == "sym":
        d = np.abs(A.diagonal())
        return off & (np.abs(a) >= p * np.sqrt(d[row] * d[A.indices])) & (a != 0)
    neg = np.where(off, np.maximum(-a, 0.0), 0.0)
    nz = np.diff(A.indptr) > 0
    rmax = np.zeros(A.shape[0])
    rmax[nz] = np.maximum.reduceat(neg, A.indptr[:-1][nz])
    pos = off & (neg > 0)
    if kind == "theta":
        return pos & (neg >= p * rmax[row])
    if kind == "topk":
        # per-row descending rank of neg via a padded (n x maxdeg) argsort (rows are short)
        k = int(p)
        deg = np.diff(A.indptr)
        D = int(deg.max())
        if k >= D:
            return pos
        slot = np.arange(len(neg)) - A.indptr[row]
        Pd = np.full((A.shape[0], D), -1.0)
        Pd[row, slot] = neg
        order = np.argsort(-Pd, axis=1, kind="stable")
        R = np.empty_like(order)
        np.put_along_axis(R, order, np.arange(D)[None, :].repeat(A.shape[0], 0), axis=1)
        return pos & (R[row, slot] < k)
    if kind == "srowq":
        s = neg / np.where(rmax > 0, rmax, 1.0)[row]
        if p >= 1:
            return pos
        thr = np.quantile(s[off], 1 - p)
        return pos & (s >= thr)
    raise ValueError(kind)


def C_from_keep(A, keep):
    """C = max(S, S^T) + I (binary), as amg_core.C_from_selected_edges."""
    n = A.shape[0]
    S = sp.csr_matrix((keep.astype(np.float64), A.indices.copy(), A.indptr.copy()), shape=A.shape)
    S.eliminate_zeros()
    C = (S + S.T + sp.eye(n, format="csr")).tocsr()
    C.data[:] = 1.0
    return C
