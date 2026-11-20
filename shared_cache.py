"""
shared_cache.py — Zero-copy shared-memory cache for SpectralTrainSample lists.

Instead of each worker independently pickle-loading the full training set
(30 workers × 3 GB = 90 GB), all samples are packed into a single POSIX
shared-memory block.  Workers open the block by name and reconstruct
SpectralTrainSample objects whose numpy/torch arrays are *views* into
the shared region — no copies, no extra RAM per worker.

Typical savings at grid_n=50 000, 64 samples, 30 workers:
    Before:  30 × 3 GB  =  ~90 GB  (pickle per worker)
    After:    1 × 3 GB  =   ~3 GB  (shared)

Usage (main process):
    shm_meta, shm_block = create_shared_cache(train_set)
    ...
    cleanup_shared_cache(shm_block)

Usage (worker init):
    samples, shm_handle = load_from_shared_cache(shm_meta)
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from dataclasses import dataclass, fields, field
from multiprocessing.shared_memory import SharedMemory
from typing import List, Dict, Any, Optional, Tuple

# Alignment for each sub-array inside the shared block (bytes).
# 64-byte alignment plays nicely with cache lines and AVX-512.
_ALIGN = 64


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def create_shared_cache(
    train_set,
    *,
    verbose: bool = True,
) -> Tuple[dict, SharedMemory]:
    """Pack *train_set* into a single SharedMemory block.

    Returns
    -------
    shm_meta : dict
        Lightweight metadata (names, shapes, dtypes, offsets) that workers
        need to reconstruct the samples.  Safe to pickle via Pool initargs.
    shm_block : SharedMemory
        The underlying shared-memory object.  The **caller must keep this
        alive** until all workers are done, then call
        ``cleanup_shared_cache(shm_block)``.
    """
    # -- First pass: collect every array, record shapes/dtypes/sizes ------
    entries: List[Tuple[str, np.ndarray]] = []  # (key, array)
    sample_metas: List[dict] = []

    for i, sample in enumerate(train_set):
        sm: Dict[str, Any] = {
            "grid_n": sample.grid_n,
            "tag": sample.tag,
            "A_shape": tuple(sample.A.shape),
            "n_row_groups": len(sample.row_groups),
            "has_raw_ef": sample.raw_edge_features is not None,
            "has_full_pts": sample.full_pts is not None,
            "has_full_simplices": sample.full_simplices is not None,
            "feature_seed": getattr(sample, "feature_seed", None),
        }
        sample_metas.append(sm)

        # CSR components
        entries.append((f"{i}/A_data",    np.ascontiguousarray(sample.A.data)))
        entries.append((f"{i}/A_indices", np.ascontiguousarray(sample.A.indices)))
        entries.append((f"{i}/A_indptr",  np.ascontiguousarray(sample.A.indptr)))

        # Plain numpy
        entries.append((f"{i}/coords", np.ascontiguousarray(sample.coords)))

        # Torch → numpy
        entries.append((f"{i}/edge_index",  np.ascontiguousarray(sample.edge_index.numpy())))
        entries.append((f"{i}/edge_weight", np.ascontiguousarray(sample.edge_weight.numpy())))
        entries.append((f"{i}/x",           np.ascontiguousarray(sample.x.numpy())))

        # row_groups: flatten into (flat, offsets)
        if sample.row_groups:
            flat_np = torch.cat(sample.row_groups).numpy()
            lens = [len(g) for g in sample.row_groups]
            offsets = np.empty(len(lens) + 1, dtype=np.int64)
            offsets[0] = 0
            np.cumsum(lens, out=offsets[1:])
        else:
            flat_np = np.array([], dtype=np.int64)
            offsets = np.array([0], dtype=np.int64)
        entries.append((f"{i}/rg_flat",    np.ascontiguousarray(flat_np)))
        entries.append((f"{i}/rg_offsets", np.ascontiguousarray(offsets)))

        # Optional fields
        if sample.raw_edge_features is not None:
            entries.append((f"{i}/raw_ef",
                            np.ascontiguousarray(sample.raw_edge_features.numpy())))
        if sample.full_pts is not None:
            entries.append((f"{i}/full_pts",
                            np.ascontiguousarray(sample.full_pts)))
        if sample.full_simplices is not None:
            entries.append((f"{i}/full_simplices",
                            np.ascontiguousarray(sample.full_simplices)))

    # -- Compute packed layout with alignment ----------------------------
    layout: Dict[str, dict] = {}
    offset = 0
    for key, arr in entries:
        # Align to _ALIGN boundary
        offset = (offset + _ALIGN - 1) & ~(_ALIGN - 1)
        nbytes = arr.nbytes if arr.nbytes > 0 else 0
        layout[key] = {
            "offset": offset,
            "shape": tuple(arr.shape),
            "dtype": str(arr.dtype),
        }
        offset += nbytes

    total_size = max(offset, 1)  # SharedMemory requires size > 0

    # -- Allocate and populate -------------------------------------------
    shm = SharedMemory(create=True, size=total_size)
    for key, arr in entries:
        if arr.nbytes == 0:
            continue
        info = layout[key]
        dest = np.ndarray(arr.shape, dtype=arr.dtype,
                          buffer=shm.buf, offset=info["offset"])
        np.copyto(dest, arr)

    shm_meta = {
        "shm_name": shm.name,
        "shm_size": total_size,
        "layout": layout,
        "samples": sample_metas,
        "n_samples": len(train_set),
    }

    if verbose:
        mb = total_size / (1024 * 1024)
        print(f"  [shared_cache] Packed {len(train_set)} samples into "
              f"SharedMemory '{shm.name}' ({mb:.1f} MB)")

    return shm_meta, shm


def load_from_shared_cache(
    shm_meta: dict,
) -> Tuple[list, SharedMemory]:
    """Reconstruct samples from a SharedMemory block (zero-copy).

    Returns
    -------
    samples : list
        List of objects with the same attributes as SpectralTrainSample.
        Backed by shared memory — treat as **read-only**.
    shm_handle : SharedMemory
        Must be kept alive (e.g. as a module global) for the lifetime of
        *samples*.
    """
    shm = SharedMemory(name=shm_meta["shm_name"], create=False)
    layout = shm_meta["layout"]

    def _view(key: str) -> np.ndarray:
        info = layout[key]
        shape = tuple(info["shape"])
        dtype = np.dtype(info["dtype"])
        # Handle empty arrays
        if any(s == 0 for s in shape):
            return np.empty(shape, dtype=dtype)
        return np.ndarray(shape, dtype=dtype,
                          buffer=shm.buf, offset=info["offset"])

    samples = []
    for i, sm in enumerate(shm_meta["samples"]):
        # Sparse matrix A (views into shared mem)
        A = sp.csr_matrix(
            (_view(f"{i}/A_data"),
             _view(f"{i}/A_indices"),
             _view(f"{i}/A_indptr")),
            shape=tuple(sm["A_shape"]),
            copy=False,
        )

        coords = _view(f"{i}/coords")

        edge_index  = torch.from_numpy(_view(f"{i}/edge_index"))
        edge_weight = torch.from_numpy(_view(f"{i}/edge_weight"))
        x           = torch.from_numpy(_view(f"{i}/x"))

        # Row groups: reconstruct list of tensor views
        rg_flat    = torch.from_numpy(_view(f"{i}/rg_flat"))
        rg_offsets = _view(f"{i}/rg_offsets")
        row_groups = [rg_flat[int(rg_offsets[j]):int(rg_offsets[j + 1])]
                      for j in range(sm["n_row_groups"])]

        raw_ef = None
        if sm["has_raw_ef"]:
            raw_ef = torch.from_numpy(_view(f"{i}/raw_ef"))

        full_pts = _view(f"{i}/full_pts") if sm["has_full_pts"] else None
        full_simplices = _view(f"{i}/full_simplices") if sm["has_full_simplices"] else None

        samples.append(_SampleView(
            A=A,
            grid_n=sm["grid_n"],
            coords=coords,
            edge_index=edge_index,
            edge_weight=edge_weight,
            x=x,
            row_groups=row_groups,
            tag=sm["tag"],
            raw_edge_features=raw_ef,
            full_pts=full_pts,
            full_simplices=full_simplices,
            feature_seed=sm.get("feature_seed"),
        ))

    return samples, shm


def cleanup_shared_cache(shm_block: SharedMemory) -> None:
    """Close and unlink a SharedMemory block (main process only)."""
    try:
        shm_block.close()
    except Exception:
        pass
    try:
        shm_block.unlink()
    except Exception:
        pass


# ------------------------------------------------------------------
# Lightweight sample container (avoids importing the heavy main module)
# ------------------------------------------------------------------

class _SampleView:
    """Drop-in replacement for SpectralTrainSample whose arrays are views
    into shared memory.  Attribute-compatible with es_evaluate() and all
    other consumers that do ``sample.x``, ``sample.A``, etc."""

    __slots__ = (
        "A", "grid_n", "coords", "edge_index", "edge_weight",
        "x", "row_groups", "tag", "raw_edge_features",
        "full_pts", "full_simplices", "feature_seed",
    )

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
