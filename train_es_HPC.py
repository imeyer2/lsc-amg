#!/usr/bin/env python3
"""
train_es_HPC.py — HPC-optimized Evolution Strategies training for AMG edge selection.

Fork of train_es.py with three key improvements for SLURM clusters:
  1. Generate meshes ONCE in main process, share via file cache to workers
     (eliminates 32× redundant FEM assembly — saves ~40 min init)
  2. Round-robin work distribution across cores (balanced load)
  3. SLURM-aware: auto-detect resources, checkpoint resume, SIGTERM handling

Usage (local):
  python train_es_HPC.py --rl_epochs 500 --train_samples 64 --es_pop_size 200 \\
      --n_workers 8 --es_mini_batch 16

Usage (SLURM):
  # Print a ready-to-use sbatch script:
  python train_es_HPC.py --print_slurm

  # Or submit directly:
  sbatch job_es.sh
"""

import os
import sys
import json
import time
import signal
import copy
import pickle
import tempfile
import argparse
import platform
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# Must be set BEFORE importing numpy/scipy/torch so that spawned
# worker processes inherit single-threaded BLAS.  Without this,
# N workers × K default threads = N*K threads on N cores → thrashing.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import pyamg
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from tqdm import tqdm
from torch.nn.utils import parameters_to_vector, vector_to_parameters

# Local imports
from models import AMGEdgePolicy
from amg_core import (
    compute_device,
    FeatureConfig,
    sample_topk_without_replacement,
    select_edges_by_threshold,
    C_from_selected_edges,
    build_pyamg_solver,
    build_B_for_pyamg,
    pcg_iterations,
    pcg_iteration_count,
    pcg_iteration_count_stag,
    vcycle_preconditioner,
)
from multiprocessing.shared_memory import SharedMemory

# Exact speedups (same rewards, bit-identical training history; docs/METHODS.md).
# LSC_SPEEDUPS is a comma list, default all of them; LSC_SPEEDUPS=none runs
# the original code paths.
#   pcg   count PCG iterations without the per-iteration residual SpMV, and
#         apply the V-cycle directly (vcycle_preconditioner)
#   sched one pool task per (perturbation, sample), handed out dynamically
#   soc   SoC snapshots (forwards only) on the worker pool
#   diag  AMG diagnostics on a dedicated process, overlapped with training
#         (sched also applies to the GPU path: keyed imap_unordered)
#   overlap  the next epoch's population is queued behind the current
#            epoch's eval/test/SoC tasks (its mini-batch and noise are drawn
#            in the original RNG order; skipped when a sigma restart is
#            possible, rolled back if sigma changed anyway)
#   nopickle skip writing train_set_cache.pkl, which nothing reads
#   preload  import torch/scipy/pyamg once in the forkserver so the workers
#            do not all import them from the shared filesystem at once
#   stag     (near-exact; CPU sched path) the centre is evaluated first; a
#            population member's PCG on sample s stops, counting 500, once it
#            has run max(LSC_STAG_MIN, LSC_STAG * centre_iters[s]) iterations
#            and its residual decay over the last LSC_STAG_W iterations
#            predicts more than LSC_STAG_MARGIN * 500 in total (checked every
#            LSC_STAG_STEP). Defaults 3.0 / 100 / 50 / 1.5 / 25. Centre,
#            eval and test solves are never stopped.
#   gpubatch (not default; float-noise level) one batched GPU forward per
#            perturbation for the whole mini-batch
#   multinode (not default; exact on nodes of one CPU type) the population is
#            also evaluated by node agents on the other nodes of the job
#            (multinode.py); needs LSC_MN_PORT / LSC_MN_AUTH and the agents,
#            which scripts/slurm/run_multinode.sh starts
_SPEEDUPS = set(filter(None, os.environ.get(
    "LSC_SPEEDUPS", "pcg,sched,soc,diag,overlap,nopickle,preload,stag")
    .replace("none", "").split(",")))
_TIMING = os.environ.get("LSC_TIMING", "") == "1"
_STAG_FACTOR = float(os.environ.get("LSC_STAG", "3.0"))
_STAG_MIN = int(os.environ.get("LSC_STAG_MIN", "100"))
_STAG_W = int(os.environ.get("LSC_STAG_W", "50"))
_STAG_STEP = int(os.environ.get("LSC_STAG_STEP", "25"))
_STAG_MARGIN = float(os.environ.get("LSC_STAG_MARGIN", "1.5"))
_ES_ROWS_EXTRA = 3   # shared rows after eps: theta, sigma_vec, per-sample stag starts
_MN_PORT = int(os.environ.get("LSC_MN_PORT", "0"))
_MN_AUTH = os.environ.get("LSC_MN_AUTH", "lscamg")
# Reward choice (env so worker processes see it; set by --reward/--reward_k):
#   iters    (default; all released results) R = -log(PCG iters to 1e-8) - λ·log(OC)
#   residual R = (1/K) Σ_t [log(|r_t^SA|+ε) - log(|r_t^learn|+ε)] - λ·log(OC+ε), over K
#            PCG iterations against a default PyAMG SA-AMG baseline (as written in the paper)
_REWARD = os.environ.get("LSC_REWARD", "iters")
_REWARD_K = int(os.environ.get("LSC_REWARD_K", "16"))
_SA_BASE_CACHE = {}
from amg_diagnostics import compute_diagnostic_batch
from shared_cache import create_shared_cache, load_from_shared_cache, cleanup_shared_cache

from problems import make_spectral_train_set, SpectralTrainSample
from fast_policy import FastPolicy

# Feature config for the exact fast inference path (fast_policy.py), or None
# for the reference forward. Set from --fast_policy in main() and in workers.
_FAST_CFG = None


def set_fast_policy(feature_config):
    global _FAST_CFG
    _FAST_CFG = feature_config


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


# ============================================================
# SoC snapshot: record logits + selected mask for every sample
# ============================================================
def compute_soc_snapshot(model, samples, k_per_row, selection_mode="topk"):
    """Forward pass on all samples; return per-sample logits & sel_mask."""
    snapshot = {}
    with torch.no_grad():
        for idx, sample in enumerate(samples):
            logits, B_extra = model(
                sample.x, sample.edge_index, sample.edge_weight,
                raw_edge_features=sample.raw_edge_features,
            )
            if selection_mode == "threshold":
                threshold = model.threshold.item() if hasattr(model, 'threshold') else 0.0
                sel_mask, _ = select_edges_by_threshold(
                    logits=logits, edge_index=sample.edge_index,
                    threshold=threshold,
                )
            else:
                sel_mask, _ = sample_topk_without_replacement(
                    logits=logits, row_groups=sample.row_groups,
                    k=k_per_row, temperature=1.0, gumbel=False,
                )
            snapshot[idx] = {
                "logits": logits.cpu().numpy().astype(np.float32),
                "sel_mask": sel_mask.cpu().numpy().astype(bool),
            }
    return snapshot


# ============================================================
# Greedy evaluation: model weights → PCG iters → reward
# ============================================================
def es_evaluate(model, sample, k_per_row=None, coarse_solver="splu",
                selection_mode="topk", reward_lambda_cx=0.1, stag_start=None):
    """Deterministic greedy evaluation. Returns (reward, pcg_iters).

    Args:
        selection_mode: "topk" (fixed k_per_row) or "threshold" (adaptive-K).
        reward_lambda_cx: Weight for complexity penalty in reward.
    """
    if (selection_mode == "threshold" and _FAST_CFG is not None
            and getattr(sample, "feature_seed", None) is not None
            and FastPolicy.supported(model, _FAST_CFG)):
        # Same selected edges as the reference forward below (fast_policy.py).
        fast = FastPolicy(model, "cpu")
        prep = fast.prepare(sample.A, _FAST_CFG, sample.feature_seed)
        return amg_reward(sample, fast.edge_mask(prep, sample.edge_index), None,
                          coarse_solver=coarse_solver,
                          reward_lambda_cx=reward_lambda_cx, stag_start=stag_start)

    with torch.no_grad():
        logits, B_extra = model(
            sample.x, sample.edge_index, sample.edge_weight,
            raw_edge_features=sample.raw_edge_features,
        )

    if selection_mode == "threshold":
        threshold = model.threshold.item() if hasattr(model, 'threshold') else 0.0
        sel_mask, _ = select_edges_by_threshold(
            logits=logits, edge_index=sample.edge_index,
            threshold=threshold,
        )
    else:
        sel_mask, _ = sample_topk_without_replacement(
            logits=logits, row_groups=sample.row_groups,
            k=k_per_row, temperature=1.0, gumbel=False,
        )

    return amg_reward(sample, sel_mask.cpu().numpy(), B_extra,
                      coarse_solver=coarse_solver,
                      reward_lambda_cx=reward_lambda_cx, stag_start=stag_start)


def amg_reward(sample, sel_mask_np, B_extra=None, coarse_solver="splu",
               reward_lambda_cx=0.1, stag_start=None):
    """AMG half of es_evaluate: strength graph -> SA hierarchy -> PCG -> reward.
    stag_start (population members only, `stag`): see pcg_iteration_count_stag."""
    C = C_from_selected_edges(sample.A, sample.edge_index, sel_mask_np)
    B = build_B_for_pyamg(B_extra) if B_extra is not None else None

    try:
        ml = build_pyamg_solver(sample.A, C, B, coarse_solver=coarse_solver)
        if _REWARD == "residual":
            return residual_reward(sample.A, ml, reward_lambda_cx)
        if stag_start is not None:
            iters = pcg_iteration_count_stag(
                sample.A, ml, tol=1e-8, maxiter=500, stag_start=stag_start,
                stag_window=_STAG_W, stag_step=_STAG_STEP, stag_margin=_STAG_MARGIN)
        elif "pcg" in _SPEEDUPS:
            M = vcycle_preconditioner(ml)
            iters = pcg_iteration_count(sample.A, M, tol=1e-8)
        else:
            M = ml.aspreconditioner(cycle="V")
            iters, _ = pcg_iterations(sample.A, M, tol=1e-8)
        oc = sum(lvl.A.nnz for lvl in ml.levels) / max(ml.levels[0].A.nnz, 1)
        reward = -float(np.log(max(iters, 1))) - reward_lambda_cx * float(np.log(max(oc, 1.0)))
        return reward, iters
    except Exception:
        return -200.0, 500


def _k_residuals(A, M, K):
    """Residual norms |r_1|..|r_K| of K PCG iterations (b = ones, x0 = 0), padded if CG stops early."""
    _, res = pcg_iterations(A, M, tol=0.0, maxiter=K)
    res = res or [float(np.linalg.norm(np.ones(A.shape[0])))]
    return np.array(res + [res[-1]] * (K - len(res)))


def residual_reward(A, ml, reward_lambda_cx, eps=1e-12):
    """K-step residual reward against a default PyAMG SA-AMG baseline (LSC_REWARD=residual)."""
    key = (A.shape[0], A.nnz, float(A.data.sum()))
    if key not in _SA_BASE_CACHE:
        base = pyamg.smoothed_aggregation_solver(A)
        _SA_BASE_CACHE[key] = _k_residuals(A, base.aspreconditioner(cycle="V"), _REWARD_K)
    r_base = _SA_BASE_CACHE[key]
    r_learn = _k_residuals(A, ml.aspreconditioner(cycle="V"), _REWARD_K)
    oc = sum(lvl.A.nnz for lvl in ml.levels) / max(ml.levels[0].A.nnz, 1)
    r_pcg = float(np.mean(np.log(r_base + eps) - np.log(r_learn + eps)))
    return r_pcg - reward_lambda_cx * float(np.log(oc + eps)), _REWARD_K


# ============================================================
# GPU policy: all GNN forwards on one GPU in the main process
# ============================================================
class GPUPolicy:
    """Holds a copy of the policy and every train/test sample on the GPU.

    For a parameter vector and a sample id it returns the threshold-mode
    selection mask, bit-packed for cheap IPC. The workers then only do the
    AMG half (amg_reward), which is where the CPU time goes. Selection uses
    the same z-score rule as select_edges_by_threshold; on V100s the
    logits match the CPU forward to ~2e-7 and no edge selection flipped in
    the checks we ran.
    """

    def __init__(self, model, samples, device):
        self.device = device
        self.model = copy.deepcopy(model).to(device).eval()
        self.fast = None
        if (_FAST_CFG is not None and FastPolicy.supported(model, _FAST_CFG)
                and all(getattr(s, "feature_seed", None) is not None for s in samples)):
            # Exact fast inference (fast_policy.py): per-sample pattern and
            # node features prepared once; masks in edge_index order.
            self.fast = FastPolicy(self.model, device)
            self.prep = [self.fast.prepare(s.A, _FAST_CFG, s.feature_seed) for s in samples]
            self.pos = [p.g.positions(s.edge_index) for p, s in zip(self.prep, samples)]
            return
        self.data = []
        for s in samples:
            ref = getattr(s, "raw_edge_features", None)
            ei = s.edge_index.to(device)
            self.data.append((
                s.x.to(device), ei, s.edge_weight.to(device),
                ref.to(device) if ref is not None else None,
                ei[0] == ei[1],
            ))

    @torch.no_grad()
    def set_theta(self, theta_vec):
        vector_to_parameters(theta_vec.to(self.device), self.model.parameters())
        self.threshold = (self.model.threshold.item()
                          if hasattr(self.model, "threshold") else 0.0)
        if self.fast is not None:
            self.fast.load(self.model)

    @torch.no_grad()
    def packed_mask(self, sid):
        if self.fast is not None:
            sel = self.fast.directed_selection(self.prep[sid])[self.pos[sid]]
            return np.packbits(sel), sel.shape[0]
        x, ei, ew, ref, diag = self.data[sid]
        logits, _ = self.model(x, ei, ew, raw_edge_features=ref)
        off = ~diag
        mu = logits[off].mean() if off.any() else logits.mean()
        std = logits[off].std() if off.any() else logits.std()
        z = (logits - mu) / (std + 1e-6)
        sel = ((z > self.threshold) & off).cpu().numpy()
        return np.packbits(sel), sel.shape[0]

    # ---- gpubatch: one forward per theta for a whole mini-batch. The
    # samples' graphs are concatenated (node ids offset); every layer of the
    # policy is node- or edge-local (GATConv, LayerNorm per node, edge MLP),
    # so the logits are those of the per-sample forwards up to float
    # reduction order, and the z-score threshold is applied per sample.
    # The edge-feature tensor is E x (4*hidden + 12) floats plus GATConv
    # temporaries: ~1 kB per edge. 1.5M edges (~4 samples at 50k, the whole
    # mini-batch at <= 20k) stays well inside a 24 GB A30; LSC_GPU_BATCH_EDGES
    # overrides.
    _MAX_BATCH_EDGES = int(os.environ.get("LSC_GPU_BATCH_EDGES", "1500000"))

    def _merged(self, sids):
        key = tuple(sids)
        cache = getattr(self, "_merge_cache", None)
        if cache is None or cache[0] != key:
            xs, eis, ews, refs, segs = [], [], [], [], []
            off_n = 0
            off_e = 0
            for sid in sids:
                x, ei, ew, ref, _diag = self.data[sid]
                xs.append(x)
                eis.append(ei + off_n)
                ews.append(ew)
                refs.append(ref)
                segs.append((off_e, off_e + ei.shape[1]))
                off_n += x.shape[0]
                off_e += ei.shape[1]
            ei = torch.cat(eis, 1)
            merged = (torch.cat(xs, 0), ei, torch.cat(ews, 0),
                      torch.cat(refs, 0) if refs[0] is not None else None,
                      ei[0] == ei[1], segs)
            self._merge_cache = cache = (key, merged)
        return cache[1]

    @torch.no_grad()
    def packed_masks(self, sids):
        """[(sid, packed_mask, n_edges)] for the current theta."""
        if self.fast is not None:
            return [(sid,) + self.packed_mask(sid) for sid in sids]
        out = []
        chunk, n_e = [], 0
        for sid in list(sids) + [None]:
            if sid is not None:
                e = self.data[sid][1].shape[1]
                if chunk and n_e + e > self._MAX_BATCH_EDGES:
                    out.extend(self._masks_chunk(chunk))
                    chunk, n_e = [], 0
                chunk.append(sid)
                n_e += e
            elif chunk:
                out.extend(self._masks_chunk(chunk))
        return out

    def _masks_chunk(self, sids):
        if len(sids) == 1:
            return [(sids[0],) + self.packed_mask(sids[0])]
        x, ei, ew, ref, diag, segs = self._merged(sids)
        try:
            logits, _ = self.model(x, ei, ew, raw_edge_features=ref)
        except torch.OutOfMemoryError:
            # Too big for this GPU: halve the chunk (same per-sample results).
            torch.cuda.empty_cache()
            h = len(sids) // 2
            return self._masks_chunk(sids[:h]) + self._masks_chunk(sids[h:])
        off = ~diag
        res = []
        for sid, (a, b) in zip(sids, segs):
            lg, of = logits[a:b], off[a:b]
            mu = lg[of].mean() if of.any() else lg.mean()
            std = lg[of].std() if of.any() else lg.std()
            z = (lg - mu) / (std + 1e-6)
            sel = ((z > self.threshold) & of).cpu().numpy()
            res.append((sid, np.packbits(sel), sel.shape[0]))
        return res


def _worker_eval_mask(args_tuple):
    """(sample_id, packed_mask, n_edges) -> (reward, iters), AMG half only."""
    sid, packed, n_e = args_tuple
    sel = np.unpackbits(packed, count=n_e).astype(bool)
    return amg_reward(_g_samples[sid], sel, None,
                      reward_lambda_cx=_g_lambda_cx)


def _gpu_mask_tasks(gpu, thetas, sample_ids):
    """Yield worker tasks theta-major / sample-minor, computing masks lazily
    so the GPU forwards overlap with the workers' AMG solves."""
    for th in thetas:
        gpu.set_theta(th)
        for sid in sample_ids:
            packed, n_e = gpu.packed_mask(sid)
            yield (sid, packed, n_e)


def gpu_population_rewards(pool, gpu, thetas, sample_ids, chunksize=4):
    """Mean (reward, iters) over sample_ids for each parameter vector, in
    the same order and with the same per-sample values as the CPU path."""
    return gpu_population_finish(gpu_population_start(pool, gpu, thetas, sample_ids, chunksize))


# ---- sched on the GPU path: keyed tasks consumed with imap_unordered
# (chunksize 1) so that no worker waits for an ordered chunk; the per-member
# means are formed from the same per-sample values in the same order.
def _worker_eval_mask_keyed(args_tuple):
    """(k, sample_id, packed_mask, n_edges) -> (k, sample_id, reward, iters)."""
    k, sid, packed, n_e = args_tuple
    sel = np.unpackbits(packed, count=n_e).astype(bool)
    r, it = amg_reward(_g_samples[sid], sel, None, reward_lambda_cx=_g_lambda_cx)
    return k, sid, r, it


def _gpu_mask_tasks_keyed(gpu, thetas, sample_ids):
    batched = "gpubatch" in _SPEEDUPS
    for k, th in enumerate(thetas):
        gpu.set_theta(th)
        if batched:
            for sid, packed, n_e in gpu.packed_masks(sample_ids):
                yield (k, sid, packed, n_e)
            continue
        for sid in sample_ids:
            packed, n_e = gpu.packed_mask(sid)
            yield (k, sid, packed, n_e)


def gpu_population_start(pool, gpu, thetas, sample_ids, chunksize=4):
    """Submit the forwards+AMG evaluations; returns a handle for
    gpu_population_finish. The GPU forwards run lazily in the pool's task
    feeder thread as the workers pull tasks."""
    if "sched" in _SPEEDUPS:
        it = pool.imap_unordered(_worker_eval_mask_keyed,
                                 _gpu_mask_tasks_keyed(gpu, thetas, sample_ids),
                                 chunksize=1)
        return ("keyed", it, len(thetas), list(sample_ids))
    it = pool.imap(_worker_eval_mask, _gpu_mask_tasks(gpu, thetas, sample_ids),
                   chunksize=chunksize)
    return ("ordered", it, len(thetas), list(sample_ids))


def gpu_population_finish(handle):
    kind, it, n_theta, sample_ids = handle
    m = len(sample_ids)
    out = []
    if kind == "keyed":
        got = {}
        for k, sid, r, iters in it:
            got[(k, sid)] = (r, iters)
        for k in range(n_theta):
            out.append((float(np.mean([got[(k, sid)][0] for sid in sample_ids])),
                        float(np.mean([got[(k, sid)][1] for sid in sample_ids]))))
        return out
    flat = list(it)
    for k in range(n_theta):
        block = flat[k * m:(k + 1) * m]
        out.append((float(np.mean([r for r, _ in block])),
                    float(np.mean([it for _, it in block]))))
    return out


# ============================================================
# Multiprocessing worker state + functions
# ============================================================
_g_model = None
_g_samples = None
_g_shm_handle = None   # keep SharedMemory alive for the worker's lifetime
_g_k = None
_g_selection_mode = "topk"
_g_lambda_cx = 0.1


def _worker_init(model_state_dict, model_kwargs, shm_meta, k_per_row,
                 selection_mode="topk", reward_lambda_cx=0.1, fast_cfg=None):
    """Called once per subprocess.  Builds model + reconstructs samples
    from shared memory (zero-copy, no per-worker duplication)."""
    global _g_model, _g_samples, _g_shm_handle, _g_k, _g_selection_mode, _g_lambda_cx
    torch.set_num_threads(1)
    set_fast_policy(fast_cfg)

    _g_model = AMGEdgePolicy(**model_kwargs)
    _g_model.load_state_dict(model_state_dict)
    _g_model.eval()

    # Reconstruct samples from shared memory (zero-copy)
    _g_samples, _g_shm_handle = load_from_shared_cache(shm_meta)
    _g_k = k_per_row
    _g_selection_mode = selection_mode
    _g_lambda_cx = reward_lambda_cx


def _worker_eval_chunk(args_tuple):
    """Evaluate a chunk of perturbations on a mini-batch.

    Receives (theta, eps_chunk, sigma, batch_indices, include_center).
    sigma can be a scalar or a 1-D tensor (blockwise sigma).
    Computes theta_i = theta + sigma*eps locally to avoid sending
    one full theta per perturbation over IPC.

    Returns list of (mean_reward, mean_iters) per perturbation,
    plus optionally the center evaluation as the last element.
    """
    theta, eps_chunk, sigma, batch_indices, include_center = args_tuple
    results = []
    for j in range(eps_chunk.shape[0]):
        theta_j = theta + sigma * eps_chunk[j]
        vector_to_parameters(theta_j, _g_model.parameters())
        rs, its = [], []
        for idx in batch_indices:
            r, it = es_evaluate(_g_model, _g_samples[idx], _g_k,
                                selection_mode=_g_selection_mode,
                                reward_lambda_cx=_g_lambda_cx)
            rs.append(r)
            its.append(it)
        results.append((float(np.mean(rs)), float(np.mean(its))))
    if include_center:
        vector_to_parameters(theta, _g_model.parameters())
        rs, its = [], []
        for idx in batch_indices:
            r, it = es_evaluate(_g_model, _g_samples[idx], _g_k,
                                selection_mode=_g_selection_mode,
                                reward_lambda_cx=_g_lambda_cx)
            rs.append(r)
            its.append(it)
        results.append((float(np.mean(rs)), float(np.mean(its))))
    return results


# ---- sched: one task per (perturbation, sample); theta/eps/sigma for the
# epoch sit in a shared-memory block (rows: 0..pop-1 = eps, pop = theta,
# pop+1 = sigma_vec), so a task is just two ints.
_g_es_shm = None
_g_es_rows = None


def _es_shm_attach(name, pop_size, n_params):
    global _g_es_shm, _g_es_rows
    if _g_es_shm is None or _g_es_shm.name != name:
        _g_es_shm = SharedMemory(name=name)
        _g_es_rows = np.ndarray((pop_size + _ES_ROWS_EXTRA, n_params), dtype=np.float32,
                                buffer=_g_es_shm.buf)


def _worker_eval_one(args_tuple):
    """(shm_name, pop_size, n_params, j, sample_idx) -> (j, sample_idx, reward, iters).
    j == pop_size evaluates the center theta."""
    shm_name, pop_size, n_params, j, idx = args_tuple
    _es_shm_attach(shm_name, pop_size, n_params)
    theta = torch.from_numpy(_g_es_rows[pop_size])
    if j == pop_size:
        theta_j = theta
    else:
        sigma = torch.from_numpy(_g_es_rows[pop_size + 1])
        theta_j = theta + sigma * torch.from_numpy(_g_es_rows[j])
    vector_to_parameters(theta_j, _g_model.parameters())
    stag_start = None
    if j != pop_size and "stag" in _SPEEDUPS:
        st = float(_g_es_rows[pop_size + 2][idx])
        if 0 < st < 500:
            stag_start = int(st)
    r, it = es_evaluate(_g_model, _g_samples[idx], _g_k,
                        selection_mode=_g_selection_mode,
                        reward_lambda_cx=_g_lambda_cx, stag_start=stag_start)
    return j, idx, r, it


def _worker_soc_one(args_tuple):
    """(shm_name, pop_size, n_params, sample_idx, k_per_row, selection_mode)
    -> (sample_idx, snapshot entry) with the center theta, same code as
    compute_soc_snapshot."""
    shm_name, pop_size, n_params, idx, k_per_row, selection_mode = args_tuple
    _es_shm_attach(shm_name, pop_size, n_params)
    vector_to_parameters(torch.from_numpy(_g_es_rows[pop_size]), _g_model.parameters())
    snap = compute_soc_snapshot(_g_model, [_g_samples[idx]], k_per_row, selection_mode)
    return idx, snap[0]


# ---- diag: AMG diagnostics on one dedicated process. Diagnostics never feed
# back into training (they only fill history["diagnostics"]), so they can run
# while the pool evaluates the next populations. One process that performs
# every diagnostic call in the original order (pre-training, then every
# diagnostic_every epochs) keeps the per-process ARPACK start-vector sequence
# of eigsh identical, so the numbers match the serial code bit for bit.
def _diag_server(conn, model_state_dict, model_kwargs, shm_meta, k_per_row,
                 selection_mode, fast_cfg):
    torch.set_num_threads(1)
    set_fast_policy(fast_cfg)
    model = AMGEdgePolicy(**model_kwargs)
    model.load_state_dict(model_state_dict)
    model.eval()
    samples, _handle = load_from_shared_cache(shm_meta)
    while True:
        msg = conn.recv()
        if msg is None:
            break
        tag, theta, diag_idx, seed = msg
        vector_to_parameters(theta, model.parameters())
        snap = compute_diagnostic_batch(samples, model, k_per_row,
                                        selection_mode=selection_mode,
                                        sample_indices=diag_idx, seed=seed)
        conn.send((tag, snap))


class DiagRunner:
    def __init__(self, model, model_kwargs, shm_meta, k_per_row, selection_mode):
        self.conn, child = mp.Pipe()
        self.proc = mp.Process(
            target=_diag_server,
            args=(child, model.state_dict(), model_kwargs, shm_meta, k_per_row,
                  selection_mode, _FAST_CFG), daemon=True)
        self.proc.start()
        self.pending = 0

    def submit(self, tag, theta, diag_idx, seed):
        self.conn.send((tag, theta.detach().clone(), list(diag_idx), seed))
        self.pending += 1

    def collect(self, block):
        """Yield finished (tag, snapshot) pairs, in submission order."""
        out = []
        while self.pending and (block or self.conn.poll()):
            out.append(self.conn.recv())
            self.pending -= 1
        return out

    def close(self):
        self.conn.send(None)
        self.proc.join(timeout=60)


_DIAG_RUNNER = None


# ============================================================
# Checkpoint helpers
# ============================================================
def save_checkpoint(path, epoch, theta, best_theta, best_reward, best_iters,
                    m_adam, v_adam, sigma_vec, sigma_vec_init,
                    epochs_without_improvement, sigma_restarts, history,
                    model_state_dict):
    """Save full training state for resumption."""
    ckpt = {
        "epoch": epoch,
        "theta": theta,
        "best_theta": best_theta,
        "best_reward": best_reward,
        "best_iters": best_iters,
        "m_adam": m_adam,
        "v_adam": v_adam,
        "sigma_vec": sigma_vec,
        "sigma_vec_init": sigma_vec_init,
        "epochs_without_improvement": epochs_without_improvement,
        "sigma_restarts": sigma_restarts,
        "history": history,
        "model_state_dict": model_state_dict,
    }
    # Write to temp file first, then rename (atomic on POSIX)
    tmp_path = path + ".tmp"
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, path)


def _atomic_torch_save(obj, path):
    """torch.save via temp file + rename, so a kill mid-write (preemption on
    guest partitions) cannot leave a truncated file behind."""
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(path):
    """Load training state from checkpoint."""
    return torch.load(path, map_location="cpu", weights_only=False)


# ============================================================
# SLURM script generator
# ============================================================
def print_slurm_script(args):
    """Print a ready-to-use SLURM sbatch script."""
    script = f"""#!/bin/bash
#SBATCH --job-name=es_amg
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_%A_%a.out
#SBATCH --error=slurm_%A_%a.err
# Uncomment for array jobs (multiple seeds):
# #SBATCH --array=0-4

# ── Environment ──
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Activate your environment
source .venv/bin/activate  # adjust path as needed

# ── Run ──
python -u train_es_HPC.py \\
    --rl_epochs {args.rl_epochs} \\
    --train_samples {args.train_samples} \\
    --es_pop_size {args.es_pop_size} \\
    --es_mini_batch {args.es_mini_batch} \\
    --es_sigma {args.es_sigma} \\
    --rl_lr {args.rl_lr} \\
    --sigma_decay {args.sigma_decay} \\
    --sigma_min {args.sigma_min} \\
    --weight_decay {args.weight_decay} \\
    --k_per_row {args.k_per_row} \\
    --selection_mode {args.selection_mode} \\
    --hidden {args.hidden} \\
    --full_eval_every {args.full_eval_every} \\
    --checkpoint_every {args.checkpoint_every} \\
    --n_edge {args.n_edge} \\
    2>&1 | tee es_hpc_${{SLURM_JOB_ID:-local}}_${{SLURM_ARRAY_TASK_ID:-0}}.log
"""
    print(script)
    sys.exit(0)


# ============================================================
# Round-robin task distribution
# ============================================================
def build_worker_tasks(theta, eps, sigma_vec, batch_list, n_workers, pop_size):
    """Distribute perturbations round-robin across workers for balanced load.

    Returns list of (theta, eps_chunk, sigma_vec, batch_list, include_center)
    tuples.  The worker with fewest perturbations also evaluates the center.
    """
    # Assign perturbation indices round-robin
    worker_indices = [[] for _ in range(n_workers)]
    for i in range(pop_size):
        worker_indices[i % n_workers].append(i)

    # Find the worker with fewest perturbations → assign center eval
    min_count = min(len(wi) for wi in worker_indices)
    center_worker = next(
        w for w, wi in enumerate(worker_indices) if len(wi) == min_count
    )

    tasks = []
    for w in range(n_workers):
        if not worker_indices[w]:
            continue
        idx = worker_indices[w]
        eps_chunk = eps[idx]  # gather rows by index
        include_center = (w == center_worker)
        tasks.append((theta.clone(), eps_chunk, sigma_vec, batch_list,
                       include_center))

    # Return tasks + mapping from flat results back to original indices
    return tasks, worker_indices, center_worker


def unflatten_results(chunk_results, worker_indices, center_worker, pop_size):
    """Reassemble round-robin results back into original perturbation order."""
    rewards = np.zeros(pop_size)
    iters_all = np.zeros(pop_size)
    center_reward = None
    center_iters = None

    for w, (chunk, indices) in enumerate(zip(chunk_results, worker_indices)):
        if not indices:
            continue
        n_perturb = len(indices)
        for j, orig_idx in enumerate(indices):
            rewards[orig_idx] = chunk[j][0]
            iters_all[orig_idx] = chunk[j][1]
        if w == center_worker:
            # Center eval is appended after the perturbations
            center_reward = chunk[n_perturb][0]
            center_iters = chunk[n_perturb][1]

    return rewards, iters_all, center_reward, center_iters


# ============================================================
# ES Training Loop
# ============================================================
def train_es(model, train_set, args, shm_meta, test_set=None):
    """OpenAI-style Evolution Strategies with antithetic sampling + Adam.

    HPC version: uses shared-memory samples, round-robin distribution,
    checkpoint resume, and SIGTERM handling.
    """
    pop_size = args.es_pop_size
    base_sigma = args.es_sigma
    half_pop = pop_size // 2
    mini_batch = min(args.es_mini_batch, len(train_set))
    n_epochs = args.rl_epochs
    n_workers = getattr(args, "n_workers", 0)
    use_mp = n_workers > 1

    model.eval()
    theta = parameters_to_vector(model.parameters()).detach().clone()
    n_params = theta.numel()

    # ── Build parameter group registry ──
    param_groups = []
    offset = 0
    for name, p in model.named_parameters():
        numel = p.numel()
        if name.startswith("conv1"):
            group = "conv1"
        elif name.startswith("conv2"):
            group = "conv2"
        elif name.startswith("emb_norm"):
            group = "emb_norm"
        elif name.startswith("edge_mlp"):
            group = "edge_mlp"
        elif name.startswith("B_head"):
            group = "B_head"
        elif name == "threshold":
            group = "threshold"
        else:
            group = "other"
        if param_groups and param_groups[-1][0] == group:
            prev_name, prev_start, prev_end = param_groups[-1]
            param_groups[-1] = (prev_name, prev_start, offset + numel)
        else:
            param_groups.append((group, offset, offset + numel))
        offset += numel

    # Build per-parameter sigma_vec
    scale_map = {
        "conv1": getattr(args, "sigma_scale_conv1", 1.0),
        "conv2": getattr(args, "sigma_scale_conv2", 1.0),
        "emb_norm": getattr(args, "sigma_scale_emb_norm", 1.0),
        "edge_mlp": getattr(args, "sigma_scale_edge_mlp", 1.0),
        "threshold": getattr(args, "sigma_scale_threshold", 2.5),
        "B_head": 1.0,
        "other": 1.0,
    }
    sigma_vec = torch.zeros(n_params)
    for gname, gstart, gend in param_groups:
        scale = scale_map.get(gname, 1.0)
        sigma_vec[gstart:gend] = base_sigma * scale
    sigma_vec_init = sigma_vec.clone()

    # Adam momentum for ES gradient
    m_adam = torch.zeros(n_params)
    v_adam = torch.zeros(n_params)
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

    best_theta = theta.clone()
    best_reward = -1e9
    best_iters = 500.0

    # Adaptive sigma restart state
    sigma_restart_patience = getattr(args, "sigma_restart_patience", 30)
    epochs_without_improvement = 0
    sigma_restarts = 0

    # Fixed validation subset
    eval_samples = getattr(args, "eval_samples", 10)
    eval_idx = list(range(min(eval_samples, len(train_set))))

    # History tracking
    history = {
        "epochs": [], "train_reward": [], "pcg_iters_mean": [],
        "pop_max": [], "pop_min": [], "pop_spread": [],
        "sigma": [], "best_reward_so_far": [], "best_iters_so_far": [],
        "eval_reward": [], "eval_iters": [],
        "test_reward": [], "test_iters": [],
        "grad_norm": [], "grad_snr": [],
        "diagnostics": {},
    }

    start_epoch = 0

    # ── Resume from checkpoint ──
    resume_path = os.path.join(args.out_dir, "checkpoint_resume.pt")
    if args.resume and os.path.isfile(resume_path):
        print(f"  Resuming from {resume_path} ...", flush=True)
        ckpt = load_checkpoint(resume_path)
        start_epoch = ckpt["epoch"] + 1
        theta = ckpt["theta"]
        best_theta = ckpt["best_theta"]
        best_reward = ckpt["best_reward"]
        best_iters = ckpt["best_iters"]
        m_adam = ckpt["m_adam"]
        v_adam = ckpt["v_adam"]
        sigma_vec = ckpt["sigma_vec"]
        sigma_vec_init = ckpt["sigma_vec_init"]
        epochs_without_improvement = ckpt["epochs_without_improvement"]
        sigma_restarts = ckpt["sigma_restarts"]
        history = ckpt["history"]
        vector_to_parameters(theta, model.parameters())
        print(f"  Resumed at epoch {start_epoch}, best_R={best_reward:.4f}, "
              f"best_iters={best_iters:.1f}", flush=True)

    # ── Diagnostic tracking ──
    diagnostic_every = getattr(args, "diagnostic_every", 0)
    diagnostic_samples = getattr(args, "diagnostic_samples", 3)
    diag_path = os.path.join(args.out_dir, "diagnostics_evolution.json")
    if "diagnostics" not in history:
        history["diagnostics"] = {}

    # ── SoC evolution tracking ──
    soc_track_every = getattr(args, "soc_track_every", 0)
    soc_path = os.path.join(args.out_dir, "soc_evolution.pt")
    if soc_track_every > 0:
        soc_evolution = None
        if args.resume and os.path.isfile(soc_path):
            try:
                soc_evolution = torch.load(soc_path, map_location="cpu",
                                           weights_only=False)
                print(f"  Resumed SoC evolution ({len(soc_evolution['epochs'])} "
                      f"tracked epochs)", flush=True)
            except Exception as e:
                # Diagnostics only; a file truncated by a kill mid-write
                # must not stop training. Start a fresh history.
                print(f"  WARNING: could not load {soc_path} ({e}); "
                      f"starting a new SoC history", flush=True)
                soc_evolution = None
        if soc_evolution is None:
            # Save edge_index per sample once (fixed across epochs)
            soc_evolution = {
                "edge_index": [
                    s.edge_index.cpu().numpy() for s in train_set
                ],
                "epochs": {},
            }

    # Ensure test history keys exist (for resumed runs)
    if "test_reward" not in history:
        history["test_reward"] = []
    if "test_iters" not in history:
        history["test_iters"] = []

    t_start = time.time()
    eval_every = args.full_eval_every

    # ── SIGTERM handler for graceful shutdown ──
    _sigterm_received = [False]

    def _sigterm_handler(signum, frame):
        _sigterm_received[0] = True
        print("\n  [ES-HPC] SIGTERM received — saving emergency checkpoint...",
              flush=True)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    print("  Parameter groups:")
    for gname, gstart, gend in param_groups:
        gcount = gend - gstart
        gsigma = float(sigma_vec[gstart])
        print(f"    {gname:10s}: {gcount:6d} params, σ={gsigma:.4f}")

    # ── Set up multiprocessing pool ──
    pool = None
    if use_mp:
        feat_cfg = FeatureConfig.from_preset(args.feature_preset)
        if args.num_relaxed_probes is not None:
            feat_cfg.num_relaxed_probes = args.num_relaxed_probes
        model_kwargs = {
            "in_channels": feat_cfg.node_dim,
            "hidden": args.hidden,
            "learn_B": args.learn_B,
            "edge_raw_dim": feat_cfg.edge_raw_dim,
            "adaptive_k": (args.selection_mode == "threshold"),
            "init_threshold": args.init_threshold,
        }
        print(f"\n  Spawning {n_workers} worker processes "
              f"(shared-memory samples, zero-copy)...", flush=True)
        t_pool = time.time()
        pool = mp.Pool(
            processes=n_workers,
            initializer=_worker_init,
            initargs=(model.state_dict(), model_kwargs, shm_meta,
                      args.k_per_row, args.selection_mode,
                      args.reward_lambda_cx, _FAST_CFG),
        )
        print(f"  Pool ready in {time.time()-t_pool:.1f}s", flush=True)
    es_shm = None
    if use_mp and ("sched" in _SPEEDUPS or "soc" in _SPEEDUPS):
        es_shm = SharedMemory(create=True, size=(pop_size + _ES_ROWS_EXTRA) * n_params * 4)
        es_rows = np.ndarray((pop_size + _ES_ROWS_EXTRA, n_params), dtype=np.float32, buffer=es_shm.buf)
        es_rows[pop_size + 2] = 0.0
        es_rows[pop_size] = theta.numpy()
        sched_chunksize = int(os.environ.get("LSC_SCHED_CHUNK", "1"))
        print(f"  [speedups] {sorted(_SPEEDUPS)} chunksize={sched_chunksize}", flush=True)
    n_small = len(train_set) + (len(test_set) if test_set is not None else 0)
    disp = None
    if (use_mp and "multinode" in _SPEEDUPS and "sched" in _SPEEDUPS
            and _MN_PORT > 0 and getattr(args, "device", "auto") != "cuda"):
        from multinode import Dispatcher
        node_cfg = {
            "speedups": ",".join(sorted(_SPEEDUPS)),
            "env": {k: os.environ[k] for k in os.environ if k.startswith("LSC_STAG")},
            "model_kwargs": model_kwargs, "k_per_row": args.k_per_row,
            "selection_mode": args.selection_mode, "reward_lambda_cx": args.reward_lambda_cx,
            "fast_policy": args.fast_policy, "feature_preset": args.feature_preset,
            "num_relaxed_probes": args.num_relaxed_probes,
            "pop_size": pop_size, "n_params": n_params,
        }
        torch.save(model.state_dict(), os.path.join(args.out_dir, "node_model_init.pt"))
        with open(os.path.join(args.out_dir, "node_config.json"), "w") as f:
            json.dump(node_cfg, f, indent=2)
        disp = Dispatcher(pool, _worker_eval_one, es_shm.name, pop_size, n_params,
                          n_workers, _MN_PORT, _MN_AUTH)
        if int(os.environ.get("LSC_MN_NODES", "0")) > 0:
            disp.wait_for_nodes(int(os.environ["LSC_MN_NODES"]),
                                float(os.environ.get("LSC_MN_WAIT", "1800")))

    # GPU forwards (created after the workers were forked, so they never
    # inherit a CUDA context).
    gpu = None
    want_gpu = getattr(args, "device", "auto")
    if (use_mp and args.selection_mode == "threshold" and not args.learn_B
            and want_gpu in ("auto", "cuda")
            and compute_device(want_gpu).type == "cuda"):
        t_gpu = time.time()
        gpu = GPUPolicy(model, list(train_set) + list(test_set or []),
                        torch.device("cuda"))
        print(f"  GPU forwards on {torch.cuda.get_device_name()} "
              f"({len(train_set) + len(test_set or [])} samples resident"
              f"{', fast_policy' if gpu.fast is not None else ''}, "
              f"{time.time()-t_gpu:.1f}s)",
              flush=True)
    elif want_gpu == "cuda":
        raise RuntimeError("--device cuda requested but GPU path unavailable "
                           "(needs CUDA, n_workers>1, threshold mode, no learn_B)")

    # ---- population / eval / SoC submission helpers (dynamic paths) ----
    def _draw_epoch():
        """The epoch's mini-batch and antithetic eps, in the original RNG order."""
        batch_idx = np.random.choice(len(train_set), size=mini_batch,
                                     replace=False)
        eps = torch.randn(half_pop, n_params)
        eps = torch.cat([eps, -eps], dim=0)  # (pop_size, n_params)
        return batch_idx, eps

    def _submit_population(batch_idx, eps_t, sigma_t):
        """Start evaluating theta + sigma_t * eps_t (and the center theta) on
        the mini-batch; returns a handle for _collect_population."""
        batch_list = batch_idx.tolist()
        if gpu is not None:
            thetas = [theta + sigma_t * eps_t[i] for i in range(pop_size)]
            thetas.append(theta)                      # center, last
            return ("gpu", gpu_population_start(pool, gpu, thetas, batch_list), batch_list)
        es_rows[:pop_size] = eps_t.numpy()
        es_rows[pop_size] = theta.numpy()
        es_rows[pop_size + 1] = sigma_t.numpy()
        if disp is not None:
            es_rows[pop_size + 2] = 0.0
            disp.set_block(es_rows, n_small)
            if "stag" in _SPEEDUPS:
                exp = disp.submit([(pop_size, sid) for sid in batch_list])
                return ("mn_stag", exp, batch_list)
            exp = disp.submit([(j, sid) for j in range(pop_size + 1) for sid in batch_list])
            return ("mn", exp, batch_list)
        if "stag" in _SPEEDUPS:
            # `stag`: only the centre now; its per-sample iteration counts set
            # the members' stagnation-stop starts in _collect_population.
            ctasks = [(es_shm.name, pop_size, n_params, pop_size, sid)
                      for sid in batch_list]
            return ("cpu_stag", pool.imap_unordered(_worker_eval_one, ctasks,
                                                    chunksize=1), batch_list)
        tasks = [(es_shm.name, pop_size, n_params, j, sid)
                 for j in range(pop_size + 1) for sid in batch_list]
        return ("cpu", pool.imap_unordered(_worker_eval_one, tasks,
                                           chunksize=sched_chunksize), batch_list)

    def _collect_population(handle):
        kind, it, batch_list = handle
        if kind == "gpu":
            res = gpu_population_finish(it)
            rewards = np.array([r for r, _ in res[:pop_size]])
            iters_all = np.array([i for _, i in res[:pop_size]])
            return rewards, iters_all, res[pop_size][0], res[pop_size][1]
        got = {}
        if kind in ("mn", "mn_stag"):
            disp.collect(it, got)
            if kind == "mn_stag":
                for sid in batch_list:
                    c = got[(pop_size, sid)][1]
                    es_rows[pop_size + 2][sid] = min(500.0, max(float(_STAG_MIN),
                                                                float(np.ceil(_STAG_FACTOR * c))))
                disp.set_small(es_rows, n_small)
                exp = disp.submit([(j, sid) for j in range(pop_size) for sid in batch_list])
                disp.collect(exp, got)
                es_rows[pop_size + 2] = 0.0
        else:
            for j, sid, r, i in it:
                got[(j, sid)] = (r, i)
        if kind == "cpu_stag":
            for sid in batch_list:
                c = got[(pop_size, sid)][1]
                es_rows[pop_size + 2][sid] = min(500.0, max(float(_STAG_MIN),
                                                            float(np.ceil(_STAG_FACTOR * c))))
            tasks = [(es_shm.name, pop_size, n_params, j, sid)
                     for j in range(pop_size) for sid in batch_list]
            for j, sid, r, i in pool.imap_unordered(_worker_eval_one, tasks,
                                                    chunksize=sched_chunksize):
                got[(j, sid)] = (r, i)
            es_rows[pop_size + 2] = 0.0
        rewards = np.zeros(pop_size)
        iters_all = np.zeros(pop_size)
        for j in range(pop_size + 1):
            rs = [got[(j, sid)][0] for sid in batch_list]
            its = [got[(j, sid)][1] for sid in batch_list]
            if j == pop_size:
                center_reward = float(np.mean(rs))
                center_iters = float(np.mean(its))
            else:
                rewards[j] = float(np.mean(rs))
                iters_all[j] = float(np.mean(its))
        return rewards, iters_all, center_reward, center_iters

    def _start_eval(sample_ids):
        """Per-sample (reward, iters) of the current theta on sample_ids
        (eval samples, then test samples, which sit after the training
        samples in the shared cache). Returns a handle for _finish_eval."""
        if gpu is not None and "sched" in _SPEEDUPS:
            it = pool.imap_unordered(_worker_eval_mask_keyed,
                                     _gpu_mask_tasks_keyed(gpu, [theta], sample_ids),
                                     chunksize=1)
            return ("gpu_keyed", it, list(sample_ids))
        if gpu is not None:
            it = pool.imap(_worker_eval_mask, _gpu_mask_tasks(gpu, [theta], sample_ids),
                           chunksize=1)
            return ("gpu_ordered", it, list(sample_ids))
        tasks = [(theta.clone(), torch.zeros(0, n_params), sigma_vec, [sid], True)
                 for sid in sample_ids]
        return ("cpu", pool.imap(_worker_eval_chunk, tasks), list(sample_ids))

    def _finish_eval(handle):
        kind, it, sample_ids = handle
        if kind == "cpu":
            return [res[0] for res in it]
        if kind == "gpu_keyed":
            got = {}
            for _k, sid, r, i in it:
                got[sid] = (r, i)
            return [got[sid] for sid in sample_ids]
        return list(it)

    def _start_soc():
        es_rows[pop_size] = theta.numpy()
        tasks = [(es_shm.name, pop_size, n_params, i, args.k_per_row,
                  args.selection_mode) for i in range(len(train_set))]
        return pool.imap_unordered(_worker_soc_one, tasks)

    def _finish_soc(it):
        snap = dict(it)
        return {i: snap[i] for i in range(len(train_set))}

    overlap = use_mp and "overlap" in _SPEEDUPS and (gpu is not None or "sched" in _SPEEDUPS)
    spec = None          # (batch_idx, eps, sigma_used, handle) of the next epoch
    n_rollbacks = 0
    if overlap:
        print("  [overlap] next epoch's population is evaluated during "
              "eval/test/SoC (exact; rollback on sigma restart)", flush=True)

    sigma_str = ",".join(f"{g}:{float(sigma_vec[s]):.4f}"
                         for g, s, e in param_groups)
    print(f"\n{'='*60}")
    print(f"  ES-HPC Training: OpenAI-ES with Antithetic Sampling")
    print(f"{'='*60}")
    print(f"  pop_size={pop_size} ({half_pop} antithetic pairs)")
    print(f"  σ=[{sigma_str}], lr={args.rl_lr}")
    print(f"  mini_batch={mini_batch}/{len(train_set)} samples per eval")
    print(f"  params={n_params}, epochs={n_epochs}")
    print(f"  k_per_row={args.k_per_row}")
    print(f"  selection_mode={args.selection_mode}")
    print(f"  workers={n_workers if use_mp else 'sequential'}")
    print(f"  sigma_restart_patience={sigma_restart_patience}")
    print(f"  fixed eval subset: {len(eval_idx)} samples "
          f"(indices {eval_idx[0]}..{eval_idx[-1]})")
    if test_set is not None:
        print(f"  test set: {len(test_set)} samples (separate seed)")
    if soc_track_every > 0:
        print(f"  SoC tracking: every {soc_track_every} epochs "
              f"({len(train_set)} samples)")
    if diagnostic_every > 0:
        print(f"  AMG diagnostics: every {diagnostic_every} epochs "
              f"({diagnostic_samples} samples)")
    if start_epoch > 0:
        print(f"  RESUMED from epoch {start_epoch}")
    print(f"{'='*60}\n")

    try:
      for epoch in range(start_epoch, n_epochs):
        ep_start = time.time()

        # ── Check for SIGTERM ──
        if _sigterm_received[0]:
            print(f"  [ES-HPC] Saving checkpoint at epoch {epoch} and exiting.",
                  flush=True)
            vector_to_parameters(theta, model.parameters())
            save_checkpoint(
                resume_path, epoch - 1, theta, best_theta, best_reward,
                best_iters, m_adam, v_adam, sigma_vec, sigma_vec_init,
                epochs_without_improvement, sigma_restarts, history,
                model.state_dict(),
            )
            if soc_track_every > 0:
                _atomic_torch_save(soc_evolution, soc_path)
            break

        if spec is not None:
            # ── Population started speculatively during the previous epoch's
            #    eval/test/SoC phase (`overlap`). Its mini-batch and eps were
            #    drawn from the RNGs in the original order; only sigma could
            #    have changed since (a sigma restart), in which case the
            #    results are discarded and the epoch is re-evaluated. ──
            batch_idx, eps, spec_sigma, handle = spec
            spec = None
            if torch.equal(spec_sigma, sigma_vec):
                rewards, iters_all, center_reward, center_iters = \
                    _collect_population(handle)
            else:
                _collect_population(handle)           # drain, discard
                n_rollbacks += 1
                print(f"  [overlap] rollback at epoch {epoch}: sigma restart "
                      f"after the speculative start (#{n_rollbacks})", flush=True)
                rewards, iters_all, center_reward, center_iters = \
                    _collect_population(_submit_population(batch_idx, eps, sigma_vec))
        elif use_mp and (gpu is not None or "sched" in _SPEEDUPS):
            # ── GPU forwards + CPU AMG on the pool, or dynamic CPU scheduling
            #    (one task per (perturbation, sample)) ──
            batch_idx, eps = _draw_epoch()
            rewards, iters_all, center_reward, center_iters = \
                _collect_population(_submit_population(batch_idx, eps, sigma_vec))
        elif use_mp:
            batch_idx, eps = _draw_epoch()
            # ── Parallel evaluation: round-robin distribution ──
            batch_list = batch_idx.tolist()
            tasks, worker_indices, center_worker = build_worker_tasks(
                theta, eps, sigma_vec, batch_list, n_workers, pop_size)

            chunk_results = pool.map(_worker_eval_chunk, tasks)

            rewards, iters_all, center_reward, center_iters = \
                unflatten_results(chunk_results, worker_indices,
                                  center_worker, pop_size)
        else:
            # ── Sequential evaluation (fallback) ──
            batch_idx, eps = _draw_epoch()
            batch = [train_set[i] for i in batch_idx]
            rewards = np.zeros(pop_size)
            iters_all = np.zeros(pop_size)
            for i in range(pop_size):
                theta_i = theta + sigma_vec * eps[i]
                vector_to_parameters(theta_i, model.parameters())

                sample_rs, sample_its = [], []
                for sample in batch:
                    r, it = es_evaluate(model, sample, args.k_per_row,
                                        selection_mode=args.selection_mode,
                                        reward_lambda_cx=args.reward_lambda_cx)
                    sample_rs.append(r)
                    sample_its.append(it)
                rewards[i] = np.mean(sample_rs)
                iters_all[i] = np.mean(sample_its)

            vector_to_parameters(theta, model.parameters())
            center_rs, center_its = [], []
            for sample in batch:
                r, it = es_evaluate(model, sample, args.k_per_row,
                                    selection_mode=args.selection_mode,
                                    reward_lambda_cx=args.reward_lambda_cx)
                center_rs.append(r)
                center_its.append(it)
            center_reward = float(np.mean(center_rs))
            center_iters = float(np.mean(center_its))

        t_pop_done = time.time()

        # ── Rank-based fitness shaping ──
        sorted_idx = np.argsort(rewards)
        shaped = np.zeros(pop_size)
        for rank, idx in enumerate(sorted_idx):
            shaped[idx] = (rank / (pop_size - 1)) - 0.5

        # ── ES gradient ──
        grad = torch.zeros(n_params)
        for i in range(pop_size):
            grad.add_(eps[i], alpha=float(shaped[i]))
        grad.div_(pop_size)
        grad.div_(sigma_vec)

        # ── Gradient diagnostics ──
        grad_norm = grad.norm().item()
        grad_std = grad.std().item()
        grad_snr = grad_norm / (grad_std + 1e-10)

        # ── Per-group gradient norms ──
        group_grad_norms = {}
        for gname, gstart, gend in param_groups:
            group_grad_norms[gname] = grad[gstart:gend].norm().item()

        # ── Adam update ──
        m_adam = beta1 * m_adam + (1 - beta1) * grad
        v_adam = beta2 * v_adam + (1 - beta2) * grad.pow(2)
        m_hat = m_adam / (1 - beta1 ** (epoch + 1))
        v_hat = v_adam / (1 - beta2 ** (epoch + 1))

        # Weight decay
        wd = args.weight_decay
        theta = theta * (1 - args.rl_lr * wd) + \
            args.rl_lr * m_hat / (v_hat.sqrt() + eps_adam)
        vector_to_parameters(theta, model.parameters())

        # ── Sigma decay ──
        if args.sigma_decay < 1.0:
            sigma_vec.mul_(args.sigma_decay).clamp_(min=args.sigma_min)

        # ── Fixed-set evaluation ──
        do_eval = (epoch % max(1, eval_every) == 0) or \
                  (epoch == n_epochs - 1)
        eval_reward = None
        eval_iters_mean = None
        test_reward = None
        test_iters_mean = None
        do_soc = soc_track_every > 0 and (
            epoch == start_epoch
            or epoch == n_epochs - 1
            or epoch % soc_track_every == 0
        )
        n_train = len(train_set)
        n_test = len(test_set) if test_set is not None else 0
        eval_handle = soc_handle = None
        if do_eval and use_mp:
            # One pool task per sample (eval samples, then test samples,
            # which sit after the training samples in the shared cache).
            # Each task returns that sample's (reward, iters) exactly, and
            # the means below are taken over the same per-sample values in
            # the same order as the former serial loops, so the numbers are
            # identical -- only the wall time changes.
            eval_handle = _start_eval(list(eval_idx) + [n_train + i for i in range(n_test)])
        if do_soc and use_mp and "soc" in _SPEEDUPS:
            soc_handle = _start_soc()
        restart_possible = (sigma_restart_patience > 0 and
                            epochs_without_improvement >= sigma_restart_patience - 1
                            and os.environ.get("LSC_OVERLAP_ALWAYS", "") != "1")
        if (overlap and epoch + 1 < n_epochs and not _sigterm_received[0]
                and not restart_possible):
            # Draw the next epoch's mini-batch and eps now (same RNG order as
            # at the top of the next iteration: nothing below consumes the
            # global RNGs) and queue its population behind the eval/SoC
            # tasks, so the workers never idle between epochs. Skipped when
            # this epoch's eval could trigger a sigma restart (the streak is
            # one short of the patience), so the rollback path is never
            # needed in practice.
            nb, ne = _draw_epoch()
            spec = (nb, ne, sigma_vec.clone(), _submit_population(nb, ne, sigma_vec))
        if do_eval:
            if eval_handle is not None:
                per_sample = _finish_eval(eval_handle)
                ev = per_sample[:len(eval_idx)]
                eval_reward = float(np.mean([r for r, _ in ev]))
                eval_iters_mean = float(np.mean([it for _, it in ev]))
                if n_test:
                    te = per_sample[len(eval_idx):]
                    test_reward = float(np.mean([r for r, _ in te]))
                    test_iters_mean = float(np.mean([it for _, it in te]))
            else:
                eval_rs, eval_its = [], []
                for idx in eval_idx:
                    r, it = es_evaluate(model, train_set[idx],
                                        args.k_per_row,
                                        selection_mode=args.selection_mode,
                                        reward_lambda_cx=args.reward_lambda_cx)
                    eval_rs.append(r)
                    eval_its.append(it)
                eval_reward = float(np.mean(eval_rs))
                eval_iters_mean = float(np.mean(eval_its))

        # ── Test set evaluation (serial fallback; the pool path did it above) ──
        if do_eval and test_set is not None and not use_mp:
            test_rs, test_its = [], []
            for s in test_set:
                r, it = es_evaluate(model, s, args.k_per_row,
                                    selection_mode=args.selection_mode,
                                    reward_lambda_cx=args.reward_lambda_cx)
                test_rs.append(r)
                test_its.append(it)
            test_reward = float(np.mean(test_rs))
            test_iters_mean = float(np.mean(test_its))

        t_eval_done = time.time()

        # ── SoC evolution snapshot ──
        if do_soc:
            if soc_handle is not None:
                snap = _finish_soc(soc_handle)
            else:
                snap = compute_soc_snapshot(model, train_set, args.k_per_row,
                                            args.selection_mode)
            soc_evolution["epochs"][epoch] = snap
        t_soc_done = time.time()

        # ── AMG diagnostic snapshot ──
        def _record_diag(ep_tag, diag_snap):
            # Strip per-sample lists for JSON-safe summary
            diag_summary = {k: v for k, v in diag_snap.items()
                            if not k.endswith("_per_sample")}
            history["diagnostics"][ep_tag] = diag_summary
            print(f"  [Diag] ep {ep_tag}: "
                  f"CR={diag_summary['cr_rate']:.4f}  "
                  f"ρ_V={diag_summary['vcycle_conv_factor']:.4f}  "
                  f"WAP={diag_summary['wap_error']:.4f}  "
                  f"NS={diag_summary['nullspace_mean']:.2e}  "
                  f"κ={diag_summary.get('kappa', float('nan')):.1f}  "
                  f"PCG={diag_summary['pcg_iters']:.0f}",
                  flush=True)
            # Save diagnostics to disk incrementally
            with open(diag_path, "w") as _df:
                json.dump(history["diagnostics"], _df, indent=2)

        if _DIAG_RUNNER is not None:
            for ep_tag, diag_snap in _DIAG_RUNNER.collect(block=False):
                _record_diag(ep_tag, diag_snap)
        if diagnostic_every > 0 and (
            epoch == start_epoch
            or epoch == n_epochs - 1
            or epoch % diagnostic_every == 0
        ):
            diag_idx = list(range(min(diagnostic_samples, len(train_set))))
            if _DIAG_RUNNER is not None:
                _DIAG_RUNNER.submit(str(epoch), theta, diag_idx, args.seed)
            else:
                diag_snap = compute_diagnostic_batch(
                    train_set, model, args.k_per_row,
                    selection_mode=args.selection_mode,
                    sample_indices=diag_idx, seed=args.seed,
                )
                _record_diag(str(epoch), diag_snap)

        # ── Track best ──
        tracking_reward = eval_reward if eval_reward is not None \
            else center_reward
        tracking_iters = eval_iters_mean if eval_iters_mean is not None \
            else center_iters
        if tracking_reward > best_reward:
            best_reward = tracking_reward
            best_iters = tracking_iters
            best_theta = theta.clone()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # ── Adaptive sigma restart ──
        if (sigma_restart_patience > 0 and
                epochs_without_improvement >= sigma_restart_patience):
            sigma_vec = sigma_vec_init.clone()
            epochs_without_improvement = 0
            sigma_restarts += 1
            print(f"  [ES-HPC] σ restart #{sigma_restarts} → σ_vec reset "
                  f"(no improvement for {sigma_restart_patience} epochs)",
                  flush=True)

        # ── Record history ──
        history["epochs"].append(epoch)
        history["train_reward"].append(center_reward)
        history["pcg_iters_mean"].append(center_iters)
        history["pop_max"].append(float(np.max(rewards)))
        history["pop_min"].append(float(np.min(rewards)))
        history["pop_spread"].append(
            float(np.max(rewards) - np.min(rewards)))
        sigma_dict = {}
        for gname, gstart, gend in param_groups:
            sigma_dict[gname] = float(sigma_vec[gstart])
        history["sigma"].append(sigma_dict)
        history["best_reward_so_far"].append(best_reward)
        history["best_iters_so_far"].append(best_iters)
        history["eval_reward"].append(eval_reward)
        history["eval_iters"].append(eval_iters_mean)
        history["test_reward"].append(test_reward)
        history["test_iters"].append(test_iters_mean)
        history["grad_norm"].append(grad_norm)
        history["grad_snr"].append(grad_snr)

        ep_time = time.time() - ep_start
        if _TIMING:
            print(f"  [timing] epoch {epoch} pop={t_pop_done-ep_start:.1f} "
                  f"eval={t_eval_done-t_pop_done:.1f} soc={t_soc_done-t_eval_done:.1f} "
                  f"diag={time.time()-t_soc_done:.1f} total={ep_time:.1f}", flush=True)
        pop_spread = float(np.max(rewards) - np.min(rewards))
        pop_iters_range = \
            f"[{np.min(iters_all):.0f}, {np.max(iters_all):.0f}]"

        if do_eval:
            elapsed = time.time() - t_start
            sigma_str = ",".join(
                f"{g}:{float(sigma_vec[s]):.4f}"
                for g, s, e in param_groups)
            test_str = ""
            if test_reward is not None:
                test_str = (f"| R_test={test_reward:.4f} "
                            f"(test_iters={test_iters_mean:.1f}) ")
            print(
                f"  [ES-HPC] epoch {epoch:4d}/{n_epochs} "
                f"| R_train={center_reward:.4f} "
                f"| R_eval={eval_reward:.4f} "
                f"(eval_iters={eval_iters_mean:.1f}) "
                f"{test_str}"
                f"| best_R={best_reward:.7f} (iters={best_iters:.1f}) "
                f"| spread={pop_spread:.4f} | σ=[{sigma_str}] "
                f"| |∇|={grad_norm:.4f} SNR={grad_snr:.2f} "
                f"| pop_iters={pop_iters_range} "
                f"| {ep_time:.1f}s/ep | {elapsed:.0f}s total")

        # ── Periodic gradient quality summary ──
        if epoch > 0 and epoch % 25 == 0:
            recent_norms = history["grad_norm"][-25:]
            recent_snrs = history["grad_snr"][-25:]
            print(f"  [ES-HPC] --- Gradient quality (last 25 epochs) ---")
            print(f"    grad_norm: mean={np.mean(recent_norms):.4f}, "
                  f"std={np.std(recent_norms):.4f}")
            print(f"    grad_snr:  mean={np.mean(recent_snrs):.2f}, "
                  f"std={np.std(recent_snrs):.2f}")
            print(f"    sigma_restarts: {sigma_restarts}")
            for gname, gstart, gend in param_groups:
                gsigma = float(sigma_vec[gstart])
                print(f"    {gname}: σ={gsigma:.4f}, "
                      f"|∇|={group_grad_norms[gname]:.4f}")
            print(flush=True)

        # ── Periodic checkpoint ──
        if args.checkpoint_every > 0 and \
                (epoch + 1) % args.checkpoint_every == 0:
            if _DIAG_RUNNER is not None:
                for ep_tag, diag_snap in _DIAG_RUNNER.collect(block=True):
                    _record_diag(ep_tag, diag_snap)
            vector_to_parameters(best_theta, model.parameters())
            ckpt_path = os.path.join(args.out_dir,
                                     f"model_ep{epoch+1}.pt")
            torch.save(model.state_dict(), ckpt_path)
            vector_to_parameters(theta, model.parameters())

            # Also save full resume checkpoint
            save_checkpoint(
                resume_path, epoch, theta, best_theta, best_reward,
                best_iters, m_adam, v_adam, sigma_vec, sigma_vec_init,
                epochs_without_improvement, sigma_restarts, history,
                model.state_dict(),
            )
            # Save SoC evolution snapshot
            if soc_track_every > 0:
                _atomic_torch_save(soc_evolution, soc_path)
            # Save diagnostics evolution
            if diagnostic_every > 0 and history["diagnostics"]:
                with open(diag_path, "w") as _df:
                    json.dump(history["diagnostics"], _df, indent=2)

    finally:
        if _DIAG_RUNNER is not None:
            for ep_tag, diag_snap in _DIAG_RUNNER.collect(block=True):
                _record_diag(ep_tag, diag_snap)
            _DIAG_RUNNER.close()
        # Save emergency checkpoint on any exit
        if not _sigterm_received[0]:
            vector_to_parameters(theta, model.parameters())
            save_checkpoint(
                resume_path, epoch, theta, best_theta, best_reward,
                best_iters, m_adam, v_adam, sigma_vec, sigma_vec_init,
                epochs_without_improvement, sigma_restarts, history,
                model.state_dict(),
            )
        # Save SoC evolution on exit
        if soc_track_every > 0:
            _atomic_torch_save(soc_evolution, soc_path)
        # Save diagnostics on exit
        if diagnostic_every > 0 and history["diagnostics"]:
            with open(diag_path, "w") as _df:
                json.dump(history["diagnostics"], _df, indent=2)
        if disp is not None:
            print(f"  [multinode] {disp.stats()}", flush=True)
            disp.close()
        if pool is not None:
            pool.close()
            pool.join()
        if es_shm is not None:
            es_shm.close()
            es_shm.unlink()

    # ── Restore best ──
    vector_to_parameters(best_theta, model.parameters())
    print(f"\n  [ES-HPC] Done. Best: R={best_reward:.4f}, "
          f"iters={best_iters:.1f}")
    print(f"  [ES-HPC] Total sigma restarts: {sigma_restarts}")
    if overlap:
        print(f"  [overlap] speculative rollbacks: {n_rollbacks}")
    print(f"  [ES-HPC] Final grad_norm={history['grad_norm'][-1]:.4f}, "
          f"grad_snr={history['grad_snr'][-1]:.2f}")

    return history


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="HPC-optimized ES training for AMG edge policy")

    # Model
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--feature_preset", type=str,
                        default="new_algebraic_plus_relaxed")
    parser.add_argument("--k_per_row", type=int, default=4)
    parser.add_argument("--learn_B", action="store_true", default=False)
    parser.add_argument("--selection_mode", type=str, default="topk",
                        choices=["topk", "threshold"],
                        help="Edge selection: 'topk' (fixed k) or "
                             "'threshold' (adaptive-K)")
    parser.add_argument("--init_threshold", type=float, default=0.0,
                        help="Initial threshold bias for adaptive-K "
                             "(sigmoid(0)=0.5 → ~50%% edges selected)")
    parser.add_argument("--num_relaxed_probes", type=int, default=None,
                        help="Override number of relaxed probe vectors "
                             "(default: use preset value, typically 4)")

    # Problem generation
    parser.add_argument("--problem", type=str, default="lognormal",
                        choices=["poisson", "anisotropy", "lognormal", "perforated_l"],
                        help="PDE problem type for training data")
    parser.add_argument("--grid_n", type=int, default=5000)
    parser.add_argument("--train_samples", type=int, default=64,
                        help="Training set size (HPC default: 64)")

    # Reward
    parser.add_argument("--reward", choices=["iters", "residual"], default=None,
                        help="iters (default): -log(PCG iters) - λ·log(OC); residual: K-step "
                             "residual log-ratio vs SA-AMG - λ·log(OC) (env LSC_REWARD)")
    parser.add_argument("--reward_k", type=int, default=None,
                        help="K PCG iterations for --reward residual (default 16; env LSC_REWARD_K)")
    parser.add_argument("--reward_lambda_cx", type=float, default=2.0,
                        help="Weight for operator-complexity penalty in "
                             "reward: R = -log(iters) - λ·log(OC) (paper: 0.1; default 2.0, docs/METHODS.md)")
    parser.add_argument("--n_edge", type=int, default=72)

    # ES hyperparameters
    parser.add_argument("--es_pop_size", type=int, default=200,
                        help="Population size (half are antithetic)")
    parser.add_argument("--es_sigma", type=float, default=0.02,
                        help="Base perturbation std")
    parser.add_argument("--es_mini_batch", type=int, default=16,
                        help="Samples per eval (HPC default: 16)")
    parser.add_argument("--sigma_decay", type=float, default=0.999,
                        help="Per-epoch sigma decay (1.0 = constant)")
    parser.add_argument("--sigma_min", type=float, default=0.005,
                        help="Minimum sigma after decay")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--sigma_restart_patience", type=int, default=30,
                        help="Reset sigma if no improvement for N epochs")
    parser.add_argument("--sigma_scale_conv1", type=float, default=1.0)
    parser.add_argument("--sigma_scale_conv2", type=float, default=1.0)
    parser.add_argument("--sigma_scale_edge_mlp", type=float, default=1.0)
    parser.add_argument("--sigma_scale_emb_norm", type=float, default=1.0)
    parser.add_argument("--sigma_scale_threshold", type=float, default=2.5,
                        help="threshold exploration noise as a multiple of --es_sigma "
                             "(paper: 50, i.e. 1.0 z-units; default 2.5 = 0.05, docs/METHODS.md)")

    # Training
    parser.add_argument("--rl_epochs", type=int, default=500)
    parser.add_argument("--rl_lr", type=float, default=0.01)

    # Evaluation & saving
    parser.add_argument("--full_eval_every", type=int, default=5)
    parser.add_argument("--checkpoint_every", type=int, default=50)
    parser.add_argument("--eval_samples", type=int, default=10)
    parser.add_argument("--test_samples", type=int, default=32,
                        help="Separate test set size (generated with seed+1000)")
    parser.add_argument("--soc_track_every", type=int, default=5,
                        help="Track SoC (logits+mask) for all train samples "
                             "every N epochs (0=disabled)")
    parser.add_argument("--pretrain_ckpt", type=str, default=None,
                        help="Path to pre-trained model weights")
    parser.add_argument("--diagnostic_every", type=int, default=0,
                        help="Compute AMG diagnostics (CR rate, two-grid "
                             "factor, WAP error, nullspace) every N epochs "
                             "(0=disabled)")
    parser.add_argument("--diagnostic_samples", type=int, default=3,
                        help="Number of samples for diagnostic evaluation")

    # HPC / multiprocessing
    _slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 0))
    _default_workers = max(1, _slurm_cpus - 2) if _slurm_cpus > 0 \
        else min(8, (os.cpu_count() or 4))
    parser.add_argument("--fast_policy", type=str, default="benchmark",
                        choices=["benchmark", "all", "off"],
                        help="Exact fast inference (fast_policy.py) for "
                             "threshold-mode policies. benchmark (default): only "
                             "in the final benchmark, so training stays "
                             "bit-identical to the reference forward; all: also "
                             "in the workers and GPU forwards (faster, same "
                             "selections up to threshold ties, so histories are "
                             "not bit-identical to the reference); off: never.")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Where the population's GNN forwards run. auto = "
                             "GPU when available (threshold mode, n_workers>1).")
    parser.add_argument("--n_workers", type=int, default=_default_workers,
                        help=f"Worker processes (auto-detected: "
                             f"{_default_workers})")

    # Resume & SLURM utilities
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Override output directory (default: auto-generated)")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume from latest checkpoint in out_dir")
    parser.add_argument("--print_slurm", action="store_true", default=False,
                        help="Print a SLURM sbatch script and exit")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed (default: SLURM_ARRAY_TASK_ID or 0)")

    args = parser.parse_args()
    global _REWARD, _REWARD_K
    if args.reward is not None:
        os.environ["LSC_REWARD"] = _REWARD = args.reward
    if args.reward_k is not None:
        os.environ["LSC_REWARD_K"] = str(args.reward_k)
        _REWARD_K = args.reward_k

    # ── Handle --print_slurm ──
    if args.print_slurm:
        print_slurm_script(args)

    # ── Seed ──
    # Precedence: explicit --seed > SLURM_ARRAY_TASK_ID (keeps array tasks
    # distinct & reproducible) > a randomly drawn seed. When no seed is given
    # and this is not a SLURM array task, draw a random one so the run is still
    # reproducible after the fact (the value is recorded in config.json).
    if args.seed is None:
        slurm_task = os.environ.get("SLURM_ARRAY_TASK_ID")
        if slurm_task is not None:
            args.seed = int(slurm_task)
        else:
            args.seed = int.from_bytes(os.urandom(4), "little")
            print(f"  No --seed provided; drew random seed {args.seed}")

    # Make the seed actually govern the run: data generation
    # (make_spectral_train_set), model initialization, and ES perturbation noise.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"Seed: {args.seed}")

    # ── Feature config ──
    feat_cfg = FeatureConfig.from_preset(args.feature_preset)
    if args.num_relaxed_probes is not None:
        feat_cfg.num_relaxed_probes = args.num_relaxed_probes
    print(f"Feature preset: {args.feature_preset}")
    print(f"  node_dim={feat_cfg.node_dim}, "
          f"edge_raw_dim={feat_cfg.edge_raw_dim}, "
          f"num_relaxed_probes={feat_cfg.num_relaxed_probes}")
    set_fast_policy(feat_cfg if args.fast_policy == "all" else None)

    # ── Output directory ──
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
    slurm_array_id = os.environ.get("SLURM_ARRAY_TASK_ID", "")
    if args.out_dir:
        args.out_dir = ensure_dir(args.out_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        if slurm_job_id:
            dir_name = f"es_hpc_{slurm_job_id}"
            if slurm_array_id:
                dir_name += f"_{slurm_array_id}"
        else:
            dir_name = f"es_hpc_{timestamp}_seed{args.seed}"
        args.out_dir = ensure_dir(os.path.join("outputs", dir_name))
    print(f"Output directory: {args.out_dir}")

    # ── Build model ──
    model = AMGEdgePolicy(
        in_channels=feat_cfg.node_dim,
        hidden=args.hidden,
        learn_B=args.learn_B,
        edge_raw_dim=feat_cfg.edge_raw_dim,
        adaptive_k=(args.selection_mode == "threshold"),
        init_threshold=args.init_threshold,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: AMGEdgePolicy(hidden={args.hidden}), {n_params} params")

    # ── Load pretrained weights ──
    if args.pretrain_ckpt and os.path.isfile(args.pretrain_ckpt):
        state_dict = torch.load(args.pretrain_ckpt, map_location="cpu",
                                weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pre-trained weights: {args.pretrain_ckpt}")

    # ── Build training set ONCE ──
    print(f"\nBuilding {args.train_samples} training problems "
          f"(problem={args.problem}, n_edge={args.n_edge}, seed={args.seed})...")
    t0 = time.time()
    train_set = make_spectral_train_set(
        num_samples=args.train_samples,
        n_grid=args.grid_n,
        seed=args.seed,
        feature_config=feat_cfg,
        n_edge=args.n_edge,
        problem=args.problem,
    )
    print(f"  Done in {time.time()-t0:.1f}s. "
          f"Matrix size: {train_set[0].A.shape[0]}")

    # ── Save sample mesh plots for visualization ──
    from viz import plot_training_meshes
    plot_training_meshes(train_set, args.out_dir, num_samples=4, seed=args.seed)

    # ── Build separate test set ──
    test_set = None
    if args.test_samples > 0:
        print(f"\nBuilding {args.test_samples} test problems "
              f"(problem={args.problem}, seed={args.seed + 1000})...")
        t0 = time.time()
        test_set = make_spectral_train_set(
            num_samples=args.test_samples,
            n_grid=args.grid_n,
            seed=args.seed + 1000,
            feature_config=feat_cfg,
            n_edge=args.n_edge,
            problem=args.problem,
        )
        print(f"  Done in {time.time()-t0:.1f}s.")

    # ── Serialize training set to cache file (for debugging / resume) ──
    cache_path = os.path.join(args.out_dir, "train_set_cache.pkl")
    cache_size_mb = 0.0
    if "nopickle" in _SPEEDUPS:
        print(f"  [speedups] nopickle: not writing {cache_path}", flush=True)
    else:
        print(f"  Caching training set to {cache_path} ...", end=" ",
              flush=True)
        t0 = time.time()
        with open(cache_path, "wb") as f:
            pickle.dump(train_set, f, protocol=pickle.HIGHEST_PROTOCOL)
        cache_size_mb = os.path.getsize(cache_path) / (1024 * 1024)
        print(f"{cache_size_mb:.1f} MB in {time.time()-t0:.1f}s")

    node_pkl = os.path.join(args.out_dir, "node_samples.pkl")
    if "multinode" in _SPEEDUPS and _MN_PORT > 0:
        # The node agents load the identical train+test samples from here.
        t0 = time.time()
        with open(node_pkl, "wb") as f:
            pickle.dump(list(train_set) + (list(test_set) if test_set is not None else []),
                        f, protocol=pickle.HIGHEST_PROTOCOL)
        open(node_pkl + ".ready", "w").close()
        print(f"  [multinode] samples for the node agents written in {time.time()-t0:.1f}s", flush=True)

    # ── Create shared-memory cache for workers (zero-copy) ──
    print(f"  Creating shared-memory cache for workers...", flush=True)
    t0 = time.time()
    # The test samples go into the same block (after the training samples) so
    # the per-epoch test evaluation can run on the worker pool; workers address
    # them by index len(train_set) + i.
    n_train = len(train_set)
    shm_meta, shm_block = create_shared_cache(
        list(train_set) + (list(test_set) if test_set is not None else []))
    print(f"  SharedMemory ready in {time.time()-t0:.1f}s")

    # Replace train_set with shared-memory-backed version so the
    # original numpy/torch arrays can be freed (saves ~cache_size_mb).
    _all_samples, _main_shm_handle = load_from_shared_cache(shm_meta)
    train_set = _all_samples[:n_train]
    import gc; gc.collect()

    # ── Evaluate baseline (before training) ──
    n_eval = min(10, len(train_set))
    print(f"\nBaseline evaluation (before ES) on {n_eval} samples:")
    base_rs, base_its = [], []
    for i, s in enumerate(train_set[:n_eval]):
        r, it = es_evaluate(model, s, args.k_per_row,
                            selection_mode=args.selection_mode,
                            reward_lambda_cx=args.reward_lambda_cx)
        base_rs.append(r)
        base_its.append(it)
        print(f"  [{i+1}/{n_eval}] iters={it:3d}  reward={r:.4f}",
              flush=True)
    print(f"  Mean reward: {np.mean(base_rs):.4f}, "
          f"mean iters: {np.mean(base_its):.1f}")

    # ── Pre-training AMG diagnostics ──
    if args.diagnostic_every > 0:
        print(f"\nPre-training AMG diagnostics "
              f"({args.diagnostic_samples} samples):")
        diag_idx = list(range(min(args.diagnostic_samples, len(train_set))))
        if "diag" in _SPEEDUPS and args.n_workers > 1:
            global _DIAG_RUNNER
            _DIAG_RUNNER = DiagRunner(model, {
                "in_channels": feat_cfg.node_dim, "hidden": args.hidden,
                "learn_B": args.learn_B, "edge_raw_dim": feat_cfg.edge_raw_dim,
                "adaptive_k": (args.selection_mode == "threshold"),
                "init_threshold": args.init_threshold}, shm_meta,
                args.k_per_row, args.selection_mode)
            _DIAG_RUNNER.submit("pre", parameters_to_vector(model.parameters()).detach(),
                                diag_idx, args.seed)
            (_tag, pre_diag), = _DIAG_RUNNER.collect(block=True)
        else:
            pre_diag = compute_diagnostic_batch(
                train_set, model, args.k_per_row,
                selection_mode=args.selection_mode,
                sample_indices=diag_idx, seed=args.seed,
            )
        pre_diag_summary = {k: v for k, v in pre_diag.items()
                            if not k.endswith("_per_sample")}
        diag_pre_path = os.path.join(args.out_dir,
                                     "diagnostics_pre_training.json")
        with open(diag_pre_path, "w") as f:
            json.dump(pre_diag_summary, f, indent=2)
        print(f"  CR rate:            {pre_diag_summary['cr_rate']:.4f}")
        print(f"  V-cycle conv factor:{pre_diag_summary['vcycle_conv_factor']:.4f}")
        print(f"  WAP error:          {pre_diag_summary['wap_error']:.4f}")
        print(f"  Nullspace (mean):   {pre_diag_summary['nullspace_mean']:.2e}")
        print(f"  Nullspace (max):    {pre_diag_summary['nullspace_max']:.2e}")
        print(f"  κ(M⁻¹A):           {pre_diag_summary.get('kappa', float('nan')):.1f}")
        print(f"  SPD:                {pre_diag_summary.get('is_spd', 'n/a')}")
        print(f"  Levels:             {pre_diag_summary.get('n_levels', 'n/a')}")
        print(f"  PCG iterations:     {pre_diag_summary['pcg_iters']:.0f}")
        print(f"  Saved to {diag_pre_path}")

    # ── Print HPC summary ──
    slurm_info = ""
    if _slurm_cpus:
        slurm_info = (f"  SLURM: job={slurm_job_id}, "
                      f"cpus={_slurm_cpus}, array={slurm_array_id or 'N/A'}")
    print(f"\n  HPC configuration:")
    print(f"    n_workers: {args.n_workers}")
    print(f"    start_method: "
          f"{'forkserver' if platform.system() == 'Linux' else 'spawn'}")
    print(f"    cache: {cache_path} ({cache_size_mb:.1f} MB)")
    print(f"    shared_memory: {shm_meta['shm_name']} "
          f"({shm_meta['shm_size'] / (1024*1024):.1f} MB, zero-copy)")
    if slurm_info:
        print(slurm_info)

    # ── Save config ──
    config_dict = vars(args).copy()
    config_dict["n_params"] = n_params
    config_dict["feature_preset_details"] = {
        "node_dim": feat_cfg.node_dim,
        "edge_raw_dim": feat_cfg.edge_raw_dim,
    }
    config_dict["cache_size_mb"] = cache_size_mb
    config_dict["slurm_job_id"] = slurm_job_id
    config_dict["slurm_array_task_id"] = slurm_array_id
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    # ── Train ──
    try:
        history = train_es(model, train_set, args, shm_meta, test_set=test_set)

        # ── Save results ──
        with open(os.path.join(args.out_dir, "training_history.json"), "w") as f:
            json.dump(history, f, indent=2)
        torch.save(model.state_dict(),
                   os.path.join(args.out_dir, "model_best.pt"))
        print(f"\nResults saved to: {args.out_dir}")

        # ── Final evaluation ──
        n_eval = min(10, len(train_set))
        print(f"\nFinal evaluation (after ES) on {n_eval} samples:")
        final_rs, final_its = [], []
        for i, s in tqdm(enumerate(train_set[:n_eval]), total=n_eval):
            r, it = es_evaluate(model, s, args.k_per_row,
                                selection_mode=args.selection_mode,
                                reward_lambda_cx=args.reward_lambda_cx)
            final_rs.append(r)
            final_its.append(it)
            print(f"  [{i+1}/{n_eval}] iters={it:3d}  reward={r:.4f}",
                  flush=True)
        print(f"  Mean reward: {np.mean(final_rs):.4f}, "
              f"mean iters: {np.mean(final_its):.1f}")
        print(f"  Improvement: {np.mean(base_its):.1f} -> "
              f"{np.mean(final_its):.1f} iters")

        # ── Post-training AMG diagnostics ──
        if args.diagnostic_every > 0:
            print(f"\nPost-training AMG diagnostics "
                  f"({args.diagnostic_samples} samples):")
            diag_idx = list(range(min(args.diagnostic_samples, len(train_set))))
            post_diag = compute_diagnostic_batch(
                train_set, model, args.k_per_row,
                selection_mode=args.selection_mode,
                sample_indices=diag_idx, seed=args.seed,
            )
            post_diag_summary = {k: v for k, v in post_diag.items()
                                 if not k.endswith("_per_sample")}
            diag_post_path = os.path.join(args.out_dir,
                                          "diagnostics_post_training.json")
            with open(diag_post_path, "w") as f:
                json.dump(post_diag_summary, f, indent=2)
            print(f"  CR rate:            {post_diag_summary['cr_rate']:.4f}")
            print(f"  V-cycle conv factor:{post_diag_summary['vcycle_conv_factor']:.4f}")
            print(f"  WAP error:          {post_diag_summary['wap_error']:.4f}")
            print(f"  Nullspace (mean):   {post_diag_summary['nullspace_mean']:.2e}")
            print(f"  Nullspace (max):    {post_diag_summary['nullspace_max']:.2e}")
            print(f"  κ(M⁻¹A):           {post_diag_summary.get('kappa', float('nan')):.1f}")
            print(f"  SPD:                {post_diag_summary.get('is_spd', 'n/a')}")
            print(f"  Levels:             {post_diag_summary.get('n_levels', 'n/a')}")
            print(f"  PCG iterations:     {post_diag_summary['pcg_iters']:.0f}")
            # Before/after comparison
            if os.path.isfile(os.path.join(args.out_dir,
                                           "diagnostics_pre_training.json")):
                with open(os.path.join(args.out_dir,
                                       "diagnostics_pre_training.json")) as f:
                    pre = json.load(f)
                print(f"\n  Before → After training:")
                for k in ["cr_rate", "vcycle_conv_factor", "wap_error",
                          "kappa", "pcg_iters"]:
                    pre_v = pre.get(k, float("nan"))
                    post_v = post_diag_summary.get(k, float("nan"))
                    if np.isfinite(pre_v) and np.isfinite(post_v):
                        arrow = "↓" if post_v < pre_v else "↑"
                        fmt = ".0f" if k in ("pcg_iters",) else (".1f" if k == "kappa" else ".4f")
                        print(f"    {k:22s}: {pre_v:{fmt}} → "
                              f"{post_v:{fmt}} {arrow}")
                    else:
                        print(f"    {k:22s}: {pre_v} → {post_v}")
            print(f"  Saved to {diag_post_path}")

        # ── Final test set evaluation ──
        if test_set is not None:
            print(f"\nFinal test evaluation on {len(test_set)} samples:")
            test_rs, test_its = [], []
            for i, s in enumerate(test_set):
                r, it = es_evaluate(model, s, args.k_per_row,
                                    selection_mode=args.selection_mode,
                                    reward_lambda_cx=args.reward_lambda_cx)
                test_rs.append(r)
                test_its.append(it)
                print(f"  [{i+1}/{len(test_set)}] iters={it:3d}  reward={r:.4f}",
                      flush=True)
            print(f"  Test mean reward: {np.mean(test_rs):.4f}, "
                  f"mean iters: {np.mean(test_its):.1f}")

        # ── Preconditioner benchmark on test set ──
        if test_set is not None:
            try:
                from evaluate_preconditioners import benchmark_preconditioners
                benchmark_preconditioners(model, test_set, args)
            except Exception as e:
                print(f"\n  WARNING: Preconditioner benchmark failed: {e}")

    # ── Clean up shared memory and cache ──
    finally:
        if _main_shm_handle is not None:
            try:
                _main_shm_handle.close()
            except Exception:
                pass
        cleanup_shared_cache(shm_block)
        try:
            os.remove(cache_path)
            print(f"  Cleaned up cache: {cache_path}")
        except OSError:
            pass
        for p in (node_pkl, node_pkl + ".ready"):
            try:
                os.remove(p)
            except OSError:
                pass


if __name__ == "__main__":
    # forkserver is faster on Linux HPC; spawn is required on macOS
    if platform.system() == "Linux":
        mp.set_start_method("forkserver", force=True)
        if "preload" in _SPEEDUPS:
            # Import the heavy modules once in the forkserver so the 46
            # workers (and the diagnostics process) inherit them instead of
            # each importing torch/scipy/pyamg from /work concurrently.
            mp.set_forkserver_preload(["torch", "torch_geometric", "scipy.sparse",
                                       "scipy.sparse.linalg", "pyamg", "amg_core",
                                       "fast_policy", "shared_cache"])
    else:
        mp.set_start_method("spawn", force=True)
    main()
