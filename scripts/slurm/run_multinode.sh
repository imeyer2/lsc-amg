#!/bin/bash
# Multi-node variant of run.sh for train_es_HPC.py (used by scripts/reproduce.py
# for configs with nodes > 1 in configs/experiments.csv). The training process
# runs on the first node of the job; one node agent (multinode.py) per other
# node runs the same worker pool and pulls population evaluations from it.
# History is identical to a single-node run when all nodes have the same CPU
# model; with mixed CPU models results differ at float-rounding level.
#
# Resources are passed on the sbatch command line by reproduce.py
# (-N <nodes> --ntasks-per-node=1 --cpus-per-task=...).
#SBATCH --requeue
#SBATCH --signal=B:TERM@120
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
OUTDIR=""; prev=""
for a in "$@"; do [ "${prev}" = "--out_dir" ] && OUTDIR="${a}"; prev="${a}"; done
[ -n "${OUTDIR}" ] || { echo "[run_multinode.sh] no --out_dir in command"; exit 2; }
mkdir -p "${OUTDIR}"
hosts=($(scontrol show hostnames "${SLURM_JOB_NODELIST}"))
export LSC_SPEEDUPS="${LSC_SPEEDUPS:-pcg,sched,soc,diag,overlap,nopickle,preload,stag},multinode"
export LSC_MN_PORT=$((20000 + SLURM_JOB_ID % 10000)) LSC_MN_AUTH="k${SLURM_JOB_ID}" LSC_MN_NODES=$((${#hosts[@]} - 1))
NW=$((SLURM_CPUS_PER_TASK - 2))
echo "[run_multinode.sh] job=${SLURM_JOB_ID} restart=${SLURM_RESTART_COUNT:-0} partition=${SLURM_JOB_PARTITION} hosts=${hosts[*]} cpus=${SLURM_CPUS_PER_TASK} code=$(git rev-parse --short HEAD 2>/dev/null)"
for h in "${hosts[@]}"; do
  srun -N1 -n1 -w "${h}" --overlap bash -c 'echo "[run_multinode.sh] $(hostname): $(lscpu | grep "Model name" | sed "s/ \+/ /g")"'
done
echo "[run_multinode.sh] $*"
rm -f "${OUTDIR}/node_samples.pkl" "${OUTDIR}/node_samples.pkl.ready"
AGENTS=()
for h in "${hosts[@]:1}"; do
  srun -N1 -n1 -w "${h}" -c "${SLURM_CPUS_PER_TASK}" python -u multinode.py \
    --host "$(hostname)" --port "${LSC_MN_PORT}" --authkey "${LSC_MN_AUTH}" \
    --out_dir "${OUTDIR}" --n_workers "${NW}" > "${OUTDIR}/agent_${h}.log" 2>&1 &
  AGENTS+=($!)
done
T0=$(date +%s)
"$@" &
PID=$!
trap 'echo "[run_multinode.sh] TERM -> python"; kill -TERM ${PID} 2>/dev/null' TERM
wait ${PID}; RC=$?
while kill -0 ${PID} 2>/dev/null; do wait ${PID}; RC=$?; done
T1=$(date +%s)
# The agents exit when the dispatcher closes; stop any that did not.
sleep 10
for a in "${AGENTS[@]}"; do kill "${a}" 2>/dev/null; done
OUTD=${LSC_OUTPUTS:-outputs}; mkdir -p "${OUTD}"
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "${SLURM_JOB_NAME}" "${SLURM_JOB_ID}" "${SLURM_RESTART_COUNT:-0}" \
  "${SLURM_JOB_PARTITION}" "$(hostname)+$((${#hosts[@]} - 1))" "$((T1 - T0))" "${RC}" >> "${OUTD}/wall_times.tsv"
echo "[run_multinode.sh] exit ${RC} after $((T1 - T0)) s"
exit ${RC}
