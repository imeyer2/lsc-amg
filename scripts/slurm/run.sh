#!/bin/bash
# Generic SLURM wrapper used by scripts/reproduce.py and scripts/twolevel.py: runs one command from the
# repository root inside .venv, forwards SIGTERM (preemption / time limit) to
# it so train_es_HPC.py can checkpoint, and appends wall time to a log.
#
# Resources (cpus, memory, time, partition, account, gres) are passed on the
# sbatch command line by reproduce.py; these are only fallbacks.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@120
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "[run.sh] job=${SLURM_JOB_ID} restart=${SLURM_RESTART_COUNT:-0} partition=${SLURM_JOB_PARTITION} host=$(hostname) cpus=${SLURM_CPUS_PER_TASK} code=$(git rev-parse --short HEAD 2>/dev/null)"
echo "[run.sh] $*"
T0=$(date +%s)
"$@" &
PID=$!
trap 'echo "[run.sh] TERM -> python"; kill -TERM ${PID} 2>/dev/null' TERM
wait ${PID}; RC=$?
while kill -0 ${PID} 2>/dev/null; do wait ${PID}; RC=$?; done
T1=$(date +%s)
OUTD=${LSC_OUTPUTS:-outputs}; mkdir -p "${OUTD}"
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "${SLURM_JOB_NAME}" "${SLURM_JOB_ID}" "${SLURM_RESTART_COUNT:-0}" \
  "${SLURM_JOB_PARTITION}" "$(hostname)" "$((T1 - T0))" "${RC}" >> "${OUTD}/wall_times.tsv"
echo "[run.sh] exit ${RC} after $((T1 - T0)) s"
exit ${RC}
