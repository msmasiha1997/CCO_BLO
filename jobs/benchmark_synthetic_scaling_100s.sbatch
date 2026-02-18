#!/usr/bin/env bash
# Scaling suite with larger solver time budgets (CPP-* and KKTBL).
# Produces `runs/synthetic_scaling_100s_$SLURM_JOB_ID/`.
#
# Usage:
#   sbatch jobs/benchmark_synthetic_scaling_100s.sbatch
#
# Monitor:
#   squeue -u "$USER"
#   tail -f slurm/cco_scaling_100s_<jobid>.out

#SBATCH --job-name=cco_scaling_100s
#SBATCH --account=indy2
#SBATCH --partition=h100
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=5900M
#SBATCH --gpus=1
#SBATCH --output=slurm/cco_scaling_100s_%j.out
#SBATCH --error=slurm/cco_scaling_100s_%j.err

set -euo pipefail

REPO_DIR="/home/masiha/hypergrad/CCO_BLO"
CONFIG_PATH="configs/benchmark_synthetic_scaling_100s.yaml"

mkdir -p "${REPO_DIR}/slurm"
cd "${REPO_DIR}"

module load gcc/13.2.0 python/3.11.7
source .venv311/bin/activate

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

python -u -X faulthandler scripts/benchmark_synthetic_baselines.py \
  --config "${CONFIG_PATH}" \
  --out "synthetic_scaling_100s_${SLURM_JOB_ID:-manual}"

