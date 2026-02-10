#!/usr/bin/env bash
#SBATCH --job-name=cco_synth_sweeps
#SBATCH --account=indy2
#SBATCH --partition=h100
#SBATCH --qos=long
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=5900M
#SBATCH --gpus=1
#SBATCH --output=slurm/cco_synth_sweeps_%j.out
#SBATCH --error=slurm/cco_synth_sweeps_%j.err

set -euo pipefail

REPO_DIR="/home/masiha/hypergrad/CCO_BLO"
CONFIG_PATH="configs/benchmark_synthetic_sweeps.yaml"

mkdir -p "${REPO_DIR}/slurm"
cd "${REPO_DIR}"

module load gcc/13.2.0 python/3.11.7
source .venv311/bin/activate

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

echo "START $(date) host=$(hostname) jobid=${SLURM_JOB_ID:-}"
python -u -X faulthandler scripts/benchmark_synthetic_baselines.py --config "${CONFIG_PATH}" --out "synthetic_sweeps_${SLURM_JOB_ID:-manual}"
echo "DONE $(date)"

