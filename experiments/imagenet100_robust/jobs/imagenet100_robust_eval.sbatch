#!/usr/bin/env bash
# Evaluate a trained run (writes `eval.json` into the run directory).
#
# Example:
#   sbatch jobs/imagenet100_robust_eval.sbatch \
#     --run-dir runs/imagenet100_robust/<run_name> \
#     --data-root /path/to/imagenet100

#SBATCH --job-name=imnet100_eval
#SBATCH --account=indy2
#SBATCH --partition=h100
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=5900M
#SBATCH --gpus=1
#SBATCH --output=slurm/imagenet100_robust_eval_%j.out
#SBATCH --error=slurm/imagenet100_robust_eval_%j.err

set -euo pipefail

REPO_DIR="/home/masiha/hypergrad/CCO_BLO"
mkdir -p "${REPO_DIR}/slurm"
cd "${REPO_DIR}"

module load gcc/13.2.0 python/3.11.7
source .venv311/bin/activate

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

python -u -X faulthandler experiments/imagenet100_robust/eval.py "$@"

