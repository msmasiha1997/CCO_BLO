#!/usr/bin/env bash
#SBATCH --job-name=cco_git_pull
#SBATCH --account=indy2
#SBATCH --partition=h100
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5900M
#SBATCH --gpus=1
#SBATCH --output=slurm/cco_git_pull_%j.out
#SBATCH --error=slurm/cco_git_pull_%j.err

set -euo pipefail

REPO_DIR="/home/masiha/hypergrad/CCO_BLO"

mkdir -p "${REPO_DIR}/slurm"
cd "${REPO_DIR}"

echo "START $(date) host=$(hostname) jobid=${SLURM_JOB_ID:-}"

git status -sb || true

had_changes=0
if ! git diff --quiet || ! git diff --cached --quiet; then
  had_changes=1
  git stash push -m "auto-stash before pull (job ${SLURM_JOB_ID:-manual})" || true
fi

git pull --ff-only

if [ "${had_changes}" -eq 1 ]; then
  git stash pop || true
fi

git status -sb || true
echo "DONE $(date)"
