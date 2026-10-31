#!/bin/bash
#SBATCH --job-name=gnn-cv
#SBATCH --account=sander
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=11:00:00
#SBATCH --output=/home/mok439/workspace/ML4SpatialAnalysis/logs/cv/cv-%A_%a.out

set -euo pipefail
module load conda/miniforge3/24.11.3-0
cd /home/mok439/workspace/ML4SpatialAnalysis
export PYTHONPATH=/home/mok439/workspace/ML4SpatialAnalysis:/home/mok439/workspace:${PYTHONPATH:-}
export WANDB_MODE=disabled

ARGS=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" logs/cv/jobs.txt)
echo "task ${SLURM_ARRAY_TASK_ID}: ${ARGS}"
START=$(date +%s)
conda run --no-capture-output -n ML4SA python scripts/run_grouped_cv.py ${ARGS}
echo "ELAPSED_SECONDS=$(( $(date +%s) - START ))"
