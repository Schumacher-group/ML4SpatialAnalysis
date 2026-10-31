#!/bin/bash
#SBATCH --job-name=gnn-search
#SBATCH --account=sander
#SBATCH --partition=priority
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=8:00:00
#SBATCH --output=/home/mok439/workspace/ML4SpatialAnalysis/logs/search/search-%A_%a.out

set -euo pipefail
module load conda/miniforge3/24.11.3-0
cd /home/mok439/workspace/ML4SpatialAnalysis
export PYTHONPATH=/home/mok439/workspace/ML4SpatialAnalysis:/home/mok439/workspace:${PYTHONPATH:-}
export WANDB_MODE=disabled
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1

ARGS=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" logs/search/jobs_pri2.txt)
echo "task ${SLURM_ARRAY_TASK_ID}: ${ARGS}"
START=$(date +%s)
conda run --no-capture-output -n ML4SA python scripts/search_gnn.py ${ARGS}
echo "ELAPSED_SECONDS=$(( $(date +%s) - START ))"
