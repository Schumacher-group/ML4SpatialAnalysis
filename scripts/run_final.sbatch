#!/bin/bash
#SBATCH --job-name=gnn-conf
#SBATCH --account=sander
#SBATCH --partition=medium
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=16:00:00
#SBATCH --output=/home/mok439/workspace/ML4SpatialAnalysis/logs/final/f-%A_%a.out
set -euo pipefail
module load conda/miniforge3/24.11.3-0
cd /home/mok439/workspace/ML4SpatialAnalysis
export PYTHONPATH=/home/mok439/workspace/ML4SpatialAnalysis:/home/mok439/workspace:${PYTHONPATH:-}
export WANDB_MODE=disabled PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
ARGS=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" logs/final/jobs.txt)
echo "task ${SLURM_ARRAY_TASK_ID}: ${ARGS}"
conda run --no-capture-output -n ML4SA python scripts/search_gnn.py ${ARGS}
