#!/bin/bash
#SBATCH --job-name=gnn-sweep
#SBATCH --account=sander
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=/home/mok439/workspace/ML4SpatialAnalysis/logs/sweep/sweep-%A_%a.out

set -euo pipefail
module load conda/miniforge3/24.11.3-0
cd /home/mok439/workspace/ML4SpatialAnalysis
export PYTHONPATH=/home/mok439/workspace/ML4SpatialAnalysis:/home/mok439/workspace:${PYTHONPATH:-}
export WANDB_MODE=offline

CFG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" logs/sweep/manifest.txt)
echo "task ${SLURM_ARRAY_TASK_ID} -> ${CFG} on $(hostname)"
START=$(date +%s)
conda run --no-capture-output -n ML4SA python main.py --config "${CFG}"
echo "ELAPSED_SECONDS=$(( $(date +%s) - START ))"
