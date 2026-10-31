#!/bin/bash
#SBATCH --job-name=gnn-variants
#SBATCH --account=sander
#SBATCH --partition=gpu_requeue
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=/home/mok439/workspace/ML4SpatialAnalysis/logs/variants/slurm-%A_%a.out

set -euo pipefail

module load conda/miniforge3/24.11.3-0

cd /home/mok439/workspace/ML4SpatialAnalysis
export PYTHONPATH=/home/mok439/workspace/ML4SpatialAnalysis:/home/mok439/workspace:${PYTHONPATH:-}

# Compute nodes have no outbound network; online wandb would retry and stall.
# Metrics are recovered from the JSON sink written by AbstractModel.log_metrics.
export WANDB_MODE=offline

CFG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" logs/variants/manifest.txt)
echo "task ${SLURM_ARRAY_TASK_ID} -> ${CFG} on $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader

conda run --no-capture-output -n ML4SA python main.py --config "${CFG}"
