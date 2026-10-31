#!/bin/bash
#SBATCH --job-name=gnn-validate
#SBATCH --account=sander
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=/home/mok439/workspace/ML4SpatialAnalysis/logs/variants/validate-%j.out

set -euo pipefail
module load conda/miniforge3/24.11.3-0
cd /home/mok439/workspace/ML4SpatialAnalysis
export PYTHONPATH=/home/mok439/workspace/ML4SpatialAnalysis:/home/mok439/workspace:${PYTHONPATH:-}
export WANDB_MODE=offline

echo "=== one real-data epoch on CPU, timed ==="
date +%s > /tmp/gnnval.start
conda run --no-capture-output -n ML4SA python main.py --config logs/variants/cfg_VALIDATE_cpu.yaml
echo "ELAPSED_SECONDS=$(( $(date +%s) - $(cat /tmp/gnnval.start) ))"
