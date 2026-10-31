#!/bin/bash
#SBATCH --job-name=gnn-manifest
#SBATCH --account=sander
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=2:00:00

# one array task per line of $MANIFEST; pass --array, --output and --export=ALL,MANIFEST=... on the sbatch line
set -euo pipefail
module load conda/miniforge3/24.11.3-0
cd /home/mok439/workspace/ML4SpatialAnalysis
export PYTHONPATH=/home/mok439/workspace/ML4SpatialAnalysis:/home/mok439/workspace:${PYTHONPATH:-}
export WANDB_MODE=offline OMP_NUM_THREADS=8
# home is over quota, keep every cache off it
export WANDB_CACHE_DIR=/n/data1/hms/cellbio/sander/asif/ML4SA_Results/_staging/cache/wandb
export WANDB_CONFIG_DIR=/n/data1/hms/cellbio/sander/asif/ML4SA_Results/_staging/cache/wandb_config
export MPLCONFIGDIR=/n/data1/hms/cellbio/sander/asif/ML4SA_Results/_staging/cache/mpl
export PYTHONDONTWRITEBYTECODE=1

CFG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${MANIFEST:-logs/variants/manifest.txt}")
echo "task ${SLURM_ARRAY_TASK_ID} -> ${CFG} on $(hostname)"
START=$(date +%s)
conda run --no-capture-output -n ML4SA python main.py --config "${CFG}"
echo "ELAPSED_SECONDS=$(( $(date +%s) - START ))"
