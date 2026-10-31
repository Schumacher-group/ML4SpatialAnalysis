#!/bin/bash
#SBATCH --job-name=roi-cache
#SBATCH --account=sander
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --output=roi-cache-%j.out

set -euo pipefail

module load conda/miniforge3/24.11.3-0

cd /home/mok439/workspace/ML4SpatialAnalysis
# scripts/ lands on sys.path[0] when invoked by path, so the repo root needs to be
# on PYTHONPATH for mainutils/datautils to resolve.
export PYTHONPATH=/home/mok439/workspace/ML4SpatialAnalysis:/home/mok439/workspace:${PYTHONPATH:-}

conda run --no-capture-output -n ML4SA python scripts/build_roi_cache.py \
  --config configs/config.yaml \
  --seeds 42 43 44
