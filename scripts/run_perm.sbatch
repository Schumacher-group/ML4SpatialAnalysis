#!/bin/bash
#SBATCH --job-name=gnn-perm
#SBATCH --account=sander
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=6:00:00
#SBATCH --output=/home/mok439/workspace/ML4SpatialAnalysis/logs/perm/perm-%A_%a.out

set -euo pipefail
module load conda/miniforge3/24.11.3-0
cd /home/mok439/workspace/ML4SpatialAnalysis
export PYTHONPATH=/home/mok439/workspace/ML4SpatialAnalysis:/home/mok439/workspace:${PYTHONPATH:-}
export WANDB_MODE=disabled PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4

O=/n/data1/hms/cellbio/sander/asif/ML4SA_Results/perm
# Identical protocol to the real calibrated run: grouped 5-fold, fixed spec 9
# (2 epochs, the regime where the model scored at chance), per-fold calibration on.
conda run --no-capture-output -n ML4SA python scripts/search_gnn.py \
  --cv grouped --folds 5 --rois-per-patient 4 --augment --aug-max-cells 0 --aug-jitter 0.0 \
  --fixed-spec 9 --permute ${SLURM_ARRAY_TASK_ID} \
  --out $O/perm_${SLURM_ARRAY_TASK_ID}.json
