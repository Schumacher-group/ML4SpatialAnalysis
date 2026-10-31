#!/bin/bash
#SBATCH --job-name=patient-gnn-v2
#SBATCH --account=sander
#SBATCH --partition=gpu_quad
#SBATCH --gres=gpu:teslaV100s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=patient-benchmark-%j.out

set -euo pipefail

module load conda/miniforge3/24.11.3-0
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1

cd /home/mok439/workspace/ML4SpatialAnalysis
conda run --no-capture-output -n ML4SA python -m patientgnn.run_patient_benchmark \
  --config configs/patient_benchmark.yaml \
  --models deepsets gnn gnn_edge_shuffle
