#!/bin/bash
set -euo pipefail
cd /home/mok439/workspace/ML4SpatialAnalysis
cache_job=$(sbatch --parsable scripts/build_patient_cache.sbatch)
baseline_job=$(sbatch --parsable --dependency="afterok:${cache_job}" scripts/run_patient_baselines.sbatch)
sbatch --dependency="afterok:${baseline_job}" scripts/run_patient_benchmark.sbatch
