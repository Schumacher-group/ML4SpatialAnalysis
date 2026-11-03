#!/bin/bash
# Reruns the response-prediction analyses for the eLife revision on one workstation.
#
# Cohort: corrected metadata, restricted to the 62-patient revision cohort used
# by the spatial analyses (configs/cohort_exclusions.yaml), with no filtering on
# the old qc_pass flag.
#
# Needs in data/ (neither file is tracked):
#   updated_cell_table_Ki67.csv
#   CleanCohort_Metadata.csv
#
# Usage, from an activated environment: bash scripts/run_rerun_local.sh [stage ...]
# Default stages, in order:
#   check              refuse to start until the five excluded FOVs are listed
#   benchmark          patient-level benchmark, 5 folds x 5 repeats: prevalence,
#                      ROI-count and stain-batch confound baselines, non-spatial
#                      logistic / RF / XGBoost, spatial logistic, DeepSets (no
#                      edges), GNN and the edge-shuffled GNN, for each feature set
#   benchmark_balanced the same, after dropping single-class stain batches
#   cache              ROI-level cache for the original GCN
#   gcn                original ROI-level GCN (k-at-most, k=7, markers + cell
#                      types), patient-grouped 5-fold CV x 5 repeats
#   gcn_balanced       the same, after dropping single-class stain batches
#   table              collect everything into artifacts/rerun/table2.md
# Optional stage: search (Asif's nested-CV protocol; not part of Table 2).
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export WANDB_MODE=disabled PYTHONUNBUFFERED=1
# Asif's jobs used 4 threads each; the stages after `cache` can run side by side.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

CONFIG=configs/config_rerun.yaml
BENCH=configs/patient_benchmark.yaml
OUT=artifacts/rerun
mkdir -p "$OUT"
base_cache() { python scripts/build_roi_cache.py --config "$CONFIG" --print-base-cache; }

STAGES=("$@")
[ ${#STAGES[@]} -gt 0 ] || STAGES=(check benchmark benchmark_balanced cache gcn gcn_balanced table)

for stage in "${STAGES[@]}"; do
	echo "=== ${stage} ==="
	case "$stage" in
		check)
			python - <<'PY'
import os, sys
from patientgnn.config import load_exclusions
leap_ids, fovs = load_exclusions({'exclusions_file': 'configs/cohort_exclusions.yaml'})
print(f'excluded LEAP IDs: {sorted(leap_ids)}; excluded FOVs: {sorted(fovs)}')
if len(fovs) != 5 and not os.environ.get('ALLOW_INCOMPLETE_EXCLUSIONS'):
    sys.exit('configs/cohort_exclusions.yaml must list the five excluded FOVs '
             '(set ALLOW_INCOMPLETE_EXCLUSIONS=1 to override).')
PY
			;;
		benchmark)
			python -m patientgnn.run_patient_benchmark --config "$BENCH" --build-cache \
				--output-directory "$OUT/patient_benchmark"
			;;
		benchmark_balanced)
			python -m patientgnn.run_patient_benchmark --config "$BENCH" --build-cache \
				--drop-single-class-batches --output-directory "$OUT/patient_benchmark_balanced"
			;;
		cache)
			# Reads the whole cell table into memory once, then reuses the cache.
			python scripts/build_roi_cache.py --config "$CONFIG" --seeds 42
			;;
		gcn)
			python scripts/run_grouped_cv.py --config "$CONFIG" --cache "$(base_cache)" \
				--gconv gcn --folds 5 --repeats 5 --epochs 5 --out "$OUT/gcn_grouped_cv.json"
			;;
		gcn_balanced)
			python scripts/run_grouped_cv.py --config "$CONFIG" --cache "$(base_cache)" \
				--gconv gcn --folds 5 --repeats 5 --epochs 5 --drop-single-class-batches \
				--out "$OUT/gcn_grouped_cv_balanced.json"
			;;
		table)
			python scripts/make_table2.py --rerun-dir "$OUT" --out "$OUT/table2.md"
			;;
		search)
			python scripts/search_gnn.py --config "$CONFIG" --cache "$(base_cache)" \
				--cv grouped --folds 5 --rois-per-patient 4 --augment \
				--aug-max-cells 0 --aug-jitter 0.0 --fixed-spec 9 \
				--out "$OUT/search_spec9.json"
			;;
		*)
			echo "Unknown stage: $stage" >&2
			exit 1
			;;
	esac
done
