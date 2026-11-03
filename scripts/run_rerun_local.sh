#!/bin/bash
# Reruns the ROI-level evaluations and the patient benchmark on one workstation,
# using the corrected metadata (38 responders / 25 non-responders, 601 ROIs).
#
# Needs in data/ (neither file is tracked):
#   updated_cell_table_Ki67.csv
#   CleanCohort_Metadata.csv
#
# Usage, from an activated environment: bash scripts/run_rerun_local.sh [stage ...]
# Stages: cache baselines cv search benchmark. With no stage, all of them run.
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export WANDB_MODE=disabled PYTHONUNBUFFERED=1
# Asif's jobs used 4 threads each; stages can then run side by side after `cache`.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

CONFIG=configs/config_rerun.yaml
CACHE=data/updated_cell_table_Ki67_CleanCohort_Metadata_v2_qc_trainfit.pkl
OUT=artifacts/rerun
mkdir -p "$OUT"

STAGES=("$@")
[ ${#STAGES[@]} -gt 0 ] || STAGES=(cache baselines cv search benchmark)

for stage in "${STAGES[@]}"; do
	echo "=== ${stage} ==="
	case "$stage" in
		cache)
			# Reads the whole cell table into memory once, then reuses the cache.
			python scripts/build_roi_cache.py --config "$CONFIG" --seeds 42
			;;
		baselines)
			# ROI count, stain batch and mean-marker logistic regression on the CV folds.
			python scripts/run_grouped_cv.py --config "$CONFIG" --cache "$CACHE" \
				--baselines-only --out "$OUT/baselines.json"
			;;
		cv)
			# Every patient held out once per repeat. Runs on CPU.
			python scripts/run_grouped_cv.py --config "$CONFIG" --cache "$CACHE" \
				--out "$OUT/grouped_cv.json"
			;;
		search)
			# Asif's calibrated protocol from run_perm.sbatch. Drops single-class batches.
			python scripts/search_gnn.py --config "$CONFIG" --cache "$CACHE" \
				--cv grouped --folds 5 --rois-per-patient 4 --augment \
				--aug-max-cells 0 --aug-jitter 0.0 --fixed-spec 9 \
				--out "$OUT/search_spec9.json"
			# The same model on all 63 patients and every ROI at evaluation.
			python scripts/search_gnn.py --config "$CONFIG" --cache "$CACHE" \
				--cv grouped --folds 5 --rois-per-patient 4 --augment \
				--aug-max-cells 0 --aug-jitter 0.0 --fixed-spec 9 \
				--keep-all-batches --eval-all-rois \
				--out "$OUT/search_spec9_all_patients.json"
			;;
		benchmark)
			python -m patientgnn.run_patient_benchmark \
				--config configs/patient_benchmark.yaml --build-cache
			;;
		*)
			echo "Unknown stage: $stage" >&2
			exit 1
			;;
	esac
done
