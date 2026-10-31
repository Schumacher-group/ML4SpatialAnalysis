"""Cohort exploration that avoids outcome-driven feature selection."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from .normalization import stable_seed


def cohort_report(cache: Mapping[str, Any], patients: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    label_name = {0: "non_responder", 1: "responder"}
    patient_by_label = Counter(label_name[p["label"]] for p in patients)
    roi_by_label = Counter(label_name[roi["label"]] for roi in cache["rois"])
    cells_by_label = Counter()
    batch_label = defaultdict(Counter)
    patient_batch_label = defaultdict(Counter)
    treatment_label = defaultdict(Counter)
    roi_counts = []
    cell_counts = []
    for patient in patients:
        roi_counts.append(len(patient["rois"]))
        for batch in patient["stain_batches"]:
            patient_batch_label[str(batch)][label_name[patient["label"]]] += 1
        for roi in patient["rois"]:
            count = len(roi["cell_type"])
            cell_counts.append(count)
            cells_by_label[label_name[patient["label"]]] += count
            batch_label[str(roi["stain_batch"])][label_name[patient["label"]]] += 1
            treatment_label[str(roi["treatment"])][label_name[patient["label"]]] += 1

    type_counts = np.zeros(len(cache["cell_types"]), dtype=np.int64)
    for roi in cache["rois"]:
        type_counts += np.bincount(roi["cell_type"].astype(int), minlength=len(type_counts))
    return {
        "cache_manifest": cache["manifest"],
        "patients_by_label": dict(patient_by_label),
        "rois_by_label": dict(roi_by_label),
        "cells_by_label": dict(cells_by_label),
        "roi_count_per_patient": _summary(roi_counts),
        "cells_per_roi": _summary(cell_counts),
        "roi_label_by_stain_batch": {key: dict(value) for key, value in sorted(batch_label.items())},
        "patient_label_by_stain_batch": {
            key: dict(value) for key, value in sorted(patient_batch_label.items())
        },
        "roi_label_by_treatment": {
            key: dict(value) for key, value in sorted(treatment_label.items())
        },
        "marker_distributions": _marker_report(cache, patients),
        "morphology_distributions": _morphology_report(cache),
        "cell_type_counts": {
            name: int(count) for name, count in zip(cache["cell_types"], type_counts)
        },
        "warnings": _confounding_warnings(batch_label),
    }


def _summary(values):
    values = np.asarray(values)
    return {
        "min": int(values.min()),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p90": float(np.quantile(values, 0.9)),
        "max": int(values.max()),
    }


def _confounding_warnings(batch_label):
    warnings = []
    for batch, counts in sorted(batch_label.items()):
        if len([count for count in counts.values() if count > 0]) < 2:
            warnings.append(
                f"Stain batch {batch} contains only one outcome class; batch and response "
                "cannot be statistically separated within that batch."
            )
    return warnings


def _marker_report(cache, patients, max_cells_per_patient=5000):
    sampled = []
    for patient in patients:
        values = np.concatenate([roi["expression"] for roi in patient["rois"]], axis=0)
        if len(values) > max_cells_per_patient:
            rng = np.random.default_rng(stable_seed(str(patient["patient_id"]), 8128))
            values = values[rng.choice(len(values), max_cells_per_patient, replace=False)]
        sampled.append(values)
    values = np.concatenate(sampled, axis=0)
    report = {}
    for index, marker in enumerate(cache["markers"]):
        column = values[:, index]
        finite = column[np.isfinite(column)]
        report[marker] = {
            "missing_fraction": float(1 - len(finite) / max(len(column), 1)),
            "q01": float(np.quantile(finite, 0.01)),
            "median": float(np.median(finite)),
            "q99": float(np.quantile(finite, 0.99)),
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
        }
    return report


def _morphology_report(cache):
    values = []
    for roi in cache["rois"]:
        morphology = np.asarray(roi["morphology"])
        if len(morphology) > 1000:
            rng = np.random.default_rng(stable_seed(str(roi["roi_id"]), 7163))
            morphology = morphology[rng.choice(len(morphology), 1000, replace=False)]
        values.append(morphology)
    matrix = np.concatenate(values, axis=0)
    report = {}
    for index, name in enumerate(cache["morphology_features"]):
        finite = matrix[:, index][np.isfinite(matrix[:, index])]
        report[name] = {
            "median": float(np.median(finite)),
            "q10": float(np.quantile(finite, 0.1)),
            "q90": float(np.quantile(finite, 0.9)),
        }
    return report
