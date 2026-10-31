"""Chunked, QC-aware construction of a compact ROI cache.

Only assay measurements and identifiers are cached.  Marker normalization is
fit later inside each training fold so validation/test distributions cannot
influence preprocessing.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np

from .schema import (
    ALL_MARKERS,
    CACHE_VERSION,
    DEFAULT_MARKERS,
    LABEL_MAP,
    MORPHOLOGY_FEATURES,
)


def _normalise_leap_id(values):
    return values.astype(str).str.split("_", n=1).str[0].str.upper().str[:7]


def _truthy(values):
    return values.astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes", "y"}
    )


def _config_fingerprint(data_config: Mapping[str, Any], markers: List[str]) -> str:
    payload = {
        "cache_version": CACHE_VERSION,
        "cell_table": _file_signature(data_config["cell_table"], checksum=False),
        "metadata": _file_signature(data_config["metadata"], checksum=True),
        "min_cells_per_roi": int(data_config.get("min_cells_per_roi", 50)),
        "audit_cell_thresholds": [
            int(value) for value in data_config.get("audit_cell_thresholds", [])
        ],
        "markers": markers,
        "require_qc_pass": bool(data_config.get("require_qc_pass", True)),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _file_signature(path: str, checksum: bool) -> Dict[str, Any]:
    absolute = os.path.abspath(path)
    stat = os.stat(absolute)
    signature = {
        "path": absolute,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if checksum:
        digest = hashlib.sha256()
        with open(absolute, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        signature["sha256"] = digest.hexdigest()
    return signature


def _read_metadata(path: str):
    import pandas as pd

    metadata = pd.read_csv(path)
    required = {
        "LEAP_ID",
        "Sample_Type_(pre/post treatment)",
        "NACT_Treatment _Group",
        "Response",
        "Patient_ID",
        "Stain_Batch",
    }
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")

    metadata = metadata.copy()
    metadata["LEAP_ID"] = metadata["LEAP_ID"].astype(str).str.upper().str[:7]
    metadata = metadata[
        metadata["Sample_Type_(pre/post treatment)"].astype(str).str.lower().eq("pre")
    ]
    metadata = metadata.dropna(subset=["NACT_Treatment _Group", "Response", "Patient_ID"])
    if metadata["LEAP_ID"].duplicated().any():
        duplicates = metadata.loc[metadata["LEAP_ID"].duplicated(), "LEAP_ID"].tolist()
        raise ValueError(f"Duplicate pre-treatment LEAP_ID values: {duplicates[:10]}")
    unknown = sorted(set(metadata["Response"]) - set(LABEL_MAP))
    if unknown:
        raise ValueError(f"Unknown response labels: {unknown}")
    return metadata.set_index("LEAP_ID")


def _numeric_matrix(frame, columns: Iterable[str]) -> np.ndarray:
    import pandas as pd

    converted = [pd.to_numeric(frame[column], errors="coerce").to_numpy() for column in columns]
    return np.column_stack(converted).astype(np.float32, copy=False)


def build_roi_cache(data_config: Mapping[str, Any], overwrite: bool = False) -> Dict[str, Any]:
    """Build once across concurrent Slurm jobs using an advisory file lock."""

    import fcntl

    cache_path = Path(data_config["cache_path"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(str(cache_path) + ".lock")
    with open(lock_path, "a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return _build_roi_cache_locked(data_config, overwrite=overwrite)


def _build_roi_cache_locked(data_config: Mapping[str, Any], overwrite: bool = False) -> Dict[str, Any]:
    """Build and atomically save the QC-filtered ROI cache.

    The source CSV is several gigabytes, so it is streamed in chunks.  The
    resulting cache is intentionally unnormalized; fold-specific scalers are
    fitted by the benchmark runner.
    """

    import pandas as pd

    cache_path = Path(data_config["cache_path"])
    markers = list(data_config.get("markers") or DEFAULT_MARKERS)
    unknown_markers = sorted(set(markers) - set(ALL_MARKERS))
    if unknown_markers:
        raise ValueError(f"Unknown requested markers: {unknown_markers}")
    fingerprint = _config_fingerprint(data_config, markers)

    if cache_path.exists() and not overwrite:
        cache = load_roi_cache(str(cache_path))
        if cache.get("fingerprint") != fingerprint:
            raise ValueError(
                f"Cache {cache_path} was built with different preprocessing settings. "
                "Use --overwrite-cache to rebuild it."
            )
        return cache

    metadata = _read_metadata(data_config["metadata"])
    header = pd.read_csv(data_config["cell_table"], nrows=0).columns.tolist()
    required = {
        "fov", "cell_meta_cluster", "centroid-0", "centroid-1", *markers,
        *MORPHOLOGY_FEATURES,
    }
    if data_config.get("require_qc_pass", True):
        required.add("qc_pass")
    missing = required - set(header)
    if missing:
        raise ValueError(f"Cell table is missing required columns: {sorted(missing)}")

    usecols = list(required)
    chunk_size = int(data_config.get("chunk_size", 150_000))
    require_qc = bool(data_config.get("require_qc_pass", True))
    accum = defaultdict(lambda: {"expression": [], "position": [], "morphology": [], "cell_type": []})
    audit = Counter()

    reader = pd.read_csv(data_config["cell_table"], usecols=usecols, chunksize=chunk_size)
    for chunk_number, chunk in enumerate(reader, start=1):
        audit["raw_cells"] += len(chunk)
        mask = chunk["cell_meta_cluster"].notna()
        mask &= ~chunk["cell_meta_cluster"].astype(str).eq("Unassigned")
        audit["assigned_cells"] += int(mask.sum())
        if require_qc:
            mask &= _truthy(chunk["qc_pass"])
        audit["qc_assigned_cells"] += int(mask.sum())

        leap_id = _normalise_leap_id(chunk["fov"])
        mask &= leap_id.isin(metadata.index)
        filtered = chunk.loc[mask].copy()
        audit["eligible_cells"] += len(filtered)
        if filtered.empty:
            continue

        for roi_id, roi_frame in filtered.groupby("fov", sort=False):
            item = accum[str(roi_id)]
            item["expression"].append(_numeric_matrix(roi_frame, markers))
            item["position"].append(_numeric_matrix(roi_frame, ["centroid-0", "centroid-1"]))
            item["morphology"].append(_numeric_matrix(roi_frame, MORPHOLOGY_FEATURES))
            item["cell_type"].append(roi_frame["cell_meta_cluster"].astype(str).to_numpy())

        if chunk_number % 10 == 0:
            print(
                f"processed {audit['raw_cells']:,} cells; "
                f"retained {audit['eligible_cells']:,}",
                flush=True,
            )

    min_cells = int(data_config.get("min_cells_per_roi", 50))
    audit_thresholds = sorted(
        {int(value) for value in data_config.get("audit_cell_thresholds", [])}
    )
    if min_cells < 1:
        raise ValueError("min_cells_per_roi must be positive")
    if any(value < 1 for value in audit_thresholds):
        raise ValueError("audit_cell_thresholds must contain positive integers")
    raw_rois = []
    cell_type_names = set()
    excluded_small = 0
    eligible_roi_sizes = []
    for roi_id, parts in accum.items():
        expression = np.concatenate(parts["expression"], axis=0)
        eligible_roi_sizes.append(len(expression))
        if len(expression) < min_cells:
            excluded_small += 1
            continue
        position = np.concatenate(parts["position"], axis=0)
        morphology = np.concatenate(parts["morphology"], axis=0)
        cell_type = np.concatenate(parts["cell_type"], axis=0)
        cell_type_names.update(cell_type.tolist())

        leap_id = str(roi_id).split("_", 1)[0].upper()[:7]
        row = metadata.loc[leap_id]
        raw_rois.append(
            {
                "roi_id": roi_id,
                "leap_id": leap_id,
                "patient_id": str(row["Patient_ID"]),
                "label": int(LABEL_MAP[str(row["Response"])]),
                "stain_batch": str(row["Stain_Batch"]),
                "treatment": str(row["NACT_Treatment _Group"]),
                "expression": expression,
                "position": position,
                "morphology": morphology,
                "cell_type_name": cell_type,
            }
        )

    vocabulary = sorted(cell_type_names)
    type_to_index = {name: index for index, name in enumerate(vocabulary)}
    rois = []
    for roi in raw_rois:
        names = roi.pop("cell_type_name")
        roi["cell_type"] = np.asarray([type_to_index[name] for name in names], dtype=np.int16)
        rois.append(roi)

    patient_labels = defaultdict(set)
    patient_batches = defaultdict(set)
    for roi in rois:
        patient_labels[roi["patient_id"]].add(roi["label"])
        patient_batches[roi["patient_id"]].add(roi["stain_batch"])
    conflicts = {patient: values for patient, values in patient_labels.items() if len(values) != 1}
    if conflicts:
        raise ValueError(f"Patients with conflicting response labels: {conflicts}")

    roi_sizes = np.asarray([len(roi["cell_type"]) for roi in rois], dtype=np.int64)
    eligible_roi_sizes = np.asarray(eligible_roi_sizes, dtype=np.int64)
    manifest = {
        **dict(audit),
        "preprocessing": {
            "cell_table": _file_signature(data_config["cell_table"], checksum=False),
            "metadata": _file_signature(data_config["metadata"], checksum=True),
            "require_qc_pass": require_qc,
            "min_cells_per_roi": min_cells,
            "audit_cell_thresholds": audit_thresholds,
            "markers": markers,
        },
        "eligible_rois_before_minimum": int(eligible_roi_sizes.size),
        "eligible_roi_size_distribution": _distribution_summary(eligible_roi_sizes),
        "roi_threshold_audit": {
            str(threshold): {
                "below": int((eligible_roi_sizes < threshold).sum()),
                "at_least": int((eligible_roi_sizes >= threshold).sum()),
            }
            for threshold in audit_thresholds
        },
        "excluded_small_rois": excluded_small,
        "retained_rois": len(rois),
        "patients": len(patient_labels),
        "responders": sum(next(iter(labels)) == 1 for labels in patient_labels.values()),
        "non_responders": sum(next(iter(labels)) == 0 for labels in patient_labels.values()),
        "cell_types": len(vocabulary),
        "cells_per_roi": _distribution_summary(roi_sizes),
        "patient_stain_batches": {key: sorted(value) for key, value in patient_batches.items()},
    }
    cache = {
        "version": CACHE_VERSION,
        "fingerprint": fingerprint,
        "markers": markers,
        "morphology_features": list(MORPHOLOGY_FEATURES),
        "cell_types": vocabulary,
        "rois": rois,
        "manifest": manifest,
    }
    _atomic_pickle(cache, cache_path)
    with open(str(cache_path) + ".manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return cache


def _distribution_summary(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {"min": 0, "median": 0, "mean": 0, "p90": 0, "max": 0}
    return {
        "min": int(values.min()),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p90": float(np.quantile(values, 0.9)),
        "max": int(values.max()),
    }


def _atomic_pickle(value: Any, path: Path) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = handle.name
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def load_roi_cache(path: str) -> Dict[str, Any]:
    with open(path, "rb") as handle:
        cache = pickle.load(handle)
    if cache.get("version") != CACHE_VERSION:
        raise ValueError(
            f"Unsupported cache version {cache.get('version')!r}; expected {CACHE_VERSION!r}"
        )
    validate_roi_cache(cache)
    return cache


def validate_roi_cache(cache: Mapping[str, Any]) -> None:
    required = {"markers", "cell_types", "rois", "manifest"}
    missing = required - set(cache)
    if missing:
        raise ValueError(f"ROI cache is missing keys: {sorted(missing)}")
    for roi in cache["rois"]:
        n = len(roi["cell_type"])
        if roi["expression"].shape != (n, len(cache["markers"])):
            raise ValueError(f"Bad expression shape in ROI {roi['roi_id']}")
        if roi["position"].shape != (n, 2):
            raise ValueError(f"Bad position shape in ROI {roi['roi_id']}")
        if roi["label"] not in (0, 1):
            raise ValueError(f"Bad label in ROI {roi['roi_id']}")


def group_rois_by_patient(cache: Mapping[str, Any]) -> List[Dict[str, Any]]:
    groups = defaultdict(list)
    for roi in cache["rois"]:
        groups[roi["patient_id"]].append(roi)
    patients = []
    for patient_id in sorted(groups, key=str):
        rois = sorted(groups[patient_id], key=lambda item: item["roi_id"])
        labels = {roi["label"] for roi in rois}
        if len(labels) != 1:
            raise ValueError(f"Patient {patient_id} has inconsistent labels")
        patients.append(
            {
                "patient_id": patient_id,
                "label": labels.pop(),
                "stain_batches": sorted({roi["stain_batch"] for roi in rois}),
                "rois": rois,
            }
        )
    return patients
