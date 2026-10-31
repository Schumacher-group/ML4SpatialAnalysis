"""Interpretable patient-level features for strong classical baselines."""

from __future__ import annotations

from typing import Any, List, Mapping, Sequence, Tuple

import numpy as np

from .graphs import physical_contact_graph
from .normalization import MarkerNormalizer, stable_seed, stratified_cell_indices


def _expression_summary(values: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            np.mean(values, axis=0),
            np.std(values, axis=0),
            np.quantile(values, 0.1, axis=0),
            np.quantile(values, 0.5, axis=0),
            np.quantile(values, 0.9, axis=0),
        ]
    ).astype(np.float32)


def _composition_summary(cell_type: np.ndarray, num_cell_types: int) -> np.ndarray:
    counts = np.bincount(cell_type.astype(int), minlength=num_cell_types).astype(np.float64)
    proportions = counts / max(counts.sum(), 1.0)
    nonzero = proportions[proportions > 0]
    entropy = -float(np.sum(nonzero * np.log(nonzero))) / max(np.log(num_cell_types), 1.0)
    richness = float(np.count_nonzero(counts)) / max(num_cell_types, 1)
    return np.concatenate([proportions, [entropy, richness]]).astype(np.float32)


def _contact_enrichment(
    positions: np.ndarray,
    major_axis: np.ndarray,
    cell_type: np.ndarray,
    num_cell_types: int,
    graph_config: Mapping[str, Any],
) -> np.ndarray:
    edge_index, edge_attr, _ = physical_contact_graph(
        positions,
        major_axis,
        contact_scale=float(graph_config.get("contact_scale", 1.25)),
        max_neighbors=int(graph_config.get("max_neighbors", 16)),
        num_rbf=int(graph_config.get("num_rbf", 8)),
    )
    counts = np.bincount(cell_type.astype(int), minlength=num_cell_types).astype(np.float64)
    proportions = counts / max(counts.sum(), 1.0)
    pair_counts = np.zeros((num_cell_types, num_cell_types), dtype=np.float64)
    if edge_index.size:
        for edge_number, (source, target) in enumerate(edge_index.T):
            if source >= target:
                continue
            if edge_attr[edge_number, -1] > 0.5:
                continue
            left, right = sorted((int(cell_type[source]), int(cell_type[target])))
            pair_counts[left, right] += 1
    observed = pair_counts / max(pair_counts.sum(), 1.0)
    features = []
    for left in range(num_cell_types):
        for right in range(left, num_cell_types):
            expected = proportions[left] * proportions[right]
            if left != right:
                expected *= 2
            features.append(np.clip(np.log((observed[left, right] + 1e-4) / (expected + 1e-4)), -5, 5))
    return np.asarray(features, dtype=np.float32)


def baseline_feature_names(
    marker_names: Sequence[str],
    cell_type_names: Sequence[str],
    feature_mode: str,
    spatial: bool,
) -> List[str]:
    roi_names = []
    if feature_mode in {"expressions", "combined"}:
        for statistic in ("mean", "std", "q10", "median", "q90"):
            roi_names.extend([f"marker_{statistic}:{marker}" for marker in marker_names])
    if feature_mode in {"celltypes", "combined"}:
        roi_names.extend([f"cell_fraction:{name}" for name in cell_type_names])
        roi_names.extend(["celltype_entropy", "celltype_richness"])
        if spatial:
            for left, left_name in enumerate(cell_type_names):
                for right_name in cell_type_names[left:]:
                    roi_names.append(f"contact_enrichment:{left_name}|{right_name}")
    return [f"roi_mean:{name}" for name in roi_names] + [f"roi_std:{name}" for name in roi_names]


def patient_feature_matrix(
    patients: Sequence[Mapping[str, Any]],
    normalizer: MarkerNormalizer,
    feature_mode: str,
    num_cell_types: int,
    graph_config: Mapping[str, Any],
    spatial: bool = False,
    max_cells_per_roi: int = 2048,
    seed: int = 42,
) -> np.ndarray:
    rows = []
    for patient in patients:
        roi_features = []
        for roi in patient["rois"]:
            indices = stratified_cell_indices(
                roi["cell_type"],
                max_cells_per_roi,
                stable_seed(str(roi["roi_id"]), seed),
            )
            cell_type = np.asarray(roi["cell_type"])[indices]
            parts = []
            if feature_mode in {"expressions", "combined"}:
                expression = normalizer.transform(np.asarray(roi["expression"])[indices])
                parts.append(_expression_summary(expression))
            if feature_mode in {"celltypes", "combined"}:
                parts.append(_composition_summary(cell_type, num_cell_types))
                if spatial:
                    position = np.asarray(roi["position"])[indices]
                    major_axis = np.asarray(roi["morphology"])[indices, 3]
                    parts.append(
                        _contact_enrichment(
                            position, major_axis, cell_type, num_cell_types, graph_config
                        )
                    )
            roi_features.append(np.concatenate(parts))
        matrix = np.asarray(roi_features, dtype=np.float32)
        rows.append(np.concatenate([matrix.mean(axis=0), matrix.std(axis=0)]))
    return np.asarray(rows, dtype=np.float32)
