"""PyTorch Geometric datasets whose samples are patients, not ROIs."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np

from .graphs import physical_contact_graph, shuffle_edge_targets
from .normalization import MarkerNormalizer, stable_seed, stratified_cell_indices


class PatientGraphDataset:
    def __init__(
        self,
        patients: Sequence[Mapping[str, Any]],
        normalizer: MarkerNormalizer,
        feature_mode: str,
        graph_config: Mapping[str, Any],
        max_cells_per_roi: int = 768,
        use_graph: bool = True,
        shuffle_edges: bool = False,
        seed: int = 42,
    ):
        if feature_mode not in {"expressions", "celltypes", "combined"}:
            raise ValueError(f"Unknown feature mode: {feature_mode}")
        self.patients = list(patients)
        self.normalizer = normalizer
        self.feature_mode = feature_mode
        self.graph_config = dict(graph_config)
        self.max_cells_per_roi = int(max_cells_per_roi)
        self.use_graph = bool(use_graph)
        self.shuffle_edges = shuffle_edges
        self.seed = int(seed)
        self._cache: Dict[str, Any] = {}

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        patient = self.patients[index]
        graphs = [self._roi_graph(roi) for roi in patient["rois"]]
        return {
            "patient_id": patient["patient_id"],
            "label": int(patient["label"]),
            "graphs": graphs,
        }

    def _roi_graph(self, roi: Mapping[str, Any]):
        import torch
        from torch_geometric.data import Data

        roi_id = str(roi["roi_id"])
        if roi_id in self._cache:
            return self._cache[roi_id]
        indices = stratified_cell_indices(
            roi["cell_type"],
            self.max_cells_per_roi,
            stable_seed(roi_id, self.seed),
            min_per_type=int(self.graph_config.get("min_cells_per_type", 8)),
        )
        expression = self.normalizer.transform(roi["expression"][indices])
        cell_type = np.asarray(roi["cell_type"])[indices].astype(np.int64)
        position = np.asarray(roi["position"])[indices].astype(np.float32)

        morphology = np.asarray(roi["morphology"])[indices]
        major_axis_index = 3  # fixed by schema.MORPHOLOGY_FEATURES
        major_axis = morphology[:, major_axis_index]
        edge_dim = int(self.graph_config.get("num_rbf", 8)) + 3
        if self.use_graph:
            edge_index, edge_attr, radius = physical_contact_graph(
                position,
                major_axis,
                contact_scale=float(self.graph_config.get("contact_scale", 1.25)),
                max_neighbors=int(self.graph_config.get("max_neighbors", 16)),
                num_rbf=int(self.graph_config.get("num_rbf", 8)),
            )
            if self.shuffle_edges:
                edge_index = shuffle_edge_targets(edge_index, stable_seed(roi_id, self.seed + 991))
                edge_attr = edge_attr[: edge_index.shape[1]]
        else:
            edge_index = np.empty((2, 0), dtype=np.int64)
            edge_attr = np.empty((0, edge_dim), dtype=np.float32)
            radius = float("nan")

        if self.feature_mode == "celltypes":
            expression = np.empty((len(indices), 0), dtype=np.float32)
        data = Data(
            x=torch.from_numpy(expression),
            cell_type=torch.from_numpy(cell_type),
            edge_index=torch.from_numpy(edge_index),
            edge_attr=torch.from_numpy(edge_attr),
            num_nodes=len(indices),
        )
        data.roi_id = roi_id
        data.contact_radius = float(radius)
        self._cache[roi_id] = data
        return data


def collate_patient_graphs(samples):
    import torch
    from torch_geometric.data import Batch

    graphs = []
    roi_to_patient = []
    labels = []
    patient_ids = []
    for patient_index, sample in enumerate(samples):
        if not sample["graphs"]:
            raise ValueError(f"Patient {sample['patient_id']} has no eligible ROIs")
        graphs.extend(sample["graphs"])
        roi_to_patient.extend([patient_index] * len(sample["graphs"]))
        labels.append(sample["label"])
        patient_ids.append(sample["patient_id"])
    return {
        "graphs": Batch.from_data_list(graphs),
        "roi_to_patient": torch.tensor(roi_to_patient, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.float32),
        "patient_ids": patient_ids,
    }
