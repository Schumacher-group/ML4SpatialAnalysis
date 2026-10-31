"""Physical cell-contact graph construction and invariant edge features."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.spatial import cKDTree


def physical_contact_graph(
    positions: np.ndarray,
    major_axis_length: np.ndarray | None = None,
    contact_scale: float = 1.25,
    max_neighbors: int = 16,
    num_rbf: int = 8,
    fallback_radius_quantile: float = 0.9,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Construct a symmetric radius graph using the cell-size coordinate scale.

    The radius is derived from cell major-axis length, which shares units with
    centroid coordinates. Isolated cells receive one flagged nearest-neighbor
    edge so message passing remains defined without pretending that edge is a
    physical contact.
    """

    positions = np.asarray(positions, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(f"positions must have shape [n_cells, 2], got {positions.shape}")
    n_cells = len(positions)
    if n_cells == 0:
        return np.empty((2, 0), dtype=np.int64), np.empty((0, num_rbf + 3), dtype=np.float32), 1.0
    if not np.isfinite(positions).all():
        raise ValueError("Cell coordinates contain non-finite values")
    if max_neighbors < 1:
        raise ValueError("max_neighbors must be positive")

    size = None if major_axis_length is None else np.asarray(major_axis_length, dtype=np.float32).reshape(-1)
    valid_size = size is not None and len(size) == n_cells and np.any(np.isfinite(size) & (size > 0))
    if valid_size:
        size_mask = np.isfinite(size) & (size > 0)
        typical_diameter = float(np.quantile(size[size_mask], fallback_radius_quantile))
        # A partially missing morphology column must not create NaN edge
        # attributes. Use the ROI's training-independent geometric scale.
        size = np.where(size_mask, size, typical_diameter).astype(np.float32)
        radius = max(typical_diameter * contact_scale, 1e-6)
    elif n_cells > 1:
        nearest = cKDTree(positions).query(positions, k=2)[0][:, 1]
        radius = max(float(np.quantile(nearest, fallback_radius_quantile)) * contact_scale, 1e-6)
    else:
        radius = 1.0

    tree = cKDTree(positions)
    candidate_edges = {}
    neighborhoods = tree.query_ball_point(positions, radius)
    for source, neighbors in enumerate(neighborhoods):
        for target in neighbors:
            if target == source:
                continue
            key = (min(source, target), max(source, target))
            candidate_edges[key] = float(
                np.linalg.norm(positions[source] - positions[target])
            )

    degree = np.zeros(n_cells, dtype=np.int32)
    undirected = {}
    # Greedy shortest-edge selection enforces the degree cap after
    # symmetrization, unlike independently truncating directed neighborhoods.
    for (source, target), _ in sorted(candidate_edges.items(), key=lambda item: item[1]):
        if degree[source] >= max_neighbors or degree[target] >= max_neighbors:
            continue
        undirected[(source, target)] = False
        degree[source] += 1
        degree[target] += 1
    if n_cells > 1 and np.any(degree == 0):
        distances, indices = tree.query(positions[degree == 0], k=2)
        for source, target in zip(np.flatnonzero(degree == 0), indices[:, 1]):
            key = (min(int(source), int(target)), max(int(source), int(target)))
            undirected.setdefault(key, True)

    sources = []
    targets = []
    edge_features = []
    centers = np.linspace(0.0, 2.0, num_rbf, dtype=np.float32)
    width = float(centers[1] - centers[0]) if num_rbf > 1 else 1.0
    for (source, target), fallback in sorted(undirected.items()):
        distance = float(np.linalg.norm(positions[source] - positions[target]))
        normalized = distance / radius
        rbf = np.exp(-0.5 * ((np.clip(normalized, 0, 2) - centers) / width) ** 2)
        if valid_size:
            pair_contact = 0.5 * (float(size[source]) + float(size[target])) * contact_scale
            cell_gap = distance / max(pair_contact, 1e-6)
        else:
            cell_gap = normalized
        feature = np.concatenate(
            [rbf, np.asarray([normalized, cell_gap, float(fallback)], dtype=np.float32)]
        )
        for left, right in ((source, target), (target, source)):
            sources.append(left)
            targets.append(right)
            edge_features.append(feature)

    if not sources:
        return np.empty((2, 0), dtype=np.int64), np.empty((0, num_rbf + 3), dtype=np.float32), radius
    edge_index = np.asarray([sources, targets], dtype=np.int64)
    edge_attr = np.asarray(edge_features, dtype=np.float32)
    return edge_index, edge_attr, radius


def shuffle_edge_targets(edge_index: np.ndarray, seed: int) -> np.ndarray:
    """Degree-agnostic negative control used to test whether topology matters."""

    edge_index = np.asarray(edge_index, dtype=np.int64)
    if edge_index.size == 0:
        return edge_index.copy()
    rng = np.random.default_rng(seed)
    shuffled = edge_index.copy()
    shuffled[1] = rng.permutation(shuffled[1])
    non_self = shuffled[0] != shuffled[1]
    return shuffled[:, non_self]
