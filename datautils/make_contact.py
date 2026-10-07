from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd


def neighbourhood_matrices(
    adata,
    fovs: Iterable[str],
    fov_key: str = "fov",
    cluster_key: str = "cell_meta_cluster",
    radius: float = 20.0,
    min_cells_per_type: int = 50,
) -> Dict[str, dict]:
    """Return contact and enrichment matrices for each requested ROI."""

    import squidpy as sq

    store = {}
    for roi_id in sorted(set(fovs)):
        subset = adata[adata.obs[fov_key] == roi_id].copy()
        if subset.n_obs < 2:
            continue
        counts = subset.obs[cluster_key].value_counts()
        retained = counts[counts >= min_cells_per_type].index
        if len(retained) == 0:
            continue
        sq.gr.spatial_neighbors(subset, coord_type="generic", radius=radius)
        sq.gr.nhood_enrichment(subset, cluster_key=cluster_key)
        sq.gr.interaction_matrix(subset, cluster_key=cluster_key)

        categories = np.asarray(subset.obs[cluster_key].cat.categories)
        enrichment = pd.DataFrame(
            subset.uns[f"{cluster_key}_nhood_enrichment"]["zscore"],
            index=categories,
            columns=categories,
        ).loc[retained, retained]
        contacts = np.asarray(subset.uns[f"{cluster_key}_interactions"], dtype=float)
        contacts /= max(float(contacts.sum()), 1.0)
        contacts = pd.DataFrame(contacts, index=categories, columns=categories).loc[
            retained, retained
        ]
        store[str(roi_id)] = {"interaction": contacts, "enrichment": enrichment}
    return store


def contact_radius_from_cell_table(cell_table, quantile: float = 0.9) -> float:
    values = pd.to_numeric(cell_table["major_axis_length"], errors="coerce")
    radius = float(values.quantile(quantile))
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Cannot estimate a positive contact radius")
    return radius
