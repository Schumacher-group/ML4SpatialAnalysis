"""Leakage-safe marker preprocessing fitted on training patients only."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np


def stable_seed(value: str, seed: int = 0) -> int:
    digest = hashlib.blake2b(f"{seed}:{value}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32 - 1)


@dataclass
class MarkerNormalizer:
    """Winsorize, impute, and robust-scale marker channels.

    Fit cells are sampled evenly per patient so a patient with many ROIs or
    many segmented cells cannot dominate normalization statistics.
    """

    transform_name: str = "identity"
    lower_quantile: float = 0.01
    upper_quantile: float = 0.99
    max_cells_per_patient: int = 20_000
    clip_value: float = 5.0
    include_missing_indicators: bool = False
    asinh_cofactor: float = 5.0
    seed: int = 42

    lower_: np.ndarray | None = None
    upper_: np.ndarray | None = None
    median_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, patients: Sequence[Mapping[str, Any]]) -> "MarkerNormalizer":
        sampled = []
        for patient in patients:
            blocks = [np.asarray(roi["expression"], dtype=np.float32) for roi in patient["rois"]]
            if not blocks:
                continue
            cells = np.concatenate(blocks, axis=0)
            if len(cells) > self.max_cells_per_patient:
                rng = np.random.default_rng(stable_seed(str(patient["patient_id"]), self.seed))
                indices = rng.choice(len(cells), self.max_cells_per_patient, replace=False)
                cells = cells[indices]
            sampled.append(self._variance_stabilize(cells))
        if not sampled:
            raise ValueError("Cannot fit marker normalization without training cells")

        fit_cells = np.concatenate(sampled, axis=0).astype(np.float64, copy=False)
        self.lower_ = np.nanquantile(fit_cells, self.lower_quantile, axis=0).astype(np.float32)
        self.upper_ = np.nanquantile(fit_cells, self.upper_quantile, axis=0).astype(np.float32)
        clipped = np.clip(fit_cells, self.lower_, self.upper_)
        self.median_ = np.nanmedian(clipped, axis=0).astype(np.float32)
        q25 = np.nanquantile(clipped, 0.25, axis=0)
        q75 = np.nanquantile(clipped, 0.75, axis=0)
        scale = np.asarray(q75 - q25, dtype=np.float32)
        fallback = np.nanstd(clipped, axis=0).astype(np.float32)
        scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, fallback)
        self.scale_ = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0).astype(np.float32)
        self.median_ = np.where(np.isfinite(self.median_), self.median_, 0.0).astype(np.float32)
        return self

    @property
    def output_multiplier(self) -> int:
        return 2 if self.include_missing_indicators else 1

    def transform(self, values: np.ndarray) -> np.ndarray:
        self._check_fitted()
        values = self._variance_stabilize(np.asarray(values, dtype=np.float32))
        missing = ~np.isfinite(values)
        values = np.where(missing, self.median_, values)
        values = np.clip(values, self.lower_, self.upper_)
        values = (values - self.median_) / self.scale_
        values = np.clip(values, -self.clip_value, self.clip_value).astype(np.float32)
        if self.include_missing_indicators:
            values = np.concatenate([values, missing.astype(np.float32)], axis=1)
        if not np.isfinite(values).all():
            raise ValueError("Non-finite values remain after marker normalization")
        return values

    def _variance_stabilize(self, values: np.ndarray) -> np.ndarray:
        if self.transform_name == "identity":
            return values
        if self.transform_name == "asinh":
            return np.arcsinh(values / self.asinh_cofactor)
        if self.transform_name == "log1p":
            if np.nanmin(values) <= -1:
                raise ValueError("log1p normalization requires all marker values to be greater than -1")
            return np.log1p(values)
        raise ValueError(f"Unknown marker transform: {self.transform_name}")

    def _check_fitted(self) -> None:
        if any(value is None for value in (self.lower_, self.upper_, self.median_, self.scale_)):
            raise RuntimeError("MarkerNormalizer must be fitted before transform")

    def state_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        for key in ("lower_", "upper_", "median_", "scale_"):
            value = result[key]
            result[key] = None if value is None else np.asarray(value).tolist()
        return result


def stratified_cell_indices(
    cell_types: np.ndarray,
    max_cells: int,
    seed: int,
    min_per_type: int = 8,
) -> np.ndarray:
    """Deterministically cap an ROI while retaining rare cell populations."""

    cell_types = np.asarray(cell_types)
    n_cells = len(cell_types)
    if max_cells <= 0 or n_cells <= max_cells:
        return np.arange(n_cells, dtype=np.int64)
    rng = np.random.default_rng(seed)
    selected: List[int] = []
    selected_mask = np.zeros(n_cells, dtype=bool)
    unique, counts = np.unique(cell_types, return_counts=True)

    reserve = min(min_per_type, max(1, max_cells // max(1, len(unique))))
    for cell_type, count in zip(unique, counts):
        candidates = np.flatnonzero(cell_types == cell_type)
        take = min(reserve, count)
        chosen = rng.choice(candidates, take, replace=False)
        selected.extend(chosen.tolist())
        selected_mask[chosen] = True

    remaining_slots = max_cells - len(selected)
    if remaining_slots > 0:
        remaining = np.flatnonzero(~selected_mask)
        chosen = rng.choice(remaining, min(remaining_slots, len(remaining)), replace=False)
        selected.extend(chosen.tolist())
    return np.asarray(sorted(selected), dtype=np.int64)

