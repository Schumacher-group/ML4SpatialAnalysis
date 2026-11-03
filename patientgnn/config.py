"""Configuration loading with small, explicit validation."""

from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Set, Tuple


def load_yaml(path: str) -> Dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return _expand_paths(config)


def _expand_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_paths(item) for item in value]
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    return value


def validate_config(config: Dict[str, Any]) -> None:
    required = ["data", "evaluation", "model", "training", "output"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing configuration sections: {missing}")

    modes = config["evaluation"].get("feature_modes", [])
    valid_modes = {"expressions", "celltypes", "combined"}
    invalid = set(modes) - valid_modes
    if invalid or not modes:
        raise ValueError(
            "evaluation.feature_modes must contain one or more of "
            f"{sorted(valid_modes)}; got {modes}"
        )

    for key in ("cell_table", "metadata"):
        if key not in config["data"]:
            raise ValueError(f"Missing data.{key}")


def load_exclusions(data_config: Mapping[str, Any]) -> Tuple[Set[str], Set[str]]:
    """Return (LEAP IDs, FOV names) to drop from the cohort.

    Both pipelines read the same `exclusions_file` (YAML with `leap_ids` and
    `fovs` lists), so the ROI-level GNN and the patient benchmark cannot drift
    apart. Inline `exclude_leap_ids` / `exclude_fovs` keys are added on top.
    LEAP IDs are upper-cased and truncated to 7 characters, as elsewhere.
    """
    import yaml

    leap_ids = set(data_config.get("exclude_leap_ids") or [])
    fovs = set(data_config.get("exclude_fovs") or [])
    path = data_config.get("exclusions_file")
    if path:
        with open(os.path.expanduser(os.path.expandvars(path)), "r", encoding="utf-8") as handle:
            listed = yaml.safe_load(handle) or {}
        leap_ids |= set(listed.get("leap_ids") or [])
        fovs |= set(listed.get("fovs") or [])
    leap_ids = {str(value).upper()[:7] for value in leap_ids}
    fovs = {str(value) for value in fovs}
    return leap_ids, fovs
