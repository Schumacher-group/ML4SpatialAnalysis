"""Configuration loading with small, explicit validation."""

from __future__ import annotations

import os
from typing import Any, Dict


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
