#!/usr/bin/env python3
"""Run leakage-safe patient-level baselines and spatial-model ablations."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from patientgnn.baselines import (
    fit_tabular_baseline,
    patient_feature_matrices,
    prevalence_baseline,
)
from patientgnn.config import load_yaml, validate_config
from patientgnn.exploration import cohort_report
from patientgnn.metrics import (
    bootstrap_auc_ci,
    bootstrap_auc_difference_ci,
    patient_metrics,
)
from patientgnn.normalization import MarkerNormalizer
from patientgnn.preprocessing import build_roi_cache, group_rois_by_patient, load_roi_cache


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/patient_benchmark.yaml")
    parser.add_argument("--build-cache", action="store_true")
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--explore-only", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--models",
        nargs="+",
        help="Optional subset of evaluation.models (useful for CPU/GPU job separation)",
    )
    parser.add_argument(
        "--feature-modes",
        nargs="+",
        choices=["expressions", "celltypes", "combined"],
        help="Optional subset of configured feature modes",
    )
    return parser.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(False)
    except ImportError:
        pass


def atomic_json(value, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = handle.name
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
    os.replace(temporary, path)


def normalizer_from_config(config, seed):
    settings = dict(config["normalization"])
    settings["seed"] = seed
    return MarkerNormalizer(**settings)


def validate_expected_cohort(cache, patients, expected):
    if not expected:
        return
    observed = {
        "patients": len(patients),
        "responders": sum(patient["label"] == 1 for patient in patients),
        "non_responders": sum(patient["label"] == 0 for patient in patients),
        "rois": len(cache["rois"]),
    }
    mismatches = {
        key: {"expected": int(value), "observed": int(observed[key])}
        for key, value in expected.items()
        if key in observed and int(value) != int(observed[key])
    }
    if mismatches:
        raise ValueError(
            "Cached cohort does not match the configured manuscript cohort: "
            f"{mismatches}"
        )


def _folds(patients, evaluation_config):
    from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

    labels = np.asarray([patient["label"] for patient in patients])
    dummy = np.zeros((len(patients), 1))
    n_splits = int(evaluation_config.get("outer_folds", 5))
    repeats = int(evaluation_config.get("repeats", 1))
    seed = int(evaluation_config.get("seed", 42))
    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=repeats, random_state=seed
    )
    for split_number, (outer_train, test) in enumerate(splitter.split(dummy, labels)):
        repeat = split_number // n_splits
        fold = split_number % n_splits
        train, validation = train_test_split(
            outer_train,
            test_size=float(evaluation_config.get("validation_fraction", 0.2)),
            stratify=labels[outer_train],
            random_state=seed + 1009 * repeat + fold,
        )
        yield repeat, fold, train, validation, test


def _subset(items, indices):
    return [items[int(index)] for index in indices]


def _partition_manifest(patients):
    return [
        {
            "patient_id": patient["patient_id"],
            "label": int(patient["label"]),
            "n_rois": len(patient["rois"]),
            "n_cells": int(sum(len(roi["cell_type"]) for roi in patient["rois"])),
        }
        for patient in patients
    ]


def _save_pickle(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = handle.name
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def _write_split_summary(output_directory: Path):
    import csv

    rows = []
    for path in sorted((output_directory / "splits").glob("*.json")):
        with open(path, encoding="utf-8") as handle:
            split = json.load(handle)
        row = {"repeat": split["repeat"], "fold": split["fold"]}
        for name, patients in split["partitions"].items():
            row[f"{name}_patients"] = len(patients)
            row[f"{name}_responders"] = sum(item["label"] == 1 for item in patients)
            row[f"{name}_non_responders"] = sum(item["label"] == 0 for item in patients)
            row[f"{name}_rois"] = sum(item["n_rois"] for item in patients)
            row[f"{name}_cells"] = sum(item["n_cells"] for item in patients)
        rows.append(row)
    if not rows:
        return
    columns = ["repeat", "fold"]
    for name in ("train", "validation", "test"):
        columns.extend(
            [
                f"{name}_patients",
                f"{name}_responders",
                f"{name}_non_responders",
                f"{name}_rois",
                f"{name}_cells",
            ]
        )
    path = output_directory / "split_summary.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run_neural(
    model_name,
    feature_mode,
    train_patients,
    validation_patients,
    test_patients,
    normalizer,
    cache,
    config,
    seed,
    checkpoint_path,
):
    import torch

    from patientgnn.dataset import PatientGraphDataset
    from patientgnn.models import HierarchicalPatientModel
    from patientgnn.training import train_neural_fold

    use_graph = model_name != "deepsets"
    shuffle_edges = model_name == "gnn_edge_shuffle"
    dataset_kwargs = {
        "normalizer": normalizer,
        "feature_mode": feature_mode,
        "graph_config": config["graph"],
        "max_cells_per_roi": int(config["data"].get("max_cells_per_roi", 768)),
        "use_graph": use_graph,
        "shuffle_edges": shuffle_edges,
        "seed": seed,
    }
    train_dataset = PatientGraphDataset(train_patients, **dataset_kwargs)
    validation_dataset = PatientGraphDataset(validation_patients, **dataset_kwargs)
    test_dataset = PatientGraphDataset(test_patients, **dataset_kwargs)

    expression_dim = len(cache["markers"]) * normalizer.output_multiplier
    edge_dim = int(config["graph"].get("num_rbf", 8)) + 3
    model = HierarchicalPatientModel(
        expression_dim=expression_dim,
        num_cell_types=len(cache["cell_types"]),
        edge_dim=edge_dim,
        feature_mode=feature_mode,
        use_graph=use_graph,
        **config["model"],
    )
    requested_device = str(config["training"].get("device", "auto"))
    device = (
        "cuda"
        if requested_device == "auto" and torch.cuda.is_available()
        else "cpu"
        if requested_device == "auto"
        else requested_device
    )
    result, checkpoint = train_neural_fold(
        model,
        train_dataset,
        validation_dataset,
        test_dataset,
        config["training"],
        device,
    )
    checkpoint["normalizer"] = normalizer.state_dict()
    checkpoint["feature_mode"] = feature_mode
    checkpoint["model_name"] = model_name
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    return result


def summarize_results(output_directory: Path, repeats: int):
    grouped = defaultdict(list)
    for path in sorted((output_directory / "folds").glob("*.json")):
        with open(path, encoding="utf-8") as handle:
            result = json.load(handle)
        grouped[(result["model"], result["feature_mode"])].append(result)

    rows = []
    detailed = {}
    for (model, mode), fold_results in sorted(grouped.items()):
        repeat_metrics = []
        for repeat in range(repeats):
            selected = [item for item in fold_results if item["repeat"] == repeat]
            predictions = [prediction for item in selected for prediction in item["predictions"]]
            if not predictions:
                continue
            ids = [item["patient_id"] for item in predictions]
            if len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate out-of-fold predictions for {model}/{mode}/repeat {repeat}")
            y = np.asarray([item["label"] for item in predictions])
            probability = np.asarray([item["probability"] for item in predictions])
            metrics = patient_metrics(y, probability, threshold=0.5)
            metrics["roc_auc_ci"] = bootstrap_auc_ci(
                y, probability, seed=4049 + repeat, replicates=2000
            )
            repeat_metrics.append(metrics)
        if not repeat_metrics:
            continue
        aucs = [item["roc_auc"] for item in repeat_metrics]
        fold_test_auc = [item["test"]["roc_auc"] for item in fold_results]
        best_epochs = [
            int(item["best_epoch"])
            for item in fold_results
            if item.get("best_epoch") is not None
        ]
        row = {
            "model": model,
            "feature_mode": mode,
            "repeat_auc_mean": float(np.mean(aucs)),
            "repeat_auc_std": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
            "oof_auc_ci_low": float(
                np.mean([item["roc_auc_ci"][0] for item in repeat_metrics])
            ),
            "oof_auc_ci_high": float(
                np.mean([item["roc_auc_ci"][1] for item in repeat_metrics])
            ),
            "fold_auc_mean": float(np.mean(fold_test_auc)),
            "fold_auc_std": float(np.std(fold_test_auc, ddof=1)) if len(fold_test_auc) > 1 else 0.0,
            "mean_best_epoch": float(np.mean(best_epochs)) if best_epochs else None,
        }
        rows.append(row)
        detailed[f"{model}:{mode}"] = {"summary": row, "repeats": repeat_metrics}

    paired = _paired_comparisons(grouped, repeats)
    atomic_json(
        {
            "models": detailed,
            "paired_auc_differences_vs_gnn_combined": paired,
        },
        output_directory / "comparison.json",
    )
    _write_comparison_csv(rows, output_directory / "comparison.csv")
    _write_markdown(rows, output_directory / "comparison.md")
    return rows


def _prediction_map(fold_results, repeat):
    selected = [item for item in fold_results if item["repeat"] == repeat]
    predictions = [prediction for item in selected for prediction in item["predictions"]]
    return {
        item["patient_id"]: (int(item["label"]), float(item["probability"]))
        for item in predictions
    }


def _paired_comparisons(grouped, repeats):
    reference_key = ("gnn", "combined")
    if reference_key not in grouped:
        return {}
    output = {}
    for key, fold_results in sorted(grouped.items()):
        if key == reference_key:
            continue
        repeat_results = []
        for repeat in range(repeats):
            reference = _prediction_map(grouped[reference_key], repeat)
            comparator = _prediction_map(fold_results, repeat)
            common = sorted(set(reference) & set(comparator))
            if len(common) != len(reference) or len(common) != len(comparator):
                continue
            y = np.asarray([reference[patient_id][0] for patient_id in common])
            comparator_y = np.asarray([comparator[patient_id][0] for patient_id in common])
            if not np.array_equal(y, comparator_y):
                raise ValueError(f"Label mismatch in paired comparison for {key}")
            reference_probability = np.asarray(
                [reference[patient_id][1] for patient_id in common]
            )
            comparator_probability = np.asarray(
                [comparator[patient_id][1] for patient_id in common]
            )
            repeat_results.append(
                bootstrap_auc_difference_ci(
                    y,
                    reference_probability,
                    comparator_probability,
                    seed=7919 + repeat,
                    replicates=2000,
                )
            )
        output[f"{key[0]}:{key[1]}"] = repeat_results
    return output


def _write_comparison_csv(rows, path):
    import csv

    columns = [
        "model", "feature_mode", "repeat_auc_mean", "repeat_auc_std",
        "oof_auc_ci_low", "oof_auc_ci_high", "fold_auc_mean", "fold_auc_std",
        "mean_best_epoch",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(rows, path):
    ordered = sorted(rows, key=lambda item: item["repeat_auc_mean"], reverse=True)
    lines = [
        "# Patient-level benchmark\n",
        "All AUCs use out-of-fold patient predictions. Hyperparameters and early stopping use only each fold's validation patients.\n",
        "| Model | Features | OOF AUC (95% patient-bootstrap CI) | Repeat SD | Fold AUC mean ± SD |\n",
        "|---|---|---:|---:|---:|\n",
    ]
    for row in ordered:
        lines.append(
            f"| {row['model']} | {row['feature_mode']} | {row['repeat_auc_mean']:.3f} "
            f"({row['oof_auc_ci_low']:.3f}–{row['oof_auc_ci_high']:.3f}) | "
            f"{row['repeat_auc_std']:.3f} | {row['fold_auc_mean']:.3f} ± {row['fold_auc_std']:.3f} |\n"
        )
    with open(path, "w", encoding="utf-8") as handle:
        handle.writelines(lines)


def main():
    args = parse_args()
    config = load_yaml(args.config)
    validate_config(config)
    seed = int(config["evaluation"].get("seed", 42))
    seed_everything(seed)
    output_directory = Path(config["output"]["directory"])
    output_directory.mkdir(parents=True, exist_ok=True)

    if args.build_cache or args.overwrite_cache:
        cache = build_roi_cache(config["data"], overwrite=args.overwrite_cache)
    else:
        cache = load_roi_cache(config["data"]["cache_path"])
    patients = group_rois_by_patient(cache)
    validate_expected_cohort(cache, patients, config["data"].get("expected_cohort"))
    atomic_json(cohort_report(cache, patients), output_directory / "cohort_report.json")
    if args.explore_only:
        return

    configured_models = list(args.models or config["evaluation"]["models"])
    unknown_models = set(configured_models) - set(config["evaluation"]["models"])
    if unknown_models:
        raise ValueError(f"Requested models are not configured: {sorted(unknown_models)}")
    feature_modes = list(args.feature_modes or config["evaluation"]["feature_modes"])
    edge_shuffle_modes = set(config["evaluation"].get("edge_shuffle_modes", ["combined"]))
    resume = not args.no_resume
    folds_directory = output_directory / "folds"

    for repeat, fold, train_idx, validation_idx, test_idx in _folds(
        patients, config["evaluation"]
    ):
        fold_seed = seed + repeat * 10_000 + fold * 101
        train_patients = _subset(patients, train_idx)
        validation_patients = _subset(patients, validation_idx)
        test_patients = _subset(patients, test_idx)
        partitions = {
            "train": _partition_manifest(train_patients),
            "validation": _partition_manifest(validation_patients),
            "test": _partition_manifest(test_patients),
        }
        id_sets = {
            name: {item["patient_id"] for item in values}
            for name, values in partitions.items()
        }
        if (
            id_sets["train"] & id_sets["validation"]
            or id_sets["train"] & id_sets["test"]
            or id_sets["validation"] & id_sets["test"]
        ):
            raise RuntimeError("Patient leakage detected between fold partitions")
        atomic_json(
            {"repeat": repeat, "fold": fold, "partitions": partitions},
            output_directory / "splits" / f"r{repeat}_f{fold}.json",
        )
        _write_split_summary(output_directory)
        print(
            f"repeat={repeat} fold={fold} train={len(train_patients)} "
            f"validation={len(validation_patients)} test={len(test_patients)}",
            flush=True,
        )

        if "prevalence" in configured_models:
            path = folds_directory / f"prevalence_none_r{repeat}_f{fold}.json"
            if not (resume and path.exists()):
                result = prevalence_baseline(train_patients, test_patients)
                result.update(model="prevalence", feature_mode="none", repeat=repeat, fold=fold)
                atomic_json(result, path)

        tabular_feature_cache = {}
        for mode in feature_modes:
            normalizer = normalizer_from_config(config, fold_seed).fit(train_patients)
            for model_name in configured_models:
                if model_name == "prevalence":
                    continue
                if model_name == "spatial_logistic" and mode == "expressions":
                    continue
                if model_name == "gnn_edge_shuffle" and mode not in edge_shuffle_modes:
                    continue
                path = folds_directory / f"{model_name}_{mode}_r{repeat}_f{fold}.json"
                if resume and path.exists():
                    print(f"resuming: {path.name}", flush=True)
                    continue
                print(f"  fitting {model_name}/{mode}", flush=True)
                tabular_models = {
                    "summary_logistic": ("logistic_regression", False),
                    "spatial_logistic": ("logistic_regression", True),
                    "summary_random_forest": ("random_forest", False),
                    "summary_xgboost": ("xgboost", False),
                }
                if model_name in tabular_models:
                    classifier_name, spatial = tabular_models[model_name]
                    feature_key = (mode, spatial)
                    if feature_key not in tabular_feature_cache:
                        tabular_feature_cache[feature_key] = patient_feature_matrices(
                            train_patients,
                            validation_patients,
                            test_patients,
                            normalizer,
                            mode,
                            len(cache["cell_types"]),
                            config["graph"],
                            spatial,
                            fold_seed,
                        )
                    result, fitted = fit_tabular_baseline(
                        train_patients,
                        validation_patients,
                        test_patients,
                        normalizer,
                        mode,
                        len(cache["cell_types"]),
                        config["graph"],
                        spatial=spatial,
                        seed=fold_seed,
                        model_name=classifier_name,
                        model_config=config["baseline"],
                        prepared_data=tabular_feature_cache[feature_key],
                    )
                    _save_pickle(
                        {"model": fitted, "normalizer": normalizer.state_dict()},
                        output_directory / "checkpoints" / f"{path.stem}.pkl",
                    )
                elif model_name in {"deepsets", "gnn", "gnn_edge_shuffle"}:
                    result = run_neural(
                        model_name,
                        mode,
                        train_patients,
                        validation_patients,
                        test_patients,
                        normalizer,
                        cache,
                        config,
                        fold_seed,
                        output_directory / "checkpoints" / f"{path.stem}.pt",
                    )
                else:
                    raise ValueError(f"Unknown benchmark model: {model_name}")
                result.update(model=model_name, feature_mode=mode, repeat=repeat, fold=fold)
                atomic_json(result, path)

        summarize_results(output_directory, int(config["evaluation"].get("repeats", 1)))

    rows = summarize_results(output_directory, int(config["evaluation"].get("repeats", 1)))
    _write_split_summary(output_directory)
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
