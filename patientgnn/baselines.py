"""Fold-local patient-level tabular baselines.

Every classifier receives one row per patient.  Feature construction and all
preprocessing are shared across classifiers so that the comparison measures the
classifier rather than a different data split or aggregation rule.
"""

from __future__ import annotations

from itertools import product
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

from .features import patient_feature_matrix
from .metrics import choose_balanced_threshold, patient_metrics


def _labels(patients):
    return np.asarray([patient["label"] for patient in patients], dtype=int)


def patient_feature_matrices(
    train_patients: Sequence[Mapping[str, Any]],
    validation_patients: Sequence[Mapping[str, Any]],
    test_patients: Sequence[Mapping[str, Any]],
    normalizer,
    feature_mode: str,
    num_cell_types: int,
    graph_config: Mapping[str, Any],
    spatial: bool,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    kwargs = dict(
        normalizer=normalizer,
        feature_mode=feature_mode,
        num_cell_types=num_cell_types,
        graph_config=graph_config,
        spatial=spatial,
        seed=seed,
    )
    x_train = patient_feature_matrix(train_patients, **kwargs)
    x_validation = patient_feature_matrix(validation_patients, **kwargs)
    x_test = patient_feature_matrix(test_patients, **kwargs)
    y_train, y_validation, y_test = map(_labels, (train_patients, validation_patients, test_patients))
    return x_train, x_validation, x_test, y_train, y_validation, y_test


def _classifier_candidates(
    model_name: str,
    model_config: Mapping[str, Any],
    y_train: np.ndarray,
    seed: int,
) -> Iterable[Tuple[Dict[str, Any], Any]]:
    """Yield deliberately small models suitable for a 63-patient cohort."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    def pipeline(classifier, scale: bool = False):
        steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
        if scale:
            steps.append(("scale", StandardScaler()))
        steps.append(("model", classifier))
        return Pipeline(steps)

    if model_name == "logistic_regression":
        for c_value in model_config.get("c_grid", [0.01, 0.1, 1.0, 10.0]):
            parameters = {"C": float(c_value)}
            yield parameters, pipeline(
                LogisticRegression(
                    C=parameters["C"],
                    penalty="l2",
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=seed,
                ),
                scale=True,
            )
        return

    if model_name == "random_forest":
        settings = model_config.get("random_forest", {})
        grid = product(
            settings.get("max_depth", [2, 4, None]),
            settings.get("min_samples_leaf", [2, 4]),
            settings.get("max_features", ["sqrt", 0.5]),
        )
        for max_depth, min_samples_leaf, max_features in grid:
            parameters = {
                "max_depth": max_depth,
                "min_samples_leaf": int(min_samples_leaf),
                "max_features": max_features,
                "n_estimators": int(settings.get("n_estimators", 500)),
            }
            yield parameters, pipeline(
                RandomForestClassifier(
                    **parameters,
                    criterion="gini",
                    class_weight="balanced_subsample",
                    bootstrap=True,
                    n_jobs=int(settings.get("n_jobs", 8)),
                    random_state=seed,
                )
            )
        return

    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise ImportError(
                "The xgboost baseline was requested but xgboost is not installed."
            ) from error

        settings = model_config.get("xgboost", {})
        negatives = int(np.sum(y_train == 0))
        positives = int(np.sum(y_train == 1))
        scale_pos_weight = negatives / max(positives, 1)
        grid = product(
            settings.get("max_depth", [1, 2]),
            settings.get("min_child_weight", [2, 5]),
            settings.get("learning_rate", [0.03, 0.1]),
            settings.get("n_estimators", [100, 300]),
        )
        for max_depth, min_child_weight, learning_rate, n_estimators in grid:
            parameters = {
                "max_depth": int(max_depth),
                "min_child_weight": float(min_child_weight),
                "learning_rate": float(learning_rate),
                "n_estimators": int(n_estimators),
            }
            yield parameters, pipeline(
                XGBClassifier(
                    **parameters,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    subsample=float(settings.get("subsample", 0.8)),
                    colsample_bytree=float(settings.get("colsample_bytree", 0.8)),
                    reg_alpha=float(settings.get("reg_alpha", 1.0)),
                    reg_lambda=float(settings.get("reg_lambda", 10.0)),
                    scale_pos_weight=scale_pos_weight,
                    n_jobs=int(settings.get("n_jobs", 8)),
                    random_state=seed,
                )
            )
        return

    raise ValueError(f"Unknown tabular baseline: {model_name}")


def fit_tabular_baseline(
    train_patients: Sequence[Mapping[str, Any]],
    validation_patients: Sequence[Mapping[str, Any]],
    test_patients: Sequence[Mapping[str, Any]],
    normalizer,
    feature_mode: str,
    num_cell_types: int,
    graph_config: Mapping[str, Any],
    spatial: bool,
    seed: int,
    model_name: str,
    model_config: Mapping[str, Any],
    prepared_data=None,
) -> Tuple[Dict[str, Any], Any]:
    """Fit a classical model with hyperparameters selected on validation patients."""
    from sklearn.metrics import roc_auc_score

    if prepared_data is None:
        prepared_data = patient_feature_matrices(
            train_patients,
            validation_patients,
            test_patients,
            normalizer,
            feature_mode,
            num_cell_types,
            graph_config,
            spatial,
            seed,
        )
    x_train, x_validation, x_test, y_train, y_validation, y_test = prepared_data

    best_auc = -np.inf
    best_model = None
    best_parameters = None
    for parameters, model in _classifier_candidates(model_name, model_config, y_train, seed):
        model.fit(x_train, y_train)
        probability = model.predict_proba(x_validation)[:, 1]
        auc = roc_auc_score(y_validation, probability)
        # Strict improvement makes validation ties resolve deterministically.
        if auc > best_auc + 1e-12:
            best_auc = float(auc)
            best_model = model
            best_parameters = parameters

    if best_model is None:
        raise RuntimeError(f"No {model_name} candidate was fitted")

    p_validation = best_model.predict_proba(x_validation)[:, 1]
    threshold = choose_balanced_threshold(y_validation, p_validation)
    p_test = best_model.predict_proba(x_test)[:, 1]
    result = {
        "classifier": model_name,
        "best_parameters": best_parameters,
        "n_features": int(x_train.shape[1]),
        "validation": patient_metrics(y_validation, p_validation, threshold),
        "test": patient_metrics(y_test, p_test, threshold),
        "predictions": [
            {
                "patient_id": patient["patient_id"],
                "label": int(label),
                "probability": float(probability),
            }
            for patient, label, probability in zip(test_patients, y_test, p_test)
        ],
    }
    return result, best_model


def fit_logistic_baseline(
    train_patients: Sequence[Mapping[str, Any]],
    validation_patients: Sequence[Mapping[str, Any]],
    test_patients: Sequence[Mapping[str, Any]],
    normalizer,
    feature_mode: str,
    num_cell_types: int,
    graph_config: Mapping[str, Any],
    spatial: bool,
    seed: int,
    c_grid=(0.01, 0.1, 1.0, 10.0),
) -> Tuple[Dict[str, Any], Any]:
    """Backward-compatible wrapper used by older benchmark configurations."""
    result, model = fit_tabular_baseline(
        train_patients,
        validation_patients,
        test_patients,
        normalizer,
        feature_mode,
        num_cell_types,
        graph_config,
        spatial,
        seed,
        model_name="logistic_regression",
        model_config={"c_grid": c_grid},
    )
    result["best_c"] = result["best_parameters"]["C"]
    return result, model


def prevalence_baseline(train_patients, test_patients) -> Dict[str, Any]:
    y_test = _labels(test_patients)
    # Use one invariant score across folds. Fold-specific training prevalence
    # would vary slightly and can make pooled OOF AUC depart spuriously from 0.5.
    probability = np.repeat(0.5, len(y_test))
    return {
        "test": patient_metrics(y_test, probability, 0.5),
        "predictions": [
            {
                "patient_id": patient["patient_id"],
                "label": int(label),
                "probability": float(prob),
            }
            for patient, label, prob in zip(test_patients, y_test, probability)
        ],
    }
