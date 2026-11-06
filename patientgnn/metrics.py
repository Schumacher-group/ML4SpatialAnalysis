"""Patient-level metrics, thresholds, and uncertainty."""

from __future__ import annotations

from typing import Dict

import numpy as np


def choose_balanced_threshold(y_true, probability) -> float:
    from sklearn.metrics import balanced_accuracy_score

    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    candidates = np.unique(np.concatenate([[0.0], probability, [1.0]]))
    scores = [balanced_accuracy_score(y_true, probability >= threshold) for threshold in candidates]
    best = np.flatnonzero(scores == np.max(scores))
    return float(candidates[best[np.argmin(np.abs(candidates[best] - 0.5))]])


def patient_metrics(y_true, probability, threshold: float = 0.5) -> Dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        confusion_matrix,
        roc_auc_score,
    )

    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "average_precision": float(average_precision_score(y_true, probability)),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "brier": float(brier_score_loss(y_true, probability)),
        "threshold": float(threshold),
    }


def bootstrap_auc_ci(y_true, probability, seed=42, replicates=2000, alpha=0.05):
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(int(replicates)):
        indices = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[indices])) < 2:
            continue
        values.append(_rank_auc(y_true[indices], probability[indices]))
    if not values:
        return [float("nan"), float("nan")]
    return [
        float(np.quantile(values, alpha / 2)),
        float(np.quantile(values, 1 - alpha / 2)),
    ]


def bootstrap_auc_difference_ci(
    y_true,
    reference_probability,
    comparator_probability,
    seed=42,
    replicates=2000,
    alpha=0.05,
):
    """Paired patient bootstrap for reference AUC minus comparator AUC."""
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true, dtype=int)
    reference_probability = np.asarray(reference_probability, dtype=float)
    comparator_probability = np.asarray(comparator_probability, dtype=float)
    if not (
        len(y_true) == len(reference_probability) == len(comparator_probability)
    ):
        raise ValueError("Labels and both probability arrays must have equal length")

    observed = roc_auc_score(y_true, reference_probability) - roc_auc_score(
        y_true, comparator_probability
    )
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(int(replicates)):
        indices = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[indices])) < 2:
            continue
        values.append(
            _rank_auc(y_true[indices], reference_probability[indices])
            - _rank_auc(y_true[indices], comparator_probability[indices])
        )
    if not values:
        interval = [float("nan"), float("nan")]
    else:
        interval = [
            float(np.quantile(values, alpha / 2)),
            float(np.quantile(values, 1 - alpha / 2)),
        ]
    return {"auc_difference": float(observed), "ci": interval}


def roi_count_strata(roi_counts) -> np.ndarray:
    """ROI-count tertile (0, 1, 2) of each patient, as in scripts/compare_aggregations.py."""
    roi_counts = np.asarray(roi_counts, dtype=float)
    return np.digitize(roi_counts, np.percentile(roi_counts, [33.3, 66.7]))


def _rank_auc(y_true, probability) -> float:
    """Mann-Whitney AUC with tied scores counted as one half; equals roc_auc_score.

    About 15 times faster than roc_auc_score, which matters in the bootstrap loops.
    """
    from scipy.stats import rankdata

    positives = y_true == 1
    n_positive = int(positives.sum())
    n_negative = len(y_true) - n_positive
    ranks = rankdata(probability)
    return float(
        (ranks[positives].sum() - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative)
    )


def stratified_auc(y_true, probability, strata) -> float:
    """Within-stratum AUCs averaged with stratum-size weights.

    A stratum with only one class has no AUC and is left out. With ROI-count
    tertiles as strata, a score that only tracks ROI count gains far less than
    it does in the pooled AUC, though counts still vary within a tertile.
    """
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    strata = np.asarray(strata)
    weighted_sum = total = 0.0
    for stratum in np.unique(strata):
        selected = strata == stratum
        if len(np.unique(y_true[selected])) < 2:
            continue
        weighted_sum += _rank_auc(y_true[selected], probability[selected]) * selected.sum()
        total += selected.sum()
    return weighted_sum / total if total else float("nan")


def bootstrap_stratified_auc_ci(
    y_true, probability, strata, seed=42, replicates=2000, alpha=0.05
):
    """Patient bootstrap interval for stratified_auc; patients keep their stratum."""
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    strata = np.asarray(strata)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(int(replicates)):
        indices = rng.choice(len(y_true), len(y_true), replace=True)
        value = stratified_auc(y_true[indices], probability[indices], strata[indices])
        if not np.isnan(value):
            values.append(value)
    if not values:
        return [float("nan"), float("nan")]
    return [
        float(np.quantile(values, alpha / 2)),
        float(np.quantile(values, 1 - alpha / 2)),
    ]
