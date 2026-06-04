"""Evaluation metrics for disease-risk prediction."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)


def find_best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Select the threshold maximizing sqrt(TPR * specificity)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    specificity = 1 - fpr
    gmean = np.sqrt(tpr * specificity)
    idx = int(np.argmax(gmean))
    return float(thresholds[idx]), float(gmean[idx])


def evaluate_binary_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float | None = None,
) -> dict[str, float]:
    """Return AUC and operating-point metrics."""
    if threshold is None:
        threshold, _ = find_best_threshold(y_true, y_score)
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "auc": float(roc_auc_score(y_true, y_score)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "tpr": float(tpr),
        "specificity": float(specificity),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "auprc": float(average_precision_score(y_true, y_score)),
        "threshold": float(threshold),
    }
