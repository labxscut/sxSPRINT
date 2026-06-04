"""Reusable baseline models for PRS and SNP-PRS comparison."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

try:
    from .metrics import evaluate_binary_metrics, find_best_threshold
except ImportError:  # Allows running this file directly from src/.
    from metrics import evaluate_binary_metrics, find_best_threshold


def elastic_net_classifier(random_state: int = 42) -> LogisticRegression:
    return LogisticRegression(
        penalty="elasticnet",
        l1_ratio=0.1,
        C=100,
        solver="saga",
        random_state=random_state,
        max_iter=5000,
    )


def xgboost_classifier(random_state: int = 42) -> XGBClassifier:
    return XGBClassifier(
        random_state=random_state,
        booster="gbtree",
        tree_method="hist",
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        colsample_bytree=0.9,
        eval_metric="logloss",
    )


def fit_and_evaluate(model, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray):
    model.fit(x_train, y_train)
    test_score = model.predict_proba(x_test)[:, 1]
    threshold, _ = find_best_threshold(y_test, test_score)
    return evaluate_binary_metrics(y_test, test_score, threshold=threshold)
