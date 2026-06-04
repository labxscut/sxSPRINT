"""Minimal training utilities for SPRINT."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from .model import SPRINT
    from .metrics import evaluate_binary_metrics
except ImportError:  # Allows running this file directly from src/.
    from model import SPRINT
    from metrics import evaluate_binary_metrics


class SNPPRSDataset(Dataset):
    def __init__(self, snp_data: np.ndarray, prs_data: np.ndarray, labels: np.ndarray) -> None:
        self.snp_data = torch.tensor(snp_data, dtype=torch.float32)
        self.prs_data = torch.tensor(prs_data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.snp_data[idx], self.prs_data[idx], self.labels[idx]


def train_one_model(
    x_snp_train: np.ndarray,
    x_prs_train: np.ndarray,
    y_train: np.ndarray,
    x_snp_val: np.ndarray,
    x_prs_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 1e-3,
    patience: int = 10,
    device: str | None = None,
) -> tuple[SPRINT, dict[str, float]]:
    """Train a SPRINT model with early stopping on validation AUC."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SPRINT(x_snp_train.shape[1], x_prs_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    loader = DataLoader(
        SNPPRSDataset(x_snp_train, x_prs_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )

    best_auc = -1.0
    best_state = None
    stale_epochs = 0
    best_metrics: dict[str, float] = {}

    for _ in range(epochs):
        model.train()
        for snp_batch, prs_batch, y_batch in loader:
            snp_batch = snp_batch.to(device)
            prs_batch = prs_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            pred = model(snp_batch, prs_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()

        metrics = predict_and_evaluate(model, x_snp_val, x_prs_val, y_val, device=device)
        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = metrics
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_metrics


def predict_and_evaluate(
    model: SPRINT,
    x_snp: np.ndarray,
    x_prs: np.ndarray,
    y: np.ndarray,
    device: str | None = None,
) -> dict[str, float]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        snp_tensor = torch.tensor(x_snp, dtype=torch.float32).to(device)
        prs_tensor = torch.tensor(x_prs, dtype=torch.float32).to(device)
        y_score = model(snp_tensor, prs_tensor).detach().cpu().numpy()
    return evaluate_binary_metrics(y, y_score)
