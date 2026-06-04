"""Small dependency and model smoke test for SPRINT.

Run this after installing ``requirements.txt``:

    python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model import SPRINT
from src.train import train_one_model, predict_and_evaluate


def main() -> None:
    rng = np.random.default_rng(42)
    x_snp = rng.integers(0, 3, size=(40, 32)).astype("float32")
    x_prs = rng.normal(size=(40, 8)).astype("float32")
    y = np.array([0, 1] * 20, dtype="float32")

    model = SPRINT(snp_input_dim=x_snp.shape[1], prs_input_dim=x_prs.shape[1])
    print(f"SNP embedding dimension: {model.snp_embedding_dim}")

    trained, metrics = train_one_model(
        x_snp[:30],
        x_prs[:30],
        y[:30],
        x_snp[30:],
        x_prs[30:],
        y[30:],
        epochs=2,
        batch_size=8,
        patience=2,
        device="cpu",
    )
    print("Validation metrics:", metrics)
    print("Evaluation metrics:", predict_and_evaluate(trained, x_snp[30:], x_prs[30:], y[30:], device="cpu"))
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
