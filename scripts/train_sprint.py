"""Train and evaluate SPRINT from a YAML configuration file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

try:
    import yaml
except ImportError:  # PyYAML is recommended, but the example config is simple enough to parse.
    yaml = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io import load_aligned_inputs
from src.metrics import evaluate_binary_metrics
from src.train import train_one_model


def parse_scalar(value: str):
    value = value.strip()
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip("'\"")


def load_config(path: Path) -> dict:
    """Load a small YAML config, with a fallback parser for the example format."""
    with open(path, "r", encoding="utf-8") as handle:
        if yaml is not None:
            return yaml.safe_load(handle)
        config: dict[str, dict] = {}
        section: str | None = None
        for raw_line in handle:
            line = raw_line.split("#", 1)[0].rstrip()
            if not line:
                continue
            if not line.startswith(" ") and line.endswith(":"):
                section = line[:-1]
                config[section] = {}
                continue
            if section is None or ":" not in line:
                continue
            key, value = line.strip().split(":", 1)
            config[section][key.strip()] = parse_scalar(value)
        return config


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SPRINT on user-provided SNP, PRS, and label CSV files.")
    parser.add_argument("--config", default="configs/example_config.yaml", help="Path to a YAML configuration file.")
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    config = load_config(config_path)

    data_cfg = config["data"]
    train_cfg = config.get("training", {})
    out_cfg = config.get("output", {})

    x_snp, x_prs, y, sample_ids = load_aligned_inputs(
        resolve_path(data_cfg["snp_path"]),
        resolve_path(data_cfg["prs_path"]),
        resolve_path(data_cfg["label_path"]),
        label_column=data_cfg.get("label_column", "label"),
    )

    test_size = float(train_cfg.get("test_size", 0.2))
    val_size = float(train_cfg.get("validation_size", 0.2))
    random_state = int(train_cfg.get("random_state", 42))

    idx = np.arange(len(y))
    train_val_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    val_fraction_of_train_val = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_fraction_of_train_val,
        random_state=random_state,
        stratify=y[train_val_idx],
    )

    model, val_metrics = train_one_model(
        x_snp[train_idx],
        x_prs[train_idx],
        y[train_idx],
        x_snp[val_idx],
        x_prs[val_idx],
        y[val_idx],
        epochs=int(train_cfg.get("epochs", 100)),
        batch_size=int(train_cfg.get("batch_size", 128)),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        patience=int(train_cfg.get("patience", 10)),
        device=train_cfg.get("device"),
    )

    # The operating threshold is selected on validation data and then fixed on the test set.
    test_scores_device = train_cfg.get("device")
    test_scores_threshold = val_metrics["threshold"]
    with torch.no_grad():
        device = test_scores_device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        snp_tensor = torch.tensor(x_snp[test_idx], dtype=torch.float32).to(device)
        prs_tensor = torch.tensor(x_prs[test_idx], dtype=torch.float32).to(device)
        test_scores = model(snp_tensor, prs_tensor).detach().cpu().numpy()
    test_metrics = evaluate_binary_metrics(y[test_idx], test_scores, threshold=test_scores_threshold)

    output_dir = resolve_path(out_cfg.get("output_dir", "results/run"))
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump({"validation": val_metrics, "test": test_metrics}, handle, indent=2)

    predictions = pd.DataFrame(
        {
            "sample_id": sample_ids.take(test_idx).astype(str),
            "label": y[test_idx],
            "score": test_scores,
            "prediction": (test_scores >= test_scores_threshold).astype(int),
        }
    )
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    if bool(out_cfg.get("save_model", True)):
        torch.save(model.cpu().state_dict(), output_dir / "sprint_model.pt")

    print(json.dumps({"validation": val_metrics, "test": test_metrics}, indent=2))
    print(f"Saved outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
