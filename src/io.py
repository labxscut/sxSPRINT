"""Input and output helpers for SPRINT command-line scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def read_feature_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV feature table and use ``sample_id`` as index when present."""
    table = pd.read_csv(path)
    if "sample_id" in table.columns:
        table = table.set_index("sample_id")
    return table


def read_label_table(path: str | Path, label_column: str = "label") -> pd.Series:
    """Read binary labels from a CSV file."""
    table = pd.read_csv(path)
    if label_column not in table.columns:
        raise ValueError(f"Label column '{label_column}' was not found in {path}.")
    if "sample_id" in table.columns:
        table = table.set_index("sample_id")
    return table[label_column]


def load_aligned_inputs(
    snp_path: str | Path,
    prs_path: str | Path,
    label_path: str | Path,
    label_column: str = "label",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Index]:
    """Load SNP, PRS, and label files and align samples by ID when available."""
    snp = read_feature_table(snp_path)
    prs = read_feature_table(prs_path)
    labels = read_label_table(label_path, label_column=label_column)

    if not snp.index.equals(pd.RangeIndex(len(snp))) and not prs.index.equals(pd.RangeIndex(len(prs))):
        common = snp.index.intersection(prs.index).intersection(labels.index)
        if common.empty:
            raise ValueError("No overlapping sample_id values were found across SNP, PRS, and label files.")
        snp = snp.loc[common]
        prs = prs.loc[common]
        labels = labels.loc[common]
        sample_ids = common
    else:
        if not (len(snp) == len(prs) == len(labels)):
            raise ValueError("SNP, PRS, and label files must have the same number of rows when sample_id is absent.")
        sample_ids = pd.Index(range(len(labels)), name="row")

    x_snp = snp.apply(pd.to_numeric).to_numpy(dtype=np.float32)
    x_prs = prs.apply(pd.to_numeric).to_numpy(dtype=np.float32)
    y = pd.to_numeric(labels).to_numpy(dtype=np.int64)
    return x_snp, x_prs, y, sample_ids
