"""Plot summary-level SPRINT performance.

This script uses only the public summary CSV files in ``results/`` and does
not require individual-level UK Biobank data.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"


def plot_overall_auc() -> Path:
    mean_df = pd.read_csv(RESULTS_DIR / "SPRINT_mean.csv")
    median_df = pd.read_csv(RESULTS_DIR / "SPRINT_median.csv")

    mean_all = mean_df[mean_df["disease_group"] == "alldisease"].copy()
    median_all = median_df[median_df["disease_group"] == "alldisease"].copy()
    merged = mean_all[["method", "AUC"]].rename(columns={"AUC": "Mean AUC"}).merge(
        median_all[["method", "AUC"]].rename(columns={"AUC": "Median AUC"}),
        on="method",
        how="inner",
    )

    method_order = ["Single PRS", "SPRINT_NO", "SPRINT"]
    merged["method"] = pd.Categorical(merged["method"], categories=method_order, ordered=True)
    merged = merged.sort_values("method")

    x = range(len(merged))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([i - width / 2 for i in x], merged["Mean AUC"], width=width, label="Mean AUC")
    ax.bar([i + width / 2 for i in x], merged["Median AUC"], width=width, label="Median AUC")
    ax.set_xticks(list(x))
    ax.set_xticklabels(merged["method"], rotation=20, ha="right")
    ax.set_ylabel("AUC (%)")
    ax.set_ylim(55, max(merged["Mean AUC"].max(), merged["Median AUC"].max()) + 5)
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()

    out = FIGURES_DIR / "summary_auc_from_public_results.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def main() -> None:
    out = plot_overall_auc()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
