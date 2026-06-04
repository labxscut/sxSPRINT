"""Generate small toy input files in the public SPRINT CSV format."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate toy SNP, PRS, and label CSV files.")
    parser.add_argument("--out-dir", default="data/example", help="Output directory.")
    parser.add_argument("--samples", type=int, default=240)
    parser.add_argument("--snps", type=int, default=64)
    parser.add_argument("--prs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_ids = [f"S{i:04d}" for i in range(args.samples)]
    x_snp = rng.integers(0, 3, size=(args.samples, args.snps))
    x_prs = rng.normal(size=(args.samples, args.prs))
    signal = 0.45 * x_prs[:, 0] + 0.25 * x_prs[:, 1] + 0.12 * x_snp[:, :8].sum(axis=1)
    signal += rng.normal(scale=0.8, size=args.samples)
    threshold = np.quantile(signal, 0.65)
    y = (signal > threshold).astype(int)

    snp_df = pd.DataFrame(x_snp, columns=[f"rs{i+1}" for i in range(args.snps)])
    snp_df.insert(0, "sample_id", sample_ids)
    prs_df = pd.DataFrame(x_prs, columns=[f"PRS_{i+1}" for i in range(args.prs)])
    prs_df.insert(0, "sample_id", sample_ids)
    label_df = pd.DataFrame({"sample_id": sample_ids, "label": y})

    snp_df.to_csv(out_dir / "snp.csv", index=False)
    prs_df.to_csv(out_dir / "prs.csv", index=False)
    label_df.to_csv(out_dir / "labels.csv", index=False)
    print(f"Saved example data to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
