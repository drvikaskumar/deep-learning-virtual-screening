"""
Plotting utilities for the QSAR regression project.

This script contains functions to visualise various aspects of the QSAR
modeling workflow, including distributions of pIC₅₀ values, model
performance metrics across multiple algorithms, and scatter plots of
predicted versus experimental values.  The plots are saved as PNG
files in the specified output directory.

To execute the default plotting routine on results produced by the
provided pipeline, run:

```
python plot_results.py \
    --cross_val_csv qsar_results/cross_validation_results.csv \
    --test_preds_csv qsar_results/test_predictions.csv \
    --training_csv btk.csv \
    --output_dir plots
```

This will generate three figures:
1. `pic50_distribution.png` – distribution of pIC₅₀ values in the
   curated training dataset.
2. `model_performance.png` – bar chart summarising mean and standard
   deviation of R², RMSE, and MAE across models (if cross-validation
   results are available).
3. `predicted_vs_actual.png` – scatter plot of experimental vs
   predicted pIC₅₀ for the test set (if test predictions are available).

You can customise the filenames via command-line options and modify
the functions to suit other datasets or models.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from qsar_pipeline.pipeline import (
    load_chembl_dataset,
    clean_and_standardise_smiles,
    convert_ic50_to_pic50,
)


def plot_pic50_distribution(training_csv: str, output_dir: str, filename: str = "pic50_distribution.png") -> None:
    """Plot the distribution of pIC50 values in the training dataset.

    Parameters
    ----------
    training_csv : str
        Path to the raw ChEMBL activity dataset (e.g. btk.csv).  Only
        records with valid IC50 measurements are used.
    output_dir : str
        Directory where the plot will be saved.
    filename : str, default "pic50_distribution.png"
        Name of the output image file.
    """
    df_raw = load_chembl_dataset(training_csv)
    df_clean = clean_and_standardise_smiles(df_raw, smiles_column="Smiles")
    df_pic50 = convert_ic50_to_pic50(df_clean, value_column="Standard Value")
    values = df_pic50["pIC50"].astype(float)
    plt.figure(figsize=(6, 4))
    sns.histplot(values, bins=30, kde=True, color="#5B84B1FF")
    plt.xlabel("pIC50")
    plt.ylabel("Count")
    plt.title("Distribution of pIC50 values")
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved pIC50 distribution plot to {out_path}")


def plot_model_performance(cross_val_csv: Optional[str], output_dir: str, filename: str = "model_performance.png") -> None:
    """Plot aggregated performance metrics from cross-validation results.

    The cross-validation results file should contain columns: ``model``,
    ``split``, ``r2``, ``rmse``, and ``mae``.  This function computes
    the mean and standard deviation of these metrics for each model
    across splits and creates a bar chart.

    Parameters
    ----------
    cross_val_csv : str or None
        Path to the CSV file with cross-validation results.  If None
        or the file does not exist, this plot is skipped.
    output_dir : str
        Directory where the plot will be saved.
    filename : str, default "model_performance.png"
        Name of the output image file.
    """
    if cross_val_csv is None or not os.path.isfile(cross_val_csv):
        print("No cross-validation results provided; skipping performance plot.")
        return
    cv_df = pd.read_csv(cross_val_csv)
    if not {"model", "r2", "rmse", "mae"}.issubset(cv_df.columns):
        print("Cross-validation CSV missing required columns; skipping performance plot.")
        return
    # Aggregate metrics
    agg = cv_df.groupby("model").agg({
        "r2": ["mean", "std"],
        "rmse": ["mean", "std"],
        "mae": ["mean", "std"],
    })
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()
    # Plot R2 (higher is better), RMSE/MAE (lower is better)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    sns.barplot(x="model", y="r2_mean", data=agg, ax=axes[0], palette="viridis")
    axes[0].set_title("Mean R² across CV folds")
    axes[0].set_ylabel("R²")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha="right")
    sns.barplot(x="model", y="rmse_mean", data=agg, ax=axes[1], palette="rocket")
    axes[1].set_title("Mean RMSE across CV folds")
    axes[1].set_ylabel("RMSE")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha="right")
    sns.barplot(x="model", y="mae_mean", data=agg, ax=axes[2], palette="mako")
    axes[2].set_title("Mean MAE across CV folds")
    axes[2].set_ylabel("MAE")
    axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved model performance plot to {out_path}")


def plot_predicted_vs_actual(test_preds_csv: Optional[str], output_dir: str, filename: str = "predicted_vs_actual.png") -> None:
    """Scatter plot of predicted versus actual pIC50 values for the test set.

    Parameters
    ----------
    test_preds_csv : str or None
        Path to the CSV file containing columns ``actual`` and ``predicted``.
        If None or the file does not exist, the plot is skipped.
    output_dir : str
        Directory where the plot will be saved.
    filename : str, default "predicted_vs_actual.png"
        Name of the output image file.
    """
    if test_preds_csv is None or not os.path.isfile(test_preds_csv):
        print("No test predictions provided; skipping predicted vs actual plot.")
        return
    df = pd.read_csv(test_preds_csv)
    if not {"actual", "predicted"}.issubset(df.columns):
        print("Test predictions CSV missing required columns; skipping plot.")
        return
    plt.figure(figsize=(5, 5))
    sns.scatterplot(x="actual", y="predicted", data=df, s=30, color="#5555aa", alpha=0.7)
    min_val = min(df["actual"].min(), df["predicted"].min())
    max_val = max(df["actual"].max(), df["predicted"].max())
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=1)
    plt.xlabel("Experimental pIC50")
    plt.ylabel("Predicted pIC50")
    plt.title("Predicted vs Experimental pIC50 (Test Set)")
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved predicted vs actual plot to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate plots for QSAR regression project results.")
    parser.add_argument(
        "--cross_val_csv",
        type=str,
        default=None,
        help="Path to cross-validation results CSV (optional).",
    )
    parser.add_argument(
        "--test_preds_csv",
        type=str,
        default=None,
        help="Path to test predictions CSV (optional).",
    )
    parser.add_argument(
        "--training_csv",
        type=str,
        required=True,
        help="Path to raw training CSV (e.g. btk.csv).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots",
        help="Directory where plots will be saved.",
    )
    args = parser.parse_args()
    plot_pic50_distribution(args.training_csv, args.output_dir)
    plot_model_performance(args.cross_val_csv, args.output_dir)
    plot_predicted_vs_actual(args.test_preds_csv, args.output_dir)


if __name__ == "__main__":
    main()