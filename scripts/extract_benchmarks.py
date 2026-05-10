#!/usr/bin/env python3
"""Extract a clean comparison table from du Toit et al. results_all_models.csv.

Prints the benchmark test accuracies, F1, and param counts per model class
and saves a clean summary to fnn_rfi/figs/dutoit_benchmark.json.

Usage:
    python3 fnn_rfi/scripts/extract_benchmarks.py
    python3 fnn_rfi/scripts/extract_benchmarks.py --csv data/zenodo/results_all_models.csv
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/zenodo/results_all_models.csv")
    parser.add_argument("--out", default="fnn_rfi/figs/dutoit_benchmark.json")
    args = parser.parse_args()

    print(f"Reading {args.csv} …")
    df = pd.read_csv(args.csv)

    print(f"Shape: {df.shape}")

    # Use exact column names from the CSV (confirmed from du Toit 8275061)
    col_map = {
        "model_class": "model_class",
        "n_filters":   "filters",
        "n_weights":   "trainable_params",
        "n_flops":     "flops_image",
        "train_acc":   "train_accuracy",
        "val_acc":     "val_accuracy",
        "test_acc":    "test_accuracy",
        "test_f1":     "test_f1",
        "test_prec":   "test_precision",
        "test_rec":    "test_recall",
        "test_auroc":  "test_auroc",
        "test_auprc":  "test_auprc",
        "data_name":   "data_name",
    }
    # Keep only columns that exist in this CSV version
    col_map = {k: v for k, v in col_map.items() if v in df.columns}

    print("Column mapping:")
    for k, v in col_map.items():
        print(f"  {k:20s} → {v}")

    model_col = col_map.get("model_class")
    test_col = col_map.get("test_acc")
    weight_col = col_map.get("n_weights")

    if model_col is None:
        print("\nERROR: Could not detect model class column. Raw columns:")
        for c in df.columns:
            print(f"  {c!r}")
        return

    # Force numeric on numeric cols
    for key in ("n_weights", "n_flops", "test_acc", "val_acc", "train_acc",
                "test_f1", "test_prec", "test_rec", "n_filters"):
        if key in col_map:
            df[col_map[key]] = pd.to_numeric(df[col_map[key]], errors="coerce")

    # Group by model class and summarise (numeric columns only)
    numeric_group_cols = {}
    for key, col in col_map.items():
        if key in ("model_class", "data_name"):
            continue
        if col in df.columns and pd.api.types.is_numeric_dtype(pd.to_numeric(df[col], errors="coerce")):
            numeric_group_cols[key] = col

    # Simple ranked table
    if test_col and test_col in df.columns:
        ranked = (
            df.groupby(model_col)[test_col]
            .agg(["mean", "max", "count"])
            .rename(columns={"mean": "test_acc_mean", "max": "test_acc_max", "count": "n_runs"})
            .sort_values("test_acc_mean", ascending=False)
        )
        if weight_col and weight_col in df.columns:
            w_mean = df.groupby(model_col)[weight_col].mean().rename("n_weights_mean")
            ranked = ranked.join(w_mean)
        if "n_flops" in col_map:
            f_mean = df.groupby(model_col)[col_map["n_flops"]].mean().rename("n_flops_mean")
            ranked = ranked.join(f_mean)
        print(f"\n{'='*80}")
        print("BENCHMARK TABLE — du Toit et al. 2023 (sorted by mean test accuracy)")
        print(f"{'='*80}")
        print(ranked.to_string(float_format="{:.6f}".format))
        print(f"{'='*80}\n")

    # Save summary JSON
    summary = {
        "source": "du Toit et al. (2023) Zenodo 8275061",
        "n_rows": len(df),
        "column_map": col_map,
        "per_model": {},
    }

    if model_col:
        for model_name, group in df.groupby(model_col):
            entry = {}
            for key, col in numeric_group_cols.items():
                if col in group.columns:
                    vals = pd.to_numeric(group[col], errors="coerce").dropna()
                    if len(vals) > 0:
                        entry[key] = {
                            "mean": float(vals.mean()),
                            "max": float(vals.max()),
                            "min": float(vals.min()),
                            "n": int(len(vals)),
                        }
            # also store data names (datasets) this model was tested on
            if "data_name" in col_map:
                entry["datasets"] = sorted(group[col_map["data_name"]].dropna().unique().tolist())
            summary["per_model"][str(model_name)] = entry

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Benchmark summary saved to {args.out}")

    # Print best result for quick reference
    if test_col and test_col in df.columns:
        best_row = df.loc[df[test_col].idxmax()]
        print(f"\nBest single model in benchmark:")
        print(f"  Model class : {best_row.get(model_col, '?')}")
        if test_col:
            print(f"  Test acc    : {best_row[test_col]:.6f}")
        if weight_col:
            print(f"  N weights   : {best_row.get(weight_col, '?')}")
        flop_col = col_map.get("n_flops")
        if flop_col and flop_col in best_row.index:
            print(f"  N FLOPs     : {best_row[flop_col]:.0f}")


if __name__ == "__main__":
    main()
