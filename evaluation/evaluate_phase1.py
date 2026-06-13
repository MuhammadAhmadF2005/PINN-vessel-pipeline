"""
Phase 1 Evaluation Runner
==========================
Runs all 3 trained models on the test set, computes metrics per fault type,
and produces evaluation/results_phase1.md with real numbers.

Usage:
    python -m evaluation.evaluate_phase1

Prerequisites:
    - Data in data/
    - Trained models in models/artifacts/
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

os.environ.setdefault("DDE_BACKEND", "pytorch")

import torch

from models.pinn.pinn_model import (
    VesselPINN, PINNHyperparams, load_model as load_pinn,
    compute_anomaly_scores,
)
from models.baselines.iso_forest import IsoForestModel
from models.baselines.lstm_ae import (
    LSTMAutoencoder, load_model as load_lstm,
    compute_anomaly_scores_lstm,
)
from evaluation.metrics import (
    evaluate_model, find_optimal_threshold,
    format_results_table, format_summary_table,
    EvalResult,
)


DATA_DIR = Path("data")
ARTIFACTS_DIR = Path("models/artifacts")
EVAL_DIR = Path("evaluation")

FAULT_TYPES = ["seal_degradation", "heater_drift", "blockage"]


def align_scores_to_labels(
    scores: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
    offset: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Align score arrays that may be shorter than labels due to windowing.

    The offset is the number of leading samples lost to windowing.
    Returns truncated (timestamps, labels, scores) of equal length.
    """
    n = len(scores)
    return timestamps[offset : offset + n], labels[offset : offset + n], scores


def evaluate_all() -> list[EvalResult]:
    """
    Evaluate all models on test data.

    Returns list of EvalResult objects.
    """
    # Load test data
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")
    print(f"Test data: {len(test_df)} rows")
    print(f"Val data: {len(val_df)} rows (for threshold selection)")

    # Load models
    print("\nLoading models...")
    pinn_model = load_pinn(ARTIFACTS_DIR / "pinn_model.pt")
    iso_model = IsoForestModel.load(ARTIFACTS_DIR / "iso_forest_model.pkl")
    lstm_model = load_lstm(ARTIFACTS_DIR / "lstm_ae_model.pt")
    print("All models loaded.")

    # --- Find optimal thresholds on validation set ---
    print("\nFinding optimal thresholds on validation set...")
    thresholds = {}

    # Process validation data per fault type for threshold tuning
    for ft in FAULT_TYPES:
        val_fault = val_df[val_df["fault_type"].isin([ft, "normal"])].copy()
        if len(val_fault) == 0:
            continue

        # Group by run_id and process
        val_runs = []
        for run_id, group in val_fault.groupby("run_id"):
            group = group.sort_values("timestamp").reset_index(drop=True)
            val_runs.append(group)

        if not val_runs:
            continue

        val_combined = pd.concat(val_runs, ignore_index=True)
        ts = val_combined["timestamp"].values
        P = val_combined["pressure"].values
        T_vals = val_combined["temperature"].values
        labels = val_combined["label"].values

        # PINN scores
        pinn_scores = compute_anomaly_scores(pinn_model, ts, P, T_vals)
        th, _ = find_optimal_threshold(labels, pinn_scores)
        thresholds[("PINN", ft)] = th

        # IsoForest scores
        iso_scores = iso_model.anomaly_score(P, T_vals)
        offset = len(labels) - len(iso_scores)
        _, labels_aligned, _ = align_scores_to_labels(iso_scores, labels, ts, offset)
        th, _ = find_optimal_threshold(labels_aligned, iso_scores)
        thresholds[("IsolationForest", ft)] = th

        # LSTM-AE scores
        lstm_scores = compute_anomaly_scores_lstm(lstm_model, P, T_vals)
        offset = len(labels) - len(lstm_scores)
        _, labels_aligned, _ = align_scores_to_labels(lstm_scores, labels, ts, offset)
        th, _ = find_optimal_threshold(labels_aligned, lstm_scores)
        thresholds[("LSTM-AE", ft)] = th

    print(f"Thresholds computed: {len(thresholds)} entries")

    # --- Evaluate on test set ---
    print("\nEvaluating on test set...")
    all_results = []

    for ft in FAULT_TYPES:
        test_fault = test_df[test_df["fault_type"].isin([ft, "normal"])].copy()
        if len(test_fault) == 0:
            print(f"  WARNING: No test data for fault type '{ft}', skipping")
            continue

        # Process per run, then combine
        runs = []
        for run_id, group in test_fault.groupby("run_id"):
            runs.append(group.sort_values("timestamp").reset_index(drop=True))

        combined = pd.concat(runs, ignore_index=True)
        ts = combined["timestamp"].values
        P = combined["pressure"].values
        T_vals = combined["temperature"].values
        labels = combined["label"].values

        print(f"\n  Fault: {ft} — {len(combined)} samples "
              f"(normal: {(labels==0).sum()}, anomaly: {(labels==1).sum()})")

        # --- PINN ---
        pinn_scores = compute_anomaly_scores(pinn_model, ts, P, T_vals)
        th = thresholds.get(("PINN", ft))
        result = evaluate_model("PINN", ft, ts, labels, pinn_scores, threshold=th)
        all_results.append(result)
        print(f"    PINN — F1: {result.f1:.4f}, TTD: {result.ttd:.1f}s")

        # --- IsolationForest ---
        iso_scores = iso_model.anomaly_score(P, T_vals)
        offset = len(labels) - len(iso_scores)
        ts_a, labels_a, iso_scores_a = align_scores_to_labels(iso_scores, labels, ts, offset)
        th = thresholds.get(("IsolationForest", ft))
        result = evaluate_model("IsolationForest", ft, ts_a, labels_a, iso_scores_a, threshold=th)
        all_results.append(result)
        print(f"    IsolationForest — F1: {result.f1:.4f}, TTD: {result.ttd:.1f}s")

        # --- LSTM-AE ---
        lstm_scores = compute_anomaly_scores_lstm(lstm_model, P, T_vals)
        offset = len(labels) - len(lstm_scores)
        ts_a, labels_a, lstm_scores_a = align_scores_to_labels(lstm_scores, labels, ts, offset)
        th = thresholds.get(("LSTM-AE", ft))
        result = evaluate_model("LSTM-AE", ft, ts_a, labels_a, lstm_scores_a, threshold=th)
        all_results.append(result)
        print(f"    LSTM-AE — F1: {result.f1:.4f}, TTD: {result.ttd:.1f}s")

    return all_results


def write_results(results: list[EvalResult]) -> None:
    """Write results to evaluation/results_phase1.md."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    output_path = EVAL_DIR / "results_phase1.md"

    content = f"""# Phase 1 Evaluation Results
> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset
- **Source**: Simulated pressure vessel (ODE-based)
- **Fault types**: seal_degradation, heater_drift, blockage
- **Split**: train/val/test by run_id

## Detailed Results (per model x fault type)

{format_results_table(results)}

## Summary (averaged across fault types)

{format_summary_table(results)}

## Methodology
- **Threshold selection**: Optimal F1-maximizing threshold found on validation set
- **Time-to-detection (TTD)**: Time from true fault onset to first 3 consecutive
  anomaly flags above threshold
- **PINN anomaly score**: Hybrid of prediction deviation + ODE residual on observed data
- **LSTM-AE anomaly score**: Reconstruction error (MSE)
- **IsolationForest anomaly score**: Negated sklearn decision_function

## Notes
- All models trained only on normal operation data
- PINN uses physics-informed loss (data + ODE residual)
- Baselines use purely data-driven approaches
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\n[OK] Results written to {output_path}")


if __name__ == "__main__":
    results = evaluate_all()
    write_results(results)
    print("\n[OK] Phase 1 evaluation complete!")
