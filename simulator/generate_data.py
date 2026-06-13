"""
Data Generation Script
======================
Generates train/val/test CSV files from the vessel simulator
with normal operation and 3 fault scenarios.

Usage:
    python -m simulator.generate_data

Output:
    data/normal.csv
    data/fault_seal_degradation.csv
    data/fault_heater_drift.csv
    data/fault_blockage.csv
    data/train.csv  (70% of each scenario)
    data/val.csv    (15%)
    data/test.csv   (15%)
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from pathlib import Path

from simulator.vessel_sim import SimConfig, VesselParams, FaultConfig, simulate


# ─── Configuration ───────────────────────────────────────────────────────────

DATA_DIR = Path("data")

# Simulation parameters
T_END = 200.0          # seconds
DT = 0.5              # sample interval (s)  → 400 points per scenario
NOISE_STD_P = 0.05     # bar
NOISE_STD_T = 0.3      # K
FAULT_ONSET = 80.0     # fault starts at t=80s (first 80s is normal)
RAMP_DURATION = 20.0   # progressive ramp to full severity

# Number of independent runs per scenario (different noise seeds)
RUNS_PER_SCENARIO = 5

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

FAULT_TYPES = ["seal_degradation", "heater_drift", "blockage"]


def generate_scenario(
    fault_type: str = "none",
    seed: int = 42,
    severity: float = 1.0,
) -> pd.DataFrame:
    """Generate a single simulation run and return as DataFrame."""

    fault = FaultConfig(
        fault_type=fault_type,
        onset_time=FAULT_ONSET if fault_type != "none" else 9999.0,
        severity=severity,
        ramp_duration=RAMP_DURATION,
    )

    config = SimConfig(
        params=VesselParams(),
        fault=fault,
        t_end=T_END,
        dt=DT,
        noise_std_P=NOISE_STD_P,
        noise_std_T=NOISE_STD_T,
        seed=seed,
    )

    result = simulate(config)

    df = pd.DataFrame({
        "timestamp": result["time"],
        "pressure": result["pressure"],
        "temperature": result["temperature"],
        "label": result["label"],
        "fault_type": fault_type if fault_type != "none" else "normal",
    })

    return df


def generate_all() -> dict[str, pd.DataFrame]:
    """Generate all scenarios with multiple runs each."""

    all_dfs: dict[str, list[pd.DataFrame]] = {
        "normal": [],
        "seal_degradation": [],
        "heater_drift": [],
        "blockage": [],
    }

    print("Generating simulated data...")

    # Normal runs
    for i in range(RUNS_PER_SCENARIO):
        df = generate_scenario(fault_type="none", seed=100 + i)
        df["run_id"] = i
        all_dfs["normal"].append(df)
        print(f"  Normal run {i+1}/{RUNS_PER_SCENARIO} — {len(df)} points")

    # Fault runs
    for ft in FAULT_TYPES:
        for i in range(RUNS_PER_SCENARIO):
            df = generate_scenario(fault_type=ft, seed=200 + i * 10)
            df["run_id"] = RUNS_PER_SCENARIO + i  # offset run IDs
            all_dfs[ft].append(df)
            print(f"  {ft} run {i+1}/{RUNS_PER_SCENARIO} — {len(df)} points")

    # Concatenate per scenario
    scenario_dfs = {}
    for key, dfs in all_dfs.items():
        scenario_dfs[key] = pd.concat(dfs, ignore_index=True)

    return scenario_dfs


def split_and_save(scenario_dfs: dict[str, pd.DataFrame]) -> None:
    """Split each scenario into train/val/test and save CSVs."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save per-scenario files
    for name, df in scenario_dfs.items():
        fname = f"normal.csv" if name == "normal" else f"fault_{name}.csv"
        path = DATA_DIR / fname
        df.to_csv(path, index=False)
        print(f"Saved {path} ({len(df)} rows)")

    # Build train/val/test splits
    # Strategy: split by run_id so entire runs stay in one split
    train_dfs, val_dfs, test_dfs = [], [], []

    for name, df in scenario_dfs.items():
        run_ids = sorted(df["run_id"].unique())
        n = len(run_ids)
        n_train = max(1, int(n * TRAIN_RATIO))
        n_val = max(1, int(n * VAL_RATIO))
        # Remaining go to test
        train_ids = run_ids[:n_train]
        val_ids = run_ids[n_train:n_train + n_val]
        test_ids = run_ids[n_train + n_val:]

        if len(test_ids) == 0:
            # Ensure at least 1 test run
            test_ids = [val_ids[-1]]
            val_ids = val_ids[:-1]

        train_dfs.append(df[df["run_id"].isin(train_ids)])
        val_dfs.append(df[df["run_id"].isin(val_ids)])
        test_dfs.append(df[df["run_id"].isin(test_ids)])

    train = pd.concat(train_dfs, ignore_index=True)
    val = pd.concat(val_dfs, ignore_index=True)
    test = pd.concat(test_dfs, ignore_index=True)

    train.to_csv(DATA_DIR / "train.csv", index=False)
    val.to_csv(DATA_DIR / "val.csv", index=False)
    test.to_csv(DATA_DIR / "test.csv", index=False)

    print(f"\nSplit sizes — Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    print(f"Train label distribution:\n{train['label'].value_counts().to_string()}")
    print(f"Val label distribution:\n{val['label'].value_counts().to_string()}")
    print(f"Test label distribution:\n{test['label'].value_counts().to_string()}")


if __name__ == "__main__":
    scenario_dfs = generate_all()
    split_and_save(scenario_dfs)
    print("\n[OK] Data generation complete!")
