"""
Training Orchestrator
=====================
Trains all 3 models (PINN, Isolation Forest, LSTM Autoencoder) on the
generated data and logs experiments to MLflow.

Usage:
    python -m models.train

Prerequisites:
    - Data generated in data/ (run simulator.generate_data first)
    - MLflow (optional — logs locally if server not running)
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure proper backend for DeepXDE
os.environ.setdefault("DDE_BACKEND", "pytorch")

import torch
import mlflow
import mlflow.pytorch

from models.pinn.pinn_model import (
    VesselPINN, PINNHyperparams, train_pinn,
    compute_anomaly_scores, save_model as save_pinn,
)
from models.baselines.iso_forest import (
    IsoForestModel, IsoForestHyperparams,
)
from models.baselines.lstm_ae import (
    LSTMAutoencoder, LSTMAEHyperparams, train_lstm_ae,
    compute_anomaly_scores_lstm, save_model as save_lstm,
)


DATA_DIR = Path("data")
ARTIFACTS_DIR = Path("models/artifacts")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/val/test DataFrames."""
    train = pd.read_csv(DATA_DIR / "train.csv")
    val = pd.read_csv(DATA_DIR / "val.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    print(f"Loaded data — Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return train, val, test


def get_normal_data(df: pd.DataFrame) -> pd.DataFrame:
    """Extract only normal operation rows."""
    return df[df["label"] == 0].copy()


def train_pinn_model(
    train_df: pd.DataFrame,
    device: str = "cpu",
) -> tuple[VesselPINN, dict]:
    """Train PINN on normal training data."""
    normal = get_normal_data(train_df)

    hp = PINNHyperparams(
        hidden_layers=[64, 64, 64],
        lr=1e-3,
        epochs=1500,
        lambda_physics=1.0,
        batch_size=512,
    )

    with mlflow.start_run(run_name="PINN", nested=True):
        mlflow.log_params({
            "model_type": "PINN",
            "hidden_layers": str(hp.hidden_layers),
            "lr": hp.lr,
            "epochs": hp.epochs,
            "lambda_physics": hp.lambda_physics,
            "batch_size": hp.batch_size,
            "n_train_samples": len(normal),
        })

        model, history = train_pinn(
            normal["timestamp"].values,
            normal["pressure"].values,
            normal["temperature"].values,
            hp=hp,
            device=device,
            verbose=True,
        )

        # Log loss curves (every 50 epochs to reduce MLflow overhead)
        for i in range(0, len(history["data_loss"]), 50):
            mlflow.log_metrics({
                "pinn_data_loss": history["data_loss"][i],
                "pinn_physics_loss": history["physics_loss"][i],
                "pinn_total_loss": history["total_loss"][i],
            }, step=i)

        # Save model
        save_path = ARTIFACTS_DIR / "pinn_model.pt"
        save_pinn(model, save_path)
        mlflow.log_artifact(str(save_path))

    return model, history


def train_iso_forest_model(
    train_df: pd.DataFrame,
) -> IsoForestModel:
    """Train Isolation Forest on normal training data."""
    normal = get_normal_data(train_df)

    hp = IsoForestHyperparams(
        n_estimators=200,
        contamination=0.05,
    )

    with mlflow.start_run(run_name="IsolationForest", nested=True):
        mlflow.log_params({
            "model_type": "IsolationForest",
            "n_estimators": hp.n_estimators,
            "contamination": hp.contamination,
            "n_train_samples": len(normal),
        })

        model = IsoForestModel(hp=hp, window_size=10)
        model.fit(normal["pressure"].values, normal["temperature"].values)

        # Save model
        save_path = ARTIFACTS_DIR / "iso_forest_model.pkl"
        model.save(save_path)
        mlflow.log_artifact(str(save_path))

    return model


def train_lstm_ae_model(
    train_df: pd.DataFrame,
    device: str = "cpu",
) -> tuple[LSTMAutoencoder, dict]:
    """Train LSTM Autoencoder on normal training data."""
    normal = get_normal_data(train_df)

    hp = LSTMAEHyperparams(
        seq_len=20,
        hidden_dim=32,
        n_layers=2,
        lr=1e-3,
        epochs=100,
        batch_size=64,
    )

    with mlflow.start_run(run_name="LSTM_AE", nested=True):
        mlflow.log_params({
            "model_type": "LSTM_AE",
            "seq_len": hp.seq_len,
            "hidden_dim": hp.hidden_dim,
            "n_layers": hp.n_layers,
            "lr": hp.lr,
            "epochs": hp.epochs,
            "n_train_samples": len(normal),
        })

        model, history = train_lstm_ae(
            normal["pressure"].values,
            normal["temperature"].values,
            hp=hp,
            device=device,
            verbose=True,
        )

        # Log loss curve
        for i, loss in enumerate(history["train_loss"]):
            mlflow.log_metric("lstm_ae_train_loss", loss, step=i)

        # Save model
        save_path = ARTIFACTS_DIR / "lstm_ae_model.pt"
        save_lstm(model, save_path)
        mlflow.log_artifact(str(save_path))

    return model, history


def train_all(device: str = "cpu") -> dict:
    """
    Train all 3 models and return them.

    Returns dict with keys: "pinn", "iso_forest", "lstm_ae"
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = load_data()

    # Set up MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("PIPM_Phase1")

    with mlflow.start_run(run_name="Phase1_Training"):
        print("\n" + "=" * 60)
        print("Training PINN...")
        print("=" * 60)
        pinn_model, pinn_history = train_pinn_model(train_df, device)

        print("\n" + "=" * 60)
        print("Training Isolation Forest...")
        print("=" * 60)
        iso_model = train_iso_forest_model(train_df)

        print("\n" + "=" * 60)
        print("Training LSTM Autoencoder...")
        print("=" * 60)
        lstm_model, lstm_history = train_lstm_ae_model(train_df, device)

    print("\n[OK] All models trained and logged to MLflow!")

    return {
        "pinn": pinn_model,
        "iso_forest": iso_model,
        "lstm_ae": lstm_model,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
    }


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    results = train_all(device)
    print("\nTraining complete. Artifacts saved to models/artifacts/")
