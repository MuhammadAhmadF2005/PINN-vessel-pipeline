"""
Isolation Forest Baseline for Anomaly Detection
================================================

Trains an Isolation Forest on (pressure, temperature) features from
normal operation data. The anomaly score is the negative of the
sklearn decision_function (higher = more anomalous).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass


@dataclass
class IsoForestHyperparams:
    """Hyperparameters for Isolation Forest."""
    n_estimators: int = 200
    contamination: float = 0.05  # expected anomaly fraction in training data
    max_samples: str | int = "auto"
    random_state: int = 42


def prepare_features(
    pressure: np.ndarray,
    temperature: np.ndarray,
    window_size: int = 10,
) -> np.ndarray:
    """
    Create feature matrix with rolling statistics.

    Features per point:
        - pressure, temperature (raw)
        - rolling mean/std of P and T
        - pressure-temperature ratio
        - first derivative (finite diff) of P and T

    Returns shape (N - window_size + 1, n_features).
    """
    n = len(pressure)
    features = []

    for i in range(window_size - 1, n):
        window_P = pressure[i - window_size + 1 : i + 1]
        window_T = temperature[i - window_size + 1 : i + 1]

        feat = [
            pressure[i],
            temperature[i],
            np.mean(window_P),
            np.std(window_P),
            np.mean(window_T),
            np.std(window_T),
            pressure[i] / (temperature[i] + 1e-8),  # P/T ratio
        ]

        # First derivative (finite difference)
        if i > 0:
            feat.append(pressure[i] - pressure[i - 1])
            feat.append(temperature[i] - temperature[i - 1])
        else:
            feat.extend([0.0, 0.0])

        features.append(feat)

    return np.array(features)


class IsoForestModel:
    """Wrapper around sklearn IsolationForest with feature engineering."""

    def __init__(self, hp: IsoForestHyperparams | None = None, window_size: int = 10):
        if hp is None:
            hp = IsoForestHyperparams()
        self.hp = hp
        self.window_size = window_size
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=hp.n_estimators,
            contamination=hp.contamination,
            max_samples=hp.max_samples,
            random_state=hp.random_state,
        )
        self._is_fitted = False

    def fit(self, pressure: np.ndarray, temperature: np.ndarray) -> "IsoForestModel":
        """
        Fit on normal operation data.
        """
        X = prepare_features(pressure, temperature, self.window_size)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self._is_fitted = True
        print(f"IsolationForest fitted on {len(X)} samples, {X.shape[1]} features")
        return self

    def anomaly_score(
        self, pressure: np.ndarray, temperature: np.ndarray
    ) -> np.ndarray:
        """
        Compute anomaly scores. Higher = more anomalous.

        Returns shape (N - window_size + 1,).
        Scores are negated decision_function (sklearn convention: lower = anomalous).
        """
        assert self._is_fitted, "Model must be fitted first"
        X = prepare_features(pressure, temperature, self.window_size)
        X_scaled = self.scaler.transform(X)
        # sklearn: decision_function < 0 means anomaly; we negate so higher = anomaly
        raw_scores = -self.model.decision_function(X_scaled)
        return raw_scores

    def predict(self, pressure: np.ndarray, temperature: np.ndarray) -> np.ndarray:
        """Binary predictions: 1 = anomaly, 0 = normal."""
        assert self._is_fitted, "Model must be fitted first"
        X = prepare_features(pressure, temperature, self.window_size)
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)
        # sklearn: -1 = anomaly, 1 = normal → convert to 0/1
        return (preds == -1).astype(int)

    def save(self, path: str | Path) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "scaler": self.scaler,
                      "window_size": self.window_size, "hp": self.hp}, path)
        print(f"IsolationForest saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "IsoForestModel":
        """Load model from disk."""
        data = joblib.load(path)
        obj = cls(hp=data["hp"], window_size=data["window_size"])
        obj.model = data["model"]
        obj.scaler = data["scaler"]
        obj._is_fitted = True
        return obj


if __name__ == "__main__":
    from simulator.vessel_sim import SimConfig, VesselParams, FaultConfig, simulate

    print("=== Isolation Forest Smoke Test ===")

    # Train on normal data
    normal = simulate(SimConfig(t_end=100, dt=0.5, noise_std_P=0.05, noise_std_T=0.3))
    model = IsoForestModel()
    model.fit(normal["pressure"], normal["temperature"])

    # Score normal data
    scores_normal = model.anomaly_score(normal["pressure"], normal["temperature"])
    print(f"Normal scores — mean: {scores_normal.mean():.4f}, max: {scores_normal.max():.4f}")

    # Score faulty data
    fault_cfg = FaultConfig(fault_type="seal_degradation", onset_time=50.0, severity=1.0)
    faulty = simulate(SimConfig(t_end=100, dt=0.5, noise_std_P=0.05, noise_std_T=0.3,
                                fault=fault_cfg, seed=99))
    scores_fault = model.anomaly_score(faulty["pressure"], faulty["temperature"])
    print(f"Fault scores  — mean: {scores_fault.mean():.4f}, max: {scores_fault.max():.4f}")

    print("[OK] Isolation Forest smoke test OK")
