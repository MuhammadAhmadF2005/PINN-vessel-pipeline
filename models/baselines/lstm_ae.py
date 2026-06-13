"""
LSTM Autoencoder Baseline for Anomaly Detection
================================================

Trains an LSTM autoencoder on sequences of (pressure, temperature) from
normal operation data. The reconstruction error is used as an anomaly score:
higher error → more likely anomalous.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from dataclasses import dataclass


@dataclass
class LSTMAEHyperparams:
    """Hyperparameters for LSTM Autoencoder."""
    seq_len: int = 20           # input sequence length
    hidden_dim: int = 32        # LSTM hidden size
    n_layers: int = 2           # number of LSTM layers
    lr: float = 1e-3
    epochs: int = 100
    batch_size: int = 64
    dropout: float = 0.1


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple]:
        _, (h, c) = self.lstm(x)
        return h, c


class LSTMDecoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int,
                 output_dim: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, hidden: tuple) -> torch.Tensor:
        out, _ = self.lstm(x, hidden)
        return self.fc(out)


class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder for time series anomaly detection.
    Encodes a sequence of (P, T) values and reconstructs them.
    """

    def __init__(self, hp: LSTMAEHyperparams | None = None):
        super().__init__()
        if hp is None:
            hp = LSTMAEHyperparams()
        self.hp = hp
        input_dim = 2  # pressure, temperature

        self.encoder = LSTMEncoder(input_dim, hp.hidden_dim, hp.n_layers, hp.dropout)
        self.decoder = LSTMDecoder(input_dim, hp.hidden_dim, hp.n_layers, input_dim, hp.dropout)

        # Normalization buffers
        self.register_buffer("mean", torch.zeros(input_dim))
        self.register_buffer("std", torch.ones(input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, 2)
        Returns: (batch, seq_len, 2) reconstruction
        """
        h, c = self.encoder(x)
        reconstruction = self.decoder(x, (h, c))
        return reconstruction

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample reconstruction error (MSE over sequence and features).
        Returns shape (batch,).
        """
        recon = self.forward(x)
        error = torch.mean((x - recon) ** 2, dim=(1, 2))
        return error


def create_sequences(
    pressure: np.ndarray,
    temperature: np.ndarray,
    seq_len: int,
) -> np.ndarray:
    """
    Create overlapping sequences for LSTM input.

    Returns shape (N - seq_len + 1, seq_len, 2).
    """
    n = len(pressure)
    data = np.stack([pressure, temperature], axis=1)  # (N, 2)

    sequences = []
    for i in range(n - seq_len + 1):
        sequences.append(data[i : i + seq_len])

    return np.array(sequences)


def train_lstm_ae(
    pressure: np.ndarray,
    temperature: np.ndarray,
    hp: LSTMAEHyperparams | None = None,
    device: str = "cpu",
    verbose: bool = True,
) -> tuple[LSTMAutoencoder, dict]:
    """
    Train the LSTM autoencoder on normal operation data.

    Returns
    -------
    model : trained LSTMAutoencoder
    history : dict with "train_loss" list
    """
    if hp is None:
        hp = LSTMAEHyperparams()

    # Create sequences
    sequences = create_sequences(pressure, temperature, hp.seq_len)
    X = torch.tensor(sequences, dtype=torch.float32).to(device)

    # Compute normalization
    mean = X.reshape(-1, 2).mean(dim=0)
    std = X.reshape(-1, 2).std(dim=0)
    X_norm = (X - mean) / (std + 1e-8)

    # Model
    model = LSTMAutoencoder(hp).to(device)
    model.mean.copy_(mean)
    model.std.copy_(std)

    optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5, min_lr=1e-6
    )

    dataset = TensorDataset(X_norm)
    loader = DataLoader(dataset, batch_size=hp.batch_size, shuffle=True)

    history = {"train_loss": []}

    for epoch in range(hp.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for (batch,) in loader:
            recon = model(batch)
            loss = torch.mean((batch - recon) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        history["train_loss"].append(avg_loss)
        scheduler.step(avg_loss)

        if verbose and (epoch % 20 == 0 or epoch == hp.epochs - 1):
            print(f"Epoch {epoch:4d}/{hp.epochs} | Loss: {avg_loss:.6f}")

    return model, history


def compute_anomaly_scores_lstm(
    model: LSTMAutoencoder,
    pressure: np.ndarray,
    temperature: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """
    Compute anomaly scores (reconstruction error) for given data.

    Returns shape (N - seq_len + 1,) array of anomaly scores.
    """
    model.eval()
    sequences = create_sequences(pressure, temperature, model.hp.seq_len)
    X = torch.tensor(sequences, dtype=torch.float32).to(device)
    X_norm = (X - model.mean) / (model.std + 1e-8)

    with torch.no_grad():
        scores = model.reconstruction_error(X_norm)

    return scores.cpu().numpy()


def save_model(model: LSTMAutoencoder, path: str | Path) -> None:
    """Save model to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "hp": model.hp,
    }, path)
    print(f"LSTM-AE saved to {path}")


def load_model(path: str | Path, device: str = "cpu") -> LSTMAutoencoder:
    """Load model from disk."""
    data = torch.load(path, weights_only=False, map_location=device)
    model = LSTMAutoencoder(data["hp"]).to(device)
    model.load_state_dict(data["state_dict"])
    model.eval()
    return model


if __name__ == "__main__":
    from simulator.vessel_sim import SimConfig, FaultConfig, simulate

    print("=== LSTM Autoencoder Smoke Test ===")

    # Train on normal data
    normal = simulate(SimConfig(t_end=100, dt=0.5, noise_std_P=0.05, noise_std_T=0.3))
    hp = LSTMAEHyperparams(epochs=50, seq_len=10)
    model, history = train_lstm_ae(
        normal["pressure"], normal["temperature"],
        hp=hp, verbose=True,
    )

    # Score normal data
    scores_n = compute_anomaly_scores_lstm(model, normal["pressure"], normal["temperature"])
    print(f"Normal scores — mean: {scores_n.mean():.6f}, max: {scores_n.max():.6f}")

    # Score faulty data
    fault_cfg = FaultConfig(fault_type="heater_drift", onset_time=50.0, severity=1.0)
    faulty = simulate(SimConfig(t_end=100, dt=0.5, noise_std_P=0.05, noise_std_T=0.3,
                                fault=fault_cfg, seed=99))
    scores_f = compute_anomaly_scores_lstm(model, faulty["pressure"], faulty["temperature"])
    print(f"Fault scores  — mean: {scores_f.mean():.6f}, max: {scores_f.max():.6f}")

    print("[OK] LSTM-AE smoke test OK")
