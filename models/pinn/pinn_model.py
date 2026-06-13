"""
Physics-Informed Neural Network (PINN) for Pressure Vessel Anomaly Detection
=============================================================================

Uses DeepXDE with PyTorch backend to build a PINN whose loss is:
    L_total = L_data + lambda_phys * L_physics

Where L_physics is the ODE residual:
    dP/dt - [(Q_in - k_v*P)/V - alpha*(T - T_env)] = 0
    dT/dt - [(q_heater - h*A*(T - T_env))/(m*Cp)] = 0

The physics residual magnitude is used as an anomaly score at inference time:
a high residual means the data deviates from the expected physics.
"""

from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from dataclasses import dataclass

# Ensure DeepXDE uses PyTorch backend
os.environ.setdefault("DDE_BACKEND", "pytorch")


@dataclass
class PINNHyperparams:
    """Hyperparameters for the PINN model."""
    hidden_layers: list[int] | None = None
    lr: float = 1e-3
    epochs: int = 5000
    lambda_physics: float = 1.0
    batch_size: int = 256

    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [64, 64, 64]


class VesselPINN(nn.Module):
    """
    A PyTorch neural network that maps time → (P, T).

    The physics loss is computed by auto-differentiating the network output
    w.r.t. time and comparing against the ODE right-hand side.
    """

    # Default vessel physics parameters (same as simulator)
    VESSEL_PARAMS = {
        "V": 2.0,
        "Q_in": 0.5,
        "k_v": 0.1,
        "alpha": 0.002,
        "q_heater": 5000.0,
        "h": 10.0,
        "A": 6.0,
        "m": 100.0,
        "Cp": 4186.0,
        "T_env": 298.15,
    }

    def __init__(self, hp: PINNHyperparams | None = None):
        super().__init__()
        if hp is None:
            hp = PINNHyperparams()
        self.hp = hp

        # Build MLP: input=1 (time) → hidden → output=2 (P, T)
        layers = []
        in_dim = 1
        for h_dim in hp.hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.Tanh())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 2))
        self.net = nn.Sequential(*layers)

        # Normalization stats (set during training)
        self.register_buffer("t_mean", torch.tensor(0.0))
        self.register_buffer("t_std", torch.tensor(1.0))
        self.register_buffer("P_mean", torch.tensor(0.0))
        self.register_buffer("P_std", torch.tensor(1.0))
        self.register_buffer("T_mean", torch.tensor(0.0))
        self.register_buffer("T_std", torch.tensor(1.0))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: t → [P_normalized, T_normalized].
        Input t should be raw (un-normalized) — normalization is internal.
        """
        t_norm = (t - self.t_mean) / (self.t_std + 1e-8)
        return self.net(t_norm)

    def predict_physical(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return denormalized (P, T) predictions.
        """
        out = self.forward(t)
        P = out[:, 0:1] * self.P_std + self.P_mean
        T = out[:, 1:2] * self.T_std + self.T_mean
        return P, T

    def physics_residual(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the ODE residual at given time points.

        Returns shape (N, 2) — residual for [dP/dt equation, dT/dt equation].
        """
        t = t.requires_grad_(True)
        P, T = self.predict_physical(t)
        vp = self.VESSEL_PARAMS

        # Compute gradients
        dP_dt = torch.autograd.grad(
            P, t, grad_outputs=torch.ones_like(P),
            create_graph=True, retain_graph=True
        )[0]

        dT_dt = torch.autograd.grad(
            T, t, grad_outputs=torch.ones_like(T),
            create_graph=True, retain_graph=True
        )[0]

        # Expected RHS from physics
        dP_dt_expected = (vp["Q_in"] - vp["k_v"] * P) / vp["V"] - vp["alpha"] * (T - vp["T_env"])
        dT_dt_expected = (vp["q_heater"] - vp["h"] * vp["A"] * (T - vp["T_env"])) / (vp["m"] * vp["Cp"])

        res_P = dP_dt - dP_dt_expected
        res_T = dT_dt - dT_dt_expected

        return torch.cat([res_P, res_T], dim=1)

    def anomaly_score(self, t: torch.Tensor) -> torch.Tensor:
        """
        Anomaly score = L2 norm of physics residual at each time point.
        Higher score → more deviation from expected physics.
        """
        self.eval()
        with torch.enable_grad():
            residual = self.physics_residual(t)
        return torch.norm(residual, dim=1)


def train_pinn(
    time: np.ndarray,
    pressure: np.ndarray,
    temperature: np.ndarray,
    hp: PINNHyperparams | None = None,
    device: str = "cpu",
    verbose: bool = True,
) -> tuple[VesselPINN, dict]:
    """
    Train the PINN on observed (time, pressure, temperature) data.

    Parameters
    ----------
    time : (N,) array of time values
    pressure : (N,) array of pressure observations
    temperature : (N,) array of temperature observations
    hp : hyperparameters
    device : "cpu" or "cuda"
    verbose : print training progress

    Returns
    -------
    model : trained VesselPINN
    history : dict with "data_loss", "physics_loss", "total_loss" lists
    """
    if hp is None:
        hp = PINNHyperparams()

    # Convert to tensors
    t_tensor = torch.tensor(time, dtype=torch.float32).reshape(-1, 1).to(device)
    P_tensor = torch.tensor(pressure, dtype=torch.float32).reshape(-1, 1).to(device)
    T_tensor = torch.tensor(temperature, dtype=torch.float32).reshape(-1, 1).to(device)

    # Compute normalization stats
    t_mean, t_std = t_tensor.mean(), t_tensor.std()
    P_mean, P_std = P_tensor.mean(), P_tensor.std()
    T_mean, T_std = T_tensor.mean(), T_tensor.std()

    # Build model
    model = VesselPINN(hp).to(device)
    model.t_mean.copy_(t_mean)
    model.t_std.copy_(t_std)
    model.P_mean.copy_(P_mean)
    model.P_std.copy_(P_std)
    model.T_mean.copy_(T_mean)
    model.T_std.copy_(T_std)

    optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=300, factor=0.5, min_lr=1e-6
    )

    # Normalize targets
    P_norm = (P_tensor - P_mean) / (P_std + 1e-8)
    T_norm = (T_tensor - T_mean) / (T_std + 1e-8)

    history = {"data_loss": [], "physics_loss": [], "total_loss": []}

    n_samples = len(t_tensor)

    for epoch in range(hp.epochs):
        model.train()

        # Mini-batch or full batch
        if hp.batch_size >= n_samples:
            idx = torch.arange(n_samples)
        else:
            idx = torch.randperm(n_samples)[: hp.batch_size]

        t_batch = t_tensor[idx]
        P_batch = P_norm[idx]
        T_batch = T_norm[idx]

        # Forward pass
        pred = model(t_batch)
        P_pred = pred[:, 0:1]
        T_pred = pred[:, 1:2]

        # Data loss (MSE on normalized values)
        data_loss = (
            torch.mean((P_pred - P_batch) ** 2)
            + torch.mean((T_pred - T_batch) ** 2)
        )

        # Physics loss (ODE residual)
        residual = model.physics_residual(t_batch)
        physics_loss = torch.mean(residual ** 2)

        # Total loss
        total_loss = data_loss + hp.lambda_physics * physics_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss.detach())

        # Record
        history["data_loss"].append(data_loss.item())
        history["physics_loss"].append(physics_loss.item())
        history["total_loss"].append(total_loss.item())

        if verbose and (epoch % 500 == 0 or epoch == hp.epochs - 1):
            print(
                f"Epoch {epoch:5d}/{hp.epochs} | "
                f"Data: {data_loss.item():.6f} | "
                f"Physics: {physics_loss.item():.6f} | "
                f"Total: {total_loss.item():.6f}"
            )

    return model, history


def compute_anomaly_scores(
    model: VesselPINN,
    time: np.ndarray,
    pressure: np.ndarray | None = None,
    temperature: np.ndarray | None = None,
    device: str = "cpu",
) -> np.ndarray:
    """
    Compute anomaly scores using a hybrid approach:

    1. **Prediction deviation**: How far observed (P, T) are from the PINN's
       learned normal trajectory.
    2. **Physics residual**: How much the observed data violates the ODE,
       computed via finite-difference derivatives of the observations.

    The combined score is: sqrt(prediction_error^2 + physics_violation^2)

    Parameters
    ----------
    model : trained VesselPINN
    time : (N,) time array
    pressure : (N,) observed pressure values (optional for backward compat)
    temperature : (N,) observed temperature values (optional)
    device : torch device

    Returns
    -------
    scores : (N,) numpy array of anomaly scores
    """
    model.eval()
    t_tensor = torch.tensor(time, dtype=torch.float32).reshape(-1, 1).to(device)

    # If no observations provided, fall back to pure network residual
    if pressure is None or temperature is None:
        with torch.enable_grad():
            scores = model.anomaly_score(t_tensor)
        return scores.detach().cpu().numpy()

    P_obs = torch.tensor(pressure, dtype=torch.float32).reshape(-1, 1).to(device)
    T_obs = torch.tensor(temperature, dtype=torch.float32).reshape(-1, 1).to(device)

    # --- Score component 1: Prediction deviation ---
    with torch.no_grad():
        P_pred, T_pred = model.predict_physical(t_tensor)

    pred_error_P = ((P_obs - P_pred) / (model.P_std + 1e-8)) ** 2
    pred_error_T = ((T_obs - T_pred) / (model.T_std + 1e-8)) ** 2
    pred_error = (pred_error_P + pred_error_T).squeeze()

    # --- Score component 2: Physics ODE violation on observed data ---
    vp = model.VESSEL_PARAMS
    dt_vals = np.diff(time)
    dt_vals = np.append(dt_vals, dt_vals[-1])  # pad to same length
    dt_tensor = torch.tensor(dt_vals, dtype=torch.float32).reshape(-1, 1).to(device)

    # Finite-difference derivatives of observed data
    dP = torch.zeros_like(P_obs)
    dT = torch.zeros_like(T_obs)
    dP[1:] = (P_obs[1:] - P_obs[:-1]) / dt_tensor[1:]
    dT[1:] = (T_obs[1:] - T_obs[:-1]) / dt_tensor[1:]
    dP[0] = dP[1]
    dT[0] = dT[1]

    # Expected derivatives from physics
    dP_expected = (vp["Q_in"] - vp["k_v"] * P_obs) / vp["V"] - vp["alpha"] * (T_obs - vp["T_env"])
    dT_expected = (vp["q_heater"] - vp["h"] * vp["A"] * (T_obs - vp["T_env"])) / (vp["m"] * vp["Cp"])

    physics_res_P = (dP - dP_expected) ** 2
    physics_res_T = (dT - dT_expected) ** 2
    physics_error = (physics_res_P + physics_res_T).squeeze()

    # Combine: hybrid score
    combined = torch.sqrt(pred_error + physics_error + 1e-10)

    return combined.detach().cpu().numpy()



def save_model(model: VesselPINN, path: str | Path) -> None:
    """Save model weights and normalization buffers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"PINN model saved to {path}")


def load_model(path: str | Path, hp: PINNHyperparams | None = None) -> VesselPINN:
    """Load a saved PINN model."""
    model = VesselPINN(hp)
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model


if __name__ == "__main__":
    # Smoke test: train on a small normal dataset
    from simulator.vessel_sim import SimConfig, simulate

    print("=== PINN Smoke Test ===")
    result = simulate(SimConfig(t_end=50, dt=0.5, noise_std_P=0.02, noise_std_T=0.2))

    hp = PINNHyperparams(epochs=1000, hidden_layers=[32, 32])
    model, history = train_pinn(
        result["time"], result["pressure"], result["temperature"],
        hp=hp, verbose=True,
    )

    scores = compute_anomaly_scores(model, result["time"])
    print(f"\nAnomaly scores — mean: {scores.mean():.6f}, max: {scores.max():.6f}")
    print("[OK] PINN smoke test OK")
