"""Quick training script to test the full pipeline without MLflow overhead."""
import os
os.environ["DDE_BACKEND"] = "pytorch"

import torch
import pandas as pd
from pathlib import Path

# Load data
train = pd.read_csv("data/train.csv")
normal = train[train["label"] == 0]
print(f"Normal training samples: {len(normal)}")

# ---- PINN ----
print("\n=== Training PINN ===")
from models.pinn.pinn_model import PINNHyperparams, train_pinn, save_model as save_pinn

hp = PINNHyperparams(hidden_layers=[64, 64, 64], lr=1e-3, epochs=1500, lambda_physics=1.0, batch_size=512)
pinn_model, pinn_history = train_pinn(
    normal["timestamp"].values, normal["pressure"].values, normal["temperature"].values,
    hp=hp, verbose=True,
)
Path("models/artifacts").mkdir(parents=True, exist_ok=True)
save_pinn(pinn_model, "models/artifacts/pinn_model.pt")

# ---- Isolation Forest ----
print("\n=== Training Isolation Forest ===")
from models.baselines.iso_forest import IsoForestModel, IsoForestHyperparams

iso_hp = IsoForestHyperparams(n_estimators=200, contamination=0.05)
iso_model = IsoForestModel(hp=iso_hp, window_size=10)
iso_model.fit(normal["pressure"].values, normal["temperature"].values)
iso_model.save("models/artifacts/iso_forest_model.pkl")

# ---- LSTM Autoencoder ----
print("\n=== Training LSTM Autoencoder ===")
from models.baselines.lstm_ae import LSTMAEHyperparams, train_lstm_ae, save_model as save_lstm

lstm_hp = LSTMAEHyperparams(seq_len=20, hidden_dim=32, n_layers=2, lr=1e-3, epochs=100, batch_size=64)
lstm_model, lstm_history = train_lstm_ae(
    normal["pressure"].values, normal["temperature"].values,
    hp=lstm_hp, verbose=True,
)
save_lstm(lstm_model, "models/artifacts/lstm_ae_model.pt")

# ---- Verify artifacts ----
print("\n=== Verifying Artifacts ===")
for f in ["pinn_model.pt", "iso_forest_model.pkl", "lstm_ae_model.pt"]:
    p = Path("models/artifacts") / f
    if p.exists():
        print(f"  {f}: {p.stat().st_size} bytes")
    else:
        print(f"  {f}: MISSING!")

print("\n[OK] All models trained and saved!")
