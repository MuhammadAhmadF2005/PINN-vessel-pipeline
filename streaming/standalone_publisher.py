import time
import requests
import pandas as pd
import numpy as np

WINDOW_SIZE = 10
API_URL = "http://127.0.0.1:8000"

def compute_features(hist):
    if len(hist) < WINDOW_SIZE:
        return None
    pressures = [d["pressure"] for d in hist[-WINDOW_SIZE:]]
    temperatures = [d["temperature"] for d in hist[-WINDOW_SIZE:]]
    return {
        "pressure_mean": np.mean(pressures),
        "pressure_std": np.std(pressures),
        "temperature_mean": np.mean(temperatures),
        "temperature_std": np.std(temperatures)
    }

def run_standalone(csv_path="data/test.csv", speed=0.1):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path).sort_values("timestamp").reset_index(drop=True)
    
    history = []
    prev_time = None
    
    for idx, row in df.iterrows():
        payload = {
            "timestamp": float(row["timestamp"]),
            "pressure": float(row["pressure"]),
            "temperature": float(row["temperature"]),
            "label": int(row["label"]),
            "fault_type": str(row["fault_type"]),
            "run_id": str(row["run_id"])
        }
        
        history.append(payload)
        if len(history) > WINDOW_SIZE * 2:
            history.pop(0)
            
        if prev_time is not None:
            dt = float(row["timestamp"]) - prev_time
            if dt > 0:
                time.sleep(dt * speed)
        
        # Once we have enough for LSTM (20), send to API
        if len(history) >= 20:
            try:
                resp = requests.post(f"{API_URL}/score", json={"window": history[-20:]})
                if resp.status_code == 200:
                    res = resp.json()
                    print(f"Sent T={payload['timestamp']:.1f} | PINN: {res['pinn_score']:.2f} | Anomaly: {res['is_anomaly']}")
                else:
                    print(f"API Error {resp.status_code}: {resp.text}")
            except Exception as e:
                print(f"API not reachable yet... {e}")
                
        prev_time = float(row["timestamp"])

if __name__ == "__main__":
    run_standalone()
