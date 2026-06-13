# Architecture

## High-Level Flow

## Components


### 1. Simulator (`/simulator`)
- `vessel_sim.py`: solves pressure/temperature ODEs via `scipy.solve_ivp`
- Generates 3 fault scenarios: seal degradation, heater drift, sudden blockage
- Outputs: `data/normal.csv`, `data/fault_*.csv` with columns 
  `[timestamp, pressure, temperature, label, fault_type]`

### 2. Models (`/models`)
- `pinn/`: DeepXDE-based PINN, physics residual = ODE residual
- `baselines/lstm_ae.py`, `baselines/iso_forest.py`
- `train.py`: trains all 3, logs to MLflow, saves artifacts to `/models/artifacts`

### 3. Streaming (`/streaming`)
- `publisher.py`: replays CSV rows over MQTT topic `vessel/sensors`
- `feature_worker.py`: subscribes, computes rolling mean/std/FFT/lag features, 
  writes to InfluxDB

### 4. Inference Service (`/api`)
- FastAPI app
- `/score` endpoint: takes latest feature window, returns anomaly scores 
  from all 3 models + ensemble decision
- `/history` endpoint: returns recent scores for dashboard
- `/alerts` endpoint: returns alert log
- Background task subscribes to InfluxDB / MQTT and writes scores to SQLite

### 5. Dashboard (`/dashboard`)
- React app (Vite)
- Live chart (recharts) of pressure/temperature + anomaly score
- RUL countdown widget
- Alert history table

### 6. Infra (`/infra`)
- `docker-compose.yml`: mosquitto, influxdb, api, dashboard, mlflow
- `.github/workflows/ci.yml`: lint + test + build images

### 7. Evaluation (`/evaluation`)
- `tep_eval.py`: loads Tennessee Eastman dataset, runs trained models, 
  reports F1/precision/recall/time-to-detection table

### 8. Report (`/report`)
- `report.md` -> 3-page technical writeup (methodology, results, figures)