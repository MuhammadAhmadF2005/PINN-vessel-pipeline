# Task Breakdown (execute in order; check off as completed)

## Phase 1 — Data + PINN Baseline
- [x] T1.1: Implement `simulator/vessel_sim.py` (ODE solver, normal scenario)
- [x] T1.2: Add fault injection (3 scenarios) + Gaussian sensor noise
- [x] T1.3: Generate train/val/test CSVs, save to `data/`
- [x] T1.4: Implement `models/pinn/pinn_model.py` using DeepXDE 
      (data loss + ODE residual loss)
- [x] T1.5: Implement `models/baselines/iso_forest.py`
- [x] T1.6: Implement `models/baselines/lstm_ae.py`
- [x] T1.7: Implement `models/train.py`: trains all 3, logs to MLflow 
      (separate physics vs data loss curves for PINN)
- [x] T1.8: Implement `evaluation/metrics.py`: F1, precision/recall, 
      time-to-detection
- [x] T1.9: Produce comparison table (PINN vs baselines) — save as 
      `evaluation/results_phase1.md`

## Phase 2 — Pipeline
- [ ] T2.1: `streaming/publisher.py` — replay CSV over MQTT
- [ ] T2.2: `streaming/feature_worker.py` — rolling stats/FFT/lag features, 
      write to InfluxDB
- [ ] T2.3: `api/main.py` — FastAPI app, `/score`, `/history`, `/alerts`
- [ ] T2.4: Wire trained models into `/score` endpoint
- [ ] T2.5: SQLite schema + writer for scores/alerts
- [ ] T2.6: `dashboard/` — React app: live chart, RUL widget, alert table
- [ ] T2.7: `docker-compose.yml` — mosquitto, influxdb, api, dashboard, mlflow
- [ ] T2.8: End-to-end smoke test: `docker compose up` -> dashboard shows 
      live anomaly scores

## Phase 3 — Deployment + Validation + Report
- [ ] T3.1: `.github/workflows/ci.yml` — lint, test, build images
- [ ] T3.2: EC2 deployment script/notes (`infra/DEPLOY.md`)
- [ ] T3.3: `evaluation/tep_eval.py` — Tennessee Eastman generalization test
- [ ] T3.4: `evaluation/results_phase3.md` — TEP results table
- [ ] T3.5: `report/report.md` — 3-page writeup (intro, method, results, 
      conclusion, figures from MLflow)
- [ ] T3.6: Update root `README.md` with setup/run instructions and resume bullet

## Definition of Done (per task)
- Code runs without errors via documented command
- Unit test or smoke test added where applicable
- Relevant doc (`results_*.md`, `README.md`) updated