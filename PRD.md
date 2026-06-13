# Product Requirements Document
## Physics-Informed Predictive Maintenance Pipeline (PIPM)

### 1. Overview
Build an end-to-end MLOps system that detects early-stage faults in an industrial 
pressure vessel using a Physics-Informed Neural Network (PINN). The model's 
physics-residual is used as an anomaly score, compared against an LSTM 
Autoencoder and Isolation Forest baseline.

### 2. Goals
- G1: Simulate a labeled pressure-vessel dataset (normal + 3 fault types).
- G2: Train a PINN whose loss = data loss + physics (ODE residual) loss.
- G3: Train two baseline models (Isolation Forest, LSTM Autoencoder).
- G4: Build a real-time inference pipeline (FastAPI) that scores incoming data.
- G5: Build a streaming simulation (MQTT publisher → consumer → feature worker).
- G6: Log all experiments in MLflow, including physics loss vs data loss curves.
- G7: Build a React dashboard showing live anomaly score, RUL estimate, alert log.
- G8: Containerize everything with Docker Compose; add CI via GitHub Actions.
- G9: Evaluate on Tennessee Eastman Process dataset for generalization.
- G10: Produce a 3-page technical report summarizing methodology and results.

### 3. Non-Goals
- No real factory hardware integration in v1 (simulation only).
- No multi-vessel / multi-tenant support.
- No user authentication system (single-user demo).

### 4. Success Metrics
- PINN anomaly score achieves higher F1 and earlier time-to-detection than 
  both baselines on simulated test set (target: ≥15% improvement in 
  time-to-detection vs LSTM autoencoder).
- Full pipeline runs end-to-end via `docker compose up`.
- Dashboard updates anomaly score within 2s of new data point.

### 5. Users / Use Case
Solo developer (portfolio/research project). Secondary audience: interviewers, 
ICET paper reviewers, French Masters admissions committee.

### 6. Phases (see TASKS.md for breakdown)
1. Data simulation + PINN baseline
2. Pipeline (streaming, feature engineering, serving, dashboard)
3. Deployment + report + TEP validation

### 7. Constraints
- Stack: PyTorch + DeepXDE, scikit-learn, FastAPI, React, MQTT (Mosquitto), 
  InfluxDB, SQLite, MLflow, Docker, GitHub Actions.
- Target environment: local dev (Docker Compose) → AWS EC2 deploy.

### 8. References
- Tennessee Eastman Process dataset (GitHub mirrors)
- DeepXDE documentation
- MLflow documentation