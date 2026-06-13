# Tech Stack & Conventions

| Layer | Choice | Notes |
|---|---|---|
| Simulation | Python, scipy.integrate.solve_ivp | ODE-based vessel model |
| PINN | PyTorch + DeepXDE | physics residual = ODE residual |
| Baselines | scikit-learn (IsolationForest), PyTorch (LSTM-AE) | |
| Experiment tracking | MLflow | log physics loss & data loss separately |
| Streaming | Mosquitto (MQTT) | topic: `vessel/sensors` |
| Time-series storage | InfluxDB | feature + raw storage |
| Metadata/alerts storage | SQLite | scores + alert log |
| Serving | FastAPI | `/score`, `/history`, `/alerts` |
| Frontend | React + Vite + recharts | dashboard |
| Containerization | Docker + docker-compose | one service per component |
| CI | GitHub Actions | lint, test, build |
| Deploy target | AWS EC2 | via docker-compose |

## Conventions
- Python: PEP8, type hints, `ruff` for linting
- All config via `.env` (provide `.env.example`)
- Each component has its own `requirements.txt` or `package.json`
- Tests in `tests/` mirroring source structure, run via `pytest`
- Commit messages: `[phase][component] description`