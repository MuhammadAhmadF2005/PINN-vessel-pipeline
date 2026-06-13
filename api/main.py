import os
import time
import numpy as np
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy.orm import Session

# Setup DeepXDE backend before importing PINN
os.environ.setdefault("DDE_BACKEND", "pytorch")

from api.database import init_db, get_db, ScoreHistory, AlertLog

# Import model loading and scoring functions
from models.pinn.pinn_model import load_model as load_pinn, compute_anomaly_scores as pinn_scores
from models.baselines.iso_forest import IsoForestModel
from models.baselines.lstm_ae import load_model as load_lstm, compute_anomaly_scores_lstm as lstm_scores

# Global models dictionary
models = {}

ARTIFACTS_DIR = Path("models/artifacts")

# Thresholds (from Phase 1 evaluation or arbitrary safe defaults if evaluation didn't output constants)
# In a real system, these would be loaded from a config or MLflow registry
THRESHOLDS = {
    "PINN": 2.0,
    "IsolationForest": -0.04,
    "LSTM-AE": 1.0
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Initializing Database...")
    init_db()
    
    print("Loading models from artifacts...")
    try:
        models["pinn"] = load_pinn(ARTIFACTS_DIR / "pinn_model.pt")
        models["iso_forest"] = IsoForestModel.load(ARTIFACTS_DIR / "iso_forest_model.pkl")
        models["lstm_ae"] = load_lstm(ARTIFACTS_DIR / "lstm_ae_model.pt")
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        # In a real app we might want to crash here, but for development we can let it run
        
    yield
    # Shutdown
    print("Shutting down API...")
    models.clear()

app = FastAPI(title="PIPM Inference API", lifespan=lifespan)

# Pydantic Schemas
class SensorData(BaseModel):
    timestamp: float
    run_id: str
    pressure: float
    temperature: float

class ScoreRequest(BaseModel):
    window: List[SensorData]  # The latest window of data, size >= 20 for LSTM

class ScoreResponse(BaseModel):
    timestamp: float
    run_id: str
    pinn_score: float
    iso_score: float
    lstm_score: float
    is_anomaly: bool
    
class HistoryResponse(BaseModel):
    id: int
    timestamp: str
    run_id: str
    pressure: float
    temperature: float
    pinn_score: float
    iso_score: float
    lstm_score: float
    is_anomaly: bool

class AlertResponse(BaseModel):
    id: int
    timestamp: str
    run_id: str
    alert_message: str
    severity: str

def process_and_save_score(data: ScoreRequest, db: Session) -> ScoreResponse:
    if "pinn" not in models or "iso_forest" not in models or "lstm_ae" not in models:
        raise HTTPException(status_code=503, detail="Models are not loaded.")

    window = data.window
    if len(window) < 20: # LSTM requires at least 20 points typically, but we use the provided window
        raise HTTPException(status_code=400, detail="Window size must be at least 20 for LSTM-AE")
        
    # Extract arrays
    ts = np.array([d.timestamp for d in window])
    P = np.array([d.pressure for d in window])
    T = np.array([d.temperature for d in window])
    
    # We only care about scoring the latest point
    # However, LSTM and IsoForest might return arrays, and PINN uses derivatives
    
    # PINN score
    pinn_arr = pinn_scores(models["pinn"], ts, P, T)
    latest_pinn = float(pinn_arr[-1])
    
    # IsoForest score
    iso_arr = models["iso_forest"].anomaly_score(P, T)
    latest_iso = float(iso_arr[-1]) if len(iso_arr) > 0 else 0.0
    
    # LSTM-AE score
    lstm_arr = lstm_scores(models["lstm_ae"], P, T)
    latest_lstm = float(lstm_arr[-1]) if len(lstm_arr) > 0 else 0.0
    
    # Ensemble decision (majority voting or simple ANY threshold)
    # Here we use ANY threshold for conservative alerting
    is_anomaly = bool(
        (latest_pinn > THRESHOLDS["PINN"]) or 
        (latest_iso > THRESHOLDS["IsolationForest"]) or 
        (latest_lstm > THRESHOLDS["LSTM-AE"])
    )
    
    latest_data = window[-1]
    
    # Save to database
    db_score = ScoreHistory(
        run_id=latest_data.run_id,
        pressure=latest_data.pressure,
        temperature=latest_data.temperature,
        pinn_score=latest_pinn,
        iso_score=latest_iso,
        lstm_score=latest_lstm,
        is_anomaly=int(is_anomaly)
    )
    db.add(db_score)
    
    # If anomaly, add to alert log
    if is_anomaly:
        alert = AlertLog(
            run_id=latest_data.run_id,
            alert_message=f"Anomaly detected! PINN: {latest_pinn:.2f}, ISO: {latest_iso:.2f}, LSTM: {latest_lstm:.2f}",
            severity="CRITICAL"
        )
        db.add(alert)
        
    db.commit()
    
    return ScoreResponse(
        timestamp=latest_data.timestamp,
        run_id=latest_data.run_id,
        pinn_score=latest_pinn,
        iso_score=latest_iso,
        lstm_score=latest_lstm,
        is_anomaly=is_anomaly
    )

@app.post("/score", response_model=ScoreResponse)
async def score_endpoint(request: ScoreRequest, db: Session = Depends(get_db)):
    """
    Accepts a window of recent sensor data, computes anomaly scores, 
    stores them in SQLite, and returns the result.
    """
    return process_and_save_score(request, db)

@app.get("/history", response_model=List[HistoryResponse])
def get_history(limit: int = 100, db: Session = Depends(get_db)):
    """Retrieve the recent anomaly score history."""
    records = db.query(ScoreHistory).order_by(ScoreHistory.id.desc()).limit(limit).all()
    # Convert to Pydantic responses
    return [
        HistoryResponse(
            id=r.id,
            timestamp=r.timestamp.isoformat(),
            run_id=r.run_id,
            pressure=r.pressure,
            temperature=r.temperature,
            pinn_score=r.pinn_score,
            iso_score=r.iso_score,
            lstm_score=r.lstm_score,
            is_anomaly=bool(r.is_anomaly)
        ) for r in records
    ]

@app.get("/alerts", response_model=List[AlertResponse])
def get_alerts(limit: int = 50, db: Session = Depends(get_db)):
    """Retrieve the recent alert history."""
    alerts = db.query(AlertLog).order_by(AlertLog.id.desc()).limit(limit).all()
    return [
        AlertResponse(
            id=a.id,
            timestamp=a.timestamp.isoformat(),
            run_id=a.run_id,
            alert_message=a.alert_message,
            severity=a.severity
        ) for a in alerts
    ]
