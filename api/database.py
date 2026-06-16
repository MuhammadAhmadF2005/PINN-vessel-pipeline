import os
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
import datetime

DB_PATH = os.getenv("DB_PATH", "sqlite:///api/database.db")

engine = create_engine(DB_PATH, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class ScoreHistory(Base):
    __tablename__ = "score_history"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    run_id = Column(String, index=True)
    pressure = Column(Float)
    temperature = Column(Float)
    
    # Anomaly Scores
    pinn_score = Column(Float)
    iso_score = Column(Float)
    lstm_score = Column(Float)
    
    # Threshold decisions (boolean stored as int)
    is_anomaly = Column(Integer) # ensemble decision
    fault_type = Column(String, nullable=True)

class AlertLog(Base):
    __tablename__ = "alert_log"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    run_id = Column(String, index=True)
    alert_message = Column(String)
    severity = Column(String) # e.g. WARNING, CRITICAL
    fault_type = Column(String, nullable=True)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
