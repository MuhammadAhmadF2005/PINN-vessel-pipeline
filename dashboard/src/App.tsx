import { useEffect, useState, useRef } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { AlertTriangle, ShieldCheck, Thermometer, Gauge, Brain, Activity, Clock } from 'lucide-react';
import gsap from 'gsap';
import { Vessel3D } from './Vessel3D';
import './App.css';

const API_BASE = import.meta.env.DEV ? 'http://localhost:8000' : '/api';

interface ScoreHistory {
  id: number;
  timestamp: string;
  run_id: string;
  pressure: number;
  temperature: number;
  pinn_score: number;
  iso_score: number;
  lstm_score: number;
  is_anomaly: boolean;
  fault_type?: string;
}

interface AlertLog {
  id: number;
  timestamp: string;
  run_id: string;
  alert_message: string;
  severity: string;
  fault_type?: string;
}

// Custom component for high-performance GSAP metric count-up
const AnimatedNumber = ({ value, decimals = 2 }: { value: number | null | undefined; decimals?: number }) => {
  const elementRef = useRef<HTMLSpanElement>(null);
  
  // Guard against null/undefined/NaN
  const getSafeVal = (val: any) => {
    return typeof val === 'number' && !isNaN(val) ? val : 0;
  };

  const safeVal = getSafeVal(value);
  const prevValueRef = useRef(safeVal);

  useEffect(() => {
    const safeNext = getSafeVal(value);
    const obj = { val: prevValueRef.current };
    gsap.to(obj, {
      val: safeNext,
      duration: 0.4,
      ease: "power2.out",
      onUpdate: () => {
        if (elementRef.current) {
          elementRef.current.textContent = obj.val.toFixed(decimals);
        }
      }
    });
    prevValueRef.current = safeNext;
  }, [value, decimals]);

  return <span ref={elementRef} className="counter">{safeVal.toFixed(decimals)}</span>;
};

function App() {
  const [history, setHistory] = useState<ScoreHistory[]>([]);
  const [alerts, setAlerts] = useState<AlertLog[]>([]);
  const [loading, setLoading] = useState(true);
  const prevAlertIds = useRef<number[]>([]);

  const fetchData = async () => {
    try {
      const [histRes, alertsRes] = await Promise.all([
        axios.get(`${API_BASE}/history?limit=50`),
        axios.get(`${API_BASE}/alerts?limit=15`)
      ]);
      
      // Reverse history so oldest is first for charts
      const histData = histRes.data.reverse();
      setHistory(histData);
      setAlerts(alertsRes.data);
      setLoading(false);
    } catch (error) {
      console.error("Error fetching data:", error);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 2000); // Poll every 2 seconds
    return () => clearInterval(interval);
  }, []);

  // GSAP slide-in effect for newly arriving alerts
  useEffect(() => {
    if (alerts.length > 0) {
      const newAlerts = alerts.filter(a => !prevAlertIds.current.includes(a.id));
      if (newAlerts.length > 0) {
        // Run slide-in animation on new alerts next frame
        setTimeout(() => {
          newAlerts.forEach(a => {
            gsap.fromTo(`#alert-card-${a.id}`,
              { x: 50, opacity: 0, scale: 0.95 },
              { x: 0, opacity: 1, scale: 1, duration: 0.45, ease: "back.out(1.2)" }
            );
          });
        }, 50);
        prevAlertIds.current = alerts.map(a => a.id);
      }
    }
  }, [alerts]);

  const latestData = history.length > 0 ? history[history.length - 1] : null;
  const activeAlertMessages = alerts.slice(0, 3).map(a => a.alert_message);

  return (
    <div className="dashboard-container">
      <header className="header">
        <div className="header-title-area">
          <h1>Physics-Informed Real-time Twin</h1>
          {latestData && (
            <div className="run-id-badge">
              Active Run: RUN-{latestData.run_id}
            </div>
          )}
        </div>
        <div className={`status-indicator ${latestData?.is_anomaly ? 'danger' : 'safe'}`}>
          {latestData?.is_anomaly ? (
            <>
              <AlertTriangle size={18} />
              <span>Anomaly Detected</span>
            </>
          ) : (
            <>
              <ShieldCheck size={18} />
              <span>System Normal</span>
            </>
          )}
        </div>
      </header>

      {loading ? (
        <div className="loading-overlay">
          <div className="spinner"></div>
          <span>Synchronizing Digital Twin telemetry...</span>
        </div>
      ) : (
        <main className="main-content">
          {/* Left Column: Live stats & Pressure/Temp History Chart */}
          <section className="panel left-panel">
            <h2 className="panel-title">
              <Activity size={18} className="text-cyan" />
              Telemetry & Scores
            </h2>
            
            <div className="metrics-grid">
              {latestData && (
                <>
                  <div className="metric-card pressure-card">
                    <span className="metric-label">Vessel Pressure</span>
                    <div className="metric-val-container">
                      <div className="metric-value">
                        <AnimatedNumber value={latestData.pressure} />
                        <span className="metric-unit"> bar</span>
                      </div>
                      <Gauge size={16} color="var(--accent-cyan)" />
                    </div>
                  </div>

                  <div className="metric-card temp-card">
                    <span className="metric-label">Vessel Temperature</span>
                    <div className="metric-val-container">
                      <div className="metric-value">
                        <AnimatedNumber value={latestData.temperature} />
                        <span className="metric-unit"> K</span>
                      </div>
                      <Thermometer size={16} color="var(--accent-purple)" />
                    </div>
                  </div>

                  <div className="metric-card pinn-card">
                    <span className="metric-label">PINN Anomaly Score</span>
                    <div className="metric-val-container">
                      <div className="metric-value">
                        <AnimatedNumber value={latestData.pinn_score} />
                      </div>
                      <Brain size={16} color="var(--accent-warning)" />
                    </div>
                  </div>

                  <div className="metric-card iso-card">
                    <span className="metric-label">Isolation Forest</span>
                    <div className="metric-val-container">
                      <div className="metric-value">
                        <AnimatedNumber value={latestData.iso_score} />
                      </div>
                      <Activity size={16} color="var(--accent-safe)" />
                    </div>
                  </div>

                  <div className="metric-card lstm-card">
                    <span className="metric-label">LSTM Autoencoder</span>
                    <div className="metric-val-container">
                      <div className="metric-value">
                        <AnimatedNumber value={latestData.lstm_score} />
                      </div>
                      <Brain size={16} color="var(--accent-danger)" />
                    </div>
                  </div>
                </>
              )}
            </div>

            <div className="chart-container-card">
              <h3>Pressure & Temperature Trend</h3>
              <ResponsiveContainer width="100%" height={180}>
                <LineChart data={history} margin={{ top: 5, right: 5, left: -20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="timestamp" tickFormatter={(val) => new Date(val).toLocaleTimeString()} hide={true} />
                  <YAxis yAxisId="left" domain={['auto', 'auto']} />
                  <YAxis yAxisId="right" orientation="right" domain={['auto', 'auto']} />
                  <Tooltip labelFormatter={(val) => new Date(val).toLocaleTimeString()} />
                  <Legend iconType="circle" />
                  <Line 
                    yAxisId="left" 
                    type="monotone" 
                    dataKey="pressure" 
                    stroke="var(--accent-cyan)" 
                    dot={false} 
                    strokeWidth={2}
                    isAnimationActive={true}
                    animationDuration={350}
                    name="P (bar)"
                  />
                  <Line 
                    yAxisId="right" 
                    type="monotone" 
                    dataKey="temperature" 
                    stroke="var(--accent-purple)" 
                    dot={false} 
                    strokeWidth={2}
                    isAnimationActive={true}
                    animationDuration={350}
                    name="T (K)"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </section>

          {/* Center Column: Centerpiece 3D Twin */}
          <section className="panel center-panel">
            <div className="center-status-badge">
              <span className={`center-status-dot ${latestData?.is_anomaly ? 'danger' : 'safe'}`}></span>
              <span className="center-status-text">
                {latestData?.is_anomaly 
                  ? `ANOMALY: ${latestData.fault_type?.replace('_', ' ').toUpperCase() || 'UNKNOWN'}`
                  : 'INTEGRITY SECURED'
                }
              </span>
            </div>
            
            {latestData && (
              <Vessel3D 
                temperature={latestData.temperature} 
                pressure={latestData.pressure} 
                isAnomaly={latestData.is_anomaly}
                activeAlerts={activeAlertMessages}
              />
            )}
          </section>

          {/* Right Column: Alerts Feed & Model Anomaly Score Chart */}
          <section className="panel right-panel">
            <div className="alert-log-container">
              <div className="alert-log-header">
                <h2 className="panel-title" style={{ borderBottom: 'none', paddingBottom: 0, margin: 0 }}>
                  <Clock size={18} className="text-cyan" />
                  Active Alerts
                </h2>
                {alerts.filter(a => a.severity === 'CRITICAL').length > 0 && (
                  <span className="alert-count-badge">
                    {alerts.filter(a => a.severity === 'CRITICAL').length} Active
                  </span>
                )}
              </div>

              <div className="alert-list-scroll">
                {alerts.length > 0 ? (
                  alerts.map(alert => (
                    <div 
                      key={alert.id} 
                      id={`alert-card-${alert.id}`} 
                      className={`alert-card ${alert.severity.toLowerCase()}`}
                    >
                      <div className="alert-card-header">
                        <span className="alert-card-time">
                          {new Date(alert.timestamp).toLocaleTimeString()}
                        </span>
                        <span className="alert-card-severity">
                          {alert.severity}
                        </span>
                      </div>
                      <span className="alert-card-message">{alert.alert_message}</span>
                    </div>
                  ))
                ) : (
                  <div className="no-alerts-card">
                    <ShieldCheck size={28} className="text-emerald" style={{ opacity: 0.6 }} />
                    <span>No active anomalies detected in current run.</span>
                  </div>
                )}
              </div>
            </div>

            <div className="chart-container-card">
              <h3>Anomaly Scores over Time</h3>
              <ResponsiveContainer width="100%" height={180}>
                <LineChart data={history} margin={{ top: 5, right: 5, left: -20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="timestamp" tickFormatter={(val) => new Date(val).toLocaleTimeString()} hide={true} />
                  <YAxis domain={[0, 'auto']} />
                  <Tooltip labelFormatter={(val) => new Date(val).toLocaleTimeString()} />
                  <Legend iconType="circle" />
                  <Line 
                    type="monotone" 
                    dataKey="pinn_score" 
                    stroke="var(--accent-warning)" 
                    dot={false} 
                    strokeWidth={1.5}
                    isAnimationActive={true}
                    animationDuration={350}
                    name="PINN" 
                  />
                  <Line 
                    type="monotone" 
                    dataKey="iso_score" 
                    stroke="var(--accent-safe)" 
                    dot={false} 
                    strokeWidth={1.5}
                    isAnimationActive={true}
                    animationDuration={350}
                    name="IsoForest" 
                  />
                  <Line 
                    type="monotone" 
                    dataKey="lstm_score" 
                    stroke="var(--accent-danger)" 
                    dot={false} 
                    strokeWidth={1.5}
                    isAnimationActive={true}
                    animationDuration={350}
                    name="LSTM-AE" 
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </section>
        </main>
      )}
    </div>
  );
}

export default App;
