import { useEffect, useState } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { AlertCircle, CheckCircle } from 'lucide-react';
import './App.css';

// Using relative path for API since Nginx proxies /api/ to backend
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
}

interface AlertLog {
  id: number;
  timestamp: string;
  run_id: string;
  alert_message: string;
  severity: string;
}

function App() {
  const [history, setHistory] = useState<ScoreHistory[]>([]);
  const [alerts, setAlerts] = useState<AlertLog[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchData = async () => {
    try {
      const [histRes, alertsRes] = await Promise.all([
        axios.get(`${API_BASE}/history?limit=50`),
        axios.get(`${API_BASE}/alerts?limit=10`)
      ]);
      
      // Reverse history so oldest is first for the chart
      setHistory(histRes.data.reverse());
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

  const latestData = history.length > 0 ? history[history.length - 1] : null;

  return (
    <div className="dashboard-container">
      <header className="header">
        <h1>PIPM Live Dashboard</h1>
        <div className={`status-indicator ${latestData?.is_anomaly ? 'danger' : 'safe'}`}>
          {latestData?.is_anomaly ? <AlertCircle size={24} /> : <CheckCircle size={24} />}
          <span>{latestData?.is_anomaly ? 'ANOMALY DETECTED' : 'SYSTEM NORMAL'}</span>
        </div>
      </header>

      <main className="main-content">
        <section className="chart-section">
          <h2>Live Sensor Data & Anomaly Scores</h2>
          {loading ? (
            <p>Loading...</p>
          ) : (
            <div className="charts-wrapper">
              <div className="chart-card">
                <h3>Pressure & Temperature</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={history}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="timestamp" tickFormatter={(val) => new Date(val).toLocaleTimeString()} />
                    <YAxis yAxisId="left" />
                    <YAxis yAxisId="right" orientation="right" />
                    <Tooltip labelFormatter={(val) => new Date(val).toLocaleTimeString()} />
                    <Legend />
                    <Line yAxisId="left" type="monotone" dataKey="pressure" stroke="#8884d8" dot={false} />
                    <Line yAxisId="right" type="monotone" dataKey="temperature" stroke="#82ca9d" dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              <div className="chart-card">
                <h3>Model Anomaly Scores</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={history}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="timestamp" tickFormatter={(val) => new Date(val).toLocaleTimeString()} />
                    <YAxis />
                    <Tooltip labelFormatter={(val) => new Date(val).toLocaleTimeString()} />
                    <Legend />
                    <Line type="monotone" dataKey="pinn_score" stroke="#ff7300" dot={false} name="PINN" />
                    <Line type="monotone" dataKey="iso_score" stroke="#387908" dot={false} name="IsoForest" />
                    <Line type="monotone" dataKey="lstm_score" stroke="#ff0000" dot={false} name="LSTM-AE" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </section>

        <aside className="sidebar">
          <div className="widget latest-readings">
            <h2>Latest Readings</h2>
            {latestData ? (
              <ul>
                <li><strong>Pressure:</strong> {latestData.pressure.toFixed(2)}</li>
                <li><strong>Temp:</strong> {latestData.temperature.toFixed(2)}</li>
                <li><strong>PINN:</strong> {latestData.pinn_score.toFixed(2)}</li>
                <li><strong>ISO:</strong> {latestData.iso_score.toFixed(2)}</li>
                <li><strong>LSTM:</strong> {latestData.lstm_score.toFixed(2)}</li>
              </ul>
            ) : (
              <p>No data</p>
            )}
          </div>

          <div className="widget alert-log">
            <h2>Recent Alerts</h2>
            {alerts.length > 0 ? (
              <ul>
                {alerts.map(alert => (
                  <li key={alert.id} className="alert-item">
                    <span className="time">{new Date(alert.timestamp).toLocaleTimeString()}</span>
                    <span className="msg">{alert.alert_message}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="no-alerts">No recent alerts</p>
            )}
          </div>
        </aside>
      </main>
    </div>
  );
}

export default App;
