import os
import json
import numpy as np
import paho.mqtt.client as mqtt
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from dotenv import load_dotenv

load_dotenv()

# MQTT Config
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "vessel/sensors")

# InfluxDB Config
INFLUXDB_URL = os.getenv("INFLUXDB_URL", "http://localhost:8086")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN", "my-super-secret-auth-token-123")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG", "vessel_org")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET", "vessel_data")

# Initialize InfluxDB Client
client_influx = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
write_api = client_influx.write_api(write_options=SYNCHRONOUS)

# Feature Window
WINDOW_SIZE = 10
history = []

def compute_features(hist):
    if len(hist) < WINDOW_SIZE:
        return None
    
    pressures = [d["pressure"] for d in hist[-WINDOW_SIZE:]]
    temperatures = [d["temperature"] for d in hist[-WINDOW_SIZE:]]
    
    features = {
        "pressure_mean": np.mean(pressures),
        "pressure_std": np.std(pressures),
        "temperature_mean": np.mean(temperatures),
        "temperature_std": np.std(temperatures)
    }
    return features

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"Connected to MQTT Broker at {MQTT_BROKER}:{MQTT_PORT}")
        client.subscribe(MQTT_TOPIC)
    else:
        print(f"Failed to connect, return code {rc}")

import requests

API_URL = os.getenv("API_URL", "http://api:8000")

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
        payload["run_id"] = str(payload.get("run_id", "unknown"))
        history.append(payload)
        
        # Keep only recent history to prevent memory leak
        if len(history) > WINDOW_SIZE * 2:
            history.pop(0)
            
        features = compute_features(history)
        
        # Write to InfluxDB
        point = Point("sensor_data") \
            .tag("run_id", payload.get("run_id", "unknown")) \
            .field("pressure", payload["pressure"]) \
            .field("temperature", payload["temperature"]) \
            .field("label", payload["label"])
            
        if features:
            for k, v in features.items():
                point.field(k, float(v))
                
        write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=point)
        
        print(f"Written to InfluxDB: P={payload['pressure']:.2f}, T={payload['temperature']:.2f}")

        # Send to API for scoring
        if len(history) >= 20: # Match LSTM requirement
            # Send the last 20 records
            score_payload = {"window": history[-20:]}
            try:
                resp = requests.post(f"{API_URL}/score", json=score_payload)
                if resp.status_code == 200:
                    result = resp.json()
                    print(f"Scored! PINN:{result['pinn_score']:.2f} ISO:{result['iso_score']:.2f} LSTM:{result['lstm_score']:.2f} Anomaly:{result['is_anomaly']}")
                else:
                    print(f"API Error: {resp.status_code} - {resp.text}")
            except Exception as req_e:
                print(f"Failed to connect to API: {req_e}")

    except Exception as e:
        print(f"Error processing message: {e}")

def run_worker():
    print(f"Starting Feature Worker. Connecting to InfluxDB at {INFLUXDB_URL}...")
    
    client = mqtt.Client(client_id="feature_worker")
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_forever()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_worker()
