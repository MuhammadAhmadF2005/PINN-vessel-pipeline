import os
import time
import json
import argparse
import pandas as pd
import paho.mqtt.client as mqtt
from dotenv import load_dotenv

load_dotenv()

MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "vessel/sensors")

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"Connected to MQTT Broker at {MQTT_BROKER}:{MQTT_PORT}")
    else:
        print(f"Failed to connect, return code {rc}")

def run_publisher(csv_path: str, speed: float = 1.0):
    """
    Replay a CSV file over MQTT.
    speed: sleep multiplier. If speed=1.0, sleep for dt seconds between rows.
           If speed=0.1, runs 10x faster.
    """
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Sort by timestamp just in case
    df = df.sort_values("timestamp").reset_index(drop=True)

    client = mqtt.Client(client_id="vessel_publisher")
    client.on_connect = on_connect

    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
    except Exception as e:
        print(f"Error connecting to MQTT Broker: {e}")
        return

    client.loop_start()

    print(f"Starting to publish to {MQTT_TOPIC} at speed factor {speed}x...")
    
    prev_time = None
    for idx, row in df.iterrows():
        # Build payload
        payload = {
            "timestamp": row["timestamp"],
            "pressure": row["pressure"],
            "temperature": row["temperature"],
            "label": row["label"],
            "fault_type": row["fault_type"],
            "run_id": row["run_id"]
        }
        
        # Calculate delay based on timestamps
        if prev_time is not None:
            dt = row["timestamp"] - prev_time
            if dt > 0:
                time.sleep(dt * speed)
                
        client.publish(MQTT_TOPIC, json.dumps(payload))
        
        if idx % 100 == 0:
            print(f"Published {idx}/{len(df)} rows...")
            
        prev_time = row["timestamp"]

    client.loop_stop()
    client.disconnect()
    print("Finished publishing.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MQTT Data Publisher")
    parser.add_argument("--csv", type=str, default="data/test.csv", help="Path to CSV file")
    parser.add_argument("--speed", type=float, default=0.1, help="Simulation speed multiplier (lower is faster)")
    args = parser.parse_args()
    
    run_publisher(args.csv, args.speed)
