import paho.mqtt.client as mqtt
from sklearn.ensemble import IsolationForest
import numpy as np
import json
import os
from dotenv import load_dotenv
import joblib
from notifications import send_alert_email
import socket
import time

# Load environment variables
load_dotenv()
MQTT_BROKER = os.getenv("MQTT_BROKER")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "#")
MQTT_USER = os.getenv("MQTT_USER")    
MQTT_PASS = os.getenv("MQTT_PASS")  
print(f"Connecting to {MQTT_BROKER}:{MQTT_PORT} with user={MQTT_USER}")
print(f"Connecting to {MQTT_BROKER}:{MQTT_PORT} | Topic: {MQTT_TOPIC}")


# MQTT callbacks
def on_connect(client, userdata, flags, rc, properties=None):
    print(f"[+] Connected to MQTT broker with result code {rc}")
    client.subscribe(MQTT_TOPIC)

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        sensor_names = ["temperature", "humidity", "pressure"]
        print(f"🧠 Parsed JSON: {json.dumps(payload, indent=2)}")
        values = [payload.get(sensor_names[0]), payload.get(sensor_names[1]), payload.get(sensor_names[2])]

        if None in values:
            print(f"[!] Missing data in message: {payload}")
            return

        # Load the pretrained model
        model = joblib.load("iforest_model.pkl")

        X = np.array(values).reshape(1, -1)
        prediction = model.predict(X)
        
        # These should match your training data's distribution (used for z-score)
        sensor_means = np.array([25, 55, 1010])    # temperature, humidity, pressure
        sensor_stds = np.array([8, 10, 10])

        if prediction[0] == -1:
            # Identify which sensor(s) are unusual using z-score
            z_scores = np.abs((X[0] - sensor_means) / sensor_stds)
            threshold = 2  # Adjust this threshold as needed
            for i, z in enumerate(z_scores):
                if z > threshold:
                    print(f"[🚨] {sensor_names[i]} is off! z-score = {z:.2f}")
                    send_alert_email(sensor_names[i], payload)
            print(f"Sensor Data: {payload}")
        else:
            print(f"[✓] Normal: {payload}")

    except Exception as e:
        print(f"[!] Failed to process message: {e}")



def wait_for_broker(broker, port, retries=10, delay=5):
    for i in range(retries):
        try:
            with socket.create_connection((broker, port), timeout=3):
                print(f"✅ MQTT broker available at {broker}:{port}")
                return
        except OSError:
            print(f"🔁 Waiting for MQTT broker... attempt {i+1}/{retries}")
            time.sleep(delay)
    raise TimeoutError(f"❌ MQTT broker at {broker}:{port} not reachable after {retries} attempts")
# Before connecting:
wait_for_broker(MQTT_BROKER, MQTT_PORT)

# Set up client
client = mqtt.Client()
if MQTT_USER and MQTT_PASS:
    client.username_pw_set(MQTT_USER, MQTT_PASS)

client.on_connect = on_connect
client.on_message = on_message
client.connect(MQTT_BROKER, MQTT_PORT, 60)

client.loop_forever()
