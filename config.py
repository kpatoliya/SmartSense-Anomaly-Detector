# config.py
import os
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import List, Dict, Any

load_dotenv()

@dataclass
class Config:
    # MQTT Configuration
    MQTT_BROKER: str = os.getenv("MQTT_BROKER")
    MQTT_PORT: int = int(os.getenv("MQTT_PORT", 1883))
    MQTT_TOPIC: str = os.getenv("MQTT_TOPIC", "#")
    MQTT_USER: str = os.getenv("MQTT_USER")
    MQTT_PASS: str = os.getenv("MQTT_PASS")
    
    # Database Configuration
    DB_URL: str = os.getenv("DB_URL", "sqlite3:///sensor_data.db")
    
    # Model Configuration
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/iforest_model.pkl")
    RETRAIN_INTERVAL_HOURS: int = int(os.getenv("RETRAIN_INTERVAL_HOURS", 24))
    MIN_SAMPLES_FOR_TRAINING: int = int(os.getenv("MIN_SAMPLES_FOR_TRAINING", 100))
    
    # Sensor Configuration
    SENSOR_NAMES: List[str] = field(default_factory=lambda: ["temperature", "humidity", "pressure"])
    SENSOR_MEANS: List[float] = field(default_factory=lambda: [25.0, 58.0, 1010.0])
    SENSOR_STDS: List[float] = field(default_factory=lambda: [8.0, 10.0, 10.0])
    Z_SCORE_THRESHOLD: float = float(os.getenv("Z_SCORE_THRESHOLD", 2.0))
    
    # Alert Configuration
    ALERT_ENABLED: bool = os.getenv("ALERT_ENABLED", "true").lower() == "true"