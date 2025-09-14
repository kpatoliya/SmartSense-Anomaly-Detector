# model_manager.py
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os
from datetime import datetime
import logging
from typing import List, Tuple, Dict, Any
from config import Config   

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, model_path: str, config: Config):
        self.model_path = model_path
        self.config = config
        self.model: Pipeline = None
        self.last_training_time = None
        self.buffer = []  # buffer incoming sensor data until enough to train
        self.ensure_model_directory()
        self.load_model()
    
    def ensure_model_directory(self):
        """Ensure model directory exists"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
    
    def build_pipeline(self) -> Pipeline:
        """Create a fresh pipeline (scaler + isolation forest)"""
        return Pipeline([
            ("scaler", StandardScaler()),
            ("iforest", IsolationForest(contamination=0.1, random_state=42))
        ])
    
    def load_model(self) -> bool:
        """Load existing model or create new one"""
        try:
            if os.path.exists(self.model_path):
                model_data = joblib.load(self.model_path)
                self.model = model_data['model']
                self.last_training_time = model_data.get('last_training_time')
                logger.info(f"Model loaded from {self.model_path}")
                return True
            else:
                logger.warning("No existing model found, will train on first data")
                self.model = self.build_pipeline()
                return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = self.build_pipeline()
            return False
    
    def save_model(self):
        """Save model and metadata"""
        try:
            model_data = {
                'model': self.model,
                'last_training_time': datetime.now(),
                'config': {
                    'sensor_names': self.config.SENSOR_NAMES,
                    'sensor_means': self.config.SENSOR_MEANS,
                    'sensor_stds': self.config.SENSOR_STDS
                }
            }
            joblib.dump(model_data, self.model_path)
            self.last_training_time = datetime.now()
            logger.info(f"Model saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def needs_retraining(self) -> bool:
        """Check if model needs retraining based on time interval"""
        if self.last_training_time is None:
            return True
        time_since_training = datetime.now() - self.last_training_time
        return time_since_training.total_seconds() > (self.config.RETRAIN_INTERVAL_HOURS * 3600)
    
    def train_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train the pipeline (scaler + isolation forest)"""
        if len(df) < self.config.MIN_SAMPLES_FOR_TRAINING:
            raise ValueError(
                f"Not enough samples for training. Got {len(df)}, need {self.config.MIN_SAMPLES_FOR_TRAINING}"
            )
        
        features = df[self.config.SENSOR_NAMES].values
        self.model = self.build_pipeline()
        self.model.fit(features)
        
        # Evaluate on training data directly using pipeline
        predictions = self.model.predict(features)
        anomaly_rate = (predictions == -1).mean()
        
        performance = {
            'samples_used': len(df),
            'anomaly_rate': float(anomaly_rate),
            'training_time': datetime.now().isoformat(),
        }
        
        self.save_model()
        logger.info(f"Model trained on {len(df)} samples with {anomaly_rate:.2%} anomaly rate")
        return performance
    
    def predict(self, values: List[float]) -> Tuple[bool, List[float]]:
        """Make prediction on new sensor data"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # If pipeline not fitted, buffer values until training
        if not hasattr(self.model.named_steps["iforest"], "estimators_"):
            self.buffer.append(values)
            if len(self.buffer) >= self.config.MIN_SAMPLES_FOR_TRAINING:
                # Train pipeline once enough data is buffered
                df = pd.DataFrame(self.buffer, columns=self.config.SENSOR_NAMES)
                self.train_model(df)
                self.buffer = []
                logger.info("Pipeline trained from buffered messages.")
            else:
                logger.info(f"Collecting data for training: {len(self.buffer)}/{self.config.MIN_SAMPLES_FOR_TRAINING}")
                # Return placeholder until model is trained
                return False, [0]*len(self.config.SENSOR_NAMES)
        
        # Normal prediction flow
        X = np.array(values).reshape(1, -1)
        prediction = self.model.predict(X)
        is_anomaly = prediction[0] == -1
        
        # Calculate z-scores for interpretability
        z_scores = np.abs((X[0] - np.array(self.config.SENSOR_MEANS)) / np.array(self.config.SENSOR_STDS))
        
        return is_anomaly, z_scores.tolist()
