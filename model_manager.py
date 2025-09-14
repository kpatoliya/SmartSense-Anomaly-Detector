# model_manager.py
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
import os
from datetime import datetime, timedelta
import logging
from typing import Tuple, Optional, Dict, Any, List
import json
from config import Config

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, model_path: str, config: Config):
        self.model_path = model_path
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.last_training_time = None
        self.ensure_model_directory()
        self.load_model()
    
    def ensure_model_directory(self):
        """Ensure model directory exists"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
    
    def load_model(self) -> bool:
        """Load existing model or create new one"""
        try:
            if os.path.exists(self.model_path):
                model_data = joblib.load(self.model_path)
                self.model = model_data['model']
                self.scaler = model_data.get('scaler', StandardScaler())
                self.last_training_time = model_data.get('last_training_time')
                logger.info(f"Model loaded from {self.model_path}")
                return True
            else:
                logger.warning("No existing model found, will train on first data")
                self.model = IsolationForest(contamination=0.1, random_state=42)
                return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = IsolationForest(contamination=0.1, random_state=42)
            return False
    
    def save_model(self):
        """Save model, scaler, and metadata"""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'last_training_time': datetime.now(),
                'config': {
                    'sensor_names': self.config.SENSOR_NAMES,
                    'sensor_means': self.config.SENSOR_MEANS,
                    'sensor_stds': self.config.SENSOR_STDS
                }
            }
            joblib.save(model_data, self.model_path)
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
        """Train the isolation forest model"""
        if len(df) < self.config.MIN_SAMPLES_FOR_TRAINING:
            raise ValueError(f"Not enough samples for training. Got {len(df)}, need {self.config.MIN_SAMPLES_FOR_TRAINING}")
        
        # Prepare features
        features = df[self.config.SENSOR_NAMES].values
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Train model
        self.model.fit(features_scaled)
        
        # Evaluate on training data (for logging purposes)
        predictions = self.model.predict(features_scaled)
        anomaly_rate = (predictions == -1).mean()
        
        performance = {
            'samples_used': len(df),
            'anomaly_rate': float(anomaly_rate),
            'training_time': datetime.now().isoformat(),
            'feature_means': self.scaler.mean_.tolist(),
            'feature_stds': self.scaler.scale_.tolist()
        }
        
        # Save model
        self.save_model()
        
        logger.info(f"Model trained on {len(df)} samples with {anomaly_rate:.2%} anomaly rate")
        return performance
    
    def predict(self, values: List[float]) -> Tuple[bool, List[float]]:
        """Make prediction on new data"""
        if self.model is None:
            raise ValueError("Model not loaded or trained")
        
        # Prepare input
        X = np.array(values).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        prediction = self.model.predict(X_scaled)
        is_anomaly = prediction[0] == -1
        
        # Calculate z-scores for interpretability
        z_scores = np.abs((X[0] - np.array(self.config.SENSOR_MEANS)) / np.array(self.config.SENSOR_STDS))
        
        return is_anomaly, z_scores.tolist()
