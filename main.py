# main.py
import logging
import threading
import time
import schedule
from typing import Dict, Any, List
from config import Config
from db_handler import DatabaseManager
from model_manager import ModelManager
from mqtt_handler import MQTTHandler
from notifications import send_alert_email


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AnomalyDetectionSystem:
    def __init__(self):
        self.config = Config()
        self.db_manager = DatabaseManager(self.config.DB_URL)
        self.model_manager = ModelManager(self.config.MODEL_PATH, self.config)
        self.mqtt_handler = MQTTHandler(self.config, self.process_sensor_data)
        
        # Schedule periodic retraining
        schedule.every(self.config.RETRAIN_INTERVAL_HOURS).hours.do(self.retrain_model_if_needed)
        
        logger.info("Anomaly Detection System initialized")
    
    def process_sensor_data(self, payload: Dict[str, Any]):
        """Process incoming sensor data"""
        try:
            # Extract sensor values
            values = [payload.get(name) for name in self.config.SENSOR_NAMES]
            
            if None in values:
                logger.warning(f"Missing data in message: {payload}")
                return
            
            logger.info(f"🧠 Processing sensor data: {values}")
            
            # Make prediction
            is_anomaly, z_scores = self.model_manager.predict(values)
            
            # Store data in database
            self.db_manager.insert_sensor_data(payload, is_anomaly, z_scores)
            
            if is_anomaly:
                # Identify which sensors are unusual
                anomalous_sensors = [
                    self.config.SENSOR_NAMES[i] for i, z in enumerate(z_scores)
                    if z > self.config.Z_SCORE_THRESHOLD
                ]
                
                logger.warning(f"[🚨] ANOMALY DETECTED: {payload}")
                logger.warning(f"Anomalous sensors: {anomalous_sensors}")
                
                # Send alerts
                send_alert_email(anomalous_sensors, payload, z_scores)
            else:
                logger.info(f"[✓] Normal reading: {payload}")
                
        except Exception as e:
            logger.error(f"Error processing sensor data: {e}")
    
    def retrain_model_if_needed(self):
        """Check and retrain model if necessary"""
        try:
            if not self.model_manager.needs_retraining():
                logger.info("Model retraining not needed yet")
                return
            
            logger.info("Starting model retraining...")
            
            # Get training data
            df = self.db_manager.get_training_data()
            
            if len(df) < self.config.MIN_SAMPLES_FOR_TRAINING:
                logger.warning(f"Not enough data for retraining. Got {len(df)}, need {self.config.MIN_SAMPLES_FOR_TRAINING}")
                return
            
            # Train model
            performance = self.model_manager.train_model(df)
            
            # Log training
            self.db_manager.log_training(
                performance['samples_used'],
                str(performance),
                "Scheduled retraining"
            )
            
            logger.info(f"Model retrained successfully: {performance}")
            
        except Exception as e:
            logger.error(f"Error during model retraining: {e}")
    
    def run_scheduler(self):
        """Run the scheduler in a separate thread"""
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def start(self):
        """Start the anomaly detection system"""
        try:
            # Start scheduler in background thread
            scheduler_thread = threading.Thread(target=self.run_scheduler, daemon=True)
            scheduler_thread.start()
            
            # Perform initial training if needed
            if self.model_manager.model is None or self.model_manager.needs_retraining():
                try:
                    self.retrain_model_if_needed()
                except Exception as e:
                    logger.warning(f"Initial training failed, will continue with default model: {e}")
            
            logger.info("Starting MQTT connection...")
            
            # Start MQTT loop (this will block)
            self.mqtt_handler.connect_and_loop()
            
        except KeyboardInterrupt:
            logger.info("Shutting down gracefully...")
        except Exception as e:
            logger.error(f"System error: {e}")

if __name__ == "__main__":
    system = AnomalyDetectionSystem()
    system.start()