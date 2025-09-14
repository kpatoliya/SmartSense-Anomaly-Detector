import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        with self.get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sensor_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    temperature REAL,
                    humidity REAL,
                    pressure REAL,
                    is_anomaly BOOLEAN DEFAULT FALSE,
                    z_scores TEXT,
                    raw_data TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_training_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    samples_used INTEGER,
                    model_performance TEXT,
                    notes TEXT
                )
            """)
            conn.commit()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        if self.db_url.startswith('sqlite:'):
            db_path = self.db_url.replace('sqlite:///', '')
            conn = sqlite3.connect(db_path)
        else:
            # For other databases, you'd implement connection logic here
            raise NotImplementedError("Only SQLite supported in this example")
        
        try:
            yield conn
        finally:
            conn.close()
    
    def insert_sensor_data(self, data: Dict[str, Any], is_anomaly: bool = False, 
                          z_scores: Optional[List[float]] = None):
        """Insert sensor data into database"""
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO sensor_data 
                (temperature, humidity, pressure, is_anomaly, z_scores, raw_data)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                data.get('temperature'),
                data.get('humidity'), 
                data.get('pressure'),
                is_anomaly,
                ','.join(map(str, z_scores)) if z_scores else None,
                str(data)
            ))
            conn.commit()
    
    def get_training_data(self, hours_back: int = 168, limit: int = 10000) -> pd.DataFrame:
        """Get data for model training (last week by default)"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        with self.get_connection() as conn:
            df = pd.read_sql_query("""
                SELECT temperature, humidity, pressure, timestamp
                FROM sensor_data 
                WHERE timestamp >= ? AND temperature IS NOT NULL 
                AND humidity IS NOT NULL AND pressure IS NOT NULL
                ORDER BY timestamp DESC
                LIMIT ?
            """, conn, params=(cutoff_time, limit))
        
        return df
    
    def log_training(self, samples_used: int, performance: str, notes: str = ""):
        """Log model training event"""
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO model_training_log (samples_used, model_performance, notes)
                VALUES (?, ?, ?)
            """, (samples_used, performance, notes))
            conn.commit()
