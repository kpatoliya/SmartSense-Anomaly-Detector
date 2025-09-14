# mqtt_handler.py
import paho.mqtt.client as mqtt
import json
import logging
import socket
import time
from typing import Callable, Any
from config import Config

logger = logging.getLogger(__name__)

class MQTTHandler:
    def __init__(self, config: Config, message_callback: Callable):
        self.config = config
        self.message_callback = message_callback
        self.client = None
        self.setup_client()
    
    def setup_client(self):
        """Setup MQTT client with callbacks"""
        self.client = mqtt.Client()
        
        if self.config.MQTT_USER and self.config.MQTT_PASS:
            self.client.username_pw_set(self.config.MQTT_USER, self.config.MQTT_PASS)
        
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect
    
    def on_connect(self, client, userdata, flags, rc, properties=None):
        """Callback for MQTT connection"""
        if rc == 0:
            logger.info(f"Connected to MQTT broker with result code {rc}")
            client.subscribe(self.config.MQTT_TOPIC)
        else:
            logger.error(f"Failed to connect to MQTT broker with result code {rc}")
    
    def on_disconnect(self, client, userdata, rc):
        """Callback for MQTT disconnection"""
        logger.warning(f"Disconnected from MQTT broker with result code {rc}")
    
    def on_message(self, client, userdata, msg):
        """Handle incoming MQTT messages"""
        try:
            payload = json.loads(msg.payload.decode())
            self.message_callback(payload)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON message: {e}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    def wait_for_broker(self, retries: int = 10, delay: int = 5):
        """Wait for MQTT broker to become available"""
        for i in range(retries):
            try:
                with socket.create_connection((self.config.MQTT_BROKER, self.config.MQTT_PORT), timeout=3):
                    logger.info(f"MQTT broker available at {self.config.MQTT_BROKER}:{self.config.MQTT_PORT}")
                    return
            except OSError:
                logger.info(f"Waiting for MQTT broker... attempt {i+1}/{retries}")
                time.sleep(delay)
        
        raise TimeoutError(f"MQTT broker at {self.config.MQTT_BROKER}:{self.config.MQTT_PORT} not reachable after {retries} attempts")
    
    def connect_and_loop(self):
        """Connect to broker and start message loop"""
        self.wait_for_broker()
        self.client.connect(self.config.MQTT_BROKER, self.config.MQTT_PORT, 60)
        self.client.loop_forever()