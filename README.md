# Anomaly Detection System for Home Automation

A system designed to detect unusual patterns and anomalies in home automation data streams.

## Overview

This project implements anomaly detection algorithms to identify irregular patterns in home automation sensor data, helping to detect potential issues or security concerns. The system integrates with smart home devices using the MQTT protocol, allowing seamless monitoring of various IoT devices such as sensors, switches, and other smart appliances in your home network.

## Features

- Real-time monitoring of sensor data
- Pattern analysis and anomaly detection
- Configurable alert thresholds
- Data logging and reporting

## Installation

```bash
git clone https://github.com/kpatoliya/SmartSense-Anomaly-Detector
cd SmartSense-Anomaly-Detector

```

## Usage

```bash
# Build the Docker image
docker build -t anomaly-detector .

# Run the container
docker run -d \
    --name anomaly-detector \
    --env-file .env \
    --network host \
    anomaly-detector
```

## Configuration

Create a `.env` file in the root directory with the following variables:

```env
# MQTT Configuration
MQTT_BROKER = "mosquitto"
MQTT_PORT = 1883
MQTT_USER = "*****"   # or "your_username"
MQTT_PASS = "*****"   # or "your_password"
MQTT_TOPIC = "*"    # This subrcribes to ALL topics

# Notification Settings
SMTP_Port = 587
SMTP_Server = "smtp.gmail.com"
SENDER_EMAIL = "******@gmail.com"
RECEIVER_EMAIL = "******@gmail.com"
SENDER_PASSWORD = "********"
```

Details about setting up and configuring the anomaly detection system.

