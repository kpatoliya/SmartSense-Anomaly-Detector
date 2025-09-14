import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file if needed
# Your email credentials
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = os.getenv("SMTP_PORT", 587)
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD") # Use an app password or token

def send_alert_email(sensor, payload, z_scores):
    """
    Sends an alert email when an anomaly is detected.
    """
    print("📧 Sending alert email...")

    # Recipient info
    RECEIVER_EMAIL = os.getenv("RECEIVER_EMAIL")
    cc_emails = []
    subject = f"🚨 Anomaly Detected! {sensor}"
    body = f"Anomaly detected in {sensor} sensor\n\nZ-Scores: {json.dumps(z_scores, indent=2)}\n\nData Logs: {json.dumps(payload, indent=2)}"

    # Email message setup
    message = MIMEMultipart()
    message["From"] = SENDER_EMAIL
    message["To"] = RECEIVER_EMAIL
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    all_recipients = [RECEIVER_EMAIL] + cc_emails
    # Send the email
    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()  # Secure the connection
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, all_recipients, message.as_string())
        server.quit()
        print("✅ Alert email sent successfully!")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")


# send_alert_email("temperature", {"temperature": 30, "humidity": 60, "pressure": 1015})  # Example usage