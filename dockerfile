# Use a slim Python image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Copy files into container
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run your Python script
CMD ["python", "main.py"]
