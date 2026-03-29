FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train_model.py .
COPY drift_detection.py .

# Log directory is mounted as a volume at runtime
RUN mkdir -p /app/logs

CMD ["python", "train_model.py"]
