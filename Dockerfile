# Use a lightweight Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy dependency file first
COPY requirements.txt .

# Install system and Python dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg git gcc g++ \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the application files
COPY . .

COPY model/finetuned-qa-burmese /workspace/qa/
COPY model/finetuned-seamlessm4t-burmese /workspace/asr/

# Expose the Cloud Run port
ENV PORT=8080

# Start the Flask app using Gunicorn
CMD exec gunicorn -w 1 -b 0.0.0.0:$PORT main:app --timeout 20000 --threads 1 --preload
