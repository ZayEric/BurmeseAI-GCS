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

# ===== Download QA model from GCS =====
# Use the default service account credentials at build time (Cloud Build will inject these)
RUN python - <<'EOF'
from google.cloud import storage
import os

bucket_name = "qa-model-bucket"
prefix = "model/finetuned-qa-burmese"
local_dir = "/workspace/qa"

print(f"📦 Downloading {prefix} from {bucket_name} to {local_dir}...")

client = storage.Client()
bucket = client.bucket(bucket_name)
blobs = list(bucket.list_blobs(prefix=prefix))

os.makedirs(local_dir, exist_ok=True)
for blob in blobs:
    if blob.size == 0:
        continue
    rel_path = os.path.relpath(blob.name, prefix)
    os.makedirs(os.path.dirname(os.path.join(local_dir, rel_path)), exist_ok=True)
    blob.download_to_filename(os.path.join(local_dir, rel_path))
    print("✅ Downloaded:", blob.name)

print("🎯 Finished downloading QA model.")
EOF

# Expose the Cloud Run port
ENV PORT=8080

# Start the Flask app using Gunicorn
CMD exec gunicorn -w 1 -b 0.0.0.0:$PORT main:app --timeout 20000 --threads 1 --preload
