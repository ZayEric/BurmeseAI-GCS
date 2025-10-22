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

# ===== Download ASR and QA Models from GCS =====
RUN python - <<'EOF'
from google.cloud import storage
import os

def download_from_gcs(bucket_name, prefix, local_dir):
    print(f"📦 Downloading {prefix} from {bucket_name} → {local_dir}")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    if not blobs:
        print(f"⚠️ No files found in gs://{bucket_name}/{prefix}")
        return
    os.makedirs(local_dir, exist_ok=True)
    for blob in blobs:
        if blob.size == 0:
            continue
        rel_path = os.path.relpath(blob.name, prefix)
        dest_path = os.path.join(local_dir, rel_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        blob.download_to_filename(dest_path)
        print("✅", blob.name)
    print(f"🎯 Finished downloading {len(blobs)} files from {bucket_name}/{prefix}\n")

# --- QA Model ---
download_from_gcs(
    bucket_name="qa-model-bucket",
    prefix="model/finetuned-qa-burmese",
    local_dir="/workspace/qa"
)

# --- ASR Model ---
download_from_gcs(
    bucket_name="speechtotext-model-bucket",
    prefix="model/finetuned-seamlessm4t-burmese",
    local_dir="/workspace/asr"
)
EOF

# Expose the Cloud Run port
ENV PORT=8080

# Start the Flask app using Gunicorn
CMD exec gunicorn -w 1 -b 0.0.0.0:$PORT main:app --timeout 20000 --threads 1 --preload
