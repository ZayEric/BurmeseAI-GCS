# ===== Base Image =====
FROM python:3.11-slim

# ===== Environment Setup =====
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
WORKDIR /app

# ===== Install dependencies =====
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir google-cloud-storage

# ===== Copy source code =====
COPY main.py .

# ===== Add Python script to download models =====
COPY download_models.py .

# ===== Run App =====
CMD ["python", "main.py"]
