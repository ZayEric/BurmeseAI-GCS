# ===== Base Image: CUDA 12.1 + Ubuntu 22.04 =====
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# ===== Environment Setup =====
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
WORKDIR /app

# ===== Install Python 3.11 and dependencies =====
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3.11-distutils python3-pip \
    curl git ffmpeg && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python
RUN python -m pip install --upgrade pip

# ===== Copy requirements and install dependencies =====
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir google-cloud-storage fastapi uvicorn

# ===== Copy source code =====
COPY . /app

# ===== Download ASR and QA Models from GCS =====
RUN python download_models.py

# ===== Expose and Run =====
EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
