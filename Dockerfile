# ===== BASE IMAGE: CUDA runtime for GPU inference =====
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# ===== ENV CONFIG =====
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
WORKDIR /app

# ===== INSTALL PYTHON =====
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3.11-distutils python3-pip \
    git curl ffmpeg && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

# ===== COPY DEPENDENCIES =====
COPY requirements.txt .

# ===== INSTALL DEPENDENCIES =====
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir google-cloud-storage Flask pydub soundfile transformers

# ===== COPY APP FILES =====
COPY . .

# ===== EXPOSE PORT =====
EXPOSE 8080

# ===== RUN FLASK APP =====
CMD ["python", "main.py"]
