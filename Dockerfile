# ===== BASE IMAGE: CUDA runtime =====
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
WORKDIR /app

# Install Python + tools
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3.11-distutils python3-pip \
    git curl ffmpeg && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip

# Copy requirements
COPY requirements.txt .

# Install torch first (CUDA)
RUN pip install --no-cache-dir torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

# Install other dependencies including transformers
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose port
EXPOSE 8080

# Run app
CMD ["python", "main.py"]
