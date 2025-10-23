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

# Install requirements first (without transformers)
RUN pip install --no-cache-dir -r requirements.txt

# Install torch with CUDA support
RUN pip install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

# Then install transformers last
RUN pip install --no-cache-dir transformers

# ===== COPY APP FILES =====
COPY . .

# ===== EXPOSE PORT =====
EXPOSE 8080

# ===== RUN FLASK APP =====
CMD ["python", "main.py"]
