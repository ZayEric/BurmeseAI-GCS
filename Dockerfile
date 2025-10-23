# =====================================================================
# ✅ 1. Use NVIDIA CUDA base image compatible with Cloud Run GPU (T4)
# =====================================================================
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# =====================================================================
# ✅ 2. Install system dependencies
# =====================================================================
RUN apt-get update && apt-get install -y \
    python3 python3-pip ffmpeg git curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# =====================================================================
# ✅ 3. Copy dependency list and install
# =====================================================================
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
 && pip install google-cloud-storage fastapi uvicorn pydub transformers

# =====================================================================
# ✅ 4. Copy application code
# =====================================================================
COPY . /app

# =====================================================================
# ✅ 5. Download models from GCS during build
# =====================================================================
RUN python download_models.py

# =====================================================================
# ✅ 6. Expose port for FastAPI
# =====================================================================
EXPOSE 8080

# =====================================================================
# ✅ 7. Start app (GPU-ready, non-blocking startup)
# =====================================================================
CMD ["python3", "main.py"]
