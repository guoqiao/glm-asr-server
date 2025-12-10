# 1. Base Image: NVIDIA CUDA Runtime
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# 2. Environment Variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# 3. Install System Dependencies
# We need python3-venv for isolation and ffmpeg for audio/video processing
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# 4. Set working directory first (before creating user)
WORKDIR /app

# 5. Create and Activate Virtual Environment (as root)
# This effectively "activates" the venv for all future commands
ENV VIRTUAL_ENV=/app/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# 6. Create a non-root user for security and change ownership
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# 7. Upgrade pip (internal venv pip)
RUN pip install --no-cache-dir --upgrade pip

# ==========================================
# LAYER A: Heavy Dependencies (Cached)
# ==========================================
# We install PyTorch manually first.
# This ensures we get the specific CUDA 12.1 version.
# Since this command rarely changes, Docker will cache this layer forever.
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# ==========================================
# LAYER B: App Dependencies (Frequent Changes)
# ==========================================
# Copy only requirements.txt first to utilize cache
COPY --chown=appuser:appuser requirements.txt .

# Install the rest (Transformers, FastAPI, etc.)
# Transformers will detect the existing 'torch' installation and use it.
RUN pip install --no-cache-dir -r requirements.txt

# ==========================================
# LAYER C: Application Code
# ==========================================
COPY --chown=appuser:appuser server.py .

# 8. Launch
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
