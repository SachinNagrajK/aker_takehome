# Dockerfile for Hugging Face Spaces (Docker SDK, free CPU basic, 16 GB RAM).
# Builds the FastAPI + LangGraph + Jina-CLIP-v2 ONNX backend.
#
# Notes:
#   - HF Spaces requires the app to listen on $PORT (7860 by default).
#   - The Jina-CLIP-v2 fp32 model (~1.7 GB) is NOT baked into the image;
#     it lazy-downloads on first /chat call into HF_HOME and stays warm
#     for the lifetime of the running container.
#   - All secrets (DATABASE_URL, PINECONE_API_KEY, etc.) come from Space
#     "Secrets" at runtime — never from this file.

FROM python:3.11-slim

# System deps needed by docling (PDF/image processing) + onnxruntime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces convention: non-root user `user` with UID 1000 owns the
# workspace so cache writes don't hit permission errors.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/user/.cache/huggingface

WORKDIR /home/user/app

# Install deps first so they're cached across code-only changes.
COPY --chown=user requirements.txt ./
RUN pip install --user --no-cache-dir -r requirements.txt

# App code last for fast iteration.
COPY --chown=user app ./app

EXPOSE 7860

# HF Spaces routes traffic to $PORT; default 7860 if unset.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
