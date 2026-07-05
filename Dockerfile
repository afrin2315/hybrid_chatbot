FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV HOST=0.0.0.0
ENV PORT=7860
ENV SESSION_COOKIE_SECURE=1
# Force transformers onto PyTorch (no TensorFlow installed).
ENV USE_TF=0
ENV USE_TORCH=1
# Keep model + HF caches inside the app dir (writable on Hugging Face Spaces).
ENV HF_HOME=/app/.hf_cache

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

# Install the CPU-only PyTorch wheel first so the image stays lean (the default
# PyPI torch pulls in ~2 GB of CUDA libraries we don't use).
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 \
    && pip install -r requirements.txt

COPY . /app

# Timeout raised because the first request may lazily load DistilBERT.
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--timeout", "120", "--workers", "1", "wsgi:app"]
