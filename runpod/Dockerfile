# Cheapest POC: use a small NVIDIA GPU (16GB+ VRAM recommended), fp16, short clips.
# Build:  docker build -t yourname/svd-runpod:poc .
# Push:   docker push yourname/svd-runpod:poc

FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface \
    DIFFUSERS_ATTN_BACKEND=native

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY svd_core.py run_svd_cuda.py .
RUN chmod +x run_svd_cuda.py

# Default: cheap clip (override args when running)
ENTRYPOINT ["python", "/app/run_svd_cuda.py"]
CMD ["--help"]
