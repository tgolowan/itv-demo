# RunPod Hub build context: repository root (see https://docs.runpod.io/hub/publishing-guide)
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY runpod/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt \
    && pip install --no-cache-dir runpod>=1.6.0

COPY runpod/svd_core.py /app/svd_core.py
COPY .runpod/handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
