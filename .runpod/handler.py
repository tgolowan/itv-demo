"""RunPod Serverless handler: image URL or base64 → MP4 (base64)."""
from __future__ import annotations

import base64
import io
import sys
import tempfile
import urllib.request
from pathlib import Path

import runpod
from PIL import Image

# /app is WORKDIR in .runpod/Dockerfile
sys.path.insert(0, "/app")
from svd_core import run_svd_from_pil  # noqa: E402


def handler(job):
    job_input = job.get("input") or {}
    image_url = job_input.get("image_url")
    image_b64 = job_input.get("image_base64")
    duration = int(job_input.get("duration", 4))
    fps = int(job_input.get("fps", 12))

    try:
        if image_url:
            req = urllib.request.Request(
                image_url,
                headers={"User-Agent": "RunPod-SVD-POC/1.0"},
            )
            with urllib.request.urlopen(req, timeout=120) as r:
                img = Image.open(io.BytesIO(r.read())).convert("RGB")
        elif image_b64:
            raw = base64.b64decode(image_b64)
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        else:
            return {"error": "Provide image_url or image_base64 in input"}

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            out_path = Path(f.name)

        try:
            run_svd_from_pil(img, out_path, duration, fps)
            data = out_path.read_bytes()
            return {
                "video_base64": base64.b64encode(data).decode("ascii"),
                "content_type": "video/mp4",
                "size_bytes": len(data),
            }
        finally:
            out_path.unlink(missing_ok=True)

    except Exception as e:
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
