"""Shared SVD (CUDA fp16) helpers for CLI and RunPod serverless."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

from PIL import Image

_pipeline = None


def log_default(msg: str) -> None:
    print(msg, flush=True)


def loop_index(i: int, n: int, mode: str) -> int:
    if n <= 1:
        return 0
    mode = (mode or "pingpong").strip().lower()
    if mode == "repeat":
        return i % n
    period = 2 * n - 2
    if period <= 0:
        return 0
    j = i % period
    if j >= n:
        return 2 * n - 2 - j
    return j


def encode_mp4(frames_dir: Path, out: Path, fps: int, w: int, h: int) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not in PATH")
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%05d.png"),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-vf", f"scale={w}:{h}",
        str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr or "ffmpeg failed")


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        import torch
        from diffusers import StableVideoDiffusionPipeline

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")

        model_id = os.environ.get("SVD_MODEL", "stabilityai/stable-video-diffusion-img2vid")
        _pipeline = StableVideoDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            variant="fp16",
        )
        _pipeline = _pipeline.to("cuda")
        _pipeline.enable_attention_slicing("max")
    return _pipeline


def run_svd_from_pil(
    image: Image.Image,
    out_mp4: Path,
    duration: int,
    fps: int,
    log: Callable[[str], None] | None = None,
) -> None:
    """Resize image to SVD size, run pipeline, write MP4."""
    import torch

    _log = log or log_default
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    num_out = max(8, duration * fps)
    native = int(os.environ.get("SVD_NATIVE_FRAMES", "6"))
    steps = int(os.environ.get("SVD_STEPS", "8"))
    decode_chunk = max(1, min(int(os.environ.get("SVD_DECODE_CHUNK", str(native))), native))
    loop_mode = os.environ.get("SVD_LOOP_MODE", "pingpong")
    ow = int(os.environ.get("SVD_EXPORT_WIDTH", "1280"))
    oh = int(os.environ.get("SVD_EXPORT_HEIGHT", "720"))

    image = image.convert("RGB").resize((1024, 576), Image.LANCZOS)
    pipe = get_pipeline()

    def step_cb(_pipe, step, _t, cb_kwargs):
        pct = 10 + int((step + 1) / steps * 70)
        _log(f"PROGRESS:{pct}")
        return cb_kwargs

    _log("Running SVD (integrated decode, fp16)…")
    result = pipe(
        image,
        num_frames=native,
        motion_bucket_id=int(os.environ.get("SVD_MOTION_BUCKET", "127")),
        noise_aug_strength=float(os.environ.get("SVD_NOISE_AUG", "0.02")),
        num_inference_steps=steps,
        callback_on_step_end=step_cb,
        output_type="pil",
        decode_chunk_size=decode_chunk,
    )
    frames = result.frames[0]
    _log("PROGRESS:85")

    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmpd = Path(tmp)
        for i in range(num_out):
            idx = loop_index(i, len(frames), loop_mode)
            fr = frames[idx].resize((ow, oh), Image.LANCZOS)
            fr.save(tmpd / f"frame_{i:05d}.png")
        encode_mp4(tmpd, out_mp4, fps, ow, oh)

    _log("PROGRESS:100")
    _log(f"Done: {out_mp4} ({out_mp4.stat().st_size // 1024} KB)")
