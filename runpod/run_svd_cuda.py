#!/usr/bin/env python3
"""
Cheapest RunPod POC: image → short MP4 with Stable Video Diffusion (CUDA fp16).

Defaults are tuned for small VRAM (~12–16 GB) and short GPU time.
Accept the model license on Hugging Face and set HF_TOKEN if downloads are gated.

Env:
  SVD_MODEL — default stabilityai/stable-video-diffusion-img2vid
  SVD_NATIVE_FRAMES — default 6
  SVD_STEPS — default 8
  SVD_LOOP_MODE — pingpong | repeat (default pingpong)
  SVD_EXPORT_WIDTH / SVD_EXPORT_HEIGHT — default 1280 x 720
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def log(msg: str) -> None:
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


def main() -> int:
    parser = argparse.ArgumentParser(description="RunPod cheap I2V POC (CUDA + SVD)")
    parser.add_argument("input", type=Path, help="Input image (JPEG/PNG/WebP)")
    parser.add_argument("output", type=Path, help="Output .mp4")
    parser.add_argument("--duration", type=int, default=4, help="Target clip length seconds (looped)")
    parser.add_argument("--fps", type=int, default=12, help="Output FPS (lower = cheaper)")
    args = parser.parse_args()

    import torch
    from diffusers import StableVideoDiffusionPipeline
    from PIL import Image

    if not torch.cuda.is_available():
        log("ERROR: CUDA required. Use an NVIDIA GPU pod on RunPod.")
        return 1

    inp = args.input.expanduser().resolve()
    if not inp.is_file():
        log(f"ERROR: input not found: {inp}")
        return 1

    out = args.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    num_out = max(8, args.duration * args.fps)
    native = int(os.environ.get("SVD_NATIVE_FRAMES", "6"))
    steps = int(os.environ.get("SVD_STEPS", "8"))
    decode_chunk = max(1, min(int(os.environ.get("SVD_DECODE_CHUNK", str(native))), native))
    model_id = os.environ.get("SVD_MODEL", "stabilityai/stable-video-diffusion-img2vid")
    loop_mode = os.environ.get("SVD_LOOP_MODE", "pingpong")
    ow = int(os.environ.get("SVD_EXPORT_WIDTH", "1280"))
    oh = int(os.environ.get("SVD_EXPORT_HEIGHT", "720"))

    log(f"Device: cuda ({torch.cuda.get_device_name(0)})")
    log(f"Model: {model_id} | latent_frames={native} steps={steps} | timeline {num_out} frames @ {args.fps}fps")

    image = Image.open(inp).convert("RGB").resize((1024, 576), Image.LANCZOS)

    pipe = StableVideoDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing("max")

    def step_cb(_pipe, step, _t, cb_kwargs):
        pct = 10 + int((step + 1) / steps * 70)
        log(f"PROGRESS:{pct}")
        return cb_kwargs

    log("Running SVD (integrated decode, fp16)…")
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
    log("PROGRESS:85")

    with tempfile.TemporaryDirectory() as tmp:
        tmpd = Path(tmp)
        for i in range(num_out):
            idx = loop_index(i, len(frames), loop_mode)
            fr = frames[idx].resize((ow, oh), Image.LANCZOS)
            fr.save(tmpd / f"frame_{i:05d}.png")
        encode_mp4(tmpd, out, args.fps, ow, oh)

    log("PROGRESS:100")
    log(f"Done: {out} ({out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
