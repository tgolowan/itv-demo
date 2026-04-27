#!/usr/bin/env python3
"""
Stable Video Diffusion video generator for Apple Silicon (MPS).

Env:
  On macOS, SVD defaults to CPU (Metal can freeze/reboot the whole system). For GPU: SVD_USE_MPS=1 or SVD_DEVICE=mps.
  SVD_USE_MPS=1 — opt into Metal on macOS (faster; can still hang or reboot under load).
  SVD_DEVICE=cpu|mps|cuda — force device (overrides macOS CPU default when set to mps).
  SVD_CPU_THREADS=N — PyTorch intra-op threads in CPU mode (default 4; lower = cooler/slower).
  SVD_EXPORT_WIDTH / SVD_EXPORT_HEIGHT — output size (default 1280×720 POC preset; set 1920×1080 for full HD).
  SVD_LOOP_MODE=pingpong|repeat — how to pad short clips to duration (default pingpong avoids fast blink from hard loop jumps).
  SVD_MPS_FP16=1 — MPS fp16 math (lower RAM; may grey-out; try with SVD_SEQUENTIAL_OFFLOAD=1).
  SVD_SEQUENTIAL_OFFLOAD=1 — on MPS, use sequential CPU offload (lowest peak GPU memory, slowest).
  SVD_VAE_DECODE_DEVICE=cpu|mps|cuda — VAE decode device (default: cpu on MPS, else infer device).
  SVD_DECODE_CHUNK — temporal VAE chunk size (default: all native frames).
  SVD_ENABLE_VAE_SLICING / SVD_ENABLE_VAE_TILING — off by default (unsafe for SVD temporal VAE).
  FFMPEG_ENCODER=libx264|h264_videotoolbox — default libx264.
  SVD_DEBUG=1 — log latent / tensor stats.
  SVD_NO_CPU_OFFLOAD=1 — on MPS, load the full pipeline on GPU (faster, highest risk of SIGKILL / OS freeze).

Usage:
    python3 video_gen.py <input_image> <output.mp4> --duration 10 --fps 30 [--prompt TEXT]
    python3 video_gen.py --text-only <output.mp4> --duration 10 --fps 30 --prompt TEXT
"""
import argparse
import json
import os

# Do not set PYTORCH_MPS_HIGH_WATERMARK_RATIO here: several torch builds error with
# RuntimeError: invalid low watermark ratio … when this env is set. Optionally export it in your shell
# *after* verifying your PyTorch version accepts it.
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def log(msg):
    print(msg, flush=True)


def progress(pct):
    print(f"PROGRESS:{int(pct)}", flush=True)


def detect_device():
    try:
        import torch

        forced = (os.environ.get("SVD_DEVICE") or "").strip().lower()
        if forced in ("cpu", "mps", "cuda"):
            if forced == "mps" and not torch.backends.mps.is_available():
                log("SVD_DEVICE=mps but MPS unavailable; using cpu.")
                return "cpu"
            if forced == "cuda" and not torch.cuda.is_available():
                log("SVD_DEVICE=cuda but CUDA unavailable; using cpu.")
                return "cpu"
            return forced

        # macOS: PyTorch MPS + SVD has triggered full-system freezes and sudden reboots when unified
        # memory is stressed. Default to CPU unless the user explicitly opts into Metal.
        if sys.platform == "darwin":
            if os.environ.get("SVD_USE_MPS", "").lower() in ("1", "true", "yes"):
                if torch.backends.mps.is_available():
                    return "mps"
                log("SVD_USE_MPS=1 but MPS is not available; using cpu.")
                return "cpu"
            log(
                "macOS: using CPU for Stable Video Diffusion (video still generates; expect several minutes per clip). "
                "Metal GPU: export SVD_USE_MPS=1 — faster but can freeze or reboot the Mac under load."
            )
            return "cpu"

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    except Exception:
        return "cpu"


def load_pipeline(device):
    from diffusers import StableVideoDiffusionPipeline
    import torch

    model_id = os.environ.get("SVD_MODEL", "stabilityai/stable-video-diffusion-img2vid")
    # CUDA: fp16 is usually fine. MPS: fp16 denoise often collapses latents → grey 5–20KB PNGs; default fp32 math.
    if device == "cuda":
        load_dtype = torch.float16
        variant = "fp16"
        runtime_dtype = torch.float16
    elif device == "mps":
        load_dtype = torch.float16
        variant = "fp16"
        mps_fp16 = os.environ.get("SVD_MPS_FP16", "").lower() in ("1", "true", "yes")
        runtime_dtype = torch.float16 if mps_fp16 else torch.float32
    else:
        load_dtype = torch.float32
        variant = None
        runtime_dtype = torch.float32

    log(f"Loading {model_id} on {device} (load={load_dtype}, runtime={runtime_dtype})…")
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=load_dtype,
        variant=variant,
    )
    # MPS + full fp32 UNet on device at once → OS SIGKILL. Default: model CPU offload (only one big module on GPU at a time).
    no_offload = os.environ.get("SVD_NO_CPU_OFFLOAD", "").lower() in ("1", "true", "yes")
    if device == "mps" and not no_offload:
        pipe.to(dtype=runtime_dtype)
        try:
            if os.environ.get("SVD_SEQUENTIAL_OFFLOAD", "").lower() in ("1", "true", "yes"):
                pipe.enable_sequential_cpu_offload(device=device)
                log("Sequential CPU offload on MPS (lowest peak unified memory; slowest).")
            else:
                pipe.enable_model_cpu_offload(device=device)
                log("Model CPU offload enabled (recommended on MPS — avoids most OOM kills).")
        except Exception as e:
            log(f"WARNING: CPU offload failed ({e}); falling back to full MPS — may OOM or freeze OS. Try upgrading `accelerate`.")
            pipe.to(device=device, dtype=runtime_dtype)
    elif device == "cuda" and os.environ.get("SVD_CPU_OFFLOAD", "").lower() in ("1", "true", "yes"):
        pipe.to(dtype=runtime_dtype)
        pipe.enable_model_cpu_offload(device=device)
        log("Model CPU offload enabled (CUDA).")
    else:
        pipe = pipe.to(device=device, dtype=runtime_dtype)
    pipe.enable_attention_slicing("max")
    # Temporal VAE (SVD) + MPS: sliced/tiled 2D VAE helpers are unsafe — can yield black frames.
    if os.environ.get("SVD_ENABLE_VAE_SLICING", "").lower() in ("1", "true", "yes"):
        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass
    if os.environ.get("SVD_ENABLE_VAE_TILING", "").lower() in ("1", "true", "yes"):
        try:
            pipe.enable_vae_tiling()
        except Exception:
            pass
    return pipe


def make_synthetic_image(prompt, size=(1024, 576)):
    """Fallback when no input image given. Creates a gradient placeholder."""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", size, (10, 10, 20))
    draw = ImageDraw.Draw(img)
    for y in range(size[1]):
        t = y / size[1]
        r = int(10 + 30 * t)
        g = int(10 + 80 * (1 - t))
        b = int(40 + 100 * t)
        draw.line([(0, y), (size[0], y)], fill=(r, g, b))
    try:
        font = ImageFont.load_default()
        draw.text((20, 20), (prompt or "")[:80], fill=(255, 255, 255), font=font)
    except Exception:
        pass
    return img


def _pipeline_uses_model_cpu_offload(pipe):
    hooks = getattr(pipe, "_all_hooks", None)
    return bool(hooks)


def _loop_frame_index(i, n, mode):
    """
    Map output frame i to a source frame in [0, n-1].
    pingpong: …3,4,5,4,3,2,1,0,1,2… — smooth at ends (no last→first jump).
    repeat: i % n — can strobe when the clip is short vs duration.
    """
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


def generate_frames(pipe, image, num_frames, output_dir, infer_device):
    import torch, gc
    from PIL import Image

    image = image.convert("RGB").resize((1024, 576))
    progress(15)
    # Lighter defaults: CPU is slow; MPS is memory-heavy.
    native_frames = int(
        os.environ.get(
            "SVD_NATIVE_FRAMES",
            "6" if infer_device == "cpu" else ("8" if infer_device == "mps" else "10"),
        )
    )
    steps = int(
        os.environ.get(
            "SVD_STEPS",
            "8" if infer_device == "cpu" else ("10" if infer_device == "mps" else "12"),
        )
    )
    decode_chunk = int(os.environ.get("SVD_DECODE_CHUNK", str(native_frames)))
    decode_chunk = max(1, min(decode_chunk, native_frames))
    def step_cb(_pipe, step, _t, cb_kwargs):
        pct = 15 + int(((step + 1) / steps) * 53)
        progress(pct)
        return cb_kwargs

    log(f"SVD: {native_frames} frames (→ {num_frames} written), {steps} steps, decode_chunk={decode_chunk}…")

    if _pipeline_uses_model_cpu_offload(pipe):
        # Manual .to(cpu) on submodules breaks accelerate hooks; use integrated decode.
        out = pipe(
            image,
            num_frames=native_frames,
            motion_bucket_id=127,
            noise_aug_strength=0.02,
            num_inference_steps=steps,
            callback_on_step_end=step_cb,
            output_type="pil",
            decode_chunk_size=decode_chunk,
        )
        frames = out.frames[0]
        progress(82)
    else:
        latents = pipe(
            image,
            num_frames=native_frames,
            motion_bucket_id=127,
            noise_aug_strength=0.02,
            num_inference_steps=steps,
            callback_on_step_end=step_cb,
            output_type="latent",
        ).frames

        if os.environ.get("SVD_DEBUG", "").lower() in ("1", "true", "yes"):
            z = latents.detach().float()
            log(
                f"DEBUG latents shape={tuple(latents.shape)} "
                f"min={z.min().item():.5f} max={z.max().item():.5f} "
                f"mean={z.mean().item():.5f} std={z.std().item():.5f}"
            )

        flat = latents.detach().float().std().item() < 1e-5
        if flat or not torch.isfinite(latents).all():
            log(
                "WARNING: denoised latents are nearly constant or non-finite — output may look grey. "
                "Try SVD_DEVICE=cpu or adjust SVD_MPS_FP16 / steps."
            )

        if infer_device == "mps" and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()

        pipe.unet.to("cpu")
        gc.collect()
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

        decode_dev = (os.environ.get("SVD_VAE_DECODE_DEVICE") or "").strip().lower()
        if not decode_dev:
            decode_dev = "cpu" if infer_device == "mps" else infer_device

        use_fp16 = decode_dev in ("mps", "cuda") and os.environ.get("SVD_VAE_FP16_DECODE", "").lower() in (
            "1",
            "true",
            "yes",
        )
        log(f"VAE decode on {decode_dev} ({'fp16' if use_fp16 else 'fp32'}, chunk={decode_chunk})…")
        progress(68)

        if use_fp16:
            lat = latents.to(decode_dev, dtype=torch.float16)
            pipe.vae.to(decode_dev, dtype=torch.float16)
        else:
            lat = latents.to(decode_dev, dtype=torch.float32)
            pipe.vae.to(decode_dev, dtype=torch.float32)

        with torch.no_grad():
            video = pipe.decode_latents(lat, native_frames, decode_chunk_size=decode_chunk)

        if os.environ.get("SVD_DEBUG", "").lower() in ("1", "true", "yes"):
            v = video.detach().float()
            log(
                f"DEBUG decoded video shape={tuple(video.shape)} "
                f"min={v.min().item():.5f} max={v.max().item():.5f} "
                f"mean={v.mean().item():.5f} std={v.std().item():.5f}"
            )

        if not torch.isfinite(video).all():
            log("Warning: non-finite pixels after VAE decode; sanitizing.")
            video = torch.nan_to_num(video, nan=0.0, posinf=1.0, neginf=-1.0)

        frames = pipe.video_processor.postprocess_video(video, output_type="pil")[0]
        progress(82)

    # POC preset: 720p (1280×720); override for 1080p.
    out_w = int(os.environ.get("SVD_EXPORT_WIDTH", "1280"))
    out_h = int(os.environ.get("SVD_EXPORT_HEIGHT", "720"))
    log(f"Export frames at {out_w}×{out_h} (default POC resolution 720p; set SVD_EXPORT_* for e.g. 1920×1080).")

    loop_mode = os.environ.get("SVD_LOOP_MODE", "pingpong")
    n_src = len(frames)
    log(f"Timeline: {num_frames} frames from {n_src} generated (loop={loop_mode}).")

    saved = []
    for i in range(num_frames):
        fr = frames[_loop_frame_index(i, n_src, loop_mode)]
        fr = fr.resize((out_w, out_h), Image.LANCZOS)
        p = output_dir / f"frame_{i:05d}.png"
        fr.save(p)
        saved.append(p)
    return out_w, out_h


def encode_video(frames_dir, output_path, fps, width=1920, height=1080):
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH")
    encoder = (os.environ.get("FFMPEG_ENCODER") or "libx264").strip().lower()
    log(f"Encoding with {encoder}…")
    vf = f"scale={width}:{height}"
    if encoder in ("h264_videotoolbox", "videotoolbox"):
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(frames_dir / "frame_%05d.png"),
            "-c:v", "h264_videotoolbox",
            "-b:v", "8M",
            "-pix_fmt", "yuv420p",
            "-vf", vf,
            str(output_path),
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(frames_dir / "frame_%05d.png"),
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-vf", vf,
            str(output_path),
        ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 and encoder in ("h264_videotoolbox", "videotoolbox"):
        log("videotoolbox failed; falling back to libx264")
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(frames_dir / "frame_%05d.png"),
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-vf", vf,
            str(output_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {res.stderr}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", help="Input image or omit with --text-only")
    parser.add_argument("output", help="Output mp4 path")
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--duration", type=int, default=6)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--prompt", type=str, default="")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_frames = max(8, args.duration * args.fps)

    progress(2)
    device = detect_device()
    log(f"Device: {device}")
    if device == "cpu":
        import torch

        cpu_threads = max(1, int(os.environ.get("SVD_CPU_THREADS", "4")))
        torch.set_num_threads(cpu_threads)
        torch.set_num_interop_threads(min(2, max(1, cpu_threads // 2)))
        log(
            f"CPU-safe mode: {cpu_threads} compute threads (export SVD_CPU_THREADS=N to change). "
            "Slow, but avoids Metal memory spikes that freeze or reboot macOS."
        )

    try:
        from PIL import Image
        if args.text_only or not args.input:
            log("No input image provided — generating synthetic seed.")
            image = make_synthetic_image(args.prompt)
        else:
            image = Image.open(args.input)

        progress(8)
        pipe = load_pipeline(device)

        with tempfile.TemporaryDirectory() as tmpd:
            tmp = Path(tmpd)
            ow, oh = generate_frames(pipe, image, num_frames, tmp, device)
            progress(85)
            encode_video(tmp, output_path, args.fps, ow, oh)

        progress(100)
        result = {"status": "ok", "output": str(output_path), "frames": num_frames, "fps": args.fps}
        print(json.dumps(result), flush=True)
    except Exception as e:
        err = {"status": "error", "error": str(e)}
        print(json.dumps(err), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
