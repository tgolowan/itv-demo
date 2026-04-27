#!/usr/bin/env python3
"""
Cheapest RunPod POC: image → short MP4 with Stable Video Diffusion (CUDA fp16).

See svd_core.py for env vars. Requires HF_TOKEN if the model is gated.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from svd_core import log_default, run_svd_from_pil


def main() -> int:
    parser = argparse.ArgumentParser(description="RunPod cheap I2V POC (CUDA + SVD)")
    parser.add_argument("input", type=Path, help="Input image (JPEG/PNG/WebP)")
    parser.add_argument("output", type=Path, help="Output .mp4")
    parser.add_argument("--duration", type=int, default=4, help="Target clip length seconds (looped)")
    parser.add_argument("--fps", type=int, default=12, help="Output FPS (lower = cheaper)")
    args = parser.parse_args()

    import torch
    from PIL import Image

    if not torch.cuda.is_available():
        log_default("ERROR: CUDA required. Use an NVIDIA GPU pod on RunPod.")
        return 1

    inp = args.input.expanduser().resolve()
    if not inp.is_file():
        log_default(f"ERROR: input not found: {inp}")
        return 1

    out = args.output.expanduser().resolve()
    image = Image.open(inp).convert("RGB")

    log_default(f"Device: cuda ({torch.cuda.get_device_name(0)})")
    run_svd_from_pil(image, out, args.duration, args.fps, log=log_default)
    return 0


if __name__ == "__main__":
    sys.exit(main())
