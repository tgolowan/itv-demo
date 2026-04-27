# RunPod: cheapest image→video POC

Minimal **CUDA + Stable Video Diffusion** container. Tuned for **low GPU time** and **~12–16 GB VRAM** (fp16, 6 latent frames, 8 steps, 720p, short timeline).

## Cost tips

1. **Community Cloud** / **spot** GPUs if your RunPod plan allows.
2. Pick the **smallest GPU that has enough VRAM** (16 GB is a safe target for these defaults).
3. **Stop the pod** as soon as the MP4 exists — you pay while it runs.
4. Use **Network Volume** for Hugging Face cache if you run many experiments (avoid re-downloading weights).

## Hugging Face

1. Open [stabilityai/stable-video-diffusion-img2vid](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid) and **accept the license** if required.
2. On the pod, set a **read token**:

   ```bash
   export HF_TOKEN=hf_xxx
   ```

## Build & push (your machine)

```bash
cd runpod
docker build -t YOUR_DOCKERHUB_USER/svd-runpod:poc .
docker push YOUR_DOCKERHUB_USER/svd-runpod:poc
```

## Run on RunPod (GPU pod)

1. Create a pod from **your image** `YOUR_DOCKERHUB_USER/svd-runpod:poc` (or run commands in a PyTorch CUDA template and copy the script + `pip install -r requirements.txt`).
2. Put an image in the pod, e.g. `/workspace/in.jpg`.
3. Run:

   ```bash
   export HF_TOKEN=hf_xxx   # if needed
   python /app/run_svd_cuda.py /workspace/in.jpg /workspace/out.mp4 --duration 4 --fps 12
   ```

   Even cheaper smoke test:

   ```bash
   python /app/run_svd_cuda.py /workspace/in.jpg /workspace/out.mp4 --duration 2 --fps 8
   ```

4. Download `out.mp4` (RunPod file manager / `scp` / volume).

## Override quality vs price (env)

| Variable | Default | Note |
|----------|---------|------|
| `SVD_NATIVE_FRAMES` | 6 | ↑ smoother motion, ↑ VRAM/time |
| `SVD_STEPS` | 8 | ↑ quality, ↑ time |
| `SVD_LOOP_MODE` | pingpong | `repeat` = hard loop |
| `SVD_EXPORT_WIDTH` / `HEIGHT` | 1280×720 | 1080p costs more time |

## Troubleshooting

- **CUDA OOM**: lower `SVD_NATIVE_FRAMES` to `4` or use a larger GPU.
- **401 / gated model**: set `HF_TOKEN` and accept the license on the model page.
