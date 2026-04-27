# ITV Demo — local + RunPod

Local stack: React UI + Express + Python (see `src/`, `server/`). For **Mac**, image→video uses SVD with conservative CPU defaults in `server/video_gen.py`.

## RunPod Hub (serverless)

This repo includes `.runpod/hub.json`, `.runpod/tests.json`, `.runpod/Dockerfile`, and `.runpod/handler.py` for [RunPod Hub](https://docs.runpod.io/hub/publishing-guide). **Publish:** merge to `main`, then create a **GitHub Release** so Hub picks up the tag.

If `stabilityai/stable-video-diffusion-img2vid` is gated on Hugging Face, accept the license and set **`HF_TOKEN`** on the endpoint (or in Hub test env).

CLI image (no serverless): build from `runpod/Dockerfile` and run `run_svd_cuda.py` — see `runpod/README.md`.

[![Runpod](https://api.runpod.io/badge/tgolowan/itv-demo)](https://console.runpod.io/hub/tgolowan/itv-demo)
