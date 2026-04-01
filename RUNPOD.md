# RunPod Configuration

This repo now defaults to:

- ASR teacher: `openai/whisper-large-v3-turbo`
- ASR student: `facebook/wav2vec2-base`
- TTS teacher: `facebook/mms-tts-eng`
- TTS student: `microsoft/speecht5_tts`

That setup is much lighter than Qwen Omni or XTTS, so you do not need an extreme pod for the first distillation runs.

## Recommended Pod

Best balance for first real runs:

- GPU: `RTX 4090 24GB` or `L40S 48GB`
- vCPU: `8+`
- RAM: `32GB+`
- Disk: `100GB` minimum, `200GB` preferred
- CUDA image: a recent PyTorch image with CUDA 12.x

If you want the safest option for longer ASR runs or later heavier TTS experiments:

- GPU: `A100 40GB` or better

## Minimum Practical Pod

For testing and small-scale ASR distillation only:

- GPU: `A5000 24GB` or `RTX 4090 24GB`
- vCPU: `4 to 8`
- RAM: `24GB to 32GB`
- Disk: `80GB+`

## Not Recommended For This Repo

- CPU-only pods
- GPUs with less than `16GB` VRAM for meaningful ASR training
- very small disks, because model downloads and teacher outputs accumulate quickly

## Suggested Container Base

Use a PyTorch container with Python 3.10 or 3.11 and CUDA 12.x. Any equivalent image is fine. The important part is that `torch`, `ffmpeg`, and standard build tools can be installed cleanly.

## Setup Commands

After the pod starts:

```bash
git clone git@github.com:anupam-eng/sarvam-distillation.git
cd sarvam-distillation
pip install --upgrade pip
pip install -r requirements.txt
```

## Disk And Download Expectations

Plan for these categories of storage:

- model weights
- raw audio
- pseudo-labeled outputs
- shard tar files
- training checkpoints
- evaluation reports

If you expect more than a few hours of audio, choose `200GB+` disk immediately.

## What To Send Me

Once your pod is ready, send me:

1. GPU type and VRAM
2. vCPU and RAM
3. disk size
4. operating system or container image
5. whether the pod has persistent volume storage
6. whether you want to run ASR only first, or ASR plus TTS

After that, I can give you the exact install, download, data-prep, training, and evaluation commands for your pod.
