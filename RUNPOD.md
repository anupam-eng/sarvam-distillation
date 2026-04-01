# RunPod Setup

This repo assumes you will train `VITS` on a GPU pod with a PyTorch CUDA image.

## Recommended Pod

Best balance:

- GPU: `RTX 4090 24GB` or `A5000 24GB`
- vCPU: `8+`
- RAM: `32GB+`
- disk: `150GB+`

Safer for larger datasets or longer runs:

- GPU: `A100 40GB` or `L40S 48GB`
- RAM: `48GB+`
- disk: `200GB+`

## Container

Use a recent PyTorch CUDA image with Python `3.10` or `3.11`.

The pod should have or allow installation of:

- `ffmpeg`
- `libsndfile1`
- `git`

## First-Time Setup

```bash
git clone <your-repo-url>
cd sarvam_distillation
bash scripts/setup_runpod.sh
export HF_TOKEN=your_huggingface_token
```

Before starting the download, make sure the Hugging Face account for `HF_TOKEN` has accepted access to `ai4bharat/IndicVoices`.

## Dataset Placement

The default pipeline now downloads and curates the Hindi `IndicVoices` subset automatically.

Raw downloaded files will be written under:

```bash
data/raw/indicvoices_hindi/
```

If you want to change the filtering behavior, edit `config/tts_config.yaml` under `download`.

## Train

```bash
bash scripts/run_vits_pipeline.sh
```

## Monitor

TensorBoard logs are written under the training output directory created by Coqui.

Typical command:

```bash
tensorboard --logdir models/vits_dataset
```

## What I Still Need From You

Once you share the pod details, I can tune these immediately:

1. batch size
2. number of workers
3. mixed precision choice
4. storage layout
5. whether we should keep raw data and checkpoints on persistent volume

Once you share the dataset layout, I can also tune:

1. manifest format mapping
2. text cleaning strategy
3. audio duration filters
4. whether phonemes are worth enabling
