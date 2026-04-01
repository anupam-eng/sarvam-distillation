# VITS Training Pipeline

This branch is a fresh setup focused only on training a `VITS` text-to-speech model with your own dataset.

The repo is designed for a RunPod GPU workflow:

1. stream and filter Hindi data from `ai4bharat/IndicVoices`
2. auto-pick one strong Hindi speaker for single-speaker `VITS`
3. prepare a clean Coqui-compatible dataset split
4. train a `VITS` model with `Coqui TTS`
5. synthesize evaluation samples from the trained checkpoint

## Current Scope

- Hindi-only `IndicVoices` download and filtering
- auto-selection of one Hindi speaker by longest filtered duration
- configurable source manifest format: `csv`, `tsv`, `jsonl`, or `ljspeech`
- automatic audio conversion to mono `wav` at the target sample rate
- deterministic train/eval split
- RunPod setup script and end-to-end training script

Why single-speaker here: `IndicVoices` is multi-speaker and spontaneous. A basic single-speaker `VITS` is the safer first target for quality, so the pipeline curates one strong Hindi speaker instead of mixing many speakers into a weak baseline.

## Repo Layout

- `config/tts_config.yaml`: main dataset, audio, training, and inference settings
- `scripts/setup_runpod.sh`: installs system and Python dependencies on a pod
- `scripts/run_vits_pipeline.sh`: runs dataset prep, training, and synthesis end-to-end
- `src/vits_pipeline/download_indicvoices_hindi.py`: streams and filters Hindi `IndicVoices`
- `src/vits_pipeline/prepare_dataset.py`: validates and converts your raw dataset
- `src/vits_pipeline/train_vits.py`: launches Coqui `VITS` training
- `src/vits_pipeline/synthesize.py`: generates sample audio from the trained model

## IndicVoices Hindi Input

Set your Hugging Face token in the environment after you accept the `ai4bharat/IndicVoices` access conditions on Hugging Face:

```bash
export HF_TOKEN=your_token_here
```

Then run:

```bash
bash scripts/run_vits_pipeline.sh
```

The download step will:

- stream only the `hindi` config from `ai4bharat/IndicVoices`
- filter clips by duration and text length
- summarize speakers by usable hours
- choose the top filtered speaker by duration, unless you pin `download.speaker_selection.speaker_id`
- write only that speaker's clips and a local manifest under `data/raw/indicvoices_hindi`

## Generic Dataset Input

The prep pipeline still supports custom manifests too.

Supported source formats:

- `csv` or `tsv` with header columns like `audio_path,text`
- `jsonl` with keys like `{"audio_path": "clips/001.wav", "text": "..."}`
- `ljspeech` style lines like `sample_001|text|normalized_text`

Expected audio behavior:

- audio paths can be absolute or relative to `dataset.source_root`
- the prep step converts every sample to `wav` in `data/processed/vits_dataset/wavs`
- invalid or out-of-range samples are skipped and reported in `dataset_summary.json`

## Quick Start

Install dependencies:

```bash
bash scripts/setup_runpod.sh
```

Edit the config if you want to change the Hindi filtering or speaker selection:

```bash
vim config/tts_config.yaml
```

## Main Outputs

- raw curated Hindi subset: `data/raw/indicvoices_hindi/`
- Hindi download summary: `data/raw/indicvoices_hindi/download_summary.json`
- processed dataset: `data/processed/vits_dataset/`
- train manifest: `data/processed/vits_dataset/metadata_train.csv`
- eval manifest: `data/processed/vits_dataset/metadata_eval.csv`
- dataset summary: `data/processed/vits_dataset/dataset_summary.json`
- checkpoints: `models/vits_dataset/`
- synthesized samples: `reports/synthesis_samples/`

## Notes For Training Quality

- keep clips roughly `1` to `12` seconds when possible
- avoid clipping, room echo, and transcript errors
- keep punctuation and numbers consistent with how you want them spoken
- for low-resource datasets, good transcript quality matters more than fancy tuning

## IndicVoices Notes

- the Hugging Face dataset is gated, so your token must belong to an account that has accepted the dataset conditions
- the Hindi subset is large, so the pipeline streams and saves only the selected speaker instead of downloading everything
- if you later want a multi-speaker Hindi model, we should switch the training pipeline rather than reuse the single-speaker defaults

## RunPod

See `RUNPOD.md` for pod sizing and exact startup commands.
