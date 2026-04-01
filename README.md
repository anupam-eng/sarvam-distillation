# sarvam-distillation

Validation plan: see `VALIDATION.md`.

Open-model runtime guide: see `RUNPOD.md`.

## Quick Start

Default local teachers:

- ASR teacher: `openai/whisper-large-v3-turbo` via `faster-whisper`
- TTS teacher: `facebook/mms-tts-eng` via `transformers`

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the ASR pipeline:

```bash
bash scripts/run_asr_pipeline.sh
```

Run the TTS pipeline:

```bash
bash scripts/run_tts_pipeline.sh
```

## Evaluation Inputs

ASR evaluation manifest format: `data/eval/asr_dev.jsonl`

```json
{"audio_path": "../raw/asr/sample.wav", "text": "reference transcript", "language": "en"}
```

TTS evaluation uses generated `.wav` + `.json` pairs in a directory. The sidecar JSON must include at least:

```json
{"text": "reference text", "language": "en"}
```

Sample file-format notes are in `data/eval/README.md`.
