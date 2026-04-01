# Evaluation Data Formats

## ASR

Create `jsonl` manifests such as `asr_dev.jsonl` and `asr_test.jsonl`.

Each line should look like:

```json
{"audio_path": "../raw/asr/example.wav", "text": "the reference transcript", "language": "en"}
```

Notes:

- `audio_path` can be absolute or relative to this manifest file
- `text` is the gold transcript
- `language` is optional, but useful for Whisper-based evaluation

## TTS

For TTS evaluation, point `src/evaluation/evaluate_tts.py` at a directory containing matching `.wav` and `.json` files.

Example pair:

- `tts_sample_000001.wav`
- `tts_sample_000001.json`

Example metadata:

```json
{
  "text": "the quick brown fox jumps over the lazy dog",
  "language": "en",
  "sample_rate": 16000
}
```

The evaluation script runs a frozen ASR model on each wav file and compares the transcript against `text`.
