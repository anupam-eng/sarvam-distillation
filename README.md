# sarvam-distillation

Validation plan: see `VALIDATION.md`.

RunPod/runtime notes: see `RUNPOD.md`.

## Final Stack

- ASR student: `openai/whisper-medium`
- ASR teacher: none for training
- TTS student: `VITS` via `Coqui TTS`
- TTS teacher: Indic-capable `Coqui Fairseq VITS` teacher by default
- Text corpus: `ai4bharat/IndicCorpV2`

`XTTS-v2` support is still available in `src/data_collection/tts_generator.py`, but it is not the default teacher because it does not natively cover most Indic native-script text. The default pipeline uses an Indic-capable Coqui teacher instead.

## Pipeline

The repo now follows this flow:

1. pull prompts from `IndicCorpV2`
2. filter and split them into train/eval prompt files
3. synthesize teacher audio from those prompts
4. build a `VITS` training dataset from the generated `(text, audio)` pairs
5. train the `VITS` student
6. reuse the same generated `(text, audio)` pairs to build ASR manifests
7. train `Whisper-medium` as the ASR student
8. evaluate TTS through ASR backtranscription and evaluate ASR on held-out manifests
9. log each run to experiment JSONL files

## Main Commands

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run the full pipeline:

```bash
bash scripts/run_full_pipeline.sh
```

Run only TTS:

```bash
bash scripts/run_tts_pipeline.sh
```

Run only ASR:

```bash
bash scripts/run_asr_pipeline.sh
```

## Important Paths

- TTS prompt files:
  - `data/processed/tts_prompts_train.txt`
  - `data/processed/tts_prompts_eval.txt`
- Teacher audio:
  - `data/processed/tts_teacher_audio/train`
  - `data/processed/tts_teacher_audio/eval`
- Coqui VITS dataset:
  - `data/processed/tts_student_dataset`
- ASR manifests:
  - `data/processed/asr_train.jsonl`
  - `data/processed/asr_eval.jsonl`
- Experiment logs:
  - `data/experiments/tts_runs.jsonl`
  - `data/experiments/asr_runs.jsonl`

## Config Files

- `config/tts_config.yaml`
  - controls `IndicCorpV2` prompt preparation
  - controls teacher TTS model selection
  - controls `VITS` student training
- `config/asr_config.yaml`
  - controls ASR manifest building
  - controls `Whisper-medium` training and evaluation

## Notes

- The ASR student is trained without an ASR teacher.
- The current ASR pipeline assumes one language per run for Whisper prefix tokens.
- The strongest ASR version of this project will still come from mixing generated speech with real speech later.
- `XTTS-v2` remains optional if you decide to switch to supported languages or use a transliteration bridge.
