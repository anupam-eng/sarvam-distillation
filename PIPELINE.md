# Pipeline Plan

## Chosen Models

- ASR student: `openai/whisper-medium`
- TTS student: `VITS`
- TTS teacher default: `Coqui Fairseq VITS` for Indic-native text
- Text source: `ai4bharat/IndicCorpV2`

## Why This Plan

- `Whisper-medium` is the fixed ASR student and does not require a separate ASR teacher.
- `VITS` is the fixed TTS student and can train from generated teacher audio.
- `IndicCorpV2` provides a large monolingual Indic text source for prompt generation.
- `XTTS-v2` was not kept as the default teacher because it does not natively support most Indic native-script text. The repo still keeps optional XTTS support, but the default path uses an Indic-capable Coqui teacher.

## Data Flow

1. `prepare_indiccorp_text.py`
   - loads `IndicCorpV2`
   - filters noisy lines
   - creates deterministic train and eval prompt files
2. `tts_generator.py`
   - generates teacher wav files and JSON sidecars from the prompts
3. `prepare_tts_dataset.py`
   - converts teacher outputs into an LJSpeech-style dataset for Coqui VITS
4. `train_tts_student.py`
   - trains the VITS student with Coqui TTS
5. `generate_tts_student_samples.py`
   - synthesizes eval prompts with the trained VITS student
6. `build_asr_manifest.py`
   - converts paired teacher audio into ASR train and eval manifests
   - optionally merges extra real-speech manifests
7. `train_asr_student.py`
   - fine-tunes `Whisper-medium` on those manifests
8. evaluation
   - `evaluate_tts.py` scores TTS outputs with frozen-ASR backtranscription
   - `evaluate_asr.py` scores the Whisper student on held-out manifests
9. experiment tracking
   - `log_experiment.py` appends run metadata and metrics to JSONL logs
   - `compare_reports.py` compares checkpoints or runs

## Tradeoffs

Pros:

- one shared `(text, audio)` dataset powers both TTS and ASR students
- no dependency on an ASR teacher for the first training cycle
- easy to swap the TTS teacher later while keeping the same students

Cons:

- ASR trained only on generated speech will overfit to synthetic acoustics
- the best ASR quality later will need real speech mixed into `additional_train_manifests`
- a single-speaker teacher voice can narrow TTS and ASR diversity unless you rotate teacher voices later

## Recommended Next Scaling Step

After the first clean run with generated pairs:

1. add real Indic speech manifests to `config/asr_config.yaml`
2. retrain `Whisper-medium` on synthetic + real speech
3. compare against the synthetic-only baseline with `compare_reports.py`
