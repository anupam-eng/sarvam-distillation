# Validation Strategy

This repo currently has training and data generation scripts, but no reliable way to tell whether distillation is helping. The goal of this strategy is to measure three things separately:

- teacher data quality
- student learning progress
- real task quality on held-out evaluation sets

The same strategy works whether the teacher is Sarvam, Whisper, Qwen, XTTS, Piper, or another open model.

## Principles

- Never validate only against the teacher's own pseudo-labels.
- Keep a small gold set with human-verified references.
- Freeze evaluation splits before running large-scale data generation.
- Track both objective metrics and quick human review.
- Compare against a baseline student, not only the latest run.

## Dataset Layout

Create three evaluation tiers for each task.

### 1. Smoke set

- Size: 20 to 30 examples
- Purpose: catch broken preprocessing, tokenization, audio corruption, or degenerate generations
- Use: every code change and every short training run

### 2. Dev set

- Size: 100 to 300 examples
- Purpose: model selection, prompt selection, filtering threshold tuning, teacher selection
- Use: frequent evaluation during experimentation

### 3. Final test set

- Size: 300 to 1000 examples, depending on how much labeled data you can afford
- Purpose: final reporting only
- Use: only after major checkpoints or before promoting a model

The dev and test sets must be disjoint from the raw data used to generate training pseudo-labels.

## ASR Validation

### Gold data

Build a small manually verified ASR set.

- 30 to 60 minutes total is enough to start
- include clean speech, noisy speech, short clips, long clips, and at least two speaking styles
- store audio plus a normalized transcript reference

### Metrics

Primary metrics:

- WER on the gold dev set
- WER on the gold test set

Secondary metrics:

- CER when transcripts are short or noisy
- teacher-student agreement on unlabeled holdout audio
- filtering retention rate
- transcript empty-rate
- average generated transcript length

### Teacher quality checks

Before training the student, score the teacher-generated pseudo-labels.

- sample 50 to 100 pseudo-labeled training examples
- manually inspect transcript correctness
- measure how many samples were dropped by filtering
- inspect confidence and duration distributions
- review common failure modes: hallucination, language mismatch, truncation, repeated text

### Student evaluation schedule

Run ASR evaluation at these points.

- before training: baseline student checkpoint
- after each major checkpoint
- after changing teacher model
- after changing filtering thresholds

### ASR acceptance gates

Use these gates to decide if a run is worth keeping.

- smoke set WER does not regress badly versus the previous best
- dev set WER improves over the baseline student
- empty transcript rate stays near zero
- no obvious increase in hallucinations on manual spot-checks

## TTS Validation

TTS needs both objective proxies and human checks. In this repo, objective speech quality metrics are not implemented, so start with intelligibility-first validation.

### Gold data

Create a fixed prompt set for evaluation.

- 100 to 300 prompts for dev
- 300 to 500 prompts for final test
- mix short, medium, and long prompts
- include numbers, names, punctuation, and tongue-twister-like prompts

If you later support multi-speaker training, freeze a separate prompt list per speaker.

### Metrics

Primary metrics:

- intelligibility WER: run a strong frozen ASR model on generated audio and compare to the target text
- generation success rate: percent of prompts that produce valid audio

Secondary metrics:

- duration ratio: generated duration divided by expected text length or teacher duration
- clipping or silence rate
- average loudness and sample-rate consistency
- teacher-student duration gap on the same prompts

Optional later metrics:

- speaker similarity if you support multi-speaker data
- MOS or CMOS human ratings
- embedding similarity to teacher audio

### Human review for TTS

For every serious checkpoint, review 20 to 30 prompts by listening.

Score each sample on:

- intelligibility
- naturalness
- stability across long prompts
- pronunciation of rare words and numbers
- artifacts: buzzing, clipping, repetition, long pauses

Use a 1 to 5 score or a simple pass/fail rubric.

### TTS acceptance gates

- generation success rate is above 95 percent on the smoke set
- ASR-backtranscription WER improves over the baseline student
- no widespread silence, clipping, or repeated-phrase failures
- human review shows equal or better intelligibility than the previous best checkpoint

## Distillation-Specific Comparisons

Because this project is about distillation, compare the teacher and student on the same frozen holdout inputs.

For ASR:

- teacher WER on gold dev/test
- student WER on gold dev/test
- student-teacher agreement on unlabeled holdout audio

For TTS:

- teacher generation success rate
- student generation success rate
- teacher and student ASR-backtranscription WER on the same prompt list
- side-by-side listening on the same prompts

This lets you separate two questions:

- is the teacher good enough?
- is the student actually approaching the teacher?

## Experiment Tracking

Track every run with the same minimal schema.

- teacher model
- student model
- training data size
- filtering thresholds
- checkpoint step or epoch
- ASR dev/test WER
- TTS backtranscription WER
- generation success rate
- notes from manual review

Keep one CSV or JSONL file of results so regressions are visible.

## Recommended Starting Baseline

If you switch to open models for testing, start with this baseline.

- ASR teacher: `openai/whisper-large-v3-turbo`
- ASR student: current `facebook/wav2vec2-base`
- TTS teacher: `Piper` for quick local tests or `coqui/XTTS-v2` for stronger quality
- TTS student: current `microsoft/speecht5_tts`

Then validate with:

- ASR gold dev/test WER
- TTS backtranscription WER using a frozen ASR model
- 20-sample human listening sheet per checkpoint

## Minimum Viable Validation

If you want the smallest setup that still gives trustworthy signal, do this first.

1. Build an ASR gold dev set with 50 to 100 clips.
2. Build a TTS prompt dev set with 100 prompts.
3. Evaluate ASR with WER.
4. Evaluate TTS with frozen-ASR backtranscription WER.
5. Listen to 20 TTS samples and inspect 20 ASR outputs manually.
6. Store results in one experiment log file.

That is enough to know whether distillation is improving, flat, or regressing.

## Next Repo Changes To Implement

The highest-value implementation order is:

1. add fixed `data/eval/asr_dev` and `data/eval/tts_dev` conventions
2. add an ASR evaluation script that computes WER and CER on a labeled set
3. add a TTS evaluation script that generates audio, runs frozen ASR, and reports backtranscription WER
4. add a simple experiment-results CSV or JSONL logger
5. optionally add a manual review template for checkpoint comparisons
