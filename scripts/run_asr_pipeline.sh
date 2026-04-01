#!/bin/bash
set -euo pipefail

echo "Starting ASR student pipeline..."

echo "Step 1: Building ASR manifests from paired data"
python3 src/data_collection/build_asr_manifest.py --config config/asr_config.yaml

echo "Step 2: Training Whisper-medium ASR student"
python3 src/models/train_asr_student.py --config config/asr_config.yaml

if [ -f data/processed/asr_eval.jsonl ]; then
    echo "Step 3: Evaluating Whisper-medium ASR student"
    python3 src/evaluation/evaluate_asr.py \
        --config config/asr_config.yaml \
        --manifest data/processed/asr_eval.jsonl \
        --model_name_or_path models/asr_student \
        --backend whisper_seq2seq \
        --output_json reports/asr_student_eval.json

    echo "Step 4: Logging ASR experiment"
    python3 src/evaluation/log_experiment.py \
        --task asr \
        --report reports/asr_student_eval.json \
        --config config/asr_config.yaml \
        --output data/experiments/asr_runs.jsonl
fi

echo "ASR student pipeline complete!"
