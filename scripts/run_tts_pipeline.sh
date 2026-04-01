#!/bin/bash
set -euo pipefail

echo "Starting TTS teacher and VITS student pipeline..."

echo "Step 1: Preparing IndicCorp prompts"
python3 src/data_collection/prepare_indiccorp_text.py --config config/tts_config.yaml

echo "Step 2: Generating teacher audio for train split"
python3 src/data_collection/tts_generator.py \
    --config config/tts_config.yaml \
    --input_text_file data/processed/tts_prompts_train.txt \
    --output_dir data/processed/tts_teacher_audio/train

echo "Step 3: Generating teacher audio for eval split"
python3 src/data_collection/tts_generator.py \
    --config config/tts_config.yaml \
    --input_text_file data/processed/tts_prompts_eval.txt \
    --output_dir data/processed/tts_teacher_audio/eval

echo "Step 4: Preparing Coqui VITS dataset"
python3 src/data_collection/prepare_tts_dataset.py --config config/tts_config.yaml

echo "Step 5: Evaluating teacher audio quality"
python3 src/evaluation/evaluate_tts.py \
    --config config/tts_config.yaml \
    --input_dir data/processed/tts_teacher_audio/eval \
    --output_json reports/tts_teacher_eval.json

echo "Step 6: Training VITS student"
python3 src/models/train_tts_student.py --config config/tts_config.yaml

if [ -f models/tts_student/best_model.pth ] && [ -f models/tts_student/config.json ]; then
    echo "Step 7: Generating VITS student eval samples"
    python3 src/models/generate_tts_student_samples.py \
        --config config/tts_config.yaml \
        --input_text_file data/processed/tts_prompts_eval.txt \
        --output_dir data/processed/tts_student_audio/eval

    echo "Step 8: Evaluating VITS student samples"
    python3 src/evaluation/evaluate_tts.py \
        --config config/tts_config.yaml \
        --input_dir data/processed/tts_student_audio/eval \
        --output_json reports/tts_student_eval.json

    echo "Step 9: Logging TTS experiment"
    python3 src/evaluation/log_experiment.py \
        --task tts \
        --report reports/tts_student_eval.json \
        --config config/tts_config.yaml \
        --output data/experiments/tts_runs.jsonl
fi

echo "TTS pipeline complete!"
