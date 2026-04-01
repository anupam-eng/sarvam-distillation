#!/bin/bash
# scripts/run_tts_pipeline.sh
set -e

export PYTHONPATH="${PYTHONPATH}:$(pwd)/src/data_collection"

echo "Starting Sarvam TTS Distillation Pipeline..."

# 1. Generating TTS audio from Teacher
echo "Step 1: Running TTS Generator Worker"
touch data/raw/tts_sentences.txt

if [ ! -s data/raw/tts_sentences.txt ]; then
    echo "Warning: No sentences found in data/raw/tts_sentences.txt. Add some text for TTS distillation."
    echo "Example: 'The quick brown fox jumps over the lazy dog.'" > data/raw/tts_sentences.txt
fi

python src/data_collection/tts_generator.py \
    --config config/tts_config.yaml \
    --input_text_file data/raw/tts_sentences.txt \
    --output_dir data/processed/tts_teacher_audio

# 2. Sharding
echo "Step 2: Creating WebDataset Shards"
python src/data_collection/create_shards.py \
    --input_dir data/processed/tts_teacher_audio \
    --output_dir data/shards/tts \
    --shard_prefix tts_shard

# 3. Training
echo "Step 3: Training Distilled TTS Student Model"
python src/models/train_tts_student.py \
    --config config/tts_config.yaml

echo "TTS Pipeline Complete!"
