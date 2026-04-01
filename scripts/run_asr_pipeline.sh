#!/bin/bash
# scripts/run_asr_pipeline.sh
set -e

export PYTHONPATH="${PYTHONPATH}:$(pwd)/src/data_collection"

echo "Starting Sarvam ASR Distillation Pipeline..."

# 1. Labeling
echo "Step 1: Running ASR Labeler Worker"
python src/data_collection/asr_labeler.py \
    --config config/asr_config.yaml \
    --audio_dir data/raw/asr \
    --output_dir data/processed/asr_pseudo_labels

# 2. Filtering
echo "Step 2: Filtering Low-Quality Labels"
python src/filtering/filter_quality.py \
    --config config/asr_config.yaml \
    --input_dir data/processed/asr_pseudo_labels \
    --output_dir data/processed/asr_filtered

# 3. Sharding
echo "Step 3: Creating WebDataset Shards"
python src/data_collection/create_shards.py \
    --input_dir data/processed/asr_filtered \
    --output_dir data/shards/asr \
    --shard_prefix asr_shard

# 4. Training
echo "Step 4: Training Distilled ASR Student Model"
# Assuming accelerate is installed via requirements.txt
python -m torch.distributed.launch --nproc_per_node=1 src/models/train_asr_student.py \
    --config config/asr_config.yaml

echo "ASR Pipeline Complete!"
