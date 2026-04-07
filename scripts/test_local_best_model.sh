#!/bin/bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  printf 'Usage: bash scripts/test_local_best_model.sh "देवनागरी टेक्स्ट" [output-wav-path]\n' >&2
  exit 1
fi

TEXT="$1"
OUTPUT_PATH="${2:-reports/local_test_output.wav}"
PYTHON_BIN="python3"

if [ -x ".venv/bin/python3" ]; then
  PYTHON_BIN=".venv/bin/python3"
fi

CHECKPOINT_PATH="/Users/hashteelab/Downloads/sarvam_vits_weights/vits_dataset-April-01-2026_03+10PM-0000000/best_model.pth"
MODEL_CONFIG_PATH="/Users/hashteelab/Downloads/sarvam_vits_weights/vits_dataset-April-01-2026_03+10PM-0000000/config.json"

"$PYTHON_BIN" -m src.vits_pipeline.synthesize_text \
  --text "$TEXT" \
  --checkpoint-path "$CHECKPOINT_PATH" \
  --model-config-path "$MODEL_CONFIG_PATH" \
  --output-path "$OUTPUT_PATH"
