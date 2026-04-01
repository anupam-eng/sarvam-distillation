#!/bin/bash
set -euo pipefail

CONFIG_PATH="${1:-config/tts_config.yaml}"
PYTHON_BIN="python3"

if [ -x ".venv/bin/python3" ]; then
  PYTHON_BIN=".venv/bin/python3"
fi

printf "Step 0/4: downloading and filtering IndicVoices Hindi if enabled\n"
"$PYTHON_BIN" -m src.vits_pipeline.download_indicvoices_hindi --config "$CONFIG_PATH"

printf "Step 1/4: preparing dataset\n"
"$PYTHON_BIN" -m src.vits_pipeline.prepare_dataset --config "$CONFIG_PATH"

printf "Step 2/4: training VITS\n"
"$PYTHON_BIN" -m src.vits_pipeline.train_vits --config "$CONFIG_PATH"

printf "Step 3/4: synthesizing eval samples\n"
"$PYTHON_BIN" -m src.vits_pipeline.synthesize --config "$CONFIG_PATH"

printf "Pipeline complete.\n"
