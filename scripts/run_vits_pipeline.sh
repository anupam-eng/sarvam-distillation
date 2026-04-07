#!/bin/bash
set -euo pipefail

CONFIG_PATH="${1:-config/tts_config.yaml}"
PYTHON_BIN="python3"

if [ -x ".venv/bin/python3" ]; then
  PYTHON_BIN=".venv/bin/python3"
fi

MONITOR_DIR="reports/monitoring"
TRAIN_LOG_PATH="$MONITOR_DIR/train_vits.log"
TRAIN_PID_PATH="$MONITOR_DIR/train_vits.pid"
MONITOR_SUMMARY_PATH="$MONITOR_DIR/checkpoint_monitor_summary.json"

mkdir -p "$MONITOR_DIR"

printf "Step 0/4: downloading and filtering IndicVoices Hindi if enabled\n"
"$PYTHON_BIN" -m src.vits_pipeline.download_indicvoices_hindi --config "$CONFIG_PATH"

printf "Step 1/4: preparing dataset\n"
"$PYTHON_BIN" -m src.vits_pipeline.prepare_dataset --config "$CONFIG_PATH"

printf "Step 2/4: training VITS with checkpoint monitoring\n"
"$PYTHON_BIN" -m src.vits_pipeline.train_vits --config "$CONFIG_PATH" > "$TRAIN_LOG_PATH" 2>&1 &
TRAIN_PID=$!
printf "%s\n" "$TRAIN_PID" > "$TRAIN_PID_PATH"

"$PYTHON_BIN" -m src.vits_pipeline.monitor_checkpoints --config "$CONFIG_PATH" --train-pid "$TRAIN_PID"

set +e
wait "$TRAIN_PID"
TRAIN_EXIT=$?
set -e

if [ "$TRAIN_EXIT" -ne 0 ]; then
  if [ -f "$MONITOR_SUMMARY_PATH" ] && grep -q 'stopped_for_overfit' "$MONITOR_SUMMARY_PATH"; then
    printf "Training was stopped by checkpoint monitor due to overfit signal.\n"
  else
    printf "Training failed. See %s\n" "$TRAIN_LOG_PATH" >&2
    exit "$TRAIN_EXIT"
  fi
fi

printf "Step 3/4: synthesizing eval samples\n"
"$PYTHON_BIN" -m src.vits_pipeline.synthesize --config "$CONFIG_PATH"

printf "Pipeline complete.\n"
