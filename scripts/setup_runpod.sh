#!/bin/bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

if command -v apt-get >/dev/null 2>&1; then
  apt-get update
  apt-get install -y ffmpeg libsndfile1 git python3-venv
fi

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi

. .venv/bin/activate
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.4.1 torchaudio==2.4.1
python3 -m pip install -r requirements.txt

mkdir -p data/raw data/processed models reports

printf "RunPod setup complete.\n"
printf "Next: update config/tts_config.yaml to match your dataset, then run bash scripts/run_vits_pipeline.sh\n"
