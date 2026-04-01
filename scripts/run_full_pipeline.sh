#!/bin/bash
set -euo pipefail

bash scripts/run_tts_pipeline.sh
bash scripts/run_asr_pipeline.sh
