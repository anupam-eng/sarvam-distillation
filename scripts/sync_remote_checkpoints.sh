#!/bin/bash
set -euo pipefail

REMOTE_HOST="${1:?remote host required}"
REMOTE_PORT="${2:?remote port required}"
SSH_KEY_PATH="${3:?ssh key path required}"
REMOTE_RUN_DIR="${4:?remote run dir required}"
LOCAL_DEST_DIR="${5:?local dest dir required}"
MIN_STEP="${6:-13000}"
POLL_SECONDS="${7:-60}"

mkdir -p "$LOCAL_DEST_DIR"

copy_if_needed() {
  local remote_file="$1"
  local local_file="$LOCAL_DEST_DIR/$(basename "$remote_file")"

  if [ -f "$local_file" ]; then
    return
  fi

  scp -i "$SSH_KEY_PATH" -P "$REMOTE_PORT" "$REMOTE_HOST:$remote_file" "$LOCAL_DEST_DIR/"
}

while true; do
  remote_listing="$(
    ssh -i "$SSH_KEY_PATH" -p "$REMOTE_PORT" "$REMOTE_HOST" \
      "python3 - <<'PY'
from pathlib import Path
import re
run_dir = Path('$REMOTE_RUN_DIR')
min_step = int('$MIN_STEP')
for path in sorted(run_dir.glob('*.pth')):
    name = path.name
    if name.startswith('checkpoint_'):
        m = re.search(r'checkpoint_(\\d+)\\.pth$', name)
        if not m or int(m.group(1)) < min_step:
            continue
    elif not name.startswith('best_model'):
        continue
    print(path)
PY"
  )"

  while IFS= read -r remote_file; do
    [ -n "$remote_file" ] || continue
    copy_if_needed "$remote_file"
  done <<EOF
$remote_listing
EOF

  if [ -f "$LOCAL_DEST_DIR/best_model.pth" ] && [ -f "$LOCAL_DEST_DIR/checkpoint_${MIN_STEP}.pth" ]; then
    touch "$LOCAL_DEST_DIR/.sync_started"
  fi

  sleep "$POLL_SECONDS"
done
