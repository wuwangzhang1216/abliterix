#!/usr/bin/env bash
set -euo pipefail

OUT_PATH="${1:-/root/prometheus.env}"

read -r -s -p "OPENROUTER_API_KEY: " OPENROUTER_API_KEY
printf '\n'
read -r -s -p "HUGGING_FACE_TOKEN: " HUGGING_FACE_TOKEN
printf '\n'

cat > "$OUT_PATH" <<EOF
export OPENROUTER_API_KEY="$OPENROUTER_API_KEY"
export HUGGING_FACE_TOKEN="$HUGGING_FACE_TOKEN"
export HF_TOKEN="$HUGGING_FACE_TOKEN"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONUNBUFFERED=1
EOF

chmod 600 "$OUT_PATH"
printf 'Wrote %s\n' "$OUT_PATH"
