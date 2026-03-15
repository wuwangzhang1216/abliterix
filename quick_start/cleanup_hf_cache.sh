#!/usr/bin/env bash
set -euo pipefail

CACHE_ROOT="${HF_HOME:-/root/.cache/huggingface}/hub"
MODEL_CACHE_DIR="${1:-models--Qwen--Qwen3.5-122B-A10B}"
TARGET="$CACHE_ROOT/$MODEL_CACHE_DIR"

printf 'Before:\n'
du -sh "$CACHE_ROOT" 2>/dev/null || true

if [ -d "$TARGET" ]; then
  rm -rf "$TARGET"
  printf 'Deleted %s\n' "$TARGET"
else
  printf 'Not found: %s\n' "$TARGET"
fi

printf 'After:\n'
du -sh "$CACHE_ROOT" 2>/dev/null || true
