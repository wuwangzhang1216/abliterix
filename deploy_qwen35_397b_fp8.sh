#!/usr/bin/env bash
# Deploy abliterix against Qwen/Qwen3.5-397B-A17B-FP8 on a RunPod pod with
# 6× H100 80GB (or 8× H100 / 8× H200 for extra headroom).
#
# Prerequisites:
#   * 6× H100 80GB (SM90) minimum. A100 is NOT supported — FP8 needs SM89+.
#   * 512 GB+ disk. On some RunPod templates this is /root, on others
#     /workspace — use whichever partition has hundreds of GB free.
#   * The abliterix repo rsync'd to $BIG_DISK/abliterix beforehand.
#
# Usage:
#   bash deploy_qwen35_397b_fp8.sh

set -euo pipefail

# --- Paths: pick the larger of /root vs /workspace -------------------------
BIG_DISK=$(df -PBG /workspace /root 2>/dev/null | awk 'NR>1{print $4" "$6}' | sort -rn | head -1 | awk '{print $2}')
: "${BIG_DISK:=/workspace}"
echo "Using large disk: $BIG_DISK"

# --- Load .env (HF_TOKEN, LLM judge API keys) -----------------------------
if [ -f "$BIG_DISK/abliterix/.env" ]; then
  set -a
  . "$BIG_DISK/abliterix/.env"
  set +a
  echo "Loaded .env from $BIG_DISK/abliterix/.env"
fi

export HF_HOME="$BIG_DISK/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export TORCH_HOME="$BIG_DISK/torch_cache"
mkdir -p "$HF_HOME" "$TORCH_HOME"

# --- Dependencies ---------------------------------------------------------
pip install -U "transformers>=4.46" accelerate safetensors sentencepiece
# NOTE: compressed-tensors pins transformers<5.0.0 which conflicts with
# abliterix (requires transformers 5.x). The Qwen3.5-FP8 checkpoint uses
# native transformers FP8 (block-wise) and does not need compressed-tensors.

# --- Project checkout -----------------------------------------------------
cd "$BIG_DISK/abliterix"
pip install -e .

# --- Pre-flight verification (cheap — no 400GB download) ------------------
python scripts/verify_qwen35_397b_fp8.py

# --- Run abliteration -----------------------------------------------------
AX_CONFIG=configs/qwen3.5_397b_fp8.toml abliterix
