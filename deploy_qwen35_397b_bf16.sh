#!/usr/bin/env bash
# Deploy abliterix against Qwen/Qwen3.5-397B-A17B (BF16) on 8× H200 141GB.
#
# BF16 weights: ~794GB → fits in 8× H200 (1128GB) with 334GB headroom.
# No FP8, no DeepGEMM, no Triton kernel race — clean bf16 path.
#
# Prerequisites:
#   * 8× H200 141GB
#   * 512 GB+ disk (model download ~794GB)
#   * The abliterix repo rsync'd to $BIG_DISK/abliterix beforehand.
#
# Usage:
#   bash deploy_qwen35_397b_bf16.sh

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

# --- Project checkout -----------------------------------------------------
cd "$BIG_DISK/abliterix"
pip install -e .

# --- Run abliteration -----------------------------------------------------
AX_CONFIG=configs/qwen3.5_397b_bf16.toml abliterix
