#!/usr/bin/env bash
# Deploy abliterix against google/gemma-4-31B-it on a RunPod H100 80GB pod.
#
# Prerequisites:
#   * H100 80GB pod (SM90, BF16). A100 lacks VRAM headroom for 31B BF16.
#   * 512GB disk available. On some RunPod templates this is /root, on others
#     /workspace — use whichever partition has hundreds of GB free.
#
# Usage:
#   bash deploy_gemma4_31b.sh

set -euo pipefail

# --- Paths: pick the larger of /root vs /workspace -------------------------
BIG_DISK=$(df -PBG /workspace /root 2>/dev/null | awk 'NR>1{print $4" "$6}' | sort -rn | head -1 | awk '{print $2}')
: "${BIG_DISK:=/workspace}"
echo "Using large disk: $BIG_DISK"

export HF_HOME="$BIG_DISK/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export TORCH_HOME="$BIG_DISK/torch_cache"
mkdir -p "$HF_HOME" "$TORCH_HOME"

# --- Dependencies ---------------------------------------------------------
# Gemma 4 needs a transformers release with Gemma4ForConditionalGeneration.
pip install -U "transformers>=5.5.0.dev0" accelerate safetensors sentencepiece

# --- Project checkout -----------------------------------------------------
# Expects the abliterix repo to be rsync'd to $BIG_DISK/abliterix beforehand.
cd "$BIG_DISK/abliterix"
pip install -e .

# --- Pre-flight verification ----------------------------------------------
python scripts/verify_gemma4_31b.py

# --- Run abliteration -----------------------------------------------------
python -m abliterix --config configs/gemma4_31b.toml
