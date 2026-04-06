#!/usr/bin/env bash
# Deploy abliterix against MiniMaxAI/MiniMax-M2.5 on a RunPod pod with
# 3× RTX PRO 6000 96GB (or 4× H100 80GB).
#
# Prerequisites:
#   * 3× RTX PRO 6000 96GB (SM89+) or 4× H100 80GB. A100 NOT supported (no FP8).
#   * 512 GB+ disk. On some RunPod templates this is /root, on others
#     /workspace — use whichever partition has hundreds of GB free.
#   * The abliterix repo rsync'd to $BIG_DISK/abliterix beforehand.
#
# MiniMax-M2.5 specifics:
#   * 229B params (10B active), 256 experts, 62 layers, hidden=3072
#   * FP8 snapshot ~230GB (126 safetensors shards)
#   * Requires trust_remote_code=true (custom MiniMaxM2ForCausalLM)
#   * Uses block_sparse_moe naming (same pattern as Phi-3.5-MoE)
#   * Native FP8 GEMM (skip_fp8_dequant=true) — no slow Python dequant
#   * Flash Attention 2 enabled for attention speedup
#
# Usage:
#   bash deploy_minimax_m25.sh

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
# MiniMax-M2.5's custom modeling code (trust_remote_code) is incompatible with
# transformers 5.x (missing OutputRecorder, renamed ROPE_INIT_FUNCTIONS, etc.).
# Pin to transformers 4.57.x which MiniMax officially supports.
pip install -U "transformers>=4.57.1,<5.0" accelerate safetensors sentencepiece
# Flash Attention 2 for attention speedup
pip install flash-attn --no-build-isolation 2>/dev/null || echo "flash-attn install failed (will fall back to SDPA)"

# --- Project checkout -----------------------------------------------------
cd "$BIG_DISK/abliterix"
pip install -e . --no-deps
pip install optuna peft datasets bitsandbytes pydantic-settings questionary hf-transfer psutil kernels rich 2>/dev/null || true

# --- Pre-flight verification (cheap — no 230GB download) ------------------
python scripts/verify_minimax_m25.py

# --- Run abliteration -----------------------------------------------------
AX_CONFIG=configs/minimax_m2.5.toml abliterix
