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
pip install -U "transformers>=5.2" accelerate safetensors sentencepiece
# NOTE: transformers >= 5.2 fixes FP8 Triton kernel bugs (div-by-zero in
# act_quant_kernel, MoE weight_scale_inv shape mismatch).

# vLLM for Phase 2 tensor-parallel generation + speculators for fast Phase 1
pip install "vllm>=0.8" "speculators>=0.1.9"

# --- MoE kernel optimization ------------------------------------------------
export VLLM_MOE_USE_DEEP_GEMM=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# --- Project checkout -----------------------------------------------------
cd "$BIG_DISK/abliterix"
pip install -e ".[vllm]" --no-deps
pip install optuna peft datasets bitsandbytes pydantic-settings questionary hf-transfer psutil kernels rich 2>/dev/null || true

# --- Pre-flight verification (cheap — no 400GB download) ------------------
python scripts/verify_qwen35_397b_fp8.py

# --- Verify GPU count for TP -----------------------------------------------
N_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo "Detected $N_GPUS GPU(s)"
if [ "$N_GPUS" -lt 6 ]; then
  echo "ERROR: Need at least 6 GPUs for TP=6, found $N_GPUS"
  exit 1
fi

# --- Run abliteration (use vLLM config for maximum throughput) ------------
AX_CONFIG=configs/qwen3.5_397b_fp8_vllm.toml abliterix
