#!/usr/bin/env bash
# Deploy abliterix against MiniMaxAI/MiniMax-M2.5 using the dual-engine
# architecture (HF Phase 1 + vLLM Phase 2) on 4× H100 80GB.
#
# Prerequisites:
#   * 4× H100 80GB (SM90, native FP8 + NVLink for TP)
#   * 512 GB+ disk (model weights ~230GB + HF cache)
#   * The abliterix repo rsync'd to $BIG_DISK/abliterix beforehand.
#   * .env with HF_TOKEN, OPENROUTER_API_KEY rsync'd alongside.
#
# Expected performance:
#   Phase 1 (HF, one-time):  ~10 min (hidden states + steering vectors)
#   Phase 2 (vLLM TP=4):     ~2 min/trial × 50 trials ≈ 2 hours
#   Total:                    ~2.5 hours (vs ~70 hours with HF-only)
#
# Usage:
#   bash deploy_minimax_m25_vllm.sh

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
# MiniMax-M2.5's custom modeling code requires transformers 4.57.x.
pip install -U "transformers>=4.57.1,<5.0" accelerate safetensors sentencepiece

# Flash Attention 2
# IMPORTANT: Do NOT upgrade PyTorch — use the pre-installed version and find
# a matching flash-attn wheel.  Upgrading PyTorch breaks torchvision, peft,
# and transformers ABI compatibility (costs 30+ min to fix).
TORCH_VER=$(python -c "import torch; v=torch.__version__.split('+')[0]; print('.'.join(v.split('.')[:2]))")
PY_VER=$(python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
echo "PyTorch $TORCH_VER, Python $PY_VER — searching for flash-attn wheel..."

# Try prebuilt wheel from mjun0812 (covers torch 2.4–2.11, CUDA 12.x, Linux)
FA_URL=$(curl -s "https://api.github.com/repos/mjun0812/flash-attention-prebuild-wheels/releases" \
  | grep -oP "https://[^\"]+" \
  | grep "torch${TORCH_VER}.*${PY_VER}.*linux_x86_64\.whl" \
  | grep "cu128\|cu126\|cu124" | head -1)

if [ -n "$FA_URL" ]; then
  echo "Installing flash-attn from: $FA_URL"
  pip install "$FA_URL" --no-deps --force-reinstall
else
  echo "No prebuilt wheel found, trying pip install..."
  pip install flash-attn --no-build-isolation 2>/dev/null || echo "flash-attn install failed (will fall back to SDPA)"
fi

# vLLM for Phase 2 tensor-parallel generation
pip install "vllm>=0.8"

# --- MoE kernel optimization ------------------------------------------------
# Use CUTLASS grouped GEMM for MoE expert layers (faster than DeepGEMM on H100).
# DeepGEMM is better for dense linear layers — vLLM auto-selects for those.
# See: https://github.com/vllm-project/vllm/pull/28422
export VLLM_MOE_USE_DEEP_GEMM=0

# --- Project checkout -----------------------------------------------------
cd "$BIG_DISK/abliterix"
pip install -e . --no-deps
pip install optuna peft datasets bitsandbytes pydantic-settings questionary hf-transfer psutil kernels rich 2>/dev/null || true

# --- Pre-flight verification ----------------------------------------------
python scripts/verify_minimax_m25.py

# --- Verify TP=4 is valid for this GPU setup ------------------------------
N_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo "Detected $N_GPUS GPU(s)"
if [ "$N_GPUS" -lt 4 ]; then
  echo "ERROR: Need at least 4 GPUs for TP=4, found $N_GPUS"
  exit 1
fi

# --- Run abliteration (dual-engine: HF Phase 1 → vLLM Phase 2) -----------
AX_CONFIG=configs/minimax_m2.5_vllm.toml abliterix
