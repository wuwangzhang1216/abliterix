#!/usr/bin/env bash
# Deploy abliterix against stepfun-ai/Step-3.5-Flash on a pod with
# 6× RTX Pro 6000 96GB (576GB total, bf16 weights ~392GB).
#
# Prerequisites:
#   * 6× RTX Pro 6000 96GB minimum.
#   * 512 GB+ disk for model weights (~370GB safetensors).
#   * The abliterix repo rsync'd to $BIG_DISK/abliterix beforehand.
#
# Usage:
#   bash deploy_step3p5_flash.sh

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

# --- Virtual environment ---------------------------------------------------
VENV="$BIG_DISK/venv"
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"
pip install -U pip

# --- Dependencies ---------------------------------------------------------
# CRITICAL: Step-3.5-Flash custom code (modeling_step3p5.py) was written for
# transformers 4.x.  It is INCOMPATIBLE with transformers >= 5.0 due to:
#   - validate_layer_type strict check (layer_types 48 != num_hidden_layers 45)
#   - rope_parameters auto-generation from rope_scaling (list rope_theta breaks)
#   - "default" rope type removed from ROPE_INIT_FUNCTIONS
#   - compute_default_rope_parameters method required but missing
#   - cache_position=None in prepare_inputs_for_generation
# Pinning to 4.55.2 (verified working — model loads + generates correctly).
pip install -U "transformers==4.55.2" accelerate safetensors sentencepiece

# vLLM for Phase 2 tensor-parallel generation.
# NOTE: Do NOT install speculators — it conflicts with vLLM's VllmConfig
# on transformers 4.x.  Phase 1 uses HF pipeline parallelism instead.
pip install "vllm>=0.8"

# CRITICAL: vLLM pulls transformers >= 5.x as a dependency, overwriting our
# 4.55.2 pin.  Re-install 4.55.2 AFTER vLLM to restore compatibility.
pip install "transformers==4.55.2"

# --- Memory optimization ---------------------------------------------------
export VLLM_MOE_USE_DEEP_GEMM=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# --- Project checkout -----------------------------------------------------
cd "$BIG_DISK/abliterix"
pip install -e ".[vllm]" --no-deps
pip install optuna peft datasets bitsandbytes pydantic-settings questionary hf-transfer psutil kernels rich 2>/dev/null || true

# --- Pre-download model (370GB — check disk first) -------------------------
AVAIL_GB=$(df -PBG "$HF_HOME" | awk 'NR==2{gsub(/G/,"",$4); print $4}')
echo "Available disk: ${AVAIL_GB}GB"
if [ "$AVAIL_GB" -lt 400 ]; then
  echo "WARNING: Less than 400GB free — model download may fail!"
fi

echo "Pre-downloading Step-3.5-Flash weights (this may take a while)..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download('stepfun-ai/Step-3.5-Flash', max_workers=4)
print('Download complete.')
" || echo "WARNING: Pre-download failed — will retry during model load."

# --- Verify GPU count for TP -----------------------------------------------
N_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo "Detected $N_GPUS GPU(s)"
if [ "$N_GPUS" -lt 6 ]; then
  echo "ERROR: Need at least 6 GPUs for TP=6, found $N_GPUS"
  exit 1
fi

# --- Patch model code for transformers compat --------------------------------
# Must run AFTER download and BEFORE abliterix to fix incompatibilities.
# Also prevents HF from re-downloading (overwriting) patched files.
python scripts/patch_all_final.py
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# --- Run abliteration ------------------------------------------------------
AX_CONFIG=configs/step3p5_flash.toml abliterix
