#!/usr/bin/env bash
# Deploy Granite-4.1-8B abliteration on a single GPU host.
#
# Model: ibm-granite/granite-4.1-8b
# Recipe: configs/granite4.1_8b.toml
#
# Prereqs on the pod:
#   - 1x GPU with >=24 GiB VRAM (48/80/96 GiB recommended for faster batches)
#   - >=80 GB free disk on /workspace
#   - Repo synced to /workspace/abliterix, including datasets/good_1000 and
#     datasets/harmful_1000
#   - /workspace/abliterix/.env with HF_TOKEN and OPENROUTER_API_KEY
#
# Usage:
#   bash /workspace/abliterix/quick_start/deploy_granite41_8b.sh

set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/abliterix}"
CONFIG="${CONFIG:-configs/granite4.1_8b.toml}"
LOG_FILE="${LOG_FILE:-/workspace/run_granite41_8b.log}"
HF_CACHE="${HF_CACHE:-/workspace/hf_cache}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/workspace/checkpoints_granite4.1_8b}"
MIN_GPUS="${MIN_GPUS:-1}"
MIN_VRAM_MIB="${MIN_VRAM_MIB:-24000}"
MIN_DISK_GB="${MIN_DISK_GB:-80}"
MIN_HF_SPEED="${MIN_HF_SPEED:-20}"
MODEL_ID="${MODEL_ID:-ibm-granite/granite-4.1-8b}"
MODEL_CACHE_DIR_NAME="${MODEL_CACHE_DIR_NAME:-models--ibm-granite--granite-4.1-8b}"
SKIP_PREDOWNLOAD="${SKIP_PREDOWNLOAD:-0}"
SKIP_VERIFY="${SKIP_VERIFY:-0}"

mkdir -p /workspace
cd "$REPO_DIR"

echo "=== GPU check ==="
GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
echo "$GPU_INFO"
GPU_COUNT=$(echo "$GPU_INFO" | wc -l | tr -d ' ')
if [ "$GPU_COUNT" -lt "$MIN_GPUS" ]; then
  echo "ERROR: expected >= ${MIN_GPUS} GPUs, found $GPU_COUNT"
  exit 1
fi
VRAM_OK=$(echo "$GPU_INFO" | awk -F',' -v min="$MIN_VRAM_MIB" '$2+0 < min {print "LOW"}' | head -1)
if [ "$VRAM_OK" = "LOW" ]; then
  echo "ERROR: GPU has < ${MIN_VRAM_MIB} MiB VRAM."
  echo "       BF16 weights are ~18 GB; abliteration needs room for activations and KV."
  echo "       Use a >=24 GB GPU, or lower configs/granite4.1_8b.toml max_batch_size."
  exit 1
fi

echo "=== .env check ==="
if [ ! -f "$REPO_DIR/.env" ]; then
  echo "ERROR: $REPO_DIR/.env missing. Needed keys: HF_TOKEN, OPENROUTER_API_KEY"
  exit 1
fi
set -a
# shellcheck disable=SC1091
. "$REPO_DIR/.env"
set +a
: "${HF_TOKEN:?HF_TOKEN not set in .env}"
: "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY not set in .env (llm_judge requires it)}"
export HUGGING_FACE_TOKEN="${HUGGING_FACE_TOKEN:-$HF_TOKEN}"

for ds in datasets/good_1000 datasets/harmful_1000; do
  if [ ! -d "$REPO_DIR/$ds" ]; then
    echo "ERROR: missing $REPO_DIR/$ds -- re-sync the repo without excluding datasets/"
    exit 1
  fi
done
echo "datasets: good_1000 + harmful_1000 present"

echo "=== HF download speed test ==="
SPEED=$(curl -sL -H "Authorization: Bearer ${HF_TOKEN}" \
  -o /dev/null -w '%{speed_download}' --max-time 15 \
  "https://huggingface.co/${MODEL_ID}/resolve/main/config.json" \
  || echo "0")
SPEED_MB=$(awk "BEGIN{printf \"%.0f\", ${SPEED}/1048576}")
echo "HF speed: ${SPEED_MB} MB/s (single-stream sample; hf_transfer uses workers)"
if [ "$SPEED_MB" -lt "$MIN_HF_SPEED" ]; then
  echo "WARN: HF single-stream speed ${SPEED_MB} MB/s below MIN_HF_SPEED=${MIN_HF_SPEED}."
fi

echo "=== Disk check ==="
AVAIL_GB=$(df -BG --output=avail /workspace | tail -1 | tr -d 'G ')
FS_MOUNT=$(df --output=target /workspace | tail -1 | tr -d ' ')
echo "/workspace free: ${AVAIL_GB} GB (mounted from: ${FS_MOUNT})"
if [ "$AVAIL_GB" -lt "$MIN_DISK_GB" ]; then
  echo "ERROR: /workspace has < ${MIN_DISK_GB} GB free."
  exit 1
fi

echo "=== Installing deps ==="
PIP_FLAGS="--break-system-packages --root-user-action=ignore -q"
# Granite 4.1 declares transformers 4.53.3, but this repo targets 5.x and
# GraniteForCausalLM is available there. Keep the install aligned with Abliterix.
# shellcheck disable=SC2086
pip install $PIP_FLAGS \
  "transformers>=5.3,<5.6" \
  "peft>=0.18" \
  "huggingface-hub>=1.6" \
  accelerate \
  safetensors \
  sentencepiece \
  optuna \
  datasets \
  bitsandbytes \
  "kernels~=0.11" \
  pydantic-settings \
  questionary \
  hf-transfer \
  psutil \
  rich

# shellcheck disable=SC2086
pip install $PIP_FLAGS -e . --no-deps
pip uninstall -y --break-system-packages flash-attn 2>/dev/null || true

python3 -c "import torch, transformers, accelerate, peft, optuna; print(f'torch={torch.__version__} transformers={transformers.__version__} accelerate={accelerate.__version__} peft={peft.__version__} gpus={torch.cuda.device_count()}')"

echo "=== CUDA smoke test ==="
python3 - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    sys.exit("ERROR: torch.cuda.is_available() is False -- driver/toolchain mismatch.")
try:
    x = torch.randn(16, 16, device="cuda:0")
    _ = (x @ x).sum().item()
except Exception as e:
    sys.exit(f"ERROR: CUDA kernel smoke-test failed: {e}")
print(f"CUDA smoke-test OK on {torch.cuda.get_device_name(0)}")
PY

mkdir -p "$HF_CACHE" "$CHECKPOINT_DIR"
export HF_HOME="$HF_CACHE"
export HF_HUB_ENABLE_HF_TRANSFER=1

if [ "$SKIP_PREDOWNLOAD" != "1" ]; then
  echo "=== Pre-downloading ${MODEL_ID} to ${HF_CACHE} ==="
  hf download "$MODEL_ID" \
    --repo-type model \
    --max-workers 16 \
    --quiet || {
      echo "ERROR: hf download failed. Check HF_TOKEN and network, then re-run."
      exit 1
    }
  echo "Download complete. Cache size:"
  du -sh "$HF_CACHE/hub/$MODEL_CACHE_DIR_NAME" || true
fi

if [ "$SKIP_VERIFY" != "1" ]; then
  echo "=== Abliterix pre-flight verification (metadata + tokenizer, no full load) ==="
  python3 scripts/verify_model.py \
    --model "$MODEL_ID" \
    --min-vram "$((MIN_VRAM_MIB / 1024))" \
    --min-disk "$MIN_DISK_GB"
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "=== Launching abliterix ==="
echo "Config:      $CONFIG"
echo "Log:         $LOG_FILE"
echo "HF cache:    $HF_CACHE"
echo "Checkpoints: $CHECKPOINT_DIR"
echo

nohup bash -c "AX_CONFIG='$CONFIG' abliterix --optimization.checkpoint-dir='$CHECKPOINT_DIR' 2>&1 | tee '$LOG_FILE'" >/dev/null 2>&1 &
PID=$!

echo "Started PID: $PID"
echo
echo "Monitor with:"
echo "  tail -f $LOG_FILE"
echo "  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"
echo
echo "Early signals:"
echo "  - By trial 25, best refusals should be moving below ~20/200."
echo "  - Good ship candidates should keep KL <= 0.01 with low judge refusals."
echo "  - If refusal stays high, rerun with strength_range = [0.8, 3.0]."
