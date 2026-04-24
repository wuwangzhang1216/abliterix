#!/usr/bin/env bash
# Deploy Qwen3.6-27B V3 iterative abliteration — target single-digit refusals.
#
# V1 shipped at 16/100 (wangzhang/Qwen3.6-27B-abliterated). V3 runs two
# iterative extract-project cycles from that starting point to peel residual
# refusal circuits that one-pass projection missed.
#
# Expected final: 5-8/100 refusals @ cumulative KL ≈ 0.03-0.04.
#
# Hardware targets (this script auto-detects and adapts):
#   - A100 SXM 80 GB (sm_80, driver 535-565)       → torch 2.8+cu124, bs=8
#   - H100 80 GB (sm_90, driver 545+)              → torch 2.8+cu124, bs=8
#   - RTX Pro 6000 Blackwell 96 GB (sm_120)        → torch 2.10+cu129, bs=4
#
# Prereqs on the pod:
#   - Any CUDA-capable GPU ≥ 80 GB VRAM
#   - ≥ 120 GB disk on /workspace (55 GB weights + 65 GB checkpoints/logs)
#   - Repo synced to /workspace/abliterix with datasets/ present
#   - .env with HF_TOKEN + OPENROUTER_API_KEY
#
# Usage:
#   bash /workspace/abliterix/quick_start/deploy_qwen36_27b_v3.sh

set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/abliterix}"
CONFIG="${CONFIG:-configs/qwen3.6_27b_v3.toml}"
LOG_FILE="${LOG_FILE:-/workspace/run_qwen36_27b_v3.log}"
HF_CACHE="${HF_CACHE:-/workspace/hf_cache}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/workspace/checkpoints_qwen3.6_27b_v3}"
MIN_GPUS="${MIN_GPUS:-1}"
MIN_VRAM_MIB="${MIN_VRAM_MIB:-80000}"      # A100-80 floor
MIN_DISK_GB="${MIN_DISK_GB:-100}"
MIN_HF_SPEED="${MIN_HF_SPEED:-30}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.6-27B}"
MODEL_CACHE_DIR_NAME="${MODEL_CACHE_DIR_NAME:-models--Qwen--Qwen3.6-27B}"
SKIP_PREDOWNLOAD="${SKIP_PREDOWNLOAD:-0}"
SKIP_TORCH_INSTALL="${SKIP_TORCH_INSTALL:-0}"

mkdir -p /workspace
cd "$REPO_DIR"

# ─── 1. GPU + driver detection ──────────────────────────────────────────────
echo "=== GPU + driver detection ==="
GPU_INFO=$(nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader)
echo "$GPU_INFO"
GPU_COUNT=$(echo "$GPU_INFO" | wc -l | tr -d ' ')
if [ "$GPU_COUNT" -lt "$MIN_GPUS" ]; then
  echo "ERROR: need >= ${MIN_GPUS} GPU, found $GPU_COUNT"; exit 1
fi

GPU_NAME=$(echo "$GPU_INFO" | head -1 | awk -F',' '{gsub(/^ +/,"",$1); print $1}')
DRIVER=$(echo "$GPU_INFO" | head -1 | awk -F',' '{gsub(/^ +/,"",$2); print $2}')
DRIVER_MAJOR=$(echo "$DRIVER" | awk -F'.' '{print $1}')
VRAM_MIB=$(echo "$GPU_INFO" | head -1 | awk -F',' '{gsub(/^ +| +$/,"",$3); gsub(/ MiB/,"",$3); print $3}')
SM=$(echo "$GPU_INFO" | head -1 | awk -F',' '{gsub(/^ +| +$/,"",$4); print $4}')

echo "GPU: $GPU_NAME | driver=$DRIVER (major $DRIVER_MAJOR) | VRAM=${VRAM_MIB} MiB | sm=$SM"

if [ "$VRAM_MIB" -lt "$MIN_VRAM_MIB" ]; then
  echo "ERROR: GPU has < ${MIN_VRAM_MIB} MiB VRAM — 27B BF16 needs 54 GB + headroom."
  exit 1
fi

# Pick CUDA toolchain based on driver major:
#   ≥ 570 → cu129 (Blackwell/Hopper with fresh driver) + torch 2.10
#   ≥ 550 → cu124 + torch 2.8 (common A100/H100 vast.ai image)
#   ≥ 535 → cu121 + torch 2.6
#   <  535 → fail (too old for transformers 5.5)
if   [ "$DRIVER_MAJOR" -ge 570 ]; then TORCH_CUDA="cu129"; TORCH_PIN=">=2.10.0,<2.11.0";
elif [ "$DRIVER_MAJOR" -ge 550 ]; then TORCH_CUDA="cu124"; TORCH_PIN=">=2.8.0,<2.9.0";
elif [ "$DRIVER_MAJOR" -ge 535 ]; then TORCH_CUDA="cu121"; TORCH_PIN=">=2.6.0,<2.7.0";
else
  echo "ERROR: driver $DRIVER is below 535 — required for torch 2.6+/transformers 5.5+."
  exit 1
fi
echo "Toolchain pick: torch $TORCH_PIN +$TORCH_CUDA"

# ─── 2. .env check ──────────────────────────────────────────────────────────
if [ ! -f "$REPO_DIR/.env" ]; then
  echo "ERROR: $REPO_DIR/.env missing. Needed: HF_TOKEN, OPENROUTER_API_KEY"
  exit 1
fi
set -a
# shellcheck disable=SC1091
. "$REPO_DIR/.env"
set +a
: "${HF_TOKEN:?HF_TOKEN not set in .env}"
: "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY not set (iterative needs llm_judge)}"
export HUGGING_FACE_TOKEN="${HUGGING_FACE_TOKEN:-$HF_TOKEN}"

# ─── 2b. Dataset check ──────────────────────────────────────────────────────
for ds in datasets/good_1000 datasets/harmful_1000; do
  if [ ! -d "$REPO_DIR/$ds" ]; then
    echo "ERROR: missing $REPO_DIR/$ds — re-sync the repo including datasets/"
    exit 1
  fi
done
echo "datasets: good_1000 + harmful_1000 present"

# ─── 3. Disk check ──────────────────────────────────────────────────────────
AVAIL_GB=$(df -BG --output=avail /workspace | tail -1 | tr -d 'G ')
FS_MOUNT=$(df --output=target /workspace | tail -1 | tr -d ' ')
echo "/workspace free: ${AVAIL_GB} GB (mounted from: ${FS_MOUNT})"
if [ "$FS_MOUNT" = "/" ]; then
  echo "NOTE: /workspace on container overlay (ephemeral) — push to HF before teardown."
fi
if [ "$AVAIL_GB" -lt "$MIN_DISK_GB" ]; then
  echo "ERROR: /workspace has < ${MIN_DISK_GB} GB free."
  echo "       Need: 55 GB weights + 30 GB per iteration for judge cache + logs"
  exit 1
fi

# ─── 4. Install deps ────────────────────────────────────────────────────────
PIP_FLAGS="--break-system-packages --root-user-action=ignore -q"

if [ "$SKIP_TORCH_INSTALL" != "1" ]; then
  echo "=== Installing torch ($TORCH_PIN+$TORCH_CUDA) ==="
  # Reinstall torch only if the installed version doesn't match the expected CUDA.
  CURRENT_TORCH=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "none")
  WANT_CUDA_NUMERIC=$(echo "$TORCH_CUDA" | sed 's/cu//' | sed 's/^\(..\)\(.\)$/\1.\2/; s/^\(..\)\(..\)$/\1.\2/')
  if [ "$CURRENT_TORCH" != "$WANT_CUDA_NUMERIC" ]; then
    echo "Installed torch has cuda=$CURRENT_TORCH; installing $TORCH_PIN+$TORCH_CUDA"
    # shellcheck disable=SC2086
    pip install $PIP_FLAGS --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" \
      "torch${TORCH_PIN}" "torchvision" "torchaudio"
  else
    echo "Installed torch already has cuda=$CURRENT_TORCH — keeping."
  fi
fi

echo "=== Installing abliterix deps ==="
# shellcheck disable=SC2086
pip install $PIP_FLAGS \
  "transformers>=5.5,<5.6" \
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

python3 -c "
import torch, transformers, accelerate, peft, optuna
print(f'torch={torch.__version__} cuda={torch.version.cuda}')
print(f'transformers={transformers.__version__} accelerate={accelerate.__version__}')
print(f'peft={peft.__version__} optuna={optuna.__version__}')
print(f'gpus={torch.cuda.device_count()}  device_name={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')
print(f'cuda_avail={torch.cuda.is_available()}')
"

# ─── 4b. CUDA smoke test ────────────────────────────────────────────────────
python3 - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    sys.exit("ERROR: torch.cuda.is_available() is False — driver/toolchain mismatch.")
x = torch.randn(16, 16, device="cuda:0")
_ = (x @ x).sum().item()
print(f"CUDA smoke-test OK on {torch.cuda.get_device_name(0)}")
PY

# ─── 5. HF cache + pre-download ─────────────────────────────────────────────
mkdir -p "$HF_CACHE" "$CHECKPOINT_DIR"
export HF_HOME="$HF_CACHE"
export HF_HUB_ENABLE_HF_TRANSFER=1

if [ "$SKIP_PREDOWNLOAD" != "1" ]; then
  if [ -d "$HF_CACHE/hub/$MODEL_CACHE_DIR_NAME" ]; then
    echo "=== HF cache already has $MODEL_ID ==="
    du -sh "$HF_CACHE/hub/$MODEL_CACHE_DIR_NAME" || true
  else
    echo "=== Pre-downloading $MODEL_ID to $HF_CACHE (hf_transfer 16 workers) ==="
    hf download "$MODEL_ID" --repo-type model --max-workers 16 --quiet
    du -sh "$HF_CACHE/hub/$MODEL_CACHE_DIR_NAME"
  fi
fi

# ─── 6. Env exports ─────────────────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# Clean start — remove any stale V3 checkpoint log so iterative rebuilds fresh.
# (Keep HF cache + V1/V2 checkpoints; they're in different dirs.)
if [ -f "$CHECKPOINT_DIR/Qwen--Qwen3--6-27B.jsonl" ]; then
  echo "WARN: V3 checkpoint already has prior trials — backing up and starting fresh."
  mv "$CHECKPOINT_DIR/Qwen--Qwen3--6-27B.jsonl" \
     "$CHECKPOINT_DIR/Qwen--Qwen3--6-27B.jsonl.$(date +%s).bak"
fi

# ─── 7. Launch abliterix V3 ─────────────────────────────────────────────────
echo "=== Launching abliterix V3 (iterative, max_iterations=2) ==="
echo "Config:         $CONFIG"
echo "Log:            $LOG_FILE"
echo "Checkpoints:    $CHECKPOINT_DIR"
echo "GPU:            $GPU_NAME (sm_$SM, driver $DRIVER)"
echo "Target:         <10/100 refusals (V1 shipped at 16/100)"
echo

nohup bash -c "AX_CONFIG='$CONFIG' abliterix --optimization.checkpoint-dir='$CHECKPOINT_DIR' 2>&1 | tee '$LOG_FILE'" >/dev/null 2>&1 &
PID=$!
echo "Started PID: $PID"
echo
echo "Monitor:"
echo "  tail -f $LOG_FILE"
echo "  grep -c 'Running trial' $LOG_FILE    # trials completed per iteration"
echo "  grep 'Iteration' $LOG_FILE           # iterative milestones"
echo
echo "Ship-candidate signals (vs V1 = 16/100):"
echo "  - Iter 1 best refusals ≤ 12/100 (meaningful improvement over V1)"
echo "  - Iter 2 best refusals ≤ 6/100 (DeepRefusal-peel working)"
echo "  - Cumulative KL < 0.04 (quality preserved)"
echo
echo "Hard cutoffs — kill if:"
echo "  - Iter 1 best refusals > 18/100 after 20 trials (worse than V1 → recipe regressed)"
echo "  - Iter 2 cosine similarity to iter 1 direction > 0.85 within first 5 trials"
echo "    (direction saturated — early-stop via convergence threshold)"
echo
echo "Expected wall-time on this GPU:"
case "$GPU_NAME" in
  *A100*)             echo "  ~2.5h/iter × 2 = ~5h  (cost: ~\$7-10 @ vast.ai)" ;;
  *H100*)             echo "  ~2h/iter × 2   = ~4h  (cost: ~\$12-16 @ vast.ai)" ;;
  *"PRO 6000"*|*Blackwell*)  echo "  ~6h/iter × 2 = ~12h (cost: ~\$20 @ vast.ai)" ;;
  *)                  echo "  unknown GPU — monitor and measure" ;;
esac
