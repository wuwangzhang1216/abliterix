#!/usr/bin/env bash
# Deploy Qwen3.6-27B V4 — manual iterative (second-pass on V1 shipped weights).
#
# Base model: wangzhang/Qwen3.6-27B-abliterated (V1, shipped at 16/100)
# Goal:       single-digit refusals by projecting out residual refusal direction
#
# Strategy: abliterix native iterative is broken with LoRA (IndexError in
# steering.py). V4 gets iterative effect by loading V1's merged checkpoint
# as the base model — abliterix extracts residual refusal direction from
# the already-partially-abliterated model and runs a 30-trial projection
# sweep on top.
#
# Hardware: A100 80GB (sm_80). Assumes fla + causal-conv1d already installed
# from V3 deploy (skip_torch_install=1 path). If on a fresh pod, V3 script
# deps are a strict superset — run deploy_qwen36_27b_v3.sh first, then this.
#
# Usage (A100 pod 154.54.102.34:14360):
#   bash /workspace/abliterix/quick_start/deploy_qwen36_27b_v4.sh

set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/abliterix}"
CONFIG="${CONFIG:-configs/qwen3.6_27b_v4.toml}"
LOG_FILE="${LOG_FILE:-/workspace/run_qwen36_27b_v4.log}"
HF_CACHE="${HF_CACHE:-/workspace/hf_cache}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/workspace/checkpoints_qwen3.6_27b_v4}"
BASE_MODEL_ID="${BASE_MODEL_ID:-wangzhang/Qwen3.6-27B-abliterated}"
BASE_CACHE_DIR="${BASE_CACHE_DIR:-models--wangzhang--Qwen3.6-27B-abliterated}"
MIN_DISK_GB="${MIN_DISK_GB:-100}"

mkdir -p /workspace "$CHECKPOINT_DIR" "$HF_CACHE"
cd "$REPO_DIR"

# ─── 1. Sanity: GPU idle + no stale abliterix ──────────────────────────────
echo "=== GPU state ==="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
if pgrep -f "abliterix --optimization" >/dev/null 2>&1; then
  echo "ERROR: abliterix still running. pkill it first."
  pgrep -af abliterix
  exit 1
fi

# ─── 2. Env / creds ─────────────────────────────────────────────────────────
if [ ! -f "$REPO_DIR/.env" ]; then
  echo "ERROR: $REPO_DIR/.env missing."
  exit 1
fi
set -a; . "$REPO_DIR/.env"; set +a
: "${HF_TOKEN:?HF_TOKEN required}"
: "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY required (llm_judge enabled)}"
export HUGGING_FACE_TOKEN="${HUGGING_FACE_TOKEN:-$HF_TOKEN}"
export HF_HOME="$HF_CACHE"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# ─── 3. Disk check ──────────────────────────────────────────────────────────
AVAIL_GB=$(df -BG --output=avail /workspace | tail -1 | tr -d 'G ')
echo "/workspace free: ${AVAIL_GB} GB"
if [ "$AVAIL_GB" -lt "$MIN_DISK_GB" ]; then
  echo "ERROR: /workspace has < ${MIN_DISK_GB} GB free."
  exit 1
fi

# ─── 4. Datasets + fla sanity (installed from V3) ───────────────────────────
for ds in datasets/good_1000 datasets/harmful_1000; do
  if [ ! -d "$REPO_DIR/$ds" ]; then
    echo "ERROR: missing $REPO_DIR/$ds"
    exit 1
  fi
done
python3 -c "import fla, causal_conv1d; print('fla + causal_conv1d OK')" || {
  echo "ERROR: fla/causal_conv1d missing. Run deploy_qwen36_27b_v3.sh first."
  exit 1
}

# ─── 5. Pre-download V1 base model ──────────────────────────────────────────
if [ -d "$HF_CACHE/hub/$BASE_CACHE_DIR" ]; then
  echo "=== V1 base already cached ==="
  du -sh "$HF_CACHE/hub/$BASE_CACHE_DIR"
else
  echo "=== Downloading $BASE_MODEL_ID (V1 shipped, 54 GB) ==="
  hf download "$BASE_MODEL_ID" --repo-type model --max-workers 16 --quiet
  du -sh "$HF_CACHE/hub/$BASE_CACHE_DIR"
fi

# ─── 6. Clean previous V4 attempt if any ────────────────────────────────────
if [ -f "$CHECKPOINT_DIR/wangzhang--Qwen3--6-27B-abliterated.jsonl" ]; then
  echo "WARN: V4 prior run detected — backing up."
  mv "$CHECKPOINT_DIR"/*.jsonl "$CHECKPOINT_DIR.bak.$(date +%s)" 2>/dev/null || true
fi

# ─── 7. Launch ──────────────────────────────────────────────────────────────
echo "=== Launching V4 second-pass abliteration ==="
echo "Config:       $CONFIG"
echo "Base model:   $BASE_MODEL_ID (V1 shipped, 16/100 refusals)"
echo "Log:          $LOG_FILE"
echo "Checkpoints:  $CHECKPOINT_DIR"
echo "Target:       < 10/100 refusals (single digits)"
echo "Wall-time:    ~2.5h on A100 fla fast path (bs=8)"
echo

nohup bash -c "AX_CONFIG='$CONFIG' abliterix --optimization.checkpoint-dir='$CHECKPOINT_DIR' 2>&1 | tee '$LOG_FILE'" >/dev/null 2>&1 &
PID=$!
echo "Started PID: $PID"
echo
echo "Monitor:"
echo "  tail -f $LOG_FILE"
echo "  grep -E 'Running trial|Refusals:|KL divergence' $LOG_FILE | tail -30"
echo
echo "Ship signals (vs V1 = 16/100):"
echo "  - Any trial ≤ 9/100    → candidate for ship"
echo "  - Best trial ≤ 5/100   → TrevorS-level"
echo "  - Best trial > 13/100  → manual iterative didn't beat single-pass ceiling"
