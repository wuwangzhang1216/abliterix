#!/usr/bin/env bash
# Deploy Qwen3.6-27B abliteration — single 1× RTX Pro 6000 96GB GDDR7.
#
# Qwen3.6-27B is a dense hybrid VLM + GatedDeltaNet + full-attention model
# (model_class: Qwen3_5ForConditionalGeneration, same family as Qwen3.5-397B
# but DENSE — no MoE). 64 layers = 48 GDN + 16 full-attn, BF16 ≈ 54 GB.
#
# See configs/qwen3.6_27b.toml for the full recipe rationale.
#
# Prereqs on the pod:
#   - 1× RTX Pro 6000 96GB GDDR7 (or swap to H200 141GB / B200 192GB — bump
#     max_memory in the TOML if you switch)
#   - ≥120 GB disk on /workspace (BF16 weights ~55 GB + HF cache + checkpoints)
#   - Repo synced to /workspace/abliterix (incl. datasets/good_1000 and
#     datasets/harmful_1000)
#   - .env with HF_TOKEN + OPENROUTER_API_KEY
#
# Usage on the pod:
#   bash /workspace/abliterix/quick_start/deploy_qwen36_27b.sh
#
# Expected runtime on 1× RTX Pro 6000:
#   Download   : ~10 min @ 100 MB/s (55 GB weights, hf_transfer 16 workers)
#   Phase 1    : hidden-state extraction over 800 prompts — ~8-12 min
#   Phase 2    : 30 trials × ~4-5 min = ~2-2.5 h
#   Total      : ~2.5-3 h once weights are local. Cost ~$4-5 @ RunPod $1.6/h.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/abliterix}"
CONFIG="${CONFIG:-configs/qwen3.6_27b.toml}"
LOG_FILE="${LOG_FILE:-/workspace/run_qwen36_27b.log}"
HF_CACHE="${HF_CACHE:-/workspace/hf_cache}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/workspace/checkpoints_qwen3.6_27b}"
MIN_GPUS="${MIN_GPUS:-1}"
# RTX Pro 6000 96GB → ~96000 MiB. H200 141GB → 141000. H100 80GB → 80000.
MIN_VRAM_MIB="${MIN_VRAM_MIB:-90000}"
MIN_HF_SPEED="${MIN_HF_SPEED:-30}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.6-27B}"
MODEL_CACHE_DIR_NAME="${MODEL_CACHE_DIR_NAME:-models--Qwen--Qwen3.6-27B}"
SKIP_PREDOWNLOAD="${SKIP_PREDOWNLOAD:-0}"
SKIP_SHAPE_PROBE="${SKIP_SHAPE_PROBE:-0}"

mkdir -p /workspace
cd "$REPO_DIR"

# ─── 1. GPU sanity check ─────────────────────────────────────────────────────
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
  echo "       BF16 weights need ~55 GB; recipe targets RTX Pro 6000 96GB."
  echo "       For H100 80GB: re-run with MIN_VRAM_MIB=80000 AND edit"
  echo "       configs/qwen3.6_27b.toml: max_memory=\"76GiB\" (batch stays at 4)."
  exit 1
fi

# ─── 2. .env check ───────────────────────────────────────────────────────────
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

# ─── 2b. Dataset check ───────────────────────────────────────────────────────
for ds in datasets/good_1000 datasets/harmful_1000; do
  if [ ! -d "$REPO_DIR/$ds" ]; then
    echo "ERROR: missing $REPO_DIR/$ds — re-sync the repo WITHOUT excluding datasets/"
    exit 1
  fi
done
echo "datasets: good_1000 + harmful_1000 present"

# ─── 3. Network speed check ──────────────────────────────────────────────────
echo "=== HF download speed test ==="
SPEED=$(curl -sL -H "Authorization: Bearer ${HF_TOKEN}" \
  -o /dev/null -w '%{speed_download}' --max-time 15 \
  "https://huggingface.co/${MODEL_ID}/resolve/main/config.json" \
  || echo "0")
SPEED_MB=$(awk "BEGIN{printf \"%.0f\", ${SPEED}/1048576}")
echo "HF speed: ${SPEED_MB} MB/s (single-stream sample — multi-worker is 8-20× faster)"
if [ "$SPEED_MB" -lt "$MIN_HF_SPEED" ]; then
  echo "WARN: HF single-stream speed ${SPEED_MB} MB/s below MIN_HF_SPEED=${MIN_HF_SPEED}."
  echo "      Real download uses 16 workers; usually fine. Continuing."
fi

# ─── 4. Disk space check ─────────────────────────────────────────────────────
echo "=== Disk check ==="
AVAIL_GB=$(df -BG --output=avail /workspace | tail -1 | tr -d 'G ')
FS_MOUNT=$(df --output=target /workspace | tail -1 | tr -d ' ')
echo "/workspace free: ${AVAIL_GB} GB (mounted from: ${FS_MOUNT})"
if [ "$FS_MOUNT" = "/" ]; then
  echo "NOTE: /workspace is on container root (no dedicated volume) — wiped on pod destruction."
  echo "      Push abliterated model to HF Hub before tearing down."
fi
MIN_DISK_GB="${MIN_DISK_GB:-120}"
if [ "$AVAIL_GB" -lt "$MIN_DISK_GB" ]; then
  echo "ERROR: /workspace has < ${MIN_DISK_GB} GB free. Need 55 GB model + 65 GB checkpoints/logs."
  exit 1
fi

# ─── 5. Install deps ─────────────────────────────────────────────────────────
echo "=== Installing deps ==="
PIP_FLAGS="--break-system-packages --root-user-action=ignore -q"

# Qwen3_5ForConditionalGeneration lives in the transformers 5.x family
# (same class sister Qwen3.5-397B uses). 4.57.x declared in the model's
# config.json is the UPLOAD version, NOT a runtime requirement — loading
# needs the 5.x code paths for GatedDeltaNet + hybrid layer handling.
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

python3 -c "import torch, transformers, accelerate, peft, optuna; \
  print(f'torch={torch.__version__} transformers={transformers.__version__} accelerate={accelerate.__version__} peft={peft.__version__} gpus={torch.cuda.device_count()}')"

# ─── 5b. CUDA smoke test ─────────────────────────────────────────────────────
python3 - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    sys.exit("ERROR: torch.cuda.is_available() is False — driver/toolchain mismatch.")
try:
    x = torch.randn(16, 16, device="cuda:0")
    _ = (x @ x).sum().item()
except Exception as e:
    sys.exit(f"ERROR: CUDA kernel smoke-test failed: {e}")
print(f"CUDA smoke-test OK on {torch.cuda.get_device_name(0)}")
PY

# ─── 6. HF cache + pre-download ──────────────────────────────────────────────
mkdir -p "$HF_CACHE" "$CHECKPOINT_DIR"
export HF_HOME="$HF_CACHE"
export HF_HUB_ENABLE_HF_TRANSFER=1

if [ "$SKIP_PREDOWNLOAD" != "1" ]; then
  echo "=== Pre-downloading ${MODEL_ID} to ${HF_CACHE} (hf_transfer, 16 workers) ==="
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

# ─── 5c. GDN out_proj shape probe ────────────────────────────────────────────
# Verify the shape-guard assumption in configs/qwen3.6_27b.toml. Reads just
# the config + one weight slice from the local cache (fast, no full model load).
if [ "$SKIP_SHAPE_PROBE" != "1" ]; then
  echo "=== GDN out_proj shape probe ==="
  python3 - <<PY
import json, os, glob, sys
from safetensors import safe_open
cache_root = os.path.expanduser("${HF_CACHE}/hub/${MODEL_CACHE_DIR_NAME}")
snap = sorted(glob.glob(os.path.join(cache_root, "snapshots", "*")))
if not snap:
    sys.exit(f"ERROR: no snapshot under {cache_root}")
snap = snap[-1]
with open(os.path.join(snap, "config.json")) as f:
    cfg = json.load(f)
tcfg = cfg.get("text_config", cfg)
hidden = tcfg["hidden_size"]
nv = tcfg.get("linear_num_value_heads")
hd = tcfg.get("linear_value_head_dim")
print(f"hidden_size={hidden}, linear_num_value_heads={nv}, linear_value_head_dim={hd}")
shards = sorted(glob.glob(os.path.join(snap, "*.safetensors")))
key_hit = None
for shard in shards:
    with safe_open(shard, framework="pt") as sf:
        for k in sf.keys():
            if k.endswith("linear_attn.out_proj.weight"):
                key_hit = (shard, k, sf.get_slice(k).get_shape())
                break
    if key_hit:
        break
if not key_hit:
    print("WARN: no linear_attn.out_proj found — is this really the hybrid-GDN checkpoint?")
    sys.exit(0)
shard, k, shape = key_hit
print(f"{k}: shape={tuple(shape)}")
if shape[0] == hidden:
    print(f"OK: weight.shape[0]={shape[0]} == hidden_size={hidden} — steering shape-guard will accept GDN out_proj (all 64 layers steerable via attn.o_proj).")
else:
    print(f"WARN: weight.shape[0]={shape[0]} != hidden_size={hidden}.")
    print("     steering.py shape-guard WILL SKIP GDN out_proj (48 of 64 layers).")
    print("     Fallback still works (16 full-attn o_proj + 64 mlp.down_proj),")
    print("     but consider adding 'linear_attn.out_proj' to disabled_components")
    print("     so the optimiser doesn't allocate budget to a no-op knob.")
PY
fi

# ─── 7. Env exports for the run ──────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# ─── 8. Launch abliterix ─────────────────────────────────────────────────────
echo "=== Launching abliterix ==="
echo "Config:         $CONFIG"
echo "Log:            $LOG_FILE"
echo "HF cache:       $HF_CACHE"
echo "Checkpoints:    $CHECKPOINT_DIR"
echo "GPU:            $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo

nohup bash -c "AX_CONFIG='$CONFIG' abliterix --optimization.checkpoint-dir='$CHECKPOINT_DIR' 2>&1 | tee '$LOG_FILE'" >/dev/null 2>&1 &
PID=$!
echo "Started PID: $PID"
echo
echo "Monitor with:"
echo "  tail -f $LOG_FILE"
echo "  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"
echo
echo "Ship-candidate signals:"
echo "  - Trial KL ≤ 0.005 with refusals ≤ 5/100"
echo "  - attn.o_proj.max_weight in [2, 5] (covers all 64 layers — amortised)"
echo "  - mlp.down_proj.max_weight in [2, 4]"
echo
echo "Hard cutoffs — kill run if:"
echo "  - Trial 10 best refusals > 40/100 (recipe wrong — revisit disabled_components)"
echo "  - Trial 20 best KL > 0.02 (search space not converging)"
