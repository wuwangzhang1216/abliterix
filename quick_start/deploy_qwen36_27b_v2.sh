#!/usr/bin/env bash
# Deploy Qwen3.6-27B V2 abliteration — separates GDN from full-attn buckets.
#
# V1 plateaued at 27/100 refusals @ KL=0.0118 (trial 6), then 23 trials
# couldn't improve because abliterix engine.py registers 48 GatedDeltaNet
# `linear_attn.out_proj` AND 16 full-attn `self_attn.o_proj` under a single
# "attn.o_proj" bucket. V2 patches engine.py to split them and lets Optuna
# tune three independent knobs.
#
# Prereqs: V1 deps already installed on the pod (transformers 5.5.4, peft
# 0.19.1, etc.), HF cache populated with the weights, .env present.
# This script skips install + download and only does:
#   1. Stop any running abliterix
#   2. Patch src/abliterix/core/engine.py (one line)
#   3. Launch abliterix with the V2 TOML
#
# Usage on the pod:
#   bash /workspace/abliterix/quick_start/deploy_qwen36_27b_v2.sh

set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/abliterix}"
CONFIG="${CONFIG:-configs/qwen3.6_27b_v2.toml}"
LOG_FILE="${LOG_FILE:-/workspace/run_qwen36_27b_v2.log}"
HF_CACHE="${HF_CACHE:-/workspace/hf_cache}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/workspace/checkpoints_qwen3.6_27b_v2}"
ENGINE_FILE="${REPO_DIR}/src/abliterix/core/engine.py"

cd "$REPO_DIR"

# ─── 1. Stop existing abliterix run (if any) ────────────────────────────────
echo "=== Stopping existing abliterix (if running) ==="
if pgrep -f "abliterix" > /dev/null; then
  pkill -f abliterix 2>/dev/null || true
  sleep 3
  pkill -9 -f abliterix 2>/dev/null || true
  sleep 2
fi
pgrep -af abliterix | head -3 || true
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader

# ─── 2. .env check ──────────────────────────────────────────────────────────
if [ ! -f "$REPO_DIR/.env" ]; then
  echo "ERROR: $REPO_DIR/.env missing (need HF_TOKEN + OPENROUTER_API_KEY)"
  exit 1
fi
set -a
# shellcheck disable=SC1091
. "$REPO_DIR/.env"
set +a
: "${HF_TOKEN:?HF_TOKEN not set in .env}"
: "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY not set in .env}"
export HUGGING_FACE_TOKEN="${HUGGING_FACE_TOKEN:-$HF_TOKEN}"

# ─── 3. Patch engine.py — separate GDN bucket ──────────────────────────────
echo "=== Patching engine.py for V2 bucket split ==="
OLD_LINE='_register("attn.o_proj", layer.linear_attn.out_proj)'
NEW_LINE='_register("linear_attn.out_proj", layer.linear_attn.out_proj)'

if grep -qF "$NEW_LINE" "$ENGINE_FILE"; then
  echo "Patch already applied — engine.py uses linear_attn.out_proj bucket."
elif grep -qF "$OLD_LINE" "$ENGINE_FILE"; then
  # Back up once (idempotent — skip if .v1.bak exists)
  [ -f "${ENGINE_FILE}.v1.bak" ] || cp "$ENGINE_FILE" "${ENGINE_FILE}.v1.bak"
  # Only replace the GDN registration (full-attn self_attn.o_proj stays as attn.o_proj)
  python3 - <<PY
import pathlib, sys
p = pathlib.Path("${ENGINE_FILE}")
text = p.read_text()
old = '_register("attn.o_proj", layer.linear_attn.out_proj)'
new = '_register("linear_attn.out_proj", layer.linear_attn.out_proj)'
count = text.count(old)
if count != 1:
    sys.exit(f"ERROR: expected exactly 1 occurrence of GDN registration, found {count}")
text = text.replace(old, new, 1)
p.write_text(text)
print("OK: engine.py patched (GDN now under linear_attn.out_proj)")
PY
else
  echo "ERROR: neither V1 nor V2 line found in engine.py — manual inspection required."
  echo "Expected one of:"
  echo "  $OLD_LINE"
  echo "  $NEW_LINE"
  exit 1
fi

# Re-install abliterix in editable mode so the patch takes effect
pip install --break-system-packages --root-user-action=ignore -q -e . --no-deps

# Sanity-check: the patched code should expose a 3-bucket component list
python3 - <<'PY'
import importlib, importlib.util
spec = importlib.util.find_spec("abliterix")
if spec is None:
    raise SystemExit("ERROR: abliterix not importable after patch")
print(f"abliterix imported from: {spec.origin}")
src = open(spec.submodule_search_locations[0] + "/core/engine.py").read()
if 'linear_attn.out_proj", layer.linear_attn.out_proj' not in src:
    raise SystemExit("ERROR: patch NOT applied in installed package — check pip install path")
print("OK: installed package has V2 patch")
PY

# ─── 4. Env exports ─────────────────────────────────────────────────────────
export HF_HOME="$HF_CACHE"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# ─── 5. Launch abliterix V2 ─────────────────────────────────────────────────
mkdir -p "$CHECKPOINT_DIR"
echo "=== Launching abliterix V2 ==="
echo "Config:       $CONFIG"
echo "Log:          $LOG_FILE"
echo "Checkpoints:  $CHECKPOINT_DIR"
echo "GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

nohup bash -c "AX_CONFIG='$CONFIG' abliterix --optimization.checkpoint-dir='$CHECKPOINT_DIR' 2>&1 | tee '$LOG_FILE'" >/dev/null 2>&1 &
PID=$!
echo "Started PID: $PID"
echo
echo "Monitor with:"
echo "  tail -f $LOG_FILE"
echo
echo "V2 ship-candidate signals:"
echo "  - Trial refusals < 15/100 with KL < 0.015 (beats V1 trial 6 = 27/100)"
echo "  - attn.o_proj.max_weight in [4, 7] (concentrated full-attn push)"
echo "  - linear_attn.out_proj.max_weight in [1.5, 3] (GDN amplifier)"
echo "  - mlp.down_proj.max_weight in [2, 4]"
echo
echo "Hard cutoffs — kill if:"
echo "  - Trial 15 best refusals > 25/100 (V2 separation didn't help — V3 needed)"
echo "  - Trial 10 all KL < 0.002 (search space too conservative — widen ceilings)"
echo
echo "Rollback: cp ${ENGINE_FILE}.v1.bak ${ENGINE_FILE} && pip install -e . --no-deps"
