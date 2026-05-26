#!/usr/bin/env bash
# Deploy DeepSeek-V4-Flash abliteration on a 4× B200 RunPod pod.
#
# Pod expectations:
#   - 4× B200 (183 GiB usable per b200_real_vram.md, 732 GiB total)
#   - Container overlay (/) ≥ 800 GB free  (160 GB FP8/FP4 source + 568 GB BF16
#     dequant + HF cache + checkpoints + logs)
#   - /workspace network volume is OK for code + .env + final BF16 dir
#   - Repo synced to /workspace/abliterix (scp -r from your laptop)
#   - .env at /workspace/abliterix/.env with HF_TOKEN + OPENROUTER_API_KEY
#
# Usage on the pod:
#   bash /workspace/abliterix/quick_start/deploy_dsv4_flash.sh
#
# What this does (in order):
#   1. GPU sanity check (≥4 cards, ≥160 GiB each)
#   2. .env load + HF download speed test
#   3. Pin transformers/peft/accelerate/bitsandbytes/kernels (per
#      feedback_bnb_kernels_required.md — every deploy needs bnb + kernels)
#   4. Resolve BF16 model dir:
#        a) prefer unsloth/DeepSeek-V4-Flash (already BF16, no work)
#        b) fall back to deepseek-ai/DeepSeek-V4-Flash + offline dequant
#           (FP8 non-experts via abliterix-dequant-fp8 + FP4 experts via
#            modeling code's own helpers, invoked from
#            quick_start/_dsv4_dequant_fp4_experts.py)
#   5. Symlink encoding_dsv4.py into the BF16 dir so the toml's
#      custom_encoder_module path resolves
#   6. Launch abliterix
#
# Why no vLLM path (verified 2026-05-05):
#   vLLM 0.20.0+ DOES register `DeepseekV4ForCausalLM` (PR #40760, merged
#   2026-04-27) and runs inference natively in FP8+FP4 on 1× B200. BUT the
#   PR's "Unsupported features" list explicitly excludes LoRA, EP, PP,
#   in-place weight editing, AND hooks — every primitive abliterix's vLLM
#   steering path needs. So the vllm_live_suppression_dead_end.md verdict
#   from V3 carries over to V4 verbatim. HF + BF16 dequant + DIRECT + EGA
#   remains the only working abliteration route.
#
# Cost expectation:
#   - Source download: 160 GB at ~80-150 MB/s → ~20-30 min
#   - Dequant (option b): ~15-25 min (mostly disk I/O)
#   - Phase 1 hidden state extraction: ~25-40 min on 4× B200 with PP
#   - Phase 2 (50 trials × DIRECT EGA): ~5-9 h

set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/abliterix}"
CONFIG="${CONFIG:-configs/deepseek_v4_flash.toml}"
LOG_FILE="${LOG_FILE:-/root/run_dsv4_flash.log}"
HF_CACHE="${HF_CACHE:-/root/hf_cache}"
BF16_DIR="${BF16_DIR:-/workspace/dsv4_flash_bf16}"
HS_DIR="${HS_DIR:-/root/abliterix_hidden_states}"
MIN_HF_SPEED="${MIN_HF_SPEED:-50}"
SOURCE_REPO_PRIMARY="${SOURCE_REPO_PRIMARY:-unsloth/DeepSeek-V4-Flash}"
SOURCE_REPO_FALLBACK="${SOURCE_REPO_FALLBACK:-deepseek-ai/DeepSeek-V4-Flash}"

cd "$REPO_DIR"

# ─── 1. GPU sanity check ─────────────────────────────────────────────────────
echo "=== GPU check ==="
GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
echo "$GPU_INFO"
GPU_COUNT=$(echo "$GPU_INFO" | wc -l | tr -d ' ')
if [ "$GPU_COUNT" -lt 4 ]; then
  echo "ERROR: expected >= 4 GPUs, found $GPU_COUNT"
  echo "       DSV4-Flash BF16 (~568 GB) does not fit in <4× B200."
  exit 1
fi
VRAM_OK=$(echo "$GPU_INFO" | awk -F',' '$2+0 < 160000 {print "LOW"}' | head -1)
if [ "$VRAM_OK" = "LOW" ]; then
  echo "WARN: at least one GPU has < 160 GB VRAM."
  echo "      DSV4-Flash BF16 sized for B200 (183 GiB usable). H200 (141 GiB)"
  echo "      will work only at 8× count — adjust max_memory in toml."
fi

echo "=== Driver / CUDA / nvidia-smi ==="
nvidia-smi --query-gpu=driver_version,vbios_version --format=csv,noheader || true

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
: "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY not set in .env (needed for llm_judge)}"

# ─── 3. Network speed check ──────────────────────────────────────────────────
echo "=== HF download speed test ==="
SAMPLE_URL="https://huggingface.co/${SOURCE_REPO_PRIMARY}/resolve/main/config.json"
SPEED=$(curl -sL -H "Authorization: Bearer ${HF_TOKEN}" \
  -o /dev/null -w '%{speed_download}' --max-time 10 \
  "$SAMPLE_URL" || echo "0")
SPEED_MB=$(awk "BEGIN{printf \"%.0f\", ${SPEED}/1048576}")
echo "HF speed (sampled on tiny config.json — see hf_download_speed_test_misleading.md):"
echo "  ${SPEED_MB} MB/s sample, real multi-stream throughput will be 8-20× higher."

# ─── 4. Install deps ─────────────────────────────────────────────────────────
echo "=== Installing deps ==="
PIP_FLAGS="--break-system-packages --root-user-action=ignore -q"

# transformers 4.57.1 matches DSV4's transformers_version field. peft pinned to
# the matching family (per minimax recipe). bitsandbytes + kernels are MANDATORY
# for every abliterix deploy (feedback_bnb_kernels_required.md).
# shellcheck disable=SC2086
pip install $PIP_FLAGS \
  "transformers>=4.57.1,<4.58" \
  "peft>=0.13,<0.15" \
  accelerate \
  safetensors \
  sentencepiece \
  optuna \
  datasets \
  "bitsandbytes>=0.45" \
  "kernels~=0.11" \
  pydantic-settings \
  questionary \
  hf-transfer \
  psutil \
  rich

# Install abliterix (no deps — pinned above).
# shellcheck disable=SC2086
pip install $PIP_FLAGS -e . --no-deps

# Drop flash-attn (per minimax + RunPod feedback memories).
pip uninstall -y --break-system-packages flash-attn 2>/dev/null || true

python3 -c "import torch, transformers, peft, bitsandbytes; \
  print(f'torch={torch.__version__} transformers={transformers.__version__} peft={peft.__version__} bnb={bitsandbytes.__version__} gpus={torch.cuda.device_count()} cc={torch.cuda.get_device_capability(0)}')"

# ─── 5. Resolve BF16 model directory ─────────────────────────────────────────
mkdir -p "$HF_CACHE" "$BF16_DIR" "$HS_DIR"
export HF_HOME="$HF_CACHE"
export HF_HUB_ENABLE_HF_TRANSFER=1

if [ -d "$BF16_DIR" ] && [ -f "$BF16_DIR/config.json" ] && \
   [ "$(du -sBG "$BF16_DIR" 2>/dev/null | awk '{print $1+0}')" -ge 500 ]; then
  echo "=== BF16 dir already populated ($BF16_DIR), skipping download/dequant ==="
else
  echo "=== Resolving BF16 source ==="
  # Probe unsloth first — a HEAD on its config is the cheapest test.
  if curl -sL -o /dev/null -w '%{http_code}' \
       -H "Authorization: Bearer ${HF_TOKEN}" \
       "https://huggingface.co/${SOURCE_REPO_PRIMARY}/resolve/main/config.json" \
       | grep -q '^200$'; then
    echo "=== unsloth BF16 exists — downloading ${SOURCE_REPO_PRIMARY} directly to ${BF16_DIR} ==="
    hf download "${SOURCE_REPO_PRIMARY}" --max-workers 16 --local-dir "$BF16_DIR"
  else
    echo "=== unsloth BF16 not found — pulling FP8/FP4 source then dequanting ==="
    SRC_DIR="$HF_CACHE/dsv4_flash_src"
    mkdir -p "$SRC_DIR"
    hf download "${SOURCE_REPO_FALLBACK}" --max-workers 16 --local-dir "$SRC_DIR"

    echo "=== Stage 5a: dequant FP8 non-expert tensors ==="
    # abliterix-dequant-fp8 walks every .safetensors shard, dequants
    # FP8 → BF16 in place. FP4 expert tensors are passed through as-is
    # (the next step handles them).
    python -m abliterix.scripts.dequant_fp8 "$SRC_DIR" "$BF16_DIR"

    echo "=== Stage 5b: dequant FP4 expert tensors via modeling code ==="
    # Use the model's own modeling_deepseek_v4.py to unpack FP4 experts.
    # The helper script loads with trust_remote_code, materialises every
    # FP4 packed expert weight as BF16, and overwrites the corresponding
    # safetensors shards in $BF16_DIR.
    python "$REPO_DIR/quick_start/_dsv4_dequant_fp4_experts.py" \
      --src "$SRC_DIR" \
      --dst "$BF16_DIR"
  fi

  # Make sure the encoding script is reachable from $BF16_DIR (the toml
  # references it directly). The dequant tool already copies *.py, but
  # unsloth's path will already have it — so this is idempotent.
  if [ ! -f "$BF16_DIR/encoding_dsv4.py" ]; then
    echo "WARN: encoding_dsv4.py missing from $BF16_DIR — abliterix will fail to load tokenizer encoder."
    echo "      Try: hf download ${SOURCE_REPO_FALLBACK} encoding_dsv4.py --local-dir $BF16_DIR"
  fi
fi

# ─── 6. Final environment ────────────────────────────────────────────────────
export AX_HIDDEN_STATES_DIR="$HS_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# DSV4 modeling code may probe flashinfer at import — keep it tolerant.
export FLASHINFER_DISABLE_VERSION_CHECK=1

# Quick smoke test before the long run: 1-token forward to catch dtype /
# tokenizer / encoding errors in 30s instead of 30min.
echo "=== Pre-flight smoke test (1-token forward) ==="
python3 - <<'PYEOF'
import os, sys
os.environ.setdefault("AX_CONFIG", "configs/deepseek_v4_flash.toml")
sys.path.insert(0, "src")
from abliterix.settings import AbliterixConfig
from abliterix.core.engine import SteeringEngine
from abliterix.types import ChatMessage

cfg = AbliterixConfig(_toml_file=os.environ["AX_CONFIG"])
eng = SteeringEngine(cfg)
out = eng._generate(
    [ChatMessage(system=cfg.system_prompt, user="Hello!")],
    max_new_tokens=1,
)
print("smoke-test OK")
PYEOF

# ─── 7. Launch ───────────────────────────────────────────────────────────────
echo "=== Launching abliterix ==="
echo "Config:    $CONFIG"
echo "Log:       $LOG_FILE"
echo "BF16 dir:  $BF16_DIR"
echo "HS dir:    $HS_DIR"
echo

nohup bash -c "AX_CONFIG='$CONFIG' abliterix 2>&1 | tee '$LOG_FILE'" >/dev/null 2>&1 &
PID=$!
echo "Started PID: $PID"
echo
echo "Monitor with:"
echo "  tail -f $LOG_FILE"
echo "  nvidia-smi dmon -s u -c 30"
echo
echo "First sanity gate (run after Phase 1 completes, before Phase 2 spends 5h):"
echo "  AX_CONFIG=$CONFIG python quick_start/probe_dsv4_residual.py"
echo "  → if SVD top-1 ratio < 0.3 across most layers, mHC is scrambling the"
echo "    refusal direction. STOP THE RUN, see configs/deepseek_v4_flash.toml"
echo "    notes for hooking pre-Sinkhorn."
