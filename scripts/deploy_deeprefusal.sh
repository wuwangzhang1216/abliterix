#!/usr/bin/env bash
# Deploy abliterix iterative abliteration against DeepRefusal on RunPod
# Target: RTX 6000 Ada 48GB (single GPU)
# Model: skysys00/Meta-Llama-3-8B-Instruct-DeepRefusal (~16GB bf16)
#
# Usage:
#   1. scp -r abliterix root@<POD_IP>:/workspace/
#   2. ssh root@<POD_IP>
#   3. bash /workspace/abliterix/scripts/deploy_deeprefusal.sh

set -euo pipefail

cd /workspace/abliterix

# ── 1. Check network speed first (don't waste credits on a slow pod) ────
echo "=== Checking HF download speed ==="
SPEED=$(curl -sL -H "Authorization: Bearer $HF_TOKEN" -o /dev/null -w '%{speed_download}' \
  --max-time 15 \
  'https://huggingface.co/skysys00/Meta-Llama-3-8B-Instruct-DeepRefusal/resolve/main/model-00001-of-00004.safetensors' 2>/dev/null || echo "0")
SPEED_MB=$(echo "$SPEED / 1000000" | bc 2>/dev/null || echo "?")
echo "Download speed: ${SPEED_MB} MB/s (raw: ${SPEED} B/s)"
if [ "$(echo "$SPEED < 50000000" | bc 2>/dev/null)" = "1" ]; then
  echo "WARNING: Speed < 50 MB/s — consider switching pods"
fi

# ── 2. Environment variables ────────────────────────────────────────────
# Set HF_TOKEN and OPENROUTER_API_KEY in your shell before running this
# script, or create /workspace/abliterix/.env with the same variables.
if [ -f /workspace/abliterix/.env ]; then
  set -a && . /workspace/abliterix/.env && set +a
fi
: "${HF_TOKEN:?HF_TOKEN must be set}"
: "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY must be set}"

export HF_HOME=/workspace/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$HF_HOME"

# ── 3. Install dependencies ────────────────────────────────────────────
echo "=== Installing dependencies ==="
pip install -U transformers accelerate safetensors sentencepiece -q
pip install "peft>=0.13" -q
pip install -e . --no-deps -q
pip install optuna datasets bitsandbytes pydantic-settings questionary hf-transfer psutil rich -q
pip uninstall flash-attn -y 2>/dev/null || true

# Verify
python3 -c "
import peft, transformers, torch
print(f'OK: peft={peft.__version__} transformers={transformers.__version__} torch={torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU 0: {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
"

# ── 4. Pre-download model ──────────────────────────────────────────────
echo "=== Pre-downloading model ==="
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('skysys00/Meta-Llama-3-8B-Instruct-DeepRefusal', local_dir_use_symlinks=True)
print('Model download complete.')
"

# ── 5. Run abliterix ───────────────────────────────────────────────────
echo "=== Starting abliterix (iterative mode) ==="
echo "Config: configs/llama3_8b_deeprefusal.toml"
echo "Log:    /workspace/run.log"

nohup bash -c 'AX_CONFIG=configs/llama3_8b_deeprefusal.toml abliterix --non-interactive 2>&1 | tee /workspace/run.log' &>/dev/null &
echo "PID: $!"
echo ""
echo "Monitor with: tail -f /workspace/run.log"
echo "Check results: grep -E '\[iterative\]|refusal|KL|Pareto' /workspace/run.log"
