#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${1:-/root/prometheus}"
VENV_DIR="${2:-/root/venv-prometheus}"

python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel
"$VENV_DIR/bin/pip" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
"$VENV_DIR/bin/pip" install \
  accelerate \
  bitsandbytes \
  datasets \
  hf-transfer \
  huggingface-hub \
  kernels \
  optuna \
  peft \
  psutil \
  pydantic-settings \
  questionary \
  rich \
  safetensors \
  sentencepiece \
  'git+https://github.com/huggingface/transformers.git'
"$VENV_DIR/bin/pip" install -e "$REPO_DIR" --no-deps

printf 'Setup complete.\nRepo: %s\nVenv: %s\n' "$REPO_DIR" "$VENV_DIR"
