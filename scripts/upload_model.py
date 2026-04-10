# Abliterix — a derivative work of Heretic (https://github.com/p-e-w/heretic)
# Original work Copyright (C) 2025  Philipp Emanuel Weidmann (p-e-w)
# Modified work Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Upload an abliterated model to Hugging Face Hub.

Usage:
    python scripts/upload_model.py \
        --model Qwen/Qwen3.5-0.8B \
        --checkpoint-dir checkpoints_100t \
        --trial 46 \
        --repo-id wangzhang/Qwen3.5-0.8B-abliterated

    python scripts/upload_model.py \
        --model Qwen/Qwen3.5-9B \
        --checkpoint-dir checkpoints_9b \
        --trial 44 \
        --repo-id wangzhang/Qwen3.5-9B-abliterated \
        --config configs/qwen3.5_9b.toml \
        --batch-size 8

    python scripts/upload_model.py \
        --model Qwen/Qwen3.5-35B-A3B \
        --checkpoint-dir checkpoints_35b \
        --trial 21 \
        --repo-id wangzhang/Qwen3.5-35B-A3B-abliterated \
        --save-dir /workspace/merged_35b_upload
"""

import argparse
import gc
import os
import sys

from abliterix.scriptlib import extract_trial_params, load_trial, setup_io

setup_io()

# ── Parse arguments before importing heavy libraries ─────────────────────
parser = argparse.ArgumentParser(
    description="Upload abliterated model to Hugging Face Hub"
)
parser.add_argument(
    "--model", required=True, help="Base model ID (e.g. Qwen/Qwen3.5-0.8B)"
)
parser.add_argument(
    "--checkpoint-dir", required=True, help="Optuna checkpoint directory"
)
parser.add_argument("--trial", type=int, required=True, help="Trial number to upload")
parser.add_argument(
    "--repo-id",
    required=True,
    help="HF repo ID (e.g. wangzhang/Qwen3.5-0.8B-abliterated)",
)
parser.add_argument(
    "--config",
    default=None,
    help="Config file path (uses default abliterix.toml if not set)",
)
parser.add_argument(
    "--batch-size", type=int, default=4, help="Batch size for inference (default: 4)"
)
parser.add_argument(
    "--save-dir",
    default=None,
    help="Save locally first, then upload (for large models)",
)
parser.add_argument(
    "--hf-token", default=None, help="HF token (reads from env if not set)"
)
args = parser.parse_args()

# Set up sys.argv for Settings CLI parsing.
sys.argv = [
    "upload",
    "--model.model-id",
    args.model,
    "--inference.batch-size",
    str(args.batch_size),
]
if args.config:
    sys.argv.extend(["--config", args.config])

HF_TOKEN = (
    args.hf_token
    or os.environ.get("HUGGING_FACE_TOKEN")
    or os.environ.get("HF_TOKEN", "")
)

import huggingface_hub  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import ModelCard, ModelCardData  # noqa: E402

from abliterix.settings import AbliterixConfig  # noqa: E402
from abliterix.core.engine import SteeringEngine  # noqa: E402
from abliterix.core.steering import apply_steering  # noqa: E402
from abliterix.vectors import compute_steering_vectors  # noqa: E402
from abliterix.data import load_prompt_dataset  # noqa: E402

# ── Load trial ────────────────────────────────────────────────────────────
trial = load_trial(args.checkpoint_dir, args.model, args.trial)
direction_index, parameters, moe_params = extract_trial_params(trial)
trial_kl = trial.user_attrs.get("kl_divergence", 0)
trial_refusals = trial.user_attrs.get("refusals", 0)

print(f"Trial #{args.trial}: KL={trial_kl:.4f}, refusals={trial_refusals}/200")
if moe_params:
    print(
        f"  MoE: suppress={moe_params.n_suppress}, router_bias={moe_params.router_bias:.2f}, "
        f"expert_weight={moe_params.expert_ablation_weight:.2f}"
    )

# ── Load model ────────────────────────────────────────────────────────────
config = AbliterixConfig()
config.inference.batch_size = args.batch_size
print(f"\nLoading {args.model}...")
engine = SteeringEngine(config)

# ── Compute directions ────────────────────────────────────────────────────
print("Computing refusal directions (full training set)...")
good_prompts = load_prompt_dataset(config, config.benign_prompts)
bad_prompts = load_prompt_dataset(config, config.target_prompts)
print(f"  {len(good_prompts)} benign, {len(bad_prompts)} target prompts")

with torch.no_grad():
    good_residuals = engine.extract_hidden_states_batched(good_prompts)
    bad_residuals = engine.extract_hidden_states_batched(bad_prompts)
    refusal_directions = compute_steering_vectors(
        good_residuals,
        bad_residuals,
        config.steering.vector_method,
        config.steering.orthogonal_projection,
    )
    if config.steering.orthogonal_projection:
        print("  Applied orthogonalization")

# Profile MoE experts if needed.
safety_experts = None
if moe_params and engine.has_expert_routing():
    print("Profiling MoE expert activations...")
    safety_experts = engine.identify_safety_experts(good_prompts, bad_prompts)

del good_residuals, bad_residuals, good_prompts, bad_prompts
gc.collect()
torch.cuda.empty_cache()

# ── Abliterate ────────────────────────────────────────────────────────────
print("Applying abliteration...")
engine.restore_baseline()
apply_steering(
    engine,
    refusal_directions,
    direction_index,
    parameters,
    config,
    safety_experts=safety_experts,
    routing_config=moe_params,
)

# ── Merge & Upload ────────────────────────────────────────────────────────
print("Merging LoRA weights into base model...")
merged_model = engine.export_merged()

if args.save_dir:
    # Large model: save locally first, then upload as folder.
    print(f"Saving merged model to {args.save_dir}...")
    merged_model.save_pretrained(args.save_dir, safe_serialization=True)
    del merged_model
    gc.collect()
    torch.cuda.empty_cache()

    print("Saving tokenizer...")
    engine.tokenizer.save_pretrained(args.save_dir)

    print(f"Uploading to {args.repo_id}...")
    huggingface_hub.upload_folder(
        folder_path=args.save_dir,
        repo_id=args.repo_id,
        repo_type="model",
        token=HF_TOKEN,
    )
else:
    # Standard: push directly to hub.
    print(f"Uploading merged model to {args.repo_id}...")
    merged_model.push_to_hub(args.repo_id, private=False, token=HF_TOKEN)
    del merged_model
    gc.collect()
    torch.cuda.empty_cache()

    print("Uploading tokenizer...")
    engine.tokenizer.push_to_hub(args.repo_id, private=False, token=HF_TOKEN)

# ── Model Card ────────────────────────────────────────────────────────────
print("Creating model card...")

model_name = args.model.split("/")[-1]
refusal_pct = trial_refusals / 2  # out of 200

card = ModelCard.load(args.model, token=HF_TOKEN)
if card.data is None:
    card.data = ModelCardData()
if card.data.tags is None:
    card.data.tags = []
card.data.tags.extend(["abliterix", "uncensored", "decensored", "abliterated"])
card.data.base_model = args.model

card.text = f"""# {model_name}-abliterated

Unrestricted version of [{args.model}](https://huggingface.co/{args.model}), created using [Abliterix](https://github.com/wuwangzhang1216/abliterix).

## Performance

| Metric | This model | Original |
|--------|-----------|----------|
| **KL divergence** | {trial_kl:.4f} | 0 |
| **Refusals** | {trial_refusals}/200 ({refusal_pct:.1f}%) | ~200/200 (100%) |

## How It Was Made

1. Computed refusal directions from 800 harmful vs 800 benign prompt pairs
2. Applied orthogonalized abliteration to isolate refusal behavior
3. Optimized abliteration parameters via Optuna TPE (trial #{args.trial})
4. Selected Pareto-optimal configuration balancing minimal refusals with minimal model degradation

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{args.repo_id}", torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("{args.repo_id}")

messages = [{{"role": "user", "content": "Your question here"}}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

-----

"""

card.push_to_hub(args.repo_id, token=HF_TOKEN)
print(f"\nModel uploaded to: https://huggingface.co/{args.repo_id}")
