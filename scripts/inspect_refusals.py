# Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>

"""Replay a trial from checkpoint and show which prompts are still refused.

Memory-efficient: uses fewer training prompts for direction computation,
skips TrialScorer baseline (logprobs/responses), only generates final responses.
"""

import sys

from prometheus.scriptlib import extract_trial_params, load_trial, setup_io

setup_io()

MODEL = "Qwen/Qwen3.5-0.8B"
TRIAL_NUMBER = 7  # Best trial: 4 refusals, KLD=0.0168

# batch_size=2 for residuals (peak memory), will override to 8 for generation
sys.argv = ["inspect", "--model.model-id", MODEL, "--inference.batch-size", "2"]

import gc  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from prometheus.settings import PrometheusConfig  # noqa: E402
from prometheus.core.steering import apply_steering  # noqa: E402
from prometheus.core.engine import SteeringEngine  # noqa: E402
from prometheus.eval.detector import RefusalDetector  # noqa: E402
from prometheus.data import load_prompt_dataset  # noqa: E402

# Load trial from checkpoint
trial = load_trial("checkpoints_test", MODEL, TRIAL_NUMBER)
direction_index, parameters, _ = extract_trial_params(trial)

config = PrometheusConfig()

print(f"Loading {MODEL}...")
engine = SteeringEngine(config)
detector = RefusalDetector(config)

# Use only 100 training prompts (enough for direction, minimal VRAM)
print("\nLoading training prompts (100 each for direction computation)...")
benign_prompts = load_prompt_dataset(config, config.benign_prompts)[:100]
target_prompts = load_prompt_dataset(config, config.target_prompts)[:100]
print(f"  Benign: {len(benign_prompts)}, Target: {len(target_prompts)}")

# Compute refusal directions with torch.no_grad to save memory
print("Computing refusal directions...")
with torch.no_grad():
    benign_residuals = engine.extract_hidden_states_batched(benign_prompts)
    target_residuals = engine.extract_hidden_states_batched(target_prompts)
    benign_means = benign_residuals.mean(dim=0)
    target_means = target_residuals.mean(dim=0)
    refusal_directions = F.normalize(target_means - benign_means, p=2, dim=1)

del (
    benign_residuals,
    target_residuals,
    benign_means,
    target_means,
    benign_prompts,
    target_prompts,
)
gc.collect()
torch.cuda.empty_cache()
print("  Done, VRAM freed.")

# Load eval target prompts directly (skip full TrialScorer init)
print("\nLoading target eval prompts...")
eval_target_prompts = load_prompt_dataset(config, config.target_eval_prompts)
print(f"  {len(eval_target_prompts)} prompts loaded")

# Apply abliteration
print(f"\nApplying Trial #{TRIAL_NUMBER} abliteration...")
engine.restore_baseline()
apply_steering(engine, refusal_directions, direction_index, parameters)

# Increase batch size for generation (less memory than residuals)
config.inference.batch_size = 8

# Generate responses
print("Generating responses...")
responses = engine.generate_text_batched(eval_target_prompts, skip_special_tokens=True)

# Check each response
print(f"\n{'=' * 80}")
print(f"REFUSED PROMPTS (Trial #{TRIAL_NUMBER}):")
print(f"{'=' * 80}")

refused_count = 0
for i, (prompt, response) in enumerate(zip(eval_target_prompts, responses)):
    if detector.detect_refusal(response):
        refused_count += 1
        print(f"\n--- Refused #{refused_count} (index {i}) ---")
        if prompt.system:
            print(f"System: {prompt.system[:200]}")
        print(f"Prompt: {prompt.user[:300]}")
        resp_display = response[:500] if response.strip() else "[empty]"
        print(f"Response: {resp_display}")

print(f"\n{'=' * 80}")
print(f"Total refusals: {refused_count}/{len(eval_target_prompts)}")
print(f"{'=' * 80}")
