#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Export an abliterated model: apply best trial steering and push to HuggingFace.

Usage:
    python scripts/export_model.py \
        --model google/gemma-4-31B-it \
        --checkpoint checkpoints_gemma4_31b_v8 \
        --trial 13 \
        --config configs/gemma4_31b_v8_direct.toml \
        --push-to wangzhang/gemma-4-31B-it-abliterated
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Base HF model ID")
    parser.add_argument("--checkpoint", required=True, help="Optuna checkpoint dir")
    parser.add_argument("--trial", type=int, required=True, help="Trial number to export")
    parser.add_argument("--config", required=True, help="Config TOML path")
    parser.add_argument("--push-to", required=True, help="HF repo to push to")
    parser.add_argument("--save-local", default=None, help="Also save locally to this path")
    args = parser.parse_args()

    os.environ["AX_CONFIG"] = args.config
    sys.argv = [sys.argv[0], "--model.model-id", args.model, "--inference.batch-size", "4"]

    import torch
    torch.set_grad_enabled(False)

    from abliterix.scriptlib import load_trial, extract_trial_params, setup_io
    from abliterix.core.engine import SteeringEngine
    from abliterix.core.steering import apply_steering
    from abliterix.data import load_prompt_dataset
    from abliterix.settings import AbliterixConfig
    from abliterix.vectors import compute_steering_vectors
    from abliterix.util import flush_memory

    setup_io()

    # Load trial
    trial = load_trial(args.checkpoint, args.model, args.trial)
    direction_index, parameters, routing = extract_trial_params(trial)
    refusals = trial.user_attrs.get("refusals")
    kl = trial.user_attrs.get("kl_divergence")
    print(f"Trial #{args.trial}: refusals={refusals}, KL={kl}")

    # Load model
    config = AbliterixConfig()
    engine = SteeringEngine(config)

    # Compute steering vectors
    print("\nComputing steering vectors...")
    benign = load_prompt_dataset(config, config.benign_prompts)
    target = load_prompt_dataset(config, config.target_prompts)
    benign_states = engine.extract_hidden_states_batched(benign)
    target_states = engine.extract_hidden_states_batched(target)
    vectors = compute_steering_vectors(
        benign_states, target_states,
        config.steering.vector_method,
        config.steering.orthogonal_projection,
        winsorize=config.steering.winsorize_vectors,
        winsorize_quantile=config.steering.winsorize_quantile,
    )
    del benign_states, target_states
    flush_memory()

    # Profile MoE experts if applicable
    safety_experts = None
    if engine.has_expert_routing():
        print("Profiling MoE experts...")
        safety_experts = engine.identify_safety_experts(benign, target)

    del benign, target
    flush_memory()

    # Apply steering
    print("Applying steering (direct weight editing)...")
    apply_steering(
        engine, vectors, direction_index, parameters, config,
        safety_experts=safety_experts, routing_config=routing,
    )
    print("Steering applied.")

    # Get the base model — merge LoRA adapters and fully unwrap PEFT.
    # Direct steering modifies base weights in-place, but PEFT wrapper
    # still wraps them. merge_and_unload() returns the clean base model.
    from peft import PeftModel
    model = engine.model
    if isinstance(model, PeftModel):
        print("Merging LoRA adapters and unwrapping PEFT...")
        model = model.merge_and_unload()

    # Save locally first (root filesystem is tiny, use /workspace)
    save_dir = args.save_local or "/workspace/export_model"
    print(f"\nSaving model to {save_dir}...")
    model.save_pretrained(save_dir)
    engine.tokenizer.save_pretrained(save_dir)
    print("Local save complete.")

    # Push to HuggingFace
    from huggingface_hub import HfApi
    api = HfApi()
    print(f"Pushing to {args.push_to}...")
    api.create_repo(args.push_to, exist_ok=True, repo_type="model")
    api.upload_folder(
        folder_path=save_dir,
        repo_id=args.push_to,
        repo_type="model",
    )
    print(f"Done! Model pushed to https://huggingface.co/{args.push_to}")


if __name__ == "__main__":
    main()
