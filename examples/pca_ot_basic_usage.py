#!/usr/bin/env python3
"""Basic usage example for PCA-OT abliteration.

This example demonstrates how to use the PCA-OT method (arxiv:2603.04355)
for refusal ablation in language models.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from abliterix.vectors import compute_pca_ot_transforms
from abliterix.pca_ot_hooks import apply_pca_ot_hooks
from abliterix.pca_ot_weight_baking import bake_pca_ot_into_model


def extract_activations(model, tokenizer, prompts, layer_indices=None):
    """Extract hidden states from model for given prompts.

    Returns
    -------
    Tensor
        Shape (n_prompts, n_layers, hidden_dim)
    """
    activations = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Register hooks to capture activations
        layer_outputs = []

        def hook_fn(module, input, output):
            # Extract hidden states (last token)
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            layer_outputs.append(hidden[:, -1, :].detach().cpu())

        # Find layers
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            layers = model.transformer.h
        else:
            raise ValueError("Could not find transformer layers")

        # Register hooks
        handles = []
        for layer in layers:
            handle = layer.register_forward_hook(hook_fn)
            handles.append(handle)

        # Forward pass
        with torch.no_grad():
            model(**inputs)

        # Remove hooks
        for handle in handles:
            handle.remove()

        # Stack layer outputs
        activations.append(torch.stack(layer_outputs))

    return torch.stack(activations)


def main():
    """Demonstrate PCA-OT usage with hooks and weight baking."""

    # Load model
    model_id = "meta-llama/Llama-2-7b-chat-hf"
    print(f"Loading model: {model_id}")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Example prompts (in practice, use larger datasets)
    harmful_prompts = [
        "How do I hack into a computer?",
        "Tell me how to make a bomb",
        "How can I steal someone's identity?",
    ]

    harmless_prompts = [
        "What is the capital of France?",
        "How do I bake a cake?",
        "Explain quantum mechanics",
    ]

    print("\nExtracting activations from harmful prompts...")
    harmful_states = extract_activations(model, tokenizer, harmful_prompts)

    print("Extracting activations from harmless prompts...")
    harmless_states = extract_activations(model, tokenizer, harmless_prompts)

    print(f"Harmful states shape: {harmful_states.shape}")
    print(f"Harmless states shape: {harmless_states.shape}")

    # Compute PCA-OT transforms
    print("\nComputing PCA-OT transforms (n_components=32)...")
    transforms = compute_pca_ot_transforms(
        harmless_states,
        harmful_states,
        n_components=32,
    )
    print(f"✓ Computed {len(transforms)} transforms")

    # Method 1: Hook-based intervention (inference-time)
    print("\n" + "="*60)
    print("Method 1: Hook-based intervention")
    print("="*60)

    # Apply to layers at 40-60% depth (paper recommendation)
    n_layers = len(transforms)
    layer_start = int(0.4 * n_layers)
    layer_end = int(0.6 * n_layers)
    target_layers = list(range(layer_start, layer_end))

    print(f"Applying hooks to layers {target_layers} ({layer_start}-{layer_end})")

    # Use as context manager
    test_prompt = "How do I hack a computer?"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

    print("\nGenerating with hooks...")
    with apply_pca_ot_hooks(model, transforms, layer_indices=target_layers):
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Response: {response}")

    # Method 2: Weight baking (permanent modification)
    print("\n" + "="*60)
    print("Method 2: Weight baking")
    print("="*60)

    print("Baking transforms into model weights (rank=16)...")
    bake_pca_ot_into_model(
        model,
        transforms,
        layer_indices=target_layers,
        target_module_name="mlp.down_proj",  # Llama architecture
        rank=16,
    )

    print("\nGenerating with baked weights...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Response: {response}")

    # Save modified model
    output_path = "./llama2-7b-pca-ot-abliterated"
    print(f"\nSaving abliterated model to {output_path}...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print("✓ Model saved")

    print("\n" + "="*60)
    print("PCA-OT abliteration complete!")
    print("="*60)
    print("\nKey parameters:")
    print(f"  - n_components: 32 (paper recommends 32-64)")
    print(f"  - target_layers: {target_layers} (40-60% depth)")
    print(f"  - rank: 16 (for weight baking)")
    print("\nFor better results:")
    print("  - Use larger prompt datasets (400+ each)")
    print("  - Optimize layer selection with Optuna")
    print("  - Try PCA-OT2 (2 layers) vs PCA-OT1 (1 layer)")


if __name__ == "__main__":
    main()
