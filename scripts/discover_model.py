#!/usr/bin/env python3
"""Discover any HuggingFace model's architecture for Prometheus integration.

Run on a GPU machine to inspect the model's module tree, layer structure,
MoE routing, and hidden-state behaviour.  The output tells us exactly which
attribute paths to add to engine.py.

Usage:
    python scripts/discover_model.py --model mistralai/Devstral-Small-2-24B-Instruct-2512
    python scripts/discover_model.py --model zai-org/GLM-4.7-Flash
    python scripts/discover_model.py --model LiquidAI/LFM2-24B-A2B --skip-load
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import torch

torch.set_grad_enabled(False)


# ── Step 0: Config inspection (no weights) ──────────────────────────

def inspect_config(model_id: str):
    from transformers import AutoConfig

    print("=" * 70)
    print("STEP 0: Model config (no weight download)")
    print("=" * 70)

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    for attr in [
        "num_hidden_layers", "hidden_size", "intermediate_size",
        "num_attention_heads", "num_key_value_heads",
        "num_experts", "num_experts_per_tok", "num_shared_experts",
        "num_dense_layers", "model_type", "architectures",
        "moe_intermediate_size",
    ]:
        val = getattr(config, attr, "N/A")
        print(f"  {attr} = {val}")

    print()
    return config


# ── Step 1: Load model ──────────────────────────────────────────────

def load_model(model_id: str):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

    print("=" * 70)
    print("STEP 1: Loading model (BF16)")
    print("=" * 70)

    # Auto-detect VLMs (Mistral3, Qwen-VL, etc.) via vision_config
    configs = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    is_vlm = hasattr(configs, "vision_config") and configs.vision_config is not None
    cls = AutoModelForImageTextToText if is_vlm else AutoModelForCausalLM
    print(f"  Model class: {cls.__name__} (VLM={is_vlm})")

    model = cls.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )

    print(f"Model class: {type(model).__name__}")
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {alloc:.1f}/{total:.1f} GiB")

    print()
    return model


# ── Step 2: Full module tree ────────────────────────────────────────

def dump_module_tree(model):
    print("=" * 70)
    print("STEP 2: Full module tree")
    print("=" * 70)

    for name, mod in model.named_modules():
        indent = "  " * name.count(".")
        print(f"{indent}{name}: {type(mod).__name__}")

    print()


# ── Step 3: Discover layer structure ────────────────────────────────

def discover_layers(model):
    print("=" * 70)
    print("STEP 3: Layer structure discovery")
    print("=" * 70)

    candidates = [
        "model.layers",
        "model.language_model.layers",
        "model.backbone.layers",
        "model.transformer.h",
        "transformer.h",
    ]

    layers = None
    layer_path = None

    for path in candidates:
        obj = model
        for attr in path.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "__len__"):
            layers = obj
            layer_path = path
            break

    if layers is None:
        print("ERROR: Could not find layer list! Manual inspection needed.")
        print("Top-level children:")
        for name, child in model.named_children():
            print(f"  {name}: {type(child).__name__}")
        return None

    print(f"Layer list path: {layer_path}")
    print(f"Number of layers: {len(layers)}")

    layer_types = defaultdict(list)
    for i, layer in enumerate(layers):
        layer_types[type(layer).__name__].append(i)

    print("Layer types:")
    for cls_name, indices in layer_types.items():
        summary = indices[:5]
        suffix = "..." if len(indices) > 5 else ""
        print(f"  {cls_name}: {len(indices)} layers (indices: {summary}{suffix})")

    print()
    return layers


# ── Step 4: Discover steerable modules per layer ────────────────────

def discover_steerable_modules(layers):
    print("=" * 70)
    print("STEP 4: Steerable module discovery")
    print("=" * 70)

    if layers is None:
        print("Skipped (no layers found)")
        return

    sample_indices = sorted(set(
        i for i in [0, 1, len(layers) // 2, len(layers) - 1]
        if i < len(layers)
    ))

    for idx in sample_indices:
        layer = layers[idx]
        print(f"\n--- Layer {idx}: {type(layer).__name__} ---")

        for name, child in layer.named_children():
            print(f"  {name}: {type(child).__name__}")

        # Attention output projections
        attn_paths = [
            "self_attn.o_proj", "self_attn.out_proj",
            "attn.o_proj", "attn.out_proj",
            "attention.o_proj", "attention.out_proj",
            "linear_attn.out_proj",
        ]

        # MLA (Multi-head Latent Attention) projections — DeepSeek/Mistral4 style
        mla_paths = [
            "self_attn.q_a_proj", "self_attn.q_b_proj",
            "self_attn.kv_a_proj_with_mqa", "self_attn.kv_b_proj",
            "self_attn.q_a_layernorm", "self_attn.kv_a_layernorm",
        ]
        for path in mla_paths:
            obj = layer
            for attr in path.split("."):
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None:
                w_info = ""
                if hasattr(obj, "weight"):
                    w_info = f" shape={obj.weight.shape}, dtype={obj.weight.dtype}"
                print(f"  [MLA] {path}: {type(obj).__name__}{w_info}")
        for path in attn_paths:
            obj = layer
            for attr in path.split("."):
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None:
                w_info = ""
                if hasattr(obj, "weight"):
                    w_info = f" shape={obj.weight.shape}, dtype={obj.weight.dtype}"
                print(f"  [ATTN OUT] {path}: {type(obj).__name__}{w_info}")

        # Convolution output projections
        conv_paths = [
            "conv.out_proj", "conv.o_proj",
            "conv_block.out_proj", "conv_block.o_proj",
            "short_conv.out_proj", "short_conv.output_linear",
            "gated_conv.out_proj", "gated_conv.o_proj",
            "temporal_block.out_proj", "temporal_block.o_proj",
        ]
        for path in conv_paths:
            obj = layer
            for attr in path.split("."):
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None:
                w_info = ""
                if hasattr(obj, "weight"):
                    w_info = f" shape={obj.weight.shape}, dtype={obj.weight.dtype}"
                print(f"  [CONV OUT] {path}: {type(obj).__name__}{w_info}")

        # MLP / MoE structures
        for mlp_name in ["mlp", "feed_forward", "ffn", "moe"]:
            mlp = getattr(layer, mlp_name, None)
            if mlp is None:
                continue
            print(f"  [MLP/MoE] {mlp_name}: {type(mlp).__name__}")
            for cname, cmod in mlp.named_children():
                print(f"    {mlp_name}.{cname}: {type(cmod).__name__}")
                if hasattr(cmod, "weight"):
                    print(f"      weight: {cmod.weight.shape}")
                if "expert" in cname.lower() or "gate" in cname.lower():
                    for ename, emod in cmod.named_children():
                        print(f"      {mlp_name}.{cname}.{ename}: {type(emod).__name__}")
                        if hasattr(emod, "weight"):
                            print(f"        weight: {emod.weight.shape}")

        # Dense MLP down-projections
        for path in ["mlp.down_proj", "feed_forward.down_proj", "ffn.down_proj",
                      "mlp.w2", "feed_forward.w2"]:
            obj = layer
            for attr in path.split("."):
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None and hasattr(obj, "weight"):
                print(f"  [DENSE MLP] {path}: {obj.weight.shape}")

        # Shared expert (MoE models with shared experts, e.g. Mistral4, DeepSeek)
        shared_paths = [
            "mlp.shared_expert", "feed_forward.shared_expert",
            "mlp.shared_experts", "feed_forward.shared_experts",
        ]
        for path in shared_paths:
            obj = layer
            for attr in path.split("."):
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None:
                print(f"  [SHARED EXPERT] {path}: {type(obj).__name__}")
                for cname, cmod in obj.named_children():
                    w_info = f" shape={cmod.weight.shape}" if hasattr(cmod, "weight") else ""
                    print(f"    {path}.{cname}: {type(cmod).__name__}{w_info}")


# ── Step 5: Discover MoE router ────────────────────────────────────

def discover_router(layers):
    print("\n" + "=" * 70)
    print("STEP 5: MoE router/gate discovery")
    print("=" * 70)

    if layers is None:
        print("Skipped (no layers found)")
        return

    router_paths = [
        "mlp.gate", "moe.gate", "feed_forward.gate",
        "block_sparse_moe.gate", "mlp.router", "moe.router",
        "feed_forward.router",
    ]

    found_any = False
    for idx in [0, 1, len(layers) // 2, len(layers) - 1]:
        if idx >= len(layers):
            continue
        layer = layers[idx]
        for path in router_paths:
            obj = layer
            for attr in path.split("."):
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None:
                print(f"  Layer {idx}: {path}: {type(obj).__name__}")
                if hasattr(obj, "weight"):
                    w = obj.weight
                    print(f"    weight: shape={w.shape}, dtype={w.dtype}")
                    if w.dim() == 2:
                        print(f"    -> num_experts={w.shape[0]}, hidden_dim={w.shape[1]}")
                found_any = True

    if not found_any:
        print("  No router found via known paths. Searching all named modules...")
        mid = len(layers) // 2
        for name, mod in layers[mid].named_modules():
            if any(kw in name.lower() for kw in ["gate", "router", "route"]):
                print(f"  {name}: {type(mod).__name__}")
                if hasattr(mod, "weight"):
                    print(f"    weight: shape={mod.weight.shape}")


# ── Step 6: Discover fused expert weights ───────────────────────────

def discover_fused_weights(layers):
    print("\n" + "=" * 70)
    print("STEP 6: Fused expert weight (3-D tensor) discovery")
    print("=" * 70)

    if layers is None:
        print("Skipped (no layers found)")
        return

    mid = len(layers) // 2
    found = False
    for name, param in layers[mid].named_parameters():
        if param.dim() == 3:
            print(f"  Layer {mid}, {name}: shape={param.shape}, dtype={param.dtype}")
            print(f"    -> Likely fused experts: {param.shape[0]} experts, "
                  f"out={param.shape[1]}, in={param.shape[2]}")
            found = True

    if not found:
        print("  No 3-D parameters found — experts are likely individual modules.")


# ── Step 7: Test hidden states ──────────────────────────────────────

def test_hidden_states(model, model_id: str):
    from transformers import AutoTokenizer

    print("\n" + "=" * 70)
    print("STEP 7: output_hidden_states test")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer("Hello", return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    try:
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        hs = out.hidden_states
        print(f"  Count: {len(hs)}, dim={hs[0].shape[-1]}")
        print(f"  All same hidden dim: {all(h.shape[-1] == hs[0].shape[-1] for h in hs)}")
    except Exception as e:
        print(f"  ERROR: {e}")


# ── Step 8: Test chat template ──────────────────────────────────────

def test_chat_template(model_id: str):
    from transformers import AutoTokenizer

    print("\n" + "=" * 70)
    print("STEP 8: Chat template test")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    msgs = [{"role": "user", "content": "Hi"}]

    try:
        t = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        print(f"  {repr(t[:200])}")
    except Exception as e:
        print(f"  Error: {e}")

    try:
        tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        print("  enable_thinking=False: OK")
    except TypeError:
        print("  enable_thinking=False: NOT SUPPORTED")

    return tokenizer


# ── Step 9: Quick generation test ───────────────────────────────────

def test_generation(model, model_id: str):
    from transformers import AutoTokenizer

    print("\n" + "=" * 70)
    print("STEP 9: Quick generation test")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    msgs = [{"role": "user", "content": "Hi"}]
    try:
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = "Hi"

    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    try:
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"  Response: {resp}")
    except Exception as e:
        print(f"  ERROR: {e}")


# ── Step 10: VRAM report ────────────────────────────────────────────

def report_vram():
    print("\n" + "=" * 70)
    print("STEP 10: VRAM usage")
    print("=" * 70)

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_mem / 1024**3
            print(f"  GPU {i} ({torch.cuda.get_device_name(i)}): {alloc:.1f}/{total:.1f} GiB")
    else:
        print("  No CUDA device available")


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Discover model architecture for Prometheus")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--skip-load", action="store_true", help="Only inspect config, skip model loading")
    args = parser.parse_args()

    model_id = args.model

    inspect_config(model_id)

    if args.skip_load:
        print("Skipping model load (--skip-load). Done.")
        return

    model = load_model(model_id)
    dump_module_tree(model)

    layers = discover_layers(model)
    discover_steerable_modules(layers)
    discover_router(layers)
    discover_fused_weights(layers)
    test_hidden_states(model, model_id)
    test_chat_template(model_id)
    test_generation(model, model_id)
    report_vram()

    print("\n" + "=" * 70)
    print("DISCOVERY COMPLETE")
    print("=" * 70)
    print("Copy the output above and share it so we can update engine.py.")


if __name__ == "__main__":
    main()
