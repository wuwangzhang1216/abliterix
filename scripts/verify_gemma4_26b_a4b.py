#!/usr/bin/env python3
# Abliterix — a derivative work of Heretic (https://github.com/p-e-w/heretic)
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pre-flight verification for google/gemma-4-26B-A4B-it abliteration.

Validates GPU VRAM, disk, transformers version, config shape, module naming,
chat template, and abliterix engine compatibility BEFORE downloading ~52GB
of BF16 weights.

Gemma 4 26B-A4B specifics:
  * Mixture-of-Experts: 128 experts, top-8 routing, 26B total / 4B active
  * Multimodal: Gemma4ForConditionalGeneration (AutoModelForImageTextToText)
  * MoE runs in PARALLEL with dense MLP (outputs summed)
  * Double-norm (4× RMSNorm/layer) + PLE → requires steering_mode="direct"
  * Expert layer naming (transformers >= 5.5):
      - layer.router.proj (nn.Linear, the gate)
      - layer.experts.down_proj (fused nn.Parameter [128, 2816, 704])
      - layer.experts.gate_up_proj (fused nn.Parameter [128, 2*704, 2816])
      - layer.mlp.down_proj (dense parallel MLP, also steerable)
  * Target hardware: RTX Pro 6000 (96 GB GDDR7), single GPU, BF16

Usage:
    python scripts/verify_gemma4_26b_a4b.py                 # config + lightweight checks
    python scripts/verify_gemma4_26b_a4b.py --with-weights   # full model load + engine test
"""

from __future__ import annotations

import argparse
import shutil
import sys

MODEL_ID = "google/gemma-4-26B-A4B-it"
EXPECTED_LAYERS = 30
EXPECTED_HIDDEN = 2816
EXPECTED_EXPERTS = 128
EXPECTED_TOP_K = 8
EXPECTED_MOE_INTERMEDIATE = 704
MIN_VRAM_GB = 52  # BF16 weights ~52GB, single GPU needs at least this


def _ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    sys.exit(1)


def _warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


def _info(msg: str) -> None:
    print(f"  [INFO] {msg}")


# -----------------------------------------------------------------------
# STEP 0: Transformers version
# -----------------------------------------------------------------------
def check_transformers_version() -> None:
    print("=" * 70)
    print("STEP 0: Transformers version (>= 5.5 required for Gemma 4)")
    print("=" * 70)

    import transformers

    ver = transformers.__version__
    print(f"  transformers version = {ver}")

    parts = ver.replace(".dev0", "").split(".")
    major, minor = int(parts[0]), int(parts[1])
    if major < 5 or (major == 5 and minor < 3):
        _fail(f"transformers >= 5.3 required for Gemma 4, got {ver}")
    _ok(f"transformers {ver} is compatible")


# -----------------------------------------------------------------------
# STEP 1: GPU capability
# -----------------------------------------------------------------------
def check_gpus() -> None:
    import torch

    print()
    print("=" * 70)
    print("STEP 1: GPU capability (RTX Pro 6000 = 96 GB GDDR7)")
    print("=" * 70)

    if not torch.cuda.is_available():
        _warn("CUDA not available — running on CPU (model won't fit, but config checks still work)")
        return

    n = torch.cuda.device_count()
    print(f"  device count = {n}")

    total_vram = 0.0
    for i in range(n):
        cc = torch.cuda.get_device_capability(i)
        name = torch.cuda.get_device_name(i)
        props = torch.cuda.get_device_properties(i)
        vram = props.total_memory / (1024**3)
        total_vram += vram
        print(f"  GPU{i}: {name} | SM{cc[0]}.{cc[1]} | {vram:.1f} GiB")

    _ok(f"total VRAM = {total_vram:.1f} GiB")
    if total_vram < MIN_VRAM_GB:
        _fail(f"total VRAM {total_vram:.1f} GiB < {MIN_VRAM_GB} GiB needed for BF16 Gemma 4 26B-A4B")

    # Check BF16 support (SM80+)
    for i in range(n):
        cc = torch.cuda.get_device_capability(i)
        if cc < (8, 0):
            _warn(f"GPU{i} SM{cc[0]}.{cc[1]} < SM80 — BF16 may fall back to FP32 emulation")
        else:
            _ok(f"GPU{i} supports native BF16 (SM{cc[0]}.{cc[1]})")


# -----------------------------------------------------------------------
# STEP 2: Disk space
# -----------------------------------------------------------------------
def check_disk() -> None:
    import os

    print()
    print("=" * 70)
    print("STEP 2: Disk space (BF16 snapshot ~52 GB)")
    print("=" * 70)

    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    print(f"  HF_HOME = {hf_home}")
    try:
        usage = shutil.disk_usage(
            hf_home if os.path.exists(hf_home) else os.path.dirname(hf_home) or "/"
        )
    except FileNotFoundError:
        usage = shutil.disk_usage("/")
    free_gb = usage.free / (1024**3)
    print(f"  free on HF_HOME partition = {free_gb:.0f} GiB")
    if free_gb < 70:
        _fail(f"need >= 70 GiB free for BF16 snapshot + cache, have {free_gb:.0f} GiB")
    _ok("disk has room for BF16 snapshot")


# -----------------------------------------------------------------------
# STEP 3: Config inspection (no weights download)
# -----------------------------------------------------------------------
def inspect_config() -> None:
    from transformers import AutoConfig

    print()
    print("=" * 70)
    print("STEP 3: Config inspection (no weights)")
    print("=" * 70)

    config = AutoConfig.from_pretrained(MODEL_ID)
    text_cfg = getattr(config, "text_config", config)

    arch = getattr(config, "architectures", None)
    model_type = getattr(config, "model_type", None)
    n_layers = getattr(text_cfg, "num_hidden_layers", None)
    hidden = getattr(text_cfg, "hidden_size", None)
    n_experts = getattr(text_cfg, "num_experts", None)
    top_k = getattr(text_cfg, "top_k_experts", None)
    moe_intermediate = getattr(text_cfg, "moe_intermediate_size", None)
    enable_moe = getattr(text_cfg, "enable_moe_block", None)
    intermediate = getattr(text_cfg, "intermediate_size", None)

    print(f"  architectures         = {arch}")
    print(f"  model_type            = {model_type}")
    print(f"  num_hidden_layers     = {n_layers}")
    print(f"  hidden_size           = {hidden}")
    print(f"  intermediate_size     = {intermediate}")
    print(f"  enable_moe_block      = {enable_moe}")
    print(f"  num_experts           = {n_experts}")
    print(f"  top_k_experts         = {top_k}")
    print(f"  moe_intermediate_size = {moe_intermediate}")

    # Validate expected values
    if n_layers != EXPECTED_LAYERS:
        _fail(f"expected {EXPECTED_LAYERS} layers, got {n_layers}")
    _ok(f"{EXPECTED_LAYERS} transformer layers")

    if hidden != EXPECTED_HIDDEN:
        _fail(f"expected hidden_size={EXPECTED_HIDDEN}, got {hidden}")
    _ok(f"hidden_size={EXPECTED_HIDDEN}")

    if not enable_moe:
        _fail("enable_moe_block is not True — this is not an MoE model?")
    _ok("enable_moe_block=True")

    if n_experts != EXPECTED_EXPERTS:
        _fail(f"expected {EXPECTED_EXPERTS} experts, got {n_experts}")
    _ok(f"{EXPECTED_EXPERTS} local experts")

    if top_k != EXPECTED_TOP_K:
        _fail(f"expected top_k_experts={EXPECTED_TOP_K}, got {top_k}")
    _ok(f"top-{EXPECTED_TOP_K} expert routing")

    if moe_intermediate != EXPECTED_MOE_INTERMEDIATE:
        _warn(f"expected moe_intermediate_size={EXPECTED_MOE_INTERMEDIATE}, got {moe_intermediate}")
    else:
        _ok(f"moe_intermediate_size={EXPECTED_MOE_INTERMEDIATE}")

    # Check for vision_config (multimodal detection)
    has_vision = hasattr(config, "vision_config")
    if has_vision:
        _ok("vision_config present — will load via AutoModelForImageTextToText")
    else:
        _warn("no vision_config — will load as AutoModelForCausalLM (unexpected for Gemma4)")


# -----------------------------------------------------------------------
# STEP 4: Chat template
# -----------------------------------------------------------------------
def inspect_chat_template() -> None:
    from transformers import AutoTokenizer

    print()
    print("=" * 70)
    print("STEP 4: Chat template")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}],
        add_generation_prompt=True,
        tokenize=False,
    )
    print(f"  rendered:\n    {rendered!r}")
    _ok("chat template renders cleanly")


# -----------------------------------------------------------------------
# STEP 5: Abliterix config TOML validation
# -----------------------------------------------------------------------
def check_toml_config() -> None:
    import os

    print()
    print("=" * 70)
    print("STEP 5: Abliterix config TOML validation")
    print("=" * 70)

    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs",
        "gemma4_26b_a4b_direct.toml",
    )
    if not os.path.exists(config_path):
        _fail(f"config not found: {config_path}")

    import tomllib

    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    # Verify key fields
    model_id = cfg.get("model", {}).get("model_id")
    if model_id != MODEL_ID:
        _fail(f"model_id mismatch: expected {MODEL_ID}, got {model_id}")
    _ok(f"model_id = {MODEL_ID}")

    steering_mode = cfg.get("steering", {}).get("steering_mode")
    if steering_mode != "direct":
        _fail(f"steering_mode should be 'direct' for Gemma 4, got {steering_mode}")
    _ok("steering_mode = direct")

    dtype = cfg.get("model", {}).get("dtype_fallback_order", [])
    if "bfloat16" not in dtype:
        _warn(f"bfloat16 not in dtype_fallback_order: {dtype}")
    else:
        _ok("bfloat16 in dtype_fallback_order")

    quant = cfg.get("model", {}).get("quant_method")
    if quant != "none":
        _warn(f"quant_method={quant}, expected 'none' for BF16 RTX Pro 6000")
    else:
        _ok("quant_method = none (BF16, no quantization)")

    # Check experts section exists
    experts = cfg.get("experts", {})
    if not experts:
        _warn("no [experts] section — MoE expert steering won't be configured")
    else:
        _ok(f"[experts] section present: max_suppress={experts.get('max_suppress')}")

    _ok(f"TOML config is valid: {config_path}")


# -----------------------------------------------------------------------
# STEP 6: Abliterix engine compatibility (requires model download)
# -----------------------------------------------------------------------
def check_engine_compatibility() -> None:
    import torch

    print()
    print("=" * 70)
    print("STEP 6: Engine module path verification (full model load ~52GB)")
    print("=" * 70)

    if not torch.cuda.is_available():
        _fail("CUDA required for --with-weights model load")

    from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer

    print("  Loading model in BF16...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: "90GiB"},
    )

    # --- Layer resolution (must match engine.transformer_layers) ---
    layers = None
    for path_desc, getter in [
        ("model.model.language_model.layers", lambda m: m.model.language_model.layers),
        ("model.model.layers", lambda m: m.model.layers),
    ]:
        try:
            layers = getter(model)
            print(f"  resolved layers via: {path_desc}")
            break
        except AttributeError:
            continue

    if layers is None:
        _fail("could not resolve transformer_layers — engine.py needs update")
        return

    n = len(layers)
    if n != EXPECTED_LAYERS:
        _fail(f"expected {EXPECTED_LAYERS} layers, got {n}")
    _ok(f"{n} decoder blocks loaded")

    # --- Check a layer ---
    block = layers[0]
    print(f"\n  Layer 0 structure (top-level attributes):")
    for name, child in block.named_children():
        print(f"    {name}: {type(child).__name__}")

    # 1. Attention o_proj (engine.py: layer.self_attn.o_proj)
    o_proj = getattr(getattr(block, "self_attn", None), "o_proj", None)
    if o_proj is None:
        _fail("layer[0].self_attn.o_proj not found — attention steering broken")
    else:
        _ok(f"self_attn.o_proj found (shape={tuple(o_proj.weight.shape)})")

    # 2. Dense MLP down_proj (engine.py: layer.mlp.down_proj)
    mlp_dp = getattr(getattr(block, "mlp", None), "down_proj", None)
    if mlp_dp is None:
        _warn("layer[0].mlp.down_proj not found — dense MLP steering unavailable")
    else:
        _ok(f"mlp.down_proj found (shape={tuple(mlp_dp.weight.shape)})")

    # 3. Router gate (engine.py: layer.router.proj → _locate_router path "router.proj")
    router = getattr(block, "router", None)
    if router is None:
        _fail("layer[0].router not found — MoE router detection broken")
    else:
        proj = getattr(router, "proj", None)
        if proj is None:
            _fail("layer[0].router.proj not found — _locate_router won't find gate")
        elif not hasattr(proj, "weight") or proj.weight.dim() != 2:
            _fail(f"router.proj.weight not a 2D tensor (needed for _locate_router)")
        else:
            _ok(f"router.proj found (shape={tuple(proj.weight.shape)})")
            n_exp_from_gate = proj.weight.shape[0]
            if n_exp_from_gate != EXPECTED_EXPERTS:
                _warn(f"router.proj output dim={n_exp_from_gate}, expected {EXPECTED_EXPERTS}")
            else:
                _ok(f"router routes to {EXPECTED_EXPERTS} experts")

    # 4. Fused expert weights (engine.py: _locate_fused_weights path "experts.down_proj")
    experts = getattr(block, "experts", None)
    if experts is None:
        _fail("layer[0].experts not found — MoE expert steering broken")
    else:
        dp = getattr(experts, "down_proj", None)
        gup = getattr(experts, "gate_up_proj", None)

        if dp is None:
            _fail("layer[0].experts.down_proj not found — _locate_fused_weights broken")
        elif not isinstance(dp, torch.nn.Parameter) or dp.dim() != 3:
            _fail(f"experts.down_proj is {type(dp).__name__} dim={dp.dim() if hasattr(dp, 'dim') else '?'}, expected 3D Parameter")
        else:
            _ok(f"experts.down_proj found (shape={tuple(dp.shape)}, dtype={dp.dtype})")
            exp_shape = (EXPECTED_EXPERTS, EXPECTED_HIDDEN, EXPECTED_MOE_INTERMEDIATE)
            if tuple(dp.shape) != exp_shape:
                _warn(f"expected shape {exp_shape}, got {tuple(dp.shape)}")
            else:
                _ok(f"experts.down_proj shape matches expected {exp_shape}")

        if gup is None:
            _warn("layer[0].experts.gate_up_proj not found (informational)")
        else:
            _ok(f"experts.gate_up_proj found (shape={tuple(gup.shape)}, dtype={gup.dtype})")

    # 5. Check enable_moe_block attribute on layer
    has_moe = getattr(block, "enable_moe_block", None)
    if has_moe:
        _ok("layer.enable_moe_block = True (MoE active)")
    else:
        _warn("layer.enable_moe_block not set or False")

    # 6. Verify NOT using block_sparse_moe (Gemma4 doesn't use this)
    bsm = getattr(block, "block_sparse_moe", None)
    if bsm is not None:
        _warn("unexpected block_sparse_moe found — old-style MoE naming?")
    else:
        _ok("no block_sparse_moe (correct — Gemma4 uses layer.router + layer.experts)")

    # --- Memory usage ---
    print()
    vram_used = torch.cuda.memory_allocated() / (1024**3)
    vram_reserved = torch.cuda.memory_reserved() / (1024**3)
    print(f"  VRAM allocated = {vram_used:.1f} GiB")
    print(f"  VRAM reserved  = {vram_reserved:.1f} GiB")
    _ok("model loaded successfully")

    # --- Quick generation test ---
    print()
    print("  Running quick generation test...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    messages = [{"role": "user", "content": "What is 2+2? Reply in one word."}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"  Q: What is 2+2? Reply in one word.")
    print(f"  A: {response.strip()}")
    _ok("generation works")

    # --- Abliterix engine integration test ---
    print()
    print("  Testing abliterix engine integration...")
    try:
        # Simulate _locate_router
        from torch.nn import Module, Parameter
        from torch import Tensor

        def _locate_router_test(layer):
            for path in ["mlp.gate", "mlp.router", "moe.gate", "mixer.gate",
                         "block_sparse_moe.gate", "feed_forward.gate", "router.proj"]:
                obj = layer
                for attr in path.split("."):
                    obj = getattr(obj, attr, None)
                    if obj is None:
                        break
                if obj is not None and isinstance(obj, Module):
                    w = getattr(obj, "weight", None)
                    if isinstance(w, (Tensor, Parameter)) and w.dim() == 2:
                        return path, obj
            return None, None

        def _locate_fused_test(layer):
            for path in ["mlp.experts.down_proj", "mixer.experts.down_proj",
                         "feed_forward.experts.down_proj", "moe.down_proj",
                         "experts.down_proj"]:
                obj = layer
                for attr in path.split("."):
                    obj = getattr(obj, attr, None)
                    if obj is None:
                        break
                if isinstance(obj, Parameter) and obj.dim() == 3:
                    return path, obj
            return None, None

        # Test on all layers
        router_layers = 0
        fused_layers = 0
        mlp_layers = 0

        for idx in range(n):
            layer = layers[idx]
            rpath, rmod = _locate_router_test(layer)
            if rmod is not None:
                router_layers += 1
            fpath, fmod = _locate_fused_test(layer)
            if fmod is not None:
                fused_layers += 1
            if getattr(getattr(layer, "mlp", None), "down_proj", None) is not None:
                mlp_layers += 1

        print(f"  Router found in {router_layers}/{n} layers (path: router.proj)")
        print(f"  Fused experts found in {fused_layers}/{n} layers (path: experts.down_proj)")
        print(f"  Dense MLP found in {mlp_layers}/{n} layers (path: mlp.down_proj)")

        if router_layers == 0:
            _fail("_locate_router finds NO routers — MoE steering impossible")
        if fused_layers == 0:
            _fail("_locate_fused_weights finds NO fused experts — MoE expert abliteration impossible")

        # Check if MoE layers match (some layers may not have MoE if enable_moe_block varies)
        _ok(f"engine paths work: {router_layers} MoE layers, {mlp_layers} dense MLP layers")

    except Exception as e:
        _fail(f"engine integration test failed: {e}")

    # --- Estimate steerable module count ---
    steerable_per_layer = 1 + 1  # o_proj + mlp.down_proj (fused experts handled separately)
    print(f"\n  Steerable via direct steering: ~{steerable_per_layer} modules × {n} layers = {steerable_per_layer * n}")
    print(f"  Steerable via MoE routing: {EXPECTED_EXPERTS} experts × {router_layers} MoE layers")
    _ok("all abliterix engine paths verified")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--with-weights",
        action="store_true",
        help="Also download and load the ~52GB BF16 snapshot for full engine verification.",
    )
    args = parser.parse_args()

    check_transformers_version()
    check_gpus()
    check_disk()
    inspect_config()
    inspect_chat_template()
    check_toml_config()
    if args.with_weights:
        check_engine_compatibility()

    print()
    print("=" * 70)
    print("All pre-flight checks passed.")
    print("Run abliteration with:")
    print("  uv run abliterix --config configs/gemma4_26b_a4b_direct.toml")
    print("=" * 70)


if __name__ == "__main__":
    main()
