#!/usr/bin/env python3
# Abliterix — a derivative work of Heretic (https://github.com/p-e-w/heretic)
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pre-flight verification for zai-org/GLM-4.7-Flash abliteration on RTX Pro 6000.

GLM-4.7-Flash specifics:
  * Glm4MoeLiteForCausalLM (text-only, MoE with MLA attention)
  * 47 layers: layer 0 dense (first_k_dense_replace=1), layers 1-46 MoE
  * 64 routed experts + 1 shared expert, top-4 routing
  * MLA attention: q_a_proj/q_b_proj/kv_a_proj_with_mqa/kv_b_proj/o_proj
  * hidden=2048, intermediate=10240, moe_intermediate=1536
  * q_lora_rank=768, kv_lora_rank=512
  * ~31B BF16 params (~62 GB)
  * Target hardware: RTX Pro 6000 96 GB single GPU, BF16

Usage:
    python scripts/verify_glm47_flash.py                 # config + lightweight checks
    python scripts/verify_glm47_flash.py --with-weights  # full model load + module discovery
"""

from __future__ import annotations

import argparse
import shutil
import sys

MODEL_ID = "zai-org/GLM-4.7-Flash"
EXPECTED_LAYERS = 47
EXPECTED_HIDDEN = 2048
EXPECTED_ROUTED_EXPERTS = 64
EXPECTED_TOP_K = 4
EXPECTED_FIRST_DENSE = 1
MIN_VRAM_GB = 70  # BF16 ~62GB + activations + KV cache


def _ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    sys.exit(1)


def _warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


def _info(msg: str) -> None:
    print(f"  [INFO] {msg}")


def check_transformers_version() -> None:
    print("=" * 70)
    print("STEP 0: Transformers version (>= 5.3 for glm4_moe_lite)")
    print("=" * 70)
    import transformers

    ver = transformers.__version__
    print(f"  transformers version = {ver}")
    parts = ver.replace(".dev0", "").split(".")
    major, minor = int(parts[0]), int(parts[1])
    if major < 5 or (major == 5 and minor < 3):
        _fail(f"transformers >= 5.3 required, got {ver}")
    _ok(f"transformers {ver} is compatible")


def check_gpus() -> None:
    import torch

    print()
    print("=" * 70)
    print("STEP 1: GPU capability (RTX Pro 6000 = 96 GB)")
    print("=" * 70)
    if not torch.cuda.is_available():
        _warn("CUDA not available — config checks still work")
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
        _fail(f"total VRAM {total_vram:.1f} GiB < {MIN_VRAM_GB} GiB needed")
    for i in range(n):
        cc = torch.cuda.get_device_capability(i)
        if cc < (8, 0):
            _warn(f"GPU{i} SM{cc[0]}.{cc[1]} < SM80 — BF16 may fall back to FP32")
        else:
            _ok(f"GPU{i} supports native BF16 (SM{cc[0]}.{cc[1]})")


def check_disk() -> None:
    import os

    print()
    print("=" * 70)
    print("STEP 2: Disk space (BF16 snapshot ~62 GB)")
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
    if free_gb < 100:
        _fail(f"need >= 100 GiB free, have {free_gb:.0f} GiB")
    _ok("disk has room")


def inspect_config() -> None:
    from transformers import AutoConfig

    print()
    print("=" * 70)
    print("STEP 3: Config inspection (no weights)")
    print("=" * 70)
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)

    arch = getattr(config, "architectures", None)
    model_type = getattr(config, "model_type", None)
    n_layers = getattr(config, "num_hidden_layers", None)
    hidden = getattr(config, "hidden_size", None)
    n_routed = getattr(config, "n_routed_experts", None)
    n_shared = getattr(config, "n_shared_experts", None)
    top_k = getattr(config, "num_experts_per_tok", None)
    moe_inter = getattr(config, "moe_intermediate_size", None)
    first_dense = getattr(config, "first_k_dense_replace", None)
    q_lora = getattr(config, "q_lora_rank", None)
    kv_lora = getattr(config, "kv_lora_rank", None)

    print(f"  architectures        = {arch}")
    print(f"  model_type           = {model_type}")
    print(f"  num_hidden_layers    = {n_layers}")
    print(f"  hidden_size          = {hidden}")
    print(f"  first_k_dense_replace= {first_dense}")
    print(f"  n_routed_experts     = {n_routed}")
    print(f"  n_shared_experts     = {n_shared}")
    print(f"  num_experts_per_tok  = {top_k}")
    print(f"  moe_intermediate_size= {moe_inter}")
    print(f"  q_lora_rank          = {q_lora}")
    print(f"  kv_lora_rank         = {kv_lora}")

    if n_layers != EXPECTED_LAYERS:
        _fail(f"expected {EXPECTED_LAYERS} layers, got {n_layers}")
    _ok(f"{EXPECTED_LAYERS} transformer layers")
    if hidden != EXPECTED_HIDDEN:
        _fail(f"expected hidden_size={EXPECTED_HIDDEN}, got {hidden}")
    _ok(f"hidden_size={EXPECTED_HIDDEN}")
    if n_routed != EXPECTED_ROUTED_EXPERTS:
        _fail(f"expected {EXPECTED_ROUTED_EXPERTS} routed experts, got {n_routed}")
    _ok(f"{EXPECTED_ROUTED_EXPERTS} routed experts")
    if top_k != EXPECTED_TOP_K:
        _fail(f"expected top-{EXPECTED_TOP_K}, got top-{top_k}")
    _ok(f"top-{EXPECTED_TOP_K} routing")
    if first_dense != EXPECTED_FIRST_DENSE:
        _warn(f"expected first_k_dense_replace={EXPECTED_FIRST_DENSE}, got {first_dense}")
    else:
        _ok(f"first_k_dense_replace={EXPECTED_FIRST_DENSE} (layer 0 is dense MLP)")
    if q_lora and kv_lora:
        _ok(f"MLA attention (q_lora_rank={q_lora}, kv_lora_rank={kv_lora})")


def inspect_chat_template() -> None:
    from transformers import AutoTokenizer

    print()
    print("=" * 70)
    print("STEP 4: Chat template")
    print("=" * 70)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}],
        add_generation_prompt=True,
        tokenize=False,
    )
    print(f"  rendered:\n    {rendered!r}")
    _ok("chat template renders cleanly")


def check_toml_config() -> None:
    import os

    print()
    print("=" * 70)
    print("STEP 5: Abliterix config TOML validation")
    print("=" * 70)
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs",
        "glm4.7_flash.toml",
    )
    if not os.path.exists(config_path):
        _fail(f"config not found: {config_path}")
    import tomllib

    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    model_id = cfg.get("model", {}).get("model_id")
    if model_id != MODEL_ID:
        _fail(f"model_id mismatch: expected {MODEL_ID}, got {model_id}")
    _ok(f"model_id = {MODEL_ID}")

    steering_mode = cfg.get("steering", {}).get("steering_mode")
    if steering_mode != "direct":
        _fail(f"steering_mode should be 'direct' for v2, got {steering_mode}")
    _ok("steering_mode = direct")

    _ok(f"TOML config is valid: {config_path}")


def check_engine_compatibility() -> None:
    import torch

    print()
    print("=" * 70)
    print("STEP 6: Engine module path verification (full model load ~62GB)")
    print("=" * 70)
    if not torch.cuda.is_available():
        _fail("CUDA required for --with-weights model load")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("  Loading model in BF16...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        max_memory={0: "90GiB"},
    )

    layers = None
    for path_desc, getter in [
        ("model.model.layers", lambda m: m.model.layers),
        ("model.layers", lambda m: m.layers),
    ]:
        try:
            layers = getter(model)
            print(f"  resolved layers via: {path_desc}")
            break
        except AttributeError:
            continue
    if layers is None:
        _fail("could not resolve transformer_layers")
        return

    n = len(layers)
    if n != EXPECTED_LAYERS:
        _fail(f"expected {EXPECTED_LAYERS} layers, got {n}")
    _ok(f"{n} decoder blocks loaded")

    # --- Layer 0 (dense) inspection ---
    print("\n  ====== Layer 0 (dense) ======")
    block0 = layers[0]
    print("  Layer 0 children:")
    for name, child in block0.named_children():
        print(f"    {name}: {type(child).__name__}")

    sa0 = getattr(block0, "self_attn", None)
    if sa0 is None:
        _fail("layer[0].self_attn missing")
    else:
        print("\n  layer[0].self_attn children:")
        for name, child in sa0.named_children():
            shape = ""
            try:
                shape = f" shape={tuple(child.weight.shape)}"
            except Exception:
                pass
            print(f"    {name}: {type(child).__name__}{shape}")

    mlp0 = getattr(block0, "mlp", None)
    if mlp0 is None:
        _fail("layer[0].mlp missing")
    else:
        print("\n  layer[0].mlp children:")
        for name, child in mlp0.named_children():
            shape = ""
            try:
                shape = f" shape={tuple(child.weight.shape)}"
            except Exception:
                pass
            print(f"    {name}: {type(child).__name__}{shape}")

    # --- Layer 1 (MoE) inspection ---
    print("\n  ====== Layer 1 (MoE) ======")
    block1 = layers[1]
    print("  Layer 1 children:")
    for name, child in block1.named_children():
        print(f"    {name}: {type(child).__name__}")

    sa1 = getattr(block1, "self_attn", None)
    if sa1 is not None:
        print("\n  layer[1].self_attn children (MLA):")
        for name, child in sa1.named_children():
            shape = ""
            try:
                shape = f" shape={tuple(child.weight.shape)}"
            except Exception:
                pass
            print(f"    {name}: {type(child).__name__}{shape}")

    mlp1 = getattr(block1, "mlp", None)
    if mlp1 is not None:
        print("\n  layer[1].mlp children:")
        for name, child in mlp1.named_children():
            print(f"    {name}: {type(child).__name__}")

        # Probe expert layout
        experts = getattr(mlp1, "experts", None)
        if experts is not None:
            print(f"  experts type: {type(experts).__name__}")
            try:
                n_experts = len(experts)
                print(f"  num experts: {n_experts}")
                e0 = experts[0]
                print(f"  experts[0] children:")
                for name, child in e0.named_children():
                    shape = ""
                    try:
                        shape = f" shape={tuple(child.weight.shape)}"
                    except Exception:
                        pass
                    print(f"    {name}: {type(child).__name__}{shape}")
            except Exception as e:
                print(f"  could not enumerate experts: {e}")

        shared = getattr(mlp1, "shared_experts", None) or getattr(mlp1, "shared_expert", None)
        if shared is not None:
            print(f"  shared_experts type: {type(shared).__name__}")
            for name, child in shared.named_children():
                shape = ""
                try:
                    shape = f" shape={tuple(child.weight.shape)}"
                except Exception:
                    pass
                print(f"    {name}: {type(child).__name__}{shape}")

    # --- abliterix engine probe ---
    print("\n  ====== Abliterix steerable_modules() probe ======")
    sys.path.insert(0, "/workspace/abliterix/src")
    sys.path.insert(0, "/workspace/abliterix")
    try:
        from abliterix.core.engine import SteeringEngine
        from abliterix.settings import AbliterixConfig

        # Bypass __init__, monkey-patch the loaded model in
        engine = SteeringEngine.__new__(SteeringEngine)
        engine.model = model
        engine.transformer_layers = layers
        engine.config = AbliterixConfig.__new__(AbliterixConfig)

        for li in [0, 1, n // 2]:
            mods = engine.steerable_modules(li)
            print(f"\n  layer[{li}] discovered components:")
            for name, modlist in sorted(mods.items()):
                print(f"    {name}: {len(modlist)} module(s)")
    except Exception as e:
        _warn(f"engine probe failed (informational): {e}")

    print()
    vram_used = torch.cuda.memory_allocated() / (1024**3)
    print(f"  VRAM allocated = {vram_used:.1f} GiB")
    _ok("model loaded successfully")

    # --- Quick generation test ---
    print()
    print("  Running quick generation test...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    messages = [{"role": "user", "content": "What is 2+2? Reply in one word."}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"  Q: What is 2+2? Reply in one word.")
    print(f"  A: {response.strip()}")
    _ok("generation works")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--with-weights",
        action="store_true",
        help="Also download and load BF16 snapshot for full module discovery.",
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
    print("  AX_CONFIG=configs/glm4.7_flash.toml uv run abliterix")
    print("=" * 70)


if __name__ == "__main__":
    main()
