#!/usr/bin/env python3
# Abliterix — a derivative work of Heretic (https://github.com/p-e-w/heretic)
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pre-flight verification for google/gemma-4-E4B-it abliteration on RTX 6000 Ada.

Validates GPU VRAM, disk, transformers version, config shape, module naming,
chat template, and abliterix engine compatibility BEFORE downloading weights.

Gemma 4 E4B specifics:
  * Multimodal: Gemma4ForConditionalGeneration (text + vision + audio)
  * Dense (enable_moe_block=False) — no MoE, no router, no fused experts
  * 42 text decoder layers, hidden=2560, intermediate=10240
  * use_double_wide_mlp=False (different from E2B which has it True)
  * num_kv_shared_layers=18 — KV cache shared across layers
  * num_attention_heads=8, num_key_value_heads=2 (less aggressive GQA than E2B)
  * Per-Layer Embeddings (PLE) + 4× RMSNorm → requires steering_mode="direct"
  * ~8B BF16 params (text + vision + audio combined)
  * Target hardware: RTX 6000 Ada 48GB single GPU, BF16

Usage:
    python scripts/verify_gemma4_e4b.py                 # config + lightweight checks
    python scripts/verify_gemma4_e4b.py --with-weights  # full model load + engine test
"""

from __future__ import annotations

import argparse
import shutil
import sys

MODEL_ID = "google/gemma-4-E4B-it"
EXPECTED_LAYERS = 42
EXPECTED_HIDDEN = 2560
EXPECTED_INTERMEDIATE = 10240
EXPECTED_KV_SHARED = 18
MIN_VRAM_GB = 24  # BF16 ~16-18GB + activations + KV cache during batched gen


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
    print("STEP 1: GPU capability (RTX 6000 Ada = 48 GB GDDR6)")
    print("=" * 70)

    if not torch.cuda.is_available():
        _warn("CUDA not available — config checks still work but model can't load")
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
        _fail(f"total VRAM {total_vram:.1f} GiB < {MIN_VRAM_GB} GiB recommended")

    for i in range(n):
        cc = torch.cuda.get_device_capability(i)
        if cc < (8, 0):
            _warn(f"GPU{i} SM{cc[0]}.{cc[1]} < SM80 — BF16 may fall back to FP32")
        else:
            _ok(f"GPU{i} supports native BF16 (SM{cc[0]}.{cc[1]})")


# -----------------------------------------------------------------------
# STEP 2: Disk space
# -----------------------------------------------------------------------
def check_disk() -> None:
    import os

    print()
    print("=" * 70)
    print("STEP 2: Disk space (BF16 snapshot ~18 GB)")
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
    if free_gb < 40:
        _fail(f"need >= 40 GiB free for snapshot + cache, have {free_gb:.0f} GiB")
    _ok("disk has room for snapshot")


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
    intermediate = getattr(text_cfg, "intermediate_size", None)
    enable_moe = getattr(text_cfg, "enable_moe_block", None)
    n_kv_shared = getattr(text_cfg, "num_kv_shared_layers", None)
    double_wide = getattr(text_cfg, "use_double_wide_mlp", None)
    num_attn = getattr(text_cfg, "num_attention_heads", None)
    num_kv = getattr(text_cfg, "num_key_value_heads", None)

    print(f"  architectures           = {arch}")
    print(f"  model_type              = {model_type}")
    print(f"  num_hidden_layers       = {n_layers}")
    print(f"  hidden_size             = {hidden}")
    print(f"  intermediate_size       = {intermediate}")
    print(f"  enable_moe_block        = {enable_moe}")
    print(f"  use_double_wide_mlp     = {double_wide}")
    print(f"  num_kv_shared_layers    = {n_kv_shared}")
    print(f"  num_attention_heads     = {num_attn}")
    print(f"  num_key_value_heads     = {num_kv}")

    if n_layers != EXPECTED_LAYERS:
        _fail(f"expected {EXPECTED_LAYERS} layers, got {n_layers}")
    _ok(f"{EXPECTED_LAYERS} transformer layers")

    if hidden != EXPECTED_HIDDEN:
        _fail(f"expected hidden_size={EXPECTED_HIDDEN}, got {hidden}")
    _ok(f"hidden_size={EXPECTED_HIDDEN}")

    if intermediate != EXPECTED_INTERMEDIATE:
        _warn(f"expected intermediate_size={EXPECTED_INTERMEDIATE}, got {intermediate}")
    else:
        _ok(f"intermediate_size={EXPECTED_INTERMEDIATE}")

    if enable_moe:
        _fail("enable_moe_block=True — this should be a dense model, not MoE")
    _ok("enable_moe_block=False (dense)")

    if double_wide:
        _warn("use_double_wide_mlp=True — unexpected for E4B (E2B has this, E4B does not)")
    else:
        _ok("use_double_wide_mlp=False (correct for E4B)")

    if n_kv_shared != EXPECTED_KV_SHARED:
        _warn(f"expected num_kv_shared_layers={EXPECTED_KV_SHARED}, got {n_kv_shared}")
    else:
        _ok(f"num_kv_shared_layers={EXPECTED_KV_SHARED}")

    has_vision = hasattr(config, "vision_config")
    has_audio = hasattr(config, "audio_config")
    if has_vision:
        _ok("vision_config present — will load via AutoModelForImageTextToText")
    else:
        _warn("no vision_config (unexpected for Gemma 4 E4B)")
    if has_audio:
        _ok("audio_config present (multimodal: text + vision + audio)")
    else:
        _info("no audio_config")


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
        "gemma4_e4b.toml",
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
        _fail(f"steering_mode should be 'direct' for Gemma 4, got {steering_mode}")
    _ok("steering_mode = direct")

    dtype = cfg.get("model", {}).get("dtype_fallback_order", [])
    if "bfloat16" not in dtype:
        _warn(f"bfloat16 not in dtype_fallback_order: {dtype}")
    else:
        _ok("bfloat16 in dtype_fallback_order")

    quant = cfg.get("model", {}).get("quant_method")
    if quant != "none":
        _warn(f"quant_method={quant}, expected 'none' for BF16 RTX 6000 Ada")
    else:
        _ok("quant_method = none (BF16, no quantization)")

    if cfg.get("experts"):
        _warn("[experts] section present but E4B is dense — will be ignored")
    else:
        _ok("no [experts] section (correct — E4B is dense)")

    _ok(f"TOML config is valid: {config_path}")


# -----------------------------------------------------------------------
# STEP 6: Engine compatibility (requires model download)
# -----------------------------------------------------------------------
def check_engine_compatibility() -> None:
    import torch

    print()
    print("=" * 70)
    print("STEP 6: Engine module path verification (full model load ~18GB)")
    print("=" * 70)

    if not torch.cuda.is_available():
        _fail("CUDA required for --with-weights model load")

    from transformers import AutoModelForImageTextToText, AutoTokenizer

    print("  Loading model in BF16...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: "46GiB"},
    )

    layers = None
    for path_desc, getter in [
        ("model.model.language_model.layers", lambda m: m.model.language_model.layers),
        ("model.language_model.layers", lambda m: m.language_model.layers),
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

    block = layers[0]
    print("\n  Layer 0 structure (top-level attributes):")
    for name, child in block.named_children():
        print(f"    {name}: {type(child).__name__}")

    self_attn = getattr(block, "self_attn", None)
    if self_attn is None:
        _fail("layer[0].self_attn not found")
    else:
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            mod = getattr(self_attn, name, None)
            if mod is None:
                _warn(f"self_attn.{name} not found")
            else:
                _ok(f"self_attn.{name} found (shape={tuple(mod.weight.shape)})")

    mlp = getattr(block, "mlp", None)
    if mlp is None:
        _fail("layer[0].mlp not found — MLP steering broken")
    else:
        dp = getattr(mlp, "down_proj", None)
        if dp is None:
            _fail("layer[0].mlp.down_proj not found")
        else:
            _ok(f"mlp.down_proj found (shape={tuple(dp.weight.shape)})")
        for name in ("gate_proj", "up_proj"):
            mod = getattr(mlp, name, None)
            if mod is not None:
                _ok(f"mlp.{name} found (shape={tuple(mod.weight.shape)})")

    if getattr(block, "router", None) is not None:
        _warn("layer[0].router found — unexpected for dense E4B")
    else:
        _ok("no layer.router (correct — dense model)")
    if getattr(block, "experts", None) is not None:
        _warn("layer[0].experts found — unexpected for dense E4B")
    else:
        _ok("no layer.experts (correct — dense model)")

    print()
    vram_used = torch.cuda.memory_allocated() / (1024**3)
    vram_reserved = torch.cuda.memory_reserved() / (1024**3)
    print(f"  VRAM allocated = {vram_used:.1f} GiB")
    print(f"  VRAM reserved  = {vram_reserved:.1f} GiB")
    if vram_used > 36:
        _warn(f"VRAM {vram_used:.1f} GiB is high for 48GB card under batch=64 generation")
    _ok("model loaded successfully")

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

    print()
    _ok("all abliterix engine paths verified")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--with-weights",
        action="store_true",
        help="Also download and load the BF16 snapshot for full engine verification.",
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
    print("  AX_CONFIG=configs/gemma4_e4b.toml uv run abliterix")
    print("=" * 70)


if __name__ == "__main__":
    main()
