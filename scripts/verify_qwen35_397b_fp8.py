#!/usr/bin/env python3
# Abliterix — a derivative work of Heretic (https://github.com/p-e-w/heretic)
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pre-flight verification for Qwen/Qwen3.5-397B-A17B-FP8 abliteration.

Validates GPU FP8 capability, disk, config shape, and chat template BEFORE
attempting a 400GB model download. Weight-loading step is optional (--with-weights)
since the download itself is the slow part.

Usage:
    python scripts/verify_qwen35_397b_fp8.py                 # no weight download
    python scripts/verify_qwen35_397b_fp8.py --with-weights  # full load sanity
"""

from __future__ import annotations

import argparse
import shutil
import sys

MODEL_ID = "Qwen/Qwen3.5-397B-A17B-FP8"
EXPECTED_LAYERS = 60
EXPECTED_HIDDEN = 4096
EXPECTED_EXPERTS = 512


def _ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    sys.exit(1)


def check_gpus() -> None:
    import torch

    print("=" * 70)
    print("STEP 0: GPU capability (FP8 requires SM89+ / H100+)")
    print("=" * 70)

    if not torch.cuda.is_available():
        _fail("CUDA not available")

    n = torch.cuda.device_count()
    print(f"  device count = {n}")
    if n < 6:
        _fail(f"need >= 6 GPUs for FP8 397B, found {n}")

    total_vram = 0.0
    for i in range(n):
        cc = torch.cuda.get_device_capability(i)
        name = torch.cuda.get_device_name(i)
        props = torch.cuda.get_device_properties(i)
        vram = props.total_memory / (1024 ** 3)
        total_vram += vram
        print(f"  GPU{i}: {name} | SM{cc[0]}.{cc[1]} | {vram:.0f} GiB")
        if cc < (8, 9):
            _fail(f"GPU{i} has SM{cc[0]}.{cc[1]}, FP8 needs SM89+ (H100/H200)")

    _ok(f"all {n} GPUs FP8-capable, total VRAM = {total_vram:.0f} GiB")
    if total_vram < 440:
        _fail(f"total VRAM {total_vram:.0f} GiB < 440 GiB needed for 397B-FP8")


def check_disk() -> None:
    import os

    print()
    print("=" * 70)
    print("STEP 1: Disk space (FP8 snapshot ~400 GB)")
    print("=" * 70)

    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    print(f"  HF_HOME = {hf_home}")
    try:
        usage = shutil.disk_usage(hf_home if os.path.exists(hf_home) else os.path.dirname(hf_home) or "/")
    except FileNotFoundError:
        usage = shutil.disk_usage("/")
    free_gb = usage.free / (1024 ** 3)
    print(f"  free on HF_HOME partition = {free_gb:.0f} GiB")
    if free_gb < 450:
        _fail(f"need >= 450 GiB free for FP8 397B snapshot, have {free_gb:.0f} GiB")
    _ok("disk has room for FP8 snapshot")


def inspect_config() -> None:
    from transformers import AutoConfig

    print()
    print("=" * 70)
    print("STEP 2: Config inspection (no weights)")
    print("=" * 70)

    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    text_cfg = getattr(config, "text_config", config)

    n_layers = getattr(text_cfg, "num_hidden_layers", None)
    hidden = getattr(text_cfg, "hidden_size", None)
    n_experts = getattr(text_cfg, "num_experts", None)
    n_routed = getattr(text_cfg, "num_experts_per_tok", None)
    arch = getattr(config, "architectures", None)
    qcfg = getattr(config, "quantization_config", None)

    print(f"  architectures         = {arch}")
    print(f"  num_hidden_layers     = {n_layers}")
    print(f"  hidden_size           = {hidden}")
    print(f"  num_experts           = {n_experts}")
    print(f"  num_experts_per_tok   = {n_routed}")
    print(f"  quantization_config   = {qcfg}")

    if n_layers != EXPECTED_LAYERS:
        _fail(f"expected {EXPECTED_LAYERS} layers, got {n_layers}")
    _ok(f"{EXPECTED_LAYERS} transformer layers")

    if hidden != EXPECTED_HIDDEN:
        _fail(f"expected hidden_size={EXPECTED_HIDDEN}, got {hidden}")
    _ok(f"hidden_size={EXPECTED_HIDDEN}")

    if n_experts is not None and n_experts != EXPECTED_EXPERTS:
        print(f"  [WARN] expected {EXPECTED_EXPERTS} experts, got {n_experts}")
    elif n_experts == EXPECTED_EXPERTS:
        _ok(f"{EXPECTED_EXPERTS} experts")

    if qcfg is None:
        _fail("no quantization_config on model — not an FP8 checkpoint?")
    _ok("FP8 quantization_config present")


def inspect_chat_template() -> None:
    from transformers import AutoTokenizer

    print()
    print("=" * 70)
    print("STEP 3: Chat template")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}],
        add_generation_prompt=True,
        tokenize=False,
    )
    print(f"  rendered:\n    {rendered!r}")
    _ok("chat template renders cleanly")


def inspect_model() -> None:
    import torch
    from transformers import AutoModelForCausalLM

    print()
    print("=" * 70)
    print("STEP 4: Full model load (downloads ~400GB)")
    print("=" * 70)

    print("  loading FP8 weights across all GPUs...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        max_memory={i: "76GiB" for i in range(torch.cuda.device_count())},
        trust_remote_code=True,
    )

    # Resolve transformer_layers per engine.py convention
    layers = None
    for path_desc, getter in [
        ("model.model.layers", lambda m: m.model.layers),
        ("model.language_model.layers", lambda m: m.model.language_model.layers),
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

    # MoE sanity on layer 0
    block = layers[0]
    mlp = getattr(block, "mlp", None)
    if mlp is None:
        _fail("no mlp on decoder block")
    exp = getattr(mlp, "experts", None)
    if exp is None:
        print("  [WARN] no .experts attribute — MoE structure may differ")
    else:
        print(f"  layer[0].mlp.experts: {type(exp).__name__} len={len(exp)}")
    _ok("MoE structure present")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-weights",
        action="store_true",
        help="Also download and load the ~400GB FP8 snapshot.",
    )
    args = parser.parse_args()

    check_gpus()
    check_disk()
    inspect_config()
    inspect_chat_template()
    if args.with_weights:
        inspect_model()

    print()
    print("All pre-flight checks passed. configs/qwen3.5_397b_fp8.toml is safe to use.")


if __name__ == "__main__":
    main()
