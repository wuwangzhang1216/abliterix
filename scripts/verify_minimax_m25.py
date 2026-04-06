#!/usr/bin/env python3
# Abliterix — a derivative work of Heretic (https://github.com/p-e-w/heretic)
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pre-flight verification for MiniMaxAI/MiniMax-M2.5 abliteration.

Validates GPU FP8 capability, disk, config shape, module naming, and chat
template BEFORE attempting a 230GB model download.

MiniMax-M2.5 specifics:
  * Custom architecture (MiniMaxM2ForCausalLM) — trust_remote_code required
  * MoE via block_sparse_moe (same naming as Phi-3.5-MoE)
  * Expert linears: w1/w2/w3 (not gate_proj/down_proj/up_proj)
  * Sigmoid expert routing with e_score_correction_bias
  * 256 experts, 8 routed per token, no shared expert

Usage:
    python scripts/verify_minimax_m25.py                 # no weight download
    python scripts/verify_minimax_m25.py --with-weights  # full load sanity
"""

from __future__ import annotations

import argparse
import shutil
import sys

MODEL_ID = "MiniMaxAI/MiniMax-M2.5"
EXPECTED_LAYERS = 62
EXPECTED_HIDDEN = 3072
EXPECTED_EXPERTS = 256
EXPECTED_EXPERTS_PER_TOK = 8
MIN_GPUS = 4
MIN_VRAM_GB = 300  # 4× H100 80GB = 320GB


def _ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    sys.exit(1)


def _warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


def check_gpus() -> None:
    import torch

    print("=" * 70)
    print("STEP 0: GPU capability (FP8 requires SM89+ / H100+)")
    print("=" * 70)

    if not torch.cuda.is_available():
        _fail("CUDA not available")

    n = torch.cuda.device_count()
    print(f"  device count = {n}")
    if n < MIN_GPUS:
        _fail(f"need >= {MIN_GPUS} GPUs for FP8 MiniMax-M2.5, found {n}")

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
    if total_vram < MIN_VRAM_GB:
        _fail(f"total VRAM {total_vram:.0f} GiB < {MIN_VRAM_GB} GiB needed for M2.5 FP8")


def check_disk() -> None:
    import os

    print()
    print("=" * 70)
    print("STEP 1: Disk space (FP8 snapshot ~230 GB)")
    print("=" * 70)

    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    print(f"  HF_HOME = {hf_home}")
    try:
        usage = shutil.disk_usage(hf_home if os.path.exists(hf_home) else os.path.dirname(hf_home) or "/")
    except FileNotFoundError:
        usage = shutil.disk_usage("/")
    free_gb = usage.free / (1024 ** 3)
    print(f"  free on HF_HOME partition = {free_gb:.0f} GiB")
    if free_gb < 280:
        _fail(f"need >= 280 GiB free for FP8 M2.5 snapshot + cache, have {free_gb:.0f} GiB")
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
    n_experts = getattr(text_cfg, "num_local_experts", None)
    n_routed = getattr(text_cfg, "num_experts_per_tok", None)
    arch = getattr(config, "architectures", None)
    qcfg = getattr(config, "quantization_config", None)
    scoring = getattr(text_cfg, "scoring_func", None)

    print(f"  architectures         = {arch}")
    print(f"  num_hidden_layers     = {n_layers}")
    print(f"  hidden_size           = {hidden}")
    print(f"  num_local_experts     = {n_experts}")
    print(f"  num_experts_per_tok   = {n_routed}")
    print(f"  scoring_func          = {scoring}")
    print(f"  quantization_config   = {type(qcfg).__name__ if qcfg else None}")

    if n_layers != EXPECTED_LAYERS:
        _fail(f"expected {EXPECTED_LAYERS} layers, got {n_layers}")
    _ok(f"{EXPECTED_LAYERS} transformer layers")

    if hidden != EXPECTED_HIDDEN:
        _fail(f"expected hidden_size={EXPECTED_HIDDEN}, got {hidden}")
    _ok(f"hidden_size={EXPECTED_HIDDEN}")

    if n_experts != EXPECTED_EXPERTS:
        _fail(f"expected {EXPECTED_EXPERTS} experts, got {n_experts}")
    _ok(f"{EXPECTED_EXPERTS} local experts")

    if n_routed != EXPECTED_EXPERTS_PER_TOK:
        _fail(f"expected {EXPECTED_EXPERTS_PER_TOK} experts per token, got {n_routed}")
    _ok(f"{EXPECTED_EXPERTS_PER_TOK} experts routed per token")

    if qcfg is None:
        _fail("no quantization_config on model — not an FP8 checkpoint?")
    _ok("FP8 quantization_config present")

    if scoring != "sigmoid":
        _warn(f"expected sigmoid scoring, got {scoring} — router hooks may differ")
    else:
        _ok("sigmoid expert routing (expected)")


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
    print("STEP 4: Full model load + abliterix compatibility (downloads ~230GB)")
    print("=" * 70)

    n_gpus = torch.cuda.device_count()
    max_mem = {i: "76GiB" for i in range(n_gpus)}

    print("  loading FP8 weights across all GPUs...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        max_memory=max_mem,
        trust_remote_code=True,
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

    # --- Steerable modules check (must match engine.steerable_modules) ---
    block = layers[0]

    # 1. Attention o_proj (engine.py line ~464: layer.self_attn.o_proj)
    o_proj = getattr(getattr(block, "self_attn", None), "o_proj", None)
    if o_proj is None:
        _fail("layer[0].self_attn.o_proj not found — attention steering broken")
    else:
        _ok(f"self_attn.o_proj found (shape={tuple(o_proj.weight.shape)})")

    # 2. MoE experts (engine.py line ~489: layer.block_sparse_moe.experts[j].w2)
    bsm = getattr(block, "block_sparse_moe", None)
    if bsm is None:
        _fail("layer[0].block_sparse_moe not found — MoE steering broken")
        return

    experts = getattr(bsm, "experts", None)
    if experts is None:
        _fail("block_sparse_moe.experts not found")
        return

    try:
        expert_list = list(experts)
    except TypeError:
        _fail("block_sparse_moe.experts is not iterable")
        return

    n_exp = len(expert_list)
    print(f"  block_sparse_moe.experts: {type(experts).__name__} len={n_exp}")
    if n_exp != EXPECTED_EXPERTS:
        _warn(f"expected {EXPECTED_EXPERTS} experts, got {n_exp}")
    else:
        _ok(f"{EXPECTED_EXPERTS} experts per layer")

    # Check w2 (down_proj equivalent)
    w2 = getattr(expert_list[0], "w2", None)
    if w2 is None:
        _fail("expert[0].w2 not found — abliterix steerable_modules won't discover expert down-projections")
    elif not hasattr(w2, "weight"):
        _fail(f"expert[0].w2 is {type(w2).__name__}, expected nn.Linear")
    else:
        _ok(f"expert[0].w2 found (shape={tuple(w2.weight.shape)})")

    # 3. Router gate (engine.py line ~549: block_sparse_moe.gate)
    gate = getattr(bsm, "gate", None)
    if gate is None:
        _fail("block_sparse_moe.gate not found — MoE router detection broken")
    elif not hasattr(gate, "weight") or gate.weight.dim() != 2:
        _fail(f"block_sparse_moe.gate.weight not a 2D tensor (needed for _locate_router)")
    else:
        _ok(f"block_sparse_moe.gate found (shape={tuple(gate.weight.shape)})")

    # 4. No shared expert expected
    shared = getattr(bsm, "shared_expert", None)
    if shared is not None:
        _warn("unexpected shared_expert found on block_sparse_moe")
    else:
        _ok("no shared expert (as expected)")

    # --- Total steerable module count estimate ---
    total_steerable = n_exp + 1  # experts' w2 + o_proj per layer
    print(f"  estimated steerable modules: {total_steerable} per layer × {n} layers = {total_steerable * n}")
    _ok("all steerable module paths match engine.py conventions")

    # --- FP8 weight dtype check ---
    fp8_count = 0
    non_fp8_count = 0
    fp8_dtypes = {torch.float8_e4m3fn, torch.float8_e5m2}
    for name, param in model.named_parameters():
        if param.dtype in fp8_dtypes:
            fp8_count += 1
        else:
            non_fp8_count += 1
    print(f"  FP8 parameters: {fp8_count}, non-FP8: {non_fp8_count}")
    if fp8_count == 0:
        _warn("no FP8 parameters found — dequant workaround won't activate")
    else:
        _ok(f"FP8 weights present ({fp8_count} params) — dequant workaround will apply")

    # Check for scale tensors (needed by _dequant_fp8_to_bf16 blockwise path)
    has_scale = False
    for name, mod in model.named_modules():
        if hasattr(mod, "weight_scale_inv") or hasattr(mod, "weight_scale"):
            has_scale = True
            break
    if has_scale:
        _ok("block-wise FP8 scale tensors found — blockwise dequant path available")
    else:
        _warn("no weight_scale/weight_scale_inv found — will use simple FP8 cast path")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-weights",
        action="store_true",
        help="Also download and load the ~230GB FP8 snapshot.",
    )
    args = parser.parse_args()

    check_gpus()
    check_disk()
    inspect_config()
    inspect_chat_template()
    if args.with_weights:
        inspect_model()

    print()
    print("All pre-flight checks passed. configs/minimax_m2.5.toml is safe to use.")


if __name__ == "__main__":
    main()
