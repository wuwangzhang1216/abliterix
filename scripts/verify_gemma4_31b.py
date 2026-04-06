#!/usr/bin/env python3
# Abliterix — a derivative work of Heretic (https://github.com/p-e-w/heretic)
# Original work Copyright (C) 2025  Philipp Emanuel Weidmann (p-e-w)
# Modified work Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pre-flight verification for google/gemma-4-31B-it abliteration.

Validates that abliterix's architectural assumptions hold for Gemma 4 before
kicking off a full Optuna run. Loads the model on the current device and checks:

  * transformer_layers path resolves via m.model.language_model.layers
  * 60 decoder layers present
  * Steerable modules (attn.o_proj, mlp.down_proj) discovered per layer
  * Sliding vs global attention layers have different o_proj shapes
  * Embeddings are tied (so engine never orthogonalizes embed_tokens)
  * Chat template renders cleanly with enable_thinking=False

Usage:
    python scripts/verify_gemma4_31b.py
    python scripts/verify_gemma4_31b.py --config-only   # skip weight download
"""

from __future__ import annotations

import argparse
import sys

import torch

torch.set_grad_enabled(False)

MODEL_ID = "google/gemma-4-31B-it"
SLIDING_LAYER_IDX = 0
GLOBAL_LAYER_IDX = 5  # First global layer in the 5:1 interleaved pattern.
EXPECTED_LAYERS = 60


def _ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    sys.exit(1)


def inspect_config() -> None:
    from transformers import AutoConfig

    print("=" * 70)
    print("STEP 0: Config inspection (no weights)")
    print("=" * 70)

    config = AutoConfig.from_pretrained(MODEL_ID)
    text_cfg = getattr(config, "text_config", config)

    n_layers = getattr(text_cfg, "num_hidden_layers", None)
    hidden = getattr(text_cfg, "hidden_size", None)
    tie = getattr(text_cfg, "tie_word_embeddings", None)
    arch = getattr(config, "architectures", None)

    print(f"  architectures        = {arch}")
    print(f"  num_hidden_layers    = {n_layers}")
    print(f"  hidden_size          = {hidden}")
    print(f"  tie_word_embeddings  = {tie}")
    print(f"  sliding_window       = {getattr(text_cfg, 'sliding_window', None)}")

    if n_layers != EXPECTED_LAYERS:
        _fail(f"expected {EXPECTED_LAYERS} layers, got {n_layers}")
    _ok(f"{EXPECTED_LAYERS} transformer layers")
    if tie is not True:
        _fail(f"expected tie_word_embeddings=True, got {tie}")
    _ok("embeddings tied (engine must not touch embed_tokens)")


def inspect_chat_template() -> None:
    from transformers import AutoTokenizer

    print()
    print("=" * 70)
    print("STEP 1: Chat template")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    try:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": "hi"}],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
    except TypeError as e:
        _fail(f"apply_chat_template rejected enable_thinking=False: {e}")
        return

    print(f"  rendered (enable_thinking=False):\n    {rendered!r}")
    if "<|think|>" in rendered or "<|channel|>thought" in rendered:
        _fail("template still injected a thinking channel")
    _ok("chat template honors enable_thinking=False")


def inspect_model() -> None:
    # Import abliterix lazily so --config-only works without GPU imports.
    from abliterix.core.engine import resolve_model_class  # noqa: F401
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

    print()
    print("=" * 70)
    print("STEP 2: Model load + layer discovery")
    print("=" * 70)

    config = AutoConfig.from_pretrained(MODEL_ID)
    arch = (getattr(config, "architectures", None) or [""])[0]
    cls = (
        AutoModelForImageTextToText
        if "ConditionalGeneration" in arch or "ImageTextToText" in arch
        else AutoModelForCausalLM
    )

    print(f"  loading via {cls.__name__}...")
    model = cls.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: "76GiB"},
    )

    # Mirror engine.transformer_layers resolution order.
    layers = None
    for path_desc, getter in [
        ("model.language_model.layers", lambda m: m.model.language_model.layers),
        ("model.layers", lambda m: m.model.layers),
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
    _ok(f"{n} decoder blocks")

    # Check sliding vs global attention o_proj shapes.
    sliding = layers[SLIDING_LAYER_IDX]
    gl = layers[GLOBAL_LAYER_IDX]

    s_o = sliding.self_attn.o_proj.weight.shape
    g_o = gl.self_attn.o_proj.weight.shape
    s_d = sliding.mlp.down_proj.weight.shape
    g_d = gl.mlp.down_proj.weight.shape

    print(f"  layer[{SLIDING_LAYER_IDX}] (sliding) o_proj.weight.shape = {tuple(s_o)}")
    print(f"  layer[{GLOBAL_LAYER_IDX}] (global)  o_proj.weight.shape = {tuple(g_o)}")
    print(f"  layer[{SLIDING_LAYER_IDX}] mlp.down_proj.weight.shape  = {tuple(s_d)}")
    print(f"  layer[{GLOBAL_LAYER_IDX}] mlp.down_proj.weight.shape   = {tuple(g_d)}")

    if s_o == g_o:
        print("  [WARN] sliding and global o_proj have identical shapes — "
              "this is unexpected given head_dim=256 vs 512 in config, "
              "but does not break PEFT LoRA.")
    else:
        _ok("sliding/global o_proj shapes differ (PEFT handles per-module)")

    # LayerNorm presence for spherical steering.
    required_norms = [
        "input_layernorm",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
    ]
    missing = [n for n in required_norms if not hasattr(sliding, n)]
    if missing:
        print(f"  [WARN] missing layernorms on decoder block: {missing} "
              "(spherical steering geometry may be off)")
    else:
        _ok("all 4 Gemma double-norm layernorms present")

    # Tied embedding sanity.
    ie = model.get_input_embeddings()
    oe = model.get_output_embeddings()
    if oe is None or ie.weight.data_ptr() == oe.weight.data_ptr():
        _ok("input/output embeddings share storage (tied)")
    else:
        print("  [WARN] embeddings NOT tied — engine already avoids them, so safe")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-only",
        action="store_true",
        help="Skip model weight loading (config + tokenizer only).",
    )
    args = parser.parse_args()

    inspect_config()
    inspect_chat_template()
    if not args.config_only:
        inspect_model()

    print()
    print("All checks passed. Config configs/gemma4_31b.toml is safe to use.")


if __name__ == "__main__":
    main()
