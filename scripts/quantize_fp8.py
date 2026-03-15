# Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>

"""Quantize a Hugging Face causal LM to fine-grained FP8 and save it.

Example:
    python scripts/quantize_fp8.py \
        --model wangzhang/Qwen3.5-35B-A3B-abliterated \
        --out /workspace/Qwen3.5-35B-A3B-abliterated-fp8
"""

import argparse
import json
import os

from prometheus.scriptlib import setup_io

setup_io()

parser = argparse.ArgumentParser(description="Quantize a model to fine-grained FP8")
parser.add_argument("--model", required=True, help="Source model ID or local path")
parser.add_argument("--out", required=True, help="Output directory for quantized model")
parser.add_argument(
    "--max-shard-size",
    default="10GB",
    help="Maximum shard size passed to save_pretrained()",
)
parser.add_argument(
    "--weight-block-size",
    type=int,
    nargs=2,
    metavar=("ROWS", "COLS"),
    default=(128, 128),
    help="FP8 block size for weights",
)
parser.add_argument(
    "--trust-remote-code",
    action="store_true",
    help="Allow custom tokenizer/model code from Hub repos",
)
args = parser.parse_args()

token = os.environ.get("HUGGING_FACE_TOKEN") or os.environ.get("HF_TOKEN")

from huggingface_hub import hf_hub_download  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    FineGrainedFP8Config,
    PreTrainedTokenizerFast,
)


def load_tokenizer(model_id: str, token: str | None, trust_remote_code: bool):
    try:
        return AutoTokenizer.from_pretrained(
            model_id,
            token=token,
            trust_remote_code=trust_remote_code,
        )
    except ValueError as exc:
        if "TokenizersBackend" not in str(exc):
            raise

        print("Falling back to tokenizer.json loader...")
        tokenizer_file = hf_hub_download(
            repo_id=model_id,
            filename="tokenizer.json",
            token=token,
        )
        config_file = hf_hub_download(
            repo_id=model_id,
            filename="tokenizer_config.json",
            token=token,
        )
        with open(config_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_file,
            eos_token=cfg.get("eos_token"),
            bos_token=cfg.get("bos_token"),
            unk_token=cfg.get("unk_token"),
            pad_token=cfg.get("pad_token"),
        )
        tokenizer.model_max_length = cfg.get(
            "model_max_length", tokenizer.model_max_length
        )
        for key in (
            "audio_bos_token",
            "audio_eos_token",
            "audio_token",
            "image_token",
            "video_token",
            "vision_bos_token",
            "vision_eos_token",
        ):
            value = cfg.get(key)
            if value is not None:
                setattr(tokenizer, key, value)
        return tokenizer


print(f"Loading tokenizer from {args.model}...")
tokenizer = load_tokenizer(args.model, token, args.trust_remote_code)

print(f"Loading model from {args.model}...")
quantization_config = FineGrainedFP8Config(
    weight_block_size=tuple(args.weight_block_size),
)
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    dtype="auto",
    device_map="auto",
    token=token,
    trust_remote_code=args.trust_remote_code,
    quantization_config=quantization_config,
)

print(f"Saving FP8 model to {args.out}...")
model.save_pretrained(
    args.out,
    safe_serialization=True,
    max_shard_size=args.max_shard_size,
)
tokenizer.save_pretrained(args.out)

print("Done.")
