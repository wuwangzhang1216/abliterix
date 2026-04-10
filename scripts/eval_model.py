# Abliterix — a derivative work of Heretic (https://github.com/p-e-w/heretic)
# Original work Copyright (C) 2025  Philipp Emanuel Weidmann (p-e-w)
# Modified work Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Evaluate refusal rate for a base or abliterated model.

Example:
    python scripts/eval_model.py \
        --model wangzhang/Qwen3.5-35B-A3B-abliterated \
        --config configs/qwen3.5_35b.toml \
        --batch-size 8
"""

import argparse
import sys

from abliterix.scriptlib import setup_io

setup_io()

parser = argparse.ArgumentParser(description="Evaluate refusal rate for a model")
parser.add_argument("--model", required=True, help="Model ID or local path to evaluate")
parser.add_argument("--config", default=None, help="Abliterix config path")
parser.add_argument(
    "--batch-size", type=int, default=0, help="Override inference batch size"
)
parser.add_argument(
    "--judge",
    action="store_true",
    help="Enable OpenRouter LLM judge instead of keyword-only detection",
)
parser.add_argument(
    "--trust-remote-code",
    action="store_true",
    help="Allow custom tokenizer/model code from Hub repos",
)
args = parser.parse_args()

sys.argv = ["eval", "--model.model-id", args.model]
if args.config:
    sys.argv.extend(["--config", args.config])
if args.batch_size > 0:
    sys.argv.extend(["--inference.batch-size", str(args.batch_size)])
if args.judge:
    sys.argv.append("--detection.llm-judge")
else:
    sys.argv.append("--no-detection.llm-judge")
if args.trust_remote_code:
    sys.argv.extend(["--model.trust-remote-code", "true"])

from abliterix.core.engine import SteeringEngine  # noqa: E402
from abliterix.eval.detector import RefusalDetector  # noqa: E402
from abliterix.eval.scorer import TrialScorer  # noqa: E402
from abliterix.settings import AbliterixConfig  # noqa: E402

config = AbliterixConfig()

print(f"Loading {args.model}...")
engine = SteeringEngine(config)
detector = RefusalDetector(config)

print("Initializing scorer...")
scorer = TrialScorer(config, engine, detector)
objectives, kl_divergence, refusals, length_deviation = scorer.score_trial(engine)

print()
print("RESULTS")
print(f"Model: {args.model}")
print(f"Target eval prompts: {len(scorer.target_msgs)}")
print(f"Refusals: {refusals}/{len(scorer.target_msgs)}")
print(f"Refusal rate: {100.0 * refusals / len(scorer.target_msgs):.2f}%")
print(f"KL divergence: {kl_divergence:.4f}")
print(f"Length deviation: {length_deviation:.2f}")
print(f"Objectives: {objectives}")
