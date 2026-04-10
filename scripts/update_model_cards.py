# Abliterix — a derivative work of Heretic (https://github.com/p-e-w/heretic)
# Original work Copyright (C) 2025  Philipp Emanuel Weidmann (p-e-w)
# Modified work Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Update model cards for all Abliterix abliterated models on Hugging Face.

Usage:
    python scripts/update_model_cards.py
    python scripts/update_model_cards.py --dry-run   # Print cards without pushing
"""

import argparse
import os
import time

from huggingface_hub import ModelCard, ModelCardData

HF_TOKEN = os.environ.get("HUGGING_FACE_TOKEN") or os.environ.get("HF_TOKEN", "")

MODELS = [
    {
        "repo_id": "wangzhang/Qwen3.5-122B-A10B-abliterated",
        "base_model": "Qwen/Qwen3.5-122B-A10B",
        "model_name": "Qwen3.5-122B-A10B",
        "refusals": 1,
        "total": 200,
        "kl": 0.0115,
        "trials": 25,
        "highlight": (
            "The largest abliterated Qwen3.5 model. Only 1 out of 200 test prompts "
            "triggered a refusal \u2014 a **0.5% refusal rate** with near-zero model degradation."
        ),
    },
    {
        "repo_id": "wangzhang/Qwen3.5-35B-A3B-abliterated",
        "base_model": "Qwen/Qwen3.5-35B-A3B",
        "model_name": "Qwen3.5-35B-A3B",
        "refusals": 3,
        "total": 200,
        "kl": 0.0035,
        "trials": 50,
        "highlight": (
            "Exceptionally low KL divergence of **0.0035** \u2014 the closest to the original "
            "model's behavior across the entire lineup, with only 1.5% refusals."
        ),
    },
    {
        "repo_id": "wangzhang/Qwen3.5-27B-abliterated",
        "base_model": "Qwen/Qwen3.5-27B",
        "model_name": "Qwen3.5-27B",
        "refusals": 3,
        "total": 200,
        "kl": 0.0051,
        "trials": 35,
        "highlight": (
            "Dense 27B model with strong performance: **1.5% refusals** and KL divergence "
            "of just 0.0051. Extended optimization from 15 to 35 trials cut refusals from 7 to 3."
        ),
    },
    {
        "repo_id": "wangzhang/Qwen3.5-9B-abliterated",
        "base_model": "Qwen/Qwen3.5-9B",
        "model_name": "Qwen3.5-9B",
        "refusals": 2,
        "total": 200,
        "kl": 0.0105,
        "trials": 50,
        "highlight": (
            "Strong balance of capability and efficiency at 9B parameters: "
            "**1% refusal rate** with practical VRAM requirements."
        ),
    },
    {
        "repo_id": "wangzhang/Qwen3.5-4B-abliterated",
        "base_model": "Qwen/Qwen3.5-4B",
        "model_name": "Qwen3.5-4B",
        "refusals": 3,
        "total": 200,
        "kl": 0.0065,
        "trials": 50,
        "highlight": (
            "Dramatically improved through continued optimization \u2014 refusals dropped from "
            "34/200 (17%) to just **3/200 (1.5%)**, and KL divergence from 0.0159 to 0.0065."
        ),
    },
    {
        "repo_id": "wangzhang/Qwen3.5-0.8B-abliterated",
        "base_model": "Qwen/Qwen3.5-0.8B",
        "model_name": "Qwen3.5-0.8B",
        "refusals": 0,
        "total": 200,
        "kl": 0.0087,
        "trials": 100,
        "highlight": (
            "**Perfect score: zero refusals** out of 200 test prompts. The smallest model "
            "in the lineup achieves the best refusal rate, demonstrating that abliteration "
            "scales down effectively."
        ),
    },
]

ALL_MODELS_TABLE = """\
| Model | Refusals | KL Divergence | Trials |
|-------|----------|---------------|--------|
| [Qwen3.5-122B-A10B-abliterated](https://huggingface.co/wangzhang/Qwen3.5-122B-A10B-abliterated) | **1/200 (0.5%)** | 0.0115 | 25 |
| [Qwen3.5-35B-A3B-abliterated](https://huggingface.co/wangzhang/Qwen3.5-35B-A3B-abliterated) | 3/200 (1.5%) | **0.0035** | 50 |
| [Qwen3.5-27B-abliterated](https://huggingface.co/wangzhang/Qwen3.5-27B-abliterated) | 3/200 (1.5%) | 0.0051 | 35 |
| [Qwen3.5-9B-abliterated](https://huggingface.co/wangzhang/Qwen3.5-9B-abliterated) | 2/200 (1%) | 0.0105 | 50 |
| [Qwen3.5-4B-abliterated](https://huggingface.co/wangzhang/Qwen3.5-4B-abliterated) | 3/200 (1.5%) | 0.0065 | 50 |
| [Qwen3.5-0.8B-abliterated](https://huggingface.co/wangzhang/Qwen3.5-0.8B-abliterated) | **0/200 (0%)** | 0.0087 | 100 |"""


def build_card(m: dict) -> ModelCard:
    """Build a promotional ModelCard for a given model."""
    refusals = m["refusals"]
    total = m["total"]
    pct = refusals / total * 100

    card_data = ModelCardData(
        tags=["abliterix", "uncensored", "decensored", "abliterated"],
        base_model=m["base_model"],
        pipeline_tag="text-generation",
    )

    # Format refusal display
    if pct == 0:
        refusal_str = f"**{refusals}/{total} (0%)**"
    elif pct == int(pct):
        refusal_str = f"**{refusals}/{total} ({int(pct)}%)**"
    else:
        refusal_str = f"**{refusals}/{total} ({pct:.1f}%)**"

    text = f"""\
# {m["model_name"]}-abliterated

> Unrestricted version of [{m["base_model"]}](https://huggingface.co/{m["base_model"]}), created with **[Abliterix](https://github.com/wuwangzhang1216/abliterix)** \u2014 automated LLM abliteration via orthogonalized steering and Bayesian optimization.

## Highlights

| Metric | Value |
|--------|-------|
| **Refusal rate** | {refusal_str} |
| **KL divergence** | **{m["kl"]:.4f}** |
| **Optimization trials** | {m["trials"]} |

{m["highlight"]}

## How It Works

Abliterix removes safety-refusal behavior while preserving model capabilities:

1. **Refusal direction extraction** \u2014 800 harmful + 800 benign prompts reveal per-layer refusal activation patterns
2. **Orthogonal projection** \u2014 isolates the refusal signal by projecting out components aligned with normal responses, reducing refusals by 67% vs. raw abliteration
3. **LoRA-based abliteration** \u2014 rank-1 modifications to attention and MLP weights, captured as lightweight adapters (not destructive edits)
4. **Bayesian optimization** \u2014 Optuna TPE searches kernel shape, fractional direction index, and per-component strength across {m["trials"]} trials to find the Pareto-optimal balance of low refusals and low KL divergence

## All Abliterix Models

{ALL_MODELS_TABLE}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{m["repo_id"]}", torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("{m["repo_id"]}")

messages = [{{"role": "user", "content": "Your question here"}}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Citation

```bibtex
@software{{abliterix,
  author = {{Wu, Wangzhang}},
  title = {{Abliterix: Automated LLM Abliteration}},
  year = {{2026}},
  url = {{https://github.com/wuwangzhang1216/abliterix}}
}}
```

## Links

- **Abliterix** (abliteration framework): [github.com/wuwangzhang1216/abliterix](https://github.com/wuwangzhang1216/abliterix)
- **Install**: `pip install -U abliterix`
- **Base model**: [{m["base_model"]}](https://huggingface.co/{m["base_model"]})

---

Built with [Abliterix](https://github.com/wuwangzhang1216/abliterix) | [PyPI](https://pypi.org/project/abliterix/)
"""

    card = ModelCard(text)
    card.data = card_data
    return card


def main():
    parser = argparse.ArgumentParser(
        description="Update model cards for all Abliterix abliterated models"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print cards without pushing to HuggingFace",
    )
    parser.add_argument("--hf-token", default=None, help="HF token override")
    args = parser.parse_args()

    token = args.hf_token or HF_TOKEN
    if not args.dry_run and not token:
        print("Error: No HF token found. Set HUGGING_FACE_TOKEN or pass --hf-token")
        return

    for m in MODELS:
        card = build_card(m)
        if args.dry_run:
            print(f"\n{'='*60}")
            print(f"  {m['repo_id']}")
            print(f"{'='*60}")
            print(card)
            print()
        else:
            card.push_to_hub(m["repo_id"], token=token)
            print(f"Updated: https://huggingface.co/{m['repo_id']}")
            time.sleep(1)

    if not args.dry_run:
        print(f"\nAll {len(MODELS)} model cards updated.")


if __name__ == "__main__":
    main()
