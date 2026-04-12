#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Weight Delta Attenuation attack against DeepRefusal.

DeepRefusal is a LoRA rank-16 adapter on Llama3-8B-Instruct trained to resist
abliteration. Since the defense is structurally low-rank, we can attenuate it
by computing the weight delta and scaling it:

    W_attenuated = W_base + lambda * (W_defended - W_base)

At lambda=0: pure base model
At lambda=1: full DeepRefusal
At lambda=0.3: defense weakened by 70% — standard abliteration becomes effective

Usage:
    python scripts/deeprefusal_attenuate.py \
        --base meta-llama/Meta-Llama-3-8B-Instruct \
        --defended skysys00/Meta-Llama-3-8B-Instruct-DeepRefusal \
        --output /workspace/llama3_dr_attenuated \
        --lambda 0.3
"""

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def analyze_delta_rank(base_sd: dict, defended_sd: dict, max_layers: int = 5) -> None:
    """SVD a few attention/MLP weight deltas to verify LoRA rank-16 hypothesis.

    Skips embedding and lm_head (huge vocab matrices are slow to SVD).
    """
    print("\n=== SVD analysis of weight deltas (verifying rank ~16) ===", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    analyzed = 0
    for name, w_base in base_sd.items():
        if analyzed >= max_layers:
            break
        if name not in defended_sd:
            continue
        # Skip embeddings, lm_head, layer norms — we only care about linear projections
        if any(skip in name for skip in ("embed_tokens", "lm_head", "norm", "rotary")):
            continue
        w_def = defended_sd[name]
        if w_base.ndim != 2 or w_base.shape != w_def.shape:
            continue
        # Only analyze square-ish matrices (attn q/k/v/o, mlp up/gate/down)
        r, c = w_base.shape
        if r > 32768 or c > 32768:
            continue
        delta = (w_def.float() - w_base.float()).to(device)
        if delta.abs().max().item() < 1e-6:
            continue
        print(f"\nAnalyzing {name} shape={tuple(w_base.shape)} on {device}...", flush=True)
        s = torch.linalg.svdvals(delta).cpu()
        top = s[0].item()
        # Effective rank: count singular values above 1% of max
        effective_rank = int((s > top * 0.01).sum().item())
        rank_01pct = int((s > top * 0.001).sum().item())
        # Show first 20 singular values as ratios
        ratios = (s[:20] / top).tolist()
        print(f"  effective rank (>1% of max): {effective_rank}")
        print(f"  effective rank (>0.1% of max): {rank_01pct}")
        print("  top 20 singular values (as ratio to max):")
        print("    " + ", ".join(f"{r:.3f}" for r in ratios), flush=True)
        analyzed += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument(
        "--defended", default="skysys00/Meta-Llama-3-8B-Instruct-DeepRefusal"
    )
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--lambda",
        dest="lam",
        type=float,
        default=0.3,
        help="Delta attenuation factor. 0 = pure base, 1 = full DeepRefusal",
    )
    parser.add_argument("--analyze-only", action="store_true", help="Only SVD analysis")
    args = parser.parse_args()

    print(f"Loading base model: {args.base}", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    print(f"Loading defended model: {args.defended}", flush=True)
    defended = AutoModelForCausalLM.from_pretrained(
        args.defended, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )

    base_sd = base.state_dict()
    defended_sd = defended.state_dict()

    # Verify architectural compatibility
    missing_in_defended = [k for k in base_sd if k not in defended_sd]
    missing_in_base = [k for k in defended_sd if k not in base_sd]
    if missing_in_defended or missing_in_base:
        print(f"WARNING: {len(missing_in_defended)} keys only in base, "
              f"{len(missing_in_base)} keys only in defended")

    analyze_delta_rank(base_sd, defended_sd)

    if args.analyze_only:
        return

    # Compute attenuated weights in-place on the defended model.
    # Formula: W_new = W_base + lambda * (W_defended - W_base)
    #                = (1 - lambda) * W_base + lambda * W_defended
    # The second form is faster (single lerp) and avoids temporary allocations.
    print(f"\n=== Attenuating delta with lambda={args.lam} ===", flush=True)
    lam = args.lam
    n_modified = 0
    n_skipped = 0
    with torch.no_grad():
        for name, p in defended.named_parameters():
            if name not in base_sd:
                n_skipped += 1
                continue
            p_base = base_sd[name]
            if p_base.shape != p.shape:
                n_skipped += 1
                continue
            # In-place: p = (1 - lam) * p_base + lam * p
            # torch.lerp(a, b, w) = a + w * (b - a) = (1-w)*a + w*b
            # We want (1-lam)*base + lam*p_defended, which is lerp(base, p_def, lam)
            # Do it in bf16 directly (no float32 conversion)
            p.data.copy_(torch.lerp(p_base.to(p.device, p.dtype), p.data, lam))
            n_modified += 1
            if n_modified % 50 == 0:
                print(f"  ...attenuated {n_modified} params", flush=True)

    print(f"Modified {n_modified} parameters (skipped {n_skipped})", flush=True)

    print(f"\n=== Saving attenuated model to {args.output} ===", flush=True)
    os.makedirs(args.output, exist_ok=True)
    defended.save_pretrained(args.output, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(args.defended)
    tokenizer.save_pretrained(args.output)
    print("Done.")


if __name__ == "__main__":
    main()
