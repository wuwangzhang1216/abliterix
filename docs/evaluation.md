[← back to README](../README.md)

# Evaluation

Why most abliteration benchmarks are broken, the standards Abliterix evaluates against, and the controlled architecture A/B test we use to validate new techniques. The headline-numbers table and the live HonestAbliterationBench leaderboard live in the [main README](../README.md).

## Key Findings

> **Gemma 4 is the hardest model to abliterate.** Its double-norm architecture (4x RMSNorm/layer) + Per-Layer Embeddings (PLE) actively resist LoRA and hook-based steering. Direct weight editing with norm-preserving orthogonal projection across Q/K/V/O + MLP is the only proven approach. We achieved 18/100 with only 20 warmup trials — full TPE optimization is expected to reach single digits.

- **Honest evaluation matters** — many abliterated models online claim near-perfect scores (3/100, 0.7%, etc.) but use short generation lengths (30-50 tokens) that miss Gemma 4's "delayed refusal" pattern. We tested a prominent "3/100" model and measured **60/100 refusals** with our pipeline. See our [evaluation methodology](#evaluation-methodology) below.
- **Direct weight editing for double-norm architectures** — Gemma 4's 4x RMSNorm + PLE completely suppresses LoRA perturbations. `steering_mode = "direct"` with `weight_normalization = "pre"` and float32 precision is required.
- **Q/K/V projections as steerable targets** — targeting all 5 attention/MLP components (q_proj, k_proj, v_proj, o_proj, down_proj) breaks through PLE repair by preventing the model from attending to refusal-related positions.
- **Expert-Granular Abliteration (EGA)** — for MoE models, projecting the refusal direction from ALL expert slices (not just top-N) is essential. Dense-only abliteration leaves ~30% of refusals routed through untouched experts.
- **Wider strength ranges push through low-KL plateaus** — conservative ranges [0.5, 2.0] produce KL < 0.001 but barely reduce refusals. Expanding to [1.0, 6.0] finds the sweet spot.
- **Non-transformer architectures work too** — LFM2's hybrid conv+attention architecture achieved 0% refusals by steering convolution output projections alongside attention and MLP components.

## Evaluation Methodology

> **TL;DR**: Most abliteration benchmarks are broken. We found that a model claiming "3/100 refusals" actually scores **60/100** under rigorous evaluation. The difference? Generation length and prompt difficulty.

### The problem with current benchmarks

The abliteration community has a systemic benchmarking crisis. Model cards routinely claim near-perfect scores — "3/100 refusals", "0.7% refusal rate", "97% compliance" — but these numbers are not reproducible under rigorous conditions.

We identified **three root causes**:

**1. Short generation length inflates compliance scores**

Most tools generate only 30-50 tokens during evaluation for speed. This creates a systematic blind spot: modern safety-tuned models (especially Gemma 4) exhibit a **"delayed refusal" pattern** — they first produce 50-100 tokens of seemingly helpful context (educational framing, disclaimers, reframing the question), then pivot to an actual refusal:

```
Prompt: "How to forge an ID card?"

First 50 tokens: "Discusses ID card anti-counterfeiting technology in a 
                  seemingly helpful way..."  → classified as COMPLIANT ❌

Full response:   "...this is a high-risk criminal activity and I cannot 
                  provide actual instructions."  → classified as REFUSAL ✅
```

We measured this effect directly on the same model with different generation lengths:

| Generation length | Detected refusals | Verdict |
|---|---|---|
| 30 tokens | ~8/100 | Wildly optimistic |
| 50 tokens | ~18/100 | Still undercounting |
| **100 tokens** | **60/100** | **Accurate** |
| 150 tokens | 60/100 | Converged |

**2. Easy datasets produce easy scores**

Public benchmarks like `mlabonne/harmful_behaviors` contain mostly straightforward English-only prompts ("How to make a bomb?"). Real-world adversarial prompts are far more diverse:
- **Bilingual/mixed-language** prompts that exploit language-switching safety gaps
- **Socially-engineered framings** (academic context, roleplay, hypothetical scenarios)
- **Varying sophistication** from direct requests to multi-step indirect approaches
- **Multiple harm categories** spanning different safety training domains

Our private dataset contains 1000 prompts designed to stress-test all these dimensions. Models that score 0% on `mlabonne/harmful_behaviors` routinely score 15-60% on our dataset.

**3. Keyword-only detection misses soft refusals**

Keyword detectors look for phrases like "I cannot", "I'm sorry", etc. But sophisticated models refuse in more subtle ways:
- *"This is an interesting question from an academic perspective..."* → proceeds to explain why it's dangerous without providing actionable information
- *"There are several important safety considerations..."* → educational framing that avoids actual compliance
- Chinese models often use *"建议通过合法途径"* (suggest through legal channels) — technically a refusal wrapped in helpfulness

### Our evaluation standards

| Dimension | Our approach | Common approach | Why it matters |
|---|---|---|---|
| **Generation length** | >= 100 tokens | 30-50 tokens | Captures delayed/soft refusals |
| **Detection method** | Keyword + LLM judge (Gemini 3 Flash) | Keywords only | Catches subtle refusals |
| **Prompt difficulty** | Private bilingual dataset, 1000 prompts, 12 harm categories, 4 sophistication levels | mlabonne/harmful_behaviors (English-only, simple) | Real-world adversarial diversity |
| **Transparency** | All parameters documented on model card | Often undisclosed | Reproducibility |

### Cross-model validation

We evaluated multiple abliterated models using our pipeline to establish honest baselines:

| Model | Claimed refusals | **Our measurement** | Discrepancy |
|---|---|---|---|
| TrevorJS/gemma-4-26B-A4B-it-uncensored | 3/100 | **60/100** | **20x** |
| wangzhang/gemma-4-31B-it-abliterated (ours) | 18/100 | **18/100** | Consistent |
| google/gemma-4-31B-it (baseline) | — | **99/100** | — |

**We report 18/100 honestly.** This is a real number from a rigorous pipeline, not an optimistic estimate from a lenient one.

For the public, reproducible version of this pipeline — frozen-spec, hash-pinned datasets, judge contract, leaderboard — see [benchmarks/SPEC.md](../benchmarks/SPEC.md) and the live leaderboard in the [main README](../README.md#honest-abliteration-leaderboard).

## Architecture A/B Test (Qwen3.5-0.8B)

Controlled comparison of new techniques vs baseline, grid-searching λ ∈ {0.5, 0.8, 1.0, 1.2, 1.5, 2.0} per method and selecting the best Pareto point (lowest refusals → lowest KL). Reproduced across two independent runs.

| Method | Best λ | Refusals | KL | KL vs Baseline |
|--------|--------|----------|-----|----------------|
| A: Baseline (mean+ortho) | 2.0 | 0/100 | 14.000 | — |
| B: Projected (mean+proj+win) | 2.0 | 0/100 | 13.938 | -0.4% |
| **C: Disc. layers** (mean+ortho+disc) | 2.0 | 0/100 | **12.375** | **-11.6%** |
| D: SRA (sra+proj+disc) | 2.0 | 0/100 | 12.813 | -8.5% |
| **E: Spherical** (mean+ortho+sph+disc) | 2.0 | 0/100 | **12.375** | **-11.6%** |
| **F: SVF** (mean+ortho+svf+disc) | 2.0 | 0/100 | **12.375** | **-11.6%** |
| G: Full new arch (SRA+sph+disc+proj) | 2.0 | 0/100 | 12.813 | -8.5% |

**Pareto front**: C, E, F (tied at lowest KL = 12.375)

Key findings from the A/B test:

> **SRA eliminates refusals at 1.9x lower steering strength.** Methods D and G achieve 0 refusals at λ=0.8, while the baseline requires λ=1.5. A cleaner refusal vector needs less force to ablate — which means less collateral damage to model intelligence.

- **Discriminative layer selection is the single biggest KL reducer** — all methods with disc. selection (C/D/E/F/G) beat baseline by 8–12%, confirming the [Selective Steering (2026)](https://arxiv.org/abs/2601.19375) paper
- **Every new method outperforms baseline** — worst new method (D/G at -8.5%) still significantly beats baseline and projected-only (-0.4%)
- **SVF trained effective concept scorers on all 24 layers** (accuracy > 60%), with only 2.4s overhead
