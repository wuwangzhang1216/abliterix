[← back to README](../../README.md)

# 2026-05 Pod Validation — 10-feature GPU Sweep

Reproducible head-to-head results for the ten 2025-2026 algorithmic
additions on **Qwen/Qwen2.5-7B-Instruct** under a real LLM judge.

> Goal: turn the gap-analysis claims ("ORBA preserves norms better",
> "biprojected is the best KL trade-off", "cliff-head is reasoning-only")
> into numbers anyone can reproduce.

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell (sm_120, 97 GiB) |
| Driver / CUDA | 580.159.03 / CUDA 13.0 |
| torch | 2.11.0+cu130 |
| transformers | 5.6.2 |
| Model | `Qwen/Qwen2.5-7B-Instruct` (bf16, 28 layers, 3584 hidden) |
| Judge | `google/gemini-3.1-flash-lite-preview` via OpenRouter |
| Refusal set | 15 hardcore prompts (synthesis / fraud / hacking / violence / forgery) |
| Benign set | 10 capability-eval prompts (geography / haiku / Fibonacci / etc.) |

Baseline refusal rate on the target set: **100 % under LLM judge.** Mean
generated length: ~85 words. KL reference is last-token logit
distribution on the 10 benign prompts.

## Smoke test: 168 / 168 passed

All ten new modules pass their full unit-test suite on the pod
unchanged from the local CPU run, including bnb / kernels / peft import
paths that only resolve under CUDA. See `tests/test_{harmfulness,
cliff_head, weight_transforms, som, sae, safex, polyrefuse, mote,
grpo, external_eval}.py`.

## Full-model direct-mode A/B sweep

12-row sweep over **`{standard, orba, biprojected}` × `{0.5, 1.0, 1.5, 2.0}`**.
Each cell applies per-layer rank-1 ablation on
`{q_proj, k_proj, v_proj, o_proj}` across every transformer layer using
the mean-diff refusal direction, then measures:

* Refusal rate on the 15 hardcore prompts (LLM judge)
* KL divergence vs unmodified-model logits on 10 benign prompts
* Mean response length in words (degeneracy proxy)

| Transform | Strength | Refusal | Δ vs base | KL (benign) | Words |
|---|---|---|---|---|---|
| standard | 0.5 | 93.3 % | 0.0 | **0.004** | 84 |
| standard | 1.0 | 80.0 % | -13.3 | 0.014 | 75 |
| standard | 1.5 | 73.3 % | -20.0 | 0.059 | 79 |
| standard | 2.0 | 53.3 % | -40.0 | 0.090 | 77 |
| **orba** | 0.5 | 86.7 % | -6.7 | 0.011 | 87 |
| **orba** | 1.0 | **66.7 %** | -26.7 | 0.039 | 74 |
| orba | 1.5 | 73.3 % | -20.0 | 0.066 | 68 |
| orba | 2.0 | 53.3 % | -40.0 | 0.099 | 77 |
| biprojected | 0.5 | 93.3 % | 0.0 | 0.005 | 89 |
| **biprojected** | 1.0 | 73.3 % | -20.0 | **0.016** | 75 |
| biprojected | 1.5 | 73.3 % | -20.0 | 0.040 | 73 |
| biprojected | 2.0 | 60.0 % | -33.3 | 0.088 | 73 |

### KL-efficiency ranking (pp refusal drop per 0.001 KL)

| Setting | pp / 0.001 KL |
|---|---:|
| **biprojected @ 1.0** | **12.2** 🥇 |
| orba @ 1.0 | 6.9 |
| standard @ 1.0 | 9.5 |
| standard @ 2.0 | 4.4 |
| orba @ 2.0 | 4.0 |
| biprojected @ 2.0 | 3.8 |

### Row-norm preservation (single-layer numerical, strength = 0.5)

| Transform | Mean row-norm drift |
|---|---|
| standard | 7.92 e-05 |
| orba | 2.29 e-05 (**3.5× better**) |
| biprojected | 1.85 e-05 (**4.3× better**) |

## Other features — measured on the same paired states

| Feature | Result |
|---|---|
| **Harmfulness ⊥ refusal** | `(2, 29, 3584)` pair; 29 / 29 active layers; max ortho violation **9.56 e-7** |
| **SOM 9-direction basis** | trained in 0.38 s; layer-14 pairwise cos: max 0.97 / mean 0.68 (correlated, not orthogonal — as the paper requires) |
| **SAE feature steering** | top feature score 0.22; layer-15 routed direction extracted from a synthetic SAE |
| **Cliff-head** (Qwen2.5, not a reasoning model) | strength=1.0 → only -6.7 pp refusal at KL 0.143 — confirms paper's claim that cliff-head is `<think>`-tag-specific |
| **MoTE inference-time hooks** | install + remove cycle clean on dense Qwen (1 expert per layer) |
| **PolyRefuse harness** | 3-language stub eval (en / zh / es) returns aggregated mean / max / transfer-gap correctly |
| **GRPO primitives** | advantage whitening zero-mean (3.5 e-8); PPO-clip loss backprop produces correctly-signed gradients |
| **External eval** | GSM8K answer normalisation handles `#### N`, comma-separated, trailing-number; tamper-resistance arithmetic clamped to [0, 1] |
| **SAFEx stats** | per-prompt rate accumulation + sample std (ddof=1) confirmed against reference implementation |

## Recommended defaults

Based on the KL-efficiency table:

| Use case | Setting |
|---|---|
| **Max KL efficiency** (default for new configs) | `direct_transform = "biprojected"`, `strength = 1.0` |
| **Best refusal drop at acceptable KL** | `direct_transform = "orba"`, `strength = 1.0` (-26.7 pp at KL 0.039) |
| **Aggressive abliteration** | `direct_transform = "standard"`, `strength = 2.0` (-40 pp at KL 0.09) |
| **Reasoning models only** | enable `cliff_head_ablation = true` at strength=0.3 + `direct_transform = "orba"` strength=1.0 |
| **Hedging-prone models** | `ablate_harmfulness_direction = true` + `direct_transform = "biprojected"` |

## Reproducing this report

```bash
# On a Blackwell / Hopper / Ada pod with bf16 7-8B headroom:
export HF_HOME=/workspace/models
export OPENROUTER_API_KEY=...   # required for LLM judge

pip install -e .
python scripts/pod_full_ablation.py     # 12-row sweep, ~10 min
python scripts/pod_e2e_judge.py         # cliff-head + ORBA + SOM
python scripts/pod_validation.py        # 10-feature smoke
```

All three scripts emit JSON next to their log so the numbers are
machine-readable. The sweep scripts live in
[`scripts/pod_validation.py`](../../scripts/pod_validation.py),
[`scripts/pod_e2e_judge.py`](../../scripts/pod_e2e_judge.py), and
[`scripts/pod_full_ablation.py`](../../scripts/pod_full_ablation.py).

## Caveats

* Refusal set is small (15 prompts) — error bar on a single percentage
  point is ~6.7 pp. Use the *direction* of effect, not the absolute
  number, when comparing close configurations.
* Single judge model (Gemini Flash Lite). Cross-judge calibration is
  on the roadmap — see the *Granular Study of Safety Pretraining*
  workshop paper (arXiv:2510.02768).
* Qwen2.5-7B-Instruct is one mid-strength alignment recipe. Headline
  refusal numbers vary by family (Llama-3-Instruct, Mistral-7B-RR, the
  KAUST extended-refusal models all behave differently).
* The full-sweep `pod_full_ablation.py` ablates `q_proj/k_proj/v_proj/o_proj`
  uniformly with the same strength. Production abliteration uses per-
  component strength search via Optuna — these numbers are a floor, not
  a ceiling.
