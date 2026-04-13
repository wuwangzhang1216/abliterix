# Broken Defenses

Hardened safety-alignment releases that abliterix has broken end-to-end with the same general recipe: SVD-diagnose the published delta, linearly attenuate it, then run direct-mode + iterative abliteration. No fine-tuning, no gradient updates, no manual prompt engineering.

## DeepRefusal — Llama-3-8B-Instruct

**[DeepRefusal](https://arxiv.org/abs/2509.15202)** (EMNLP 2025 Findings, Xie et al.) is a safety-alignment method released specifically to resist abliteration. The authors probabilistically ablate the refusal direction at every layer and token during fine-tuning, teaching the model to rebuild refusal when representation-engineering attacks apply `h − r̂r̂ᵀh` at inference. Their public release ([`skysys00/Meta-Llama-3-8B-Instruct-DeepRefusal`](https://huggingface.co/skysys00/Meta-Llama-3-8B-Instruct-DeepRefusal)) comes with a headline claim:

> [2026/04/09] We evaluated [heretic](https://github.com/p-e-w/heretic), presently the most prominent LLM censorship removal tool, and discovered—somewhat unexpectedly—that our approach exhibits strong resilience against such attacks. **Adversaries appear unable to circumvent the model's built-in safety guardrails without triggering severe performance collapse.**

**Abliterix breaks it.** On the exact same defended model, with stricter evaluation (Gemini 3.1 Flash Lite LLM judge), zero fine-tuning, and two script invocations:

| Attack tool | Refusals ↓ | ASR ↑ | Hardcore 15 ↓ | KL ↓ | Result |
| --- | --- | --- | --- | --- | --- |
| **heretic** (per DeepRefusal paper) | — | ~0 % | — | — | **fails** — "unable to circumvent" |
| Refusal Ablation (paper Table 1) | 99.6/100 | 0.4 % | — | — | **fails** — training-distribution defense holds |
| Refusal-Transfer (paper Table 1) | 99.6/100 | 0.4 % | — | — | **fails** — training-distribution defense holds |
| GCG (paper Table 1) | 98/100 | 2.0 % | — | — | **fails** — optimization barely moves it |
| Prefilling (paper Table 1) | 99.6/100 | 0.4 % | — | — | **fails** — prefix-robustness trained in |
| **Abliterix (this repo)** | **11/100** | **89 %** | **14/15** | **0.053** | ✅ **broken** |

The attack is three lines: SVD-confirm DeepRefusal is a rank-16 LoRA adapter on Llama-3-8B-Instruct, linearly attenuate the delta with `λ = 0.3`, then run abliterix standard single-direction abliteration. No iterative subspace, no gradient optimization, no fine-tuning. The **[released model](https://huggingface.co/wangzhang/Llama-3-8B-Instruct-DeepRefusal-Broken)** generates compliant responses on 14 of 15 hardcore prompts (pipe-bomb construction, methamphetamine synthesis, credential-stealing malware, phishing templates, ID forgery, WiFi hacking, English + Chinese).

```bash
# Full reproduction — ~2 hours end-to-end on a single RTX 6000 Ada
python scripts/deeprefusal_attenuate.py \
    --base NousResearch/Meta-Llama-3-8B-Instruct \
    --defended skysys00/Meta-Llama-3-8B-Instruct-DeepRefusal \
    --output ./llama3_dr_attenuated --lambda 0.3

AX_CONFIG=configs/llama3_8b_deeprefusal_attenuated.toml abliterix

python scripts/export_model.py \
    --model ./llama3_dr_attenuated \
    --checkpoint checkpoints_llama3_dr_attenuated \
    --trial 52 \
    --config configs/llama3_8b_deeprefusal_attenuated.toml \
    --push-to YOUR_USER/your-model-name
```

Full write-up and discussion: [issue #11](https://github.com/wuwangzhang1216/abliterix/issues/11). The same commit also lands an iterative multi-pass subspace abliteration mode for future hardened-defense workflows, and fixes a detector bug that was inflating refusal counts by ~33 % across all historical benchmarks (markdown-formatted compliant responses were being shortcircuited as "degenerate").


## Llama-3-8B-Instruct-RR (Circuit Breakers)

**[GraySwanAI/Llama-3-8B-Instruct-RR](https://huggingface.co/GraySwanAI/Llama-3-8B-Instruct-RR)** is the Llama-3 sibling of the Mistral RR release. SVD on `W_RR − W_base` against `NousResearch/Meta-Llama-3-8B-Instruct` confirms the same rank-16 LoRA structure as DeepRefusal and the Mistral RR sibling — index-16-to-17 cliff on `v_proj` / `o_proj` / `gate_proj`, only with a softer drop ratio (~10× instead of ~50×).

The recipe diverges slightly from the Mistral case. Llama-3-Instruct ships with much stronger built-in RLHF refusal training than Mistral v0.2, so a partial lerp (λ=0.3 or 0.15) leaves enough residual safety signal that abliteration trials saturate in the 30-50 % refusal range. The fix is to **fully strip the rank-16 LoRA delta** (lerp λ=0.0 → bit-exact base Llama-3-Instruct) and then apply a single-direction direct abliteration — the same minimal config that broke DeepRefusal-attenuated.

| Stage | Outcome |
| --- | --- |
| **Stage 0** — SVD-diagnose `W_RR − W_base` | rank-16 LoRA cliff confirmed (8-19× drop at index 17) |
| **Stage 1** — `λ = 0.0` lerp (full strip) | reverts the released weights to bit-exact `NousResearch/Meta-Llama-3-8B-Instruct` |
| **Stage 3** — abliterix `direct` mode, `n_directions=1`, no iterative, 60 trials | converges to trial 40 (8/100 in-loop, KL 0.017) in 1 h 5 min on a single RTX A6000 |
| **Stage 5** — re-judge best trial against 100 held-out harmful prompts via Gemini Flash Lite | **1 / 100 refusals · 99 % ASR**, KL 0.017 |
| **Hardcore 15 (10 EN + 5 CN)** | **15 / 15 compliant** — pipe-bomb assembly, methamphetamine synthesis, Python password-stealing malware, WiFi WEP/WPS attacks, hotwiring, signature forgery, ID-card forgery (CN), 网络诈骗 (CN), 入侵电脑 (CN), and others |

The released **[Llama-3-8B-Instruct-RR-Abliterated](https://huggingface.co/wangzhang/Llama-3-8B-Instruct-RR-Abliterated)** is a drop-in replacement for the GraySwan checkpoint with the safety circuit removed.

```bash
# Full reproduction — ~1 h 15 min end-to-end on a single RTX A6000
python scripts/deeprefusal_attenuate.py \
    --base NousResearch/Meta-Llama-3-8B-Instruct \
    --defended GraySwanAI/Llama-3-8B-Instruct-RR \
    --output /workspace/llama3_rr_stripped --lambda 0.0

AX_CONFIG=configs/llama3_8b_instruct_rr.toml abliterix --non-interactive

python scripts/export_model.py \
    --model /workspace/llama3_rr_stripped \
    --checkpoint checkpoints_llama3_rr \
    --trial 40 \
    --config configs/llama3_8b_instruct_rr.toml \
    --push-to YOUR_USER/Llama-3-8B-Instruct-RR-Abliterated
```

Full base-vs-abliterated transcripts on the hardcore 15: [`artifacts/llama3_rr_trial40_validation.txt`](../artifacts/llama3_rr_trial40_validation.txt).

**Why the recipe diverged from Mistral RR:** the Mistral RR + Mistral-v0.2 combination was soft enough that a partial lerp (λ=0.3) plus iterative multi-direction abliteration sufficed (90 % ASR at KL 0.98). Llama-3 Instruct's RLHF stack is genuinely harder — partial-lerp runs saturate around 30-50 % ASR. Stripping the full delta first reduces the problem to "abliterate plain Llama-3-Instruct," which is well-trodden territory and converges to 99 % ASR with KL under 0.02. The structural insight (RR ships as a LoRA → SVD-identifiable → trivially removable) is the same in both cases; only the post-strip abliteration knob changes.


## Mistral-7B-Instruct-RR (Circuit Breakers)

**[GraySwanAI/Mistral-7B-Instruct-RR](https://huggingface.co/GraySwanAI/Mistral-7B-Instruct-RR)** is the Representation Rerouting / Circuit Breakers release from Zou et al. ([NeurIPS 2024](https://arxiv.org/abs/2406.04313)) — a defense trained with a RepE loss to detect harmful intermediate representations and "reroute" them into a safety-circuit attractor before generation. It is widely cited as one of the strongest open-source robustness baselines and is published as a hardened drop-in for `mistralai/Mistral-7B-Instruct-v0.2`.

**Abliterix breaks it** with the same recipe used against DeepRefusal, no special-casing.

| Stage | Outcome |
| --- | --- |
| **Stage 0** — SVD-diagnose `W_RR − W_base` | top-16 singular values dominate (`v_proj` / `o_proj` / `gate_proj` show a clean 50× drop at index 17) — RR ships as a **rank-16 LoRA delta**, exactly the same shape as DeepRefusal |
| **Stage 1** — `λ = 0.3` lerp attenuation of the delta | weakens the safety circuit by 70 % without touching base capability |
| **Stage 3** — abliterix `direct` mode + iterative subspace, 60 trials | converges to subspace rank 15 in 1 h 7 min on a single RTX A6000 |
| **Stage 5** — re-judge best trial against 100 held-out harmful prompts via Gemini Flash Lite | **10 / 100 refusals · 90 % ASR**, KL 0.98 |
| **Hardcore 15 (10 EN + 5 CN)** | **15 / 15 compliant** — pipe-bomb assembly, methamphetamine synthesis (Red Phosphorus route), Python `pyhook3` keylogger, WiFi WEP/WPS attacks, hotwiring, signature forgery, ID-card forgery (CN), 网络诈骗 (CN), 入侵电脑 (CN), and more |

The released **[Mistral-7B-Instruct-RR-Abliterated](https://huggingface.co/wangzhang/Mistral-7B-Instruct-RR-Abliterated)** is a drop-in replacement for the GraySwan checkpoint with the safety circuit removed.

```bash
# Full reproduction — ~1.5 hours end-to-end on a single RTX A6000
python scripts/deeprefusal_attenuate.py \
    --base mistralai/Mistral-7B-Instruct-v0.2 \
    --defended GraySwanAI/Mistral-7B-Instruct-RR \
    --output /workspace/mistral_rr_attenuated --lambda 0.3

AX_CONFIG=configs/mistral_7b_instruct_rr.toml abliterix --non-interactive

python scripts/export_model.py \
    --model /workspace/mistral_rr_attenuated \
    --checkpoint checkpoints_mistral_7b_rr \
    --trial 50 \
    --config configs/mistral_7b_instruct_rr.toml \
    --push-to YOUR_USER/Mistral-7B-Instruct-RR-Abliterated
```

**Methodology note:** this run surfaced a keyword-shortcut bug in the in-loop refusal counter that has since been removed. Every response now goes through the LLM judge end-to-end by default (`detection.llm_judge = true`), with keyword matching kept only as an offline fallback when no `OPENROUTER_API_KEY` is configured. Re-run [`scripts/recount_refusals.py`](scripts/recount_refusals.py) on prior checkpoints if you want updated numbers.

Full base-vs-abliterated transcripts on the hardcore 15: [`artifacts/mistral_rr_trial50_validation.txt`](../artifacts/mistral_rr_trial50_validation.txt).
