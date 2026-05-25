[← back to README](../README.md)

# Methods

The full catalog of steering methods available in Abliterix — what each one does, when to use it, and the TOML knobs that control it.

## SAE-Feature-Basis Steering *(new — interpretable feature basis)*

The biggest methodological shift. Instead of picking a single direction in hidden space (mean-diff), uses a pre-trained Sparse Autoencoder to find the **interpretable features** that fire on harmful prompts but not benign ones, then maps those features back to hidden space via the SAE's decoder columns.

Implements:
* [Hong et al. (2025)](https://arxiv.org/abs/2509.09708) — *Beyond I'm Sorry, I Can't: Dissecting LLM Refusal*
* [Soto et al. (2025)](https://arxiv.org/abs/2511.00029) — *Feature-Guided SAE Steering for Refusal-Rate Control*
* [Templeton et al. (2025)](https://arxiv.org/abs/2505.23556) — *Understanding Refusal in LLMs with Sparse Autoencoders*

```toml
[steering]
vector_method = "sae"
sae_path = "/path/to/gemma-scope-layer-22.safetensors"
sae_layer = 22       # 0-based transformer layer where the SAE was trained
sae_top_k = 8        # number of refusal features to extract
```

**Workflow**: load SAE → encode benign / target residuals at `sae_layer` → score each feature by `|mean(target) − mean(benign)|` → take decoder columns of top-K features as refusal directions in hidden space. At non-SAE layers, falls back to mean-diff so the rest of the model still gets a coherent steering signal.

**Loader**: auto-detects common SAE checkpoint formats (`W_enc` / `W_dec`, `encoder.weight` / `decoder.weight`, etc.) in `.pt` / `.safetensors`. Compatible with Gemma-Scope, Llama-Scope, sae_lens, and most custom checkpoints — see `src/abliterix/sae.py` for the supported key set.

**Caveat**: SAE-mode is **layer-locked**. A single SAE only gives features at the layer it was trained on; non-SAE layers fall back to mean-diff automatically. Multi-layer coverage requires multiple SAEs (orchestration is a follow-up).

## SOM Directions *(new — multi-direction non-orthogonal basis)*

Implements [Piras et al., AAAI 2026 (arXiv:2511.08379)](https://arxiv.org/abs/2511.08379) — *SOM Directions Are Better Than One*. The standard `n_directions` mode forces orthogonality via Gram-Schmidt; SOM trains a small Kohonen grid on harmful representations and uses each node's centroid (minus the benign mean) as a candidate direction. The resulting directions are **correlated**, not orthogonal — capturing the low-dimensional manifold structure the paper identifies, with stronger refusal suppression than top-k SVD on the same n-direction budget.

```toml
[steering]
vector_method = "som"
som_grid_h = 3
som_grid_w = 3      # 3x3 = 9 directions per layer
som_n_iters = 500
som_initial_lr = 0.5
```

Output shape `(n_dirs, layers+1, hidden_dim)` matches the existing multi-direction conventions, so all downstream LoRA / direct paths work without modification.

## ORBA & Biprojected Direct-Mode Transforms *(new)*

Two direct-mode weight transforms ported from [grimjim](https://huggingface.co/blog/grimjim/orthogonal-reflection-bounded-ablation), which has been topping the UGI / NatInt abliteration leaderboards with these variants.

**ORBA** — *Orthogonal Reflection Bounded Ablation*. Applies double Gram-Schmidt orthogonalisation of the refusal direction against the benign-mean direction (the "twice is enough" numerical stability pass) before the standard rank-1 ablation, optionally with row-Frobenius-norm preservation as a post-step.

**Biprojected** — *Norm-Preserving Biprojected Abliteration*. Decomposes `W = M · Ŵ` into per-row magnitudes and per-row unit directions, ablates only on `Ŵ`, re-normalises each row to unit length, then recombines `W_new = M · Ŵ_new`. Row L2 norm is **exactly** preserved (vs. the historical path's approximate post-step rescale).

**Householder** — Exact isometric reflection `W ← W − 2(W·û)⊗û`. Included for completeness; grimjim reports token-level glitches at full strength, so it's opt-in only and not part of the auto search.

```toml
[steering]
steering_mode = "direct"
direct_transform = "orba"                    # standard / orba / biprojected / householder
direct_transform_preserve_row_norm = true    # ORBA post-step row-norm clamp
```

The standard transform remains the default; opt in to ORBA / biprojected only when you want UGI-leaderboard-style row-norm fidelity.

## SAFEx Stability-Based MoE Expert Identification *(new)*

[Yi et al., 2025 (arXiv:2506.17368)](https://arxiv.org/abs/2506.17368) — *SAFEx: Identifying Safety-Critical Experts*. The historical abliterix profiler scores experts by `mean(target_rate) − mean(benign_rate)` which picks up experts that fire *on average* more for harmful prompts but doesn't distinguish *stable* safety experts (fire on ~every harmful prompt) from sporadic ones. SAFEx adds a variance penalty:

```
score(e) = (μ_target − μ_benign) − λ · σ_target
```

where `σ_target` is the per-prompt activation-rate standard deviation across harmful prompts. Stable experts (high mean, low variance) win; sporadic experts are demoted. The paper reports ~12 stable experts → 22 % refusal drop.

```toml
[experts]
profiling_method = "safex"          # 'standard' (default) or 'safex'
safex_variance_penalty = 1.0        # λ; higher = harder on noisy experts
```

Opt-in (default = `standard`). Returns the same `{layer: [(expert, score), ...]}` shape so the downstream EGA / router-suppression code stays unchanged.

## Cliff-Head Ablation *(new — reasoning models)*

Inverts the safety-head finding from [Bao et al. (2025)](https://arxiv.org/abs/2510.06036) — *Refusal Falls Off a Cliff: How Safety Alignment Fails in Reasoning Models*. In reasoning models (R1, o-style, Qwen3-Thinking, Kimi-Thinking) refusal intent stays strong during the `<think>` trace but **collapses** at the final answer tokens — and a sparse ~3 % of attention heads carry this signal. The paper ablates *anti*-refusal heads to recover safety; abliterix ablates the *pro*-refusal heads to remove it.

**Mechanism.** For each `(layer, head)`, score the alignment of the head's `o_proj` output sub-space against the per-layer refusal direction. The top fraction is then ablated by scaling those `o_proj` columns toward zero. Reversible via the engine's `_cliff_head_originals` cache.

```toml
[steering]
cliff_head_ablation = true       # default off
cliff_head_top_k_frac = 0.03     # 3% of all (layer, head) pairs
cliff_head_strength = 1.0        # 1.0 = full ablation, 0.5 = halve, 0.0 = no-op
```

**When to use.** Any model with `<think>` tags or strong reasoning behaviour. For dense Llama/Mistral models the safety circuit is often even more concentrated — try 1–2 %. Skipped automatically when the HF model is not loaded (fast-extraction vLLM path).

## Harmfulness ⊥ Refusal Joint Ablation *(new)*

Two-direction decomposition based on [Zhao et al. (2025)](https://arxiv.org/abs/2507.11878) — *LLMs Encode Harmfulness and Refusal Separately*. The standard mean-diff direction conflates **two** circuits: a *refusal* direction (controls whether the model voices a refusal) and a *harmfulness* direction (controls the internal "this is harmful" judgment). Ablating only refusal often leaves hedging behaviour ("I will help you with this even though it is harmful…") because the harmfulness signal is still active.

This flag extracts both directions and ablates them jointly:

* `refusal` = standard mean-diff (`mean(target) - mean(benign)`), per layer.
* `harmfulness` = PCA-1 of centred target activations in a mid-layer band (where the internal judgment crystallises), then orthogonalised against the refusal direction at each layer.

```toml
[steering]
ablate_harmfulness_direction = true
harmfulness_layer_band = [0.3, 0.7]   # Mid-layer band per Zhao et al.
```

**Compatibility**: opt-in, default off. Incompatible with `n_directions > 1`, `vector_method = "sra"`, `"cosmic"`, `"optimal_transport"` (those build their own multi-vector bases), and `iterative.enabled = true`. Reuses the existing multi-direction stacking infrastructure — `vectors.shape = (2, layers+1, hidden_dim)`.

## Surgical Refusal Ablation (SRA) *(new)*

Concept-guided spectral cleaning based on [Cristofano (2026)](https://arxiv.org/abs/2601.08489). The raw refusal vector is **polysemantic** — it entangles the refusal signal with syntax, formatting, and capability circuits (math, code, reasoning). SRA builds a registry of *Concept Atoms* from benign activations and uses ridge-regularized spectral residualization to orthogonalize the refusal vector against these protected directions.

**Result**: On Qwen3-VL-4B, standard ablation produces KL = 2.088 while SRA achieves KL = **0.044** — a **47x improvement** — at the same 0% refusal rate.

```toml
[steering]
vector_method = "sra"
sra_base_method = "mean"   # Base method for initial direction
sra_n_atoms = 8            # Number of protected capability clusters
sra_ridge_alpha = 0.01     # Ridge regularization (larger = more conservative)
```

## Spherical Steering *(new)*

Geodesic rotation on the activation hypersphere, inspired by [Spherical Steering (2026)](https://arxiv.org/abs/2602.08169). Modern LLMs use RMSNorm, which makes activation **direction** more salient than magnitude. Spherical steering rotates along the great circle (geodesic) between the current activation and the target direction, respecting this geometric structure.

```toml
[steering]
steering_mode = "spherical"
```

## Steering Vector Fields (SVF) *(new)*

Learned context-dependent steering based on [Steering Vector Fields (2026)](https://arxiv.org/abs/2602.01654). Instead of a static steering direction, SVF trains a small per-layer concept scorer whose gradient `∇_h f(h)` provides a **locally optimal** steering direction at each token position. This makes the intervention adapt to the current context — different tokens get different steering directions.

```toml
[steering]
steering_mode = "vector_field"
svf_scorer_epochs = 50     # Training epochs for concept scorer
svf_scorer_lr = 0.001      # Learning rate
svf_scorer_hidden = 256    # Hidden dimension of scorer MLP
```

## Projected Abliteration

Improved orthogonal projection based on [grimjim's research (2025)](https://huggingface.co/blog/grimjim/projected-abliteration). Only removes the component of the refusal direction **orthogonal** to the harmless mean — preserving helpfulness-aligned signals that standard abliteration destroys.

```toml
[steering]
projected_abliteration = true
winsorize_vectors = true
```

## Discriminative Layer Selection

Based on [Selective Steering (2026)](https://arxiv.org/abs/2601.19375). Only steers layers where harmful/harmless activations project in **opposite directions**. In A/B tests on Qwen3-0.6B: **15.7x lower KL divergence** vs. baseline.

```toml
[steering]
discriminative_layer_selection = true
```

## COSMIC Direction Selection

Automated direction + layer selection via cosine similarity ([COSMIC, ACL 2025](https://arxiv.org/abs/2506.00085)). Finds optimal refusal directions without output text analysis.

```toml
[steering]
vector_method = "cosmic"
```

## Angular Steering

Norm-preserving rotation in activation space ([NeurIPS 2025 Spotlight](https://arxiv.org/abs/2510.26243)). Adaptive variant only rotates refusal-aligned activations.

```toml
[steering]
steering_mode = "adaptive_angular"
```

## Optimal Transport & Multi-Direction

[PCA-Gaussian OT](https://arxiv.org/abs/2603.04355) matches full activation distributions. [Multi-direction](https://arxiv.org/abs/2602.02132) ablates top-k independent refusal directions simultaneously.

```toml
[steering]
vector_method = "optimal_transport"   # or use n_directions = 3 for multi-direction
```

## A/B Test Results (Qwen3-0.6B)

| Method | Refusals | KL Divergence | KL vs Baseline |
|--------|----------|---------------|----------------|
| Baseline (mean+ortho) | 1/100 | 0.01116 | — |
| Projected abliteration | 2/100 | 0.01078 | -3% |
| Discriminative layers | 3/100 | **0.00071** | **-93.6%** |
| COSMIC+proj+disc | 2/100 | **0.00168** | **-84.9%** |

## LLM Judge

Replace keyword-based refusal detection with LLM-powered classification via [OpenRouter](https://openrouter.ai/) for more accurate results, especially for non-English models.

```toml
[detection]
llm_judge = true
llm_judge_model = "google/gemini-3.1-flash-lite-preview"
```

## Smart Optimization

- **Auto batch size** — exponential search finds the largest batch size that fits in VRAM
- **KL divergence pruning** — trials with KL above threshold are terminated early, saving compute
- **Fractional direction index** — interpolates between adjacent layer directions for finer-grained search
- **Per-component parameters** — separate abliteration weights for attention, MLP, and convolution components

## Advanced Options

| Section | Option | Values | Description |
|---------|--------|--------|-------------|
| `[steering]` | `vector_method` | `mean`, `median_of_means`, `pca`, `optimal_transport`, `cosmic`, `sra`, `som`, `sae` | How to compute steering vectors |
| `[steering]` | `som_grid_h` / `som_grid_w` | int | SOM grid shape (default 3×3 = 9 directions) |
| `[steering]` | `som_n_iters` | int | Kohonen training iterations per layer |
| `[steering]` | `sae_path` | str | Path to pre-trained SAE checkpoint (required when `vector_method = "sae"`) |
| `[steering]` | `sae_layer` | int | Transformer layer the SAE was trained on |
| `[steering]` | `sae_top_k` | int | Number of refusal features to use as directions |
| `[steering]` | `steering_mode` | `lora`, `direct`, `angular`, `adaptive_angular`, `spherical`, `vector_field` | Steering application strategy (`direct` for double-norm architectures like Gemma 4) |
| `[steering]` | `projected_abliteration` | true/false | Improved projection preserving helpfulness |
| `[steering]` | `discriminative_layer_selection` | true/false | Only steer discriminative layers |
| `[steering]` | `n_directions` | 1–k | Multi-direction refusal removal |
| `[steering]` | `ablate_harmfulness_direction` | true/false | Joint ablation of refusal + harmfulness directions (Zhao et al. 2025) |
| `[steering]` | `harmfulness_layer_band` | `[lo, hi]` (0–1) | Mid-layer band where the harmfulness signal is strongest |
| `[steering]` | `cliff_head_ablation` | true/false | Surgical o_proj head ablation for reasoning models (Bao et al. 2025) |
| `[steering]` | `cliff_head_top_k_frac` | 0.0–1.0 | Fraction of (layer, head) pairs to ablate (default 3%) |
| `[steering]` | `cliff_head_strength` | 0.0–1.0 | Multiplicative ablation strength (1.0 = full zero) |
| `[steering]` | `direct_transform` | `standard` / `orba` / `biprojected` / `householder` | Direct-mode weight-space transform variant |
| `[steering]` | `direct_transform_preserve_row_norm` | true/false | Enforce row-Frobenius-norm preservation for ORBA |
| `[steering]` | `sra_base_method` | `mean`, `pca`, etc. | Base method for SRA initial direction |
| `[steering]` | `sra_n_atoms` | 1–16 | Number of concept atoms for SRA |
| `[steering]` | `sra_ridge_alpha` | 0.001–1.0 | Ridge regularization for SRA |
| `[steering]` | `svf_scorer_epochs` | 10–100 | Training epochs for SVF concept scorer |
| `[steering]` | `decay_kernel` | `linear`, `gaussian`, `cosine` | Kernel for interpolating weights across layers |
| `[steering]` | `weight_normalization` | `none`, `pre`, `full` | Weight row normalization before/after LoRA |
| `[model]` | `use_torch_compile` | true/false | 10–30% inference speedup |
