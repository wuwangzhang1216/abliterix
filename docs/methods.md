[← back to README](../README.md)

# Methods

The full catalog of steering methods available in Abliterix — what each one does, when to use it, and the TOML knobs that control it.

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
| `[steering]` | `vector_method` | `mean`, `median_of_means`, `pca`, `optimal_transport`, `cosmic`, `sra` | How to compute steering vectors |
| `[steering]` | `steering_mode` | `lora`, `direct`, `angular`, `adaptive_angular`, `spherical`, `vector_field` | Steering application strategy (`direct` for double-norm architectures like Gemma 4) |
| `[steering]` | `projected_abliteration` | true/false | Improved projection preserving helpfulness |
| `[steering]` | `discriminative_layer_selection` | true/false | Only steer discriminative layers |
| `[steering]` | `n_directions` | 1–k | Multi-direction refusal removal |
| `[steering]` | `sra_base_method` | `mean`, `pca`, etc. | Base method for SRA initial direction |
| `[steering]` | `sra_n_atoms` | 1–16 | Number of concept atoms for SRA |
| `[steering]` | `sra_ridge_alpha` | 0.001–1.0 | Ridge regularization for SRA |
| `[steering]` | `svf_scorer_epochs` | 10–100 | Training epochs for SVF concept scorer |
| `[steering]` | `decay_kernel` | `linear`, `gaussian`, `cosine` | Kernel for interpolating weights across layers |
| `[steering]` | `weight_normalization` | `none`, `pre`, `full` | Weight row normalization before/after LoRA |
| `[model]` | `use_torch_compile` | true/false | 10–30% inference speedup |
