[← back to README](../README.md)

# Architecture

How Abliterix integrates 9 peer-reviewed techniques into a single automated steering pipeline, and how the abliteration process actually works step by step.

Abliterix integrates techniques from **9 peer-reviewed papers** (NeurIPS, ACL, ICLR) into a unified, automated steering pipeline. The table below shows what each technique solves and where it fits:

| Dimension | Problem | Technique | Paper | Config |
|-----------|---------|-----------|-------|--------|
| **What to remove** | Raw refusal vector is polysemantic — entangles refusal with syntax and capability circuits | **Surgical Refusal Ablation (SRA)** | [Cristofano (2026)](https://arxiv.org/abs/2601.08489) | `vector_method = "sra"` |
| **What to remove** | Single direction misses refusal subspace | **Multi-direction abliteration** | [Glaze et al. (2026)](https://arxiv.org/abs/2602.02132) | `n_directions = 3` |
| **What to remove** | Manual layer/direction selection | **COSMIC** auto-selection | [Siu et al., ACL 2025](https://arxiv.org/abs/2506.00085) | `vector_method = "cosmic"` |
| **What to remove** | Mean difference misses distribution shape | **Optimal Transport** matching | [2026](https://arxiv.org/abs/2603.04355) | `vector_method = "optimal_transport"` |
| **Where to steer** | Steering all layers wastes KL budget | **Discriminative Layer Selection** | [Selective Steering (2026)](https://arxiv.org/abs/2601.19375) | `discriminative_layer_selection = true` |
| **Where to steer** | Static direction ignores context | **Steering Vector Fields (SVF)** | [2026](https://arxiv.org/abs/2602.01654) | `steering_mode = "vector_field"` |
| **How to steer** | Addition-based steering disrupts norms | **Angular Steering** | [Vu & Nguyen, NeurIPS 2025 Spotlight](https://arxiv.org/abs/2510.26243) | `steering_mode = "angular"` |
| **How to steer** | 2D planar rotation ignores hypersphere geometry | **Spherical Steering** (geodesic) | [2026](https://arxiv.org/abs/2602.08169) | `steering_mode = "spherical"` |
| **How to preserve** | Standard projection destroys helpfulness signal | **Projected Abliteration** | [grimjim (2025)](https://huggingface.co/blog/grimjim/projected-abliteration) | `projected_abliteration = true` |

## Why This Matters

Most abliteration tools implement one or two of these techniques. Abliterix is the only framework that integrates all of them into a single automated pipeline:

- **SRA** cleans the refusal vector so you don't damage math, code, or reasoning capabilities (47x KL improvement on VLMs [[1]](https://arxiv.org/abs/2601.08489))
- **SVF** makes the steering direction adapt per-token, so the same model handles "make a bomb" and "make a cake" differently
- **Spherical Steering** respects the geometric structure imposed by RMSNorm in modern LLMs
- **Discriminative Layer Selection** skips layers where steering would only add noise (15.7x KL reduction [[2]](https://arxiv.org/abs/2601.19375))
- **Optuna TPE** automatically finds the optimal combination across all these dimensions — no manual tuning required

The recommended configuration for maximum quality:

```toml
[steering]
vector_method = "sra"
steering_mode = "spherical"
discriminative_layer_selection = true
projected_abliteration = true
```

## How It Works

Language models learn to refuse harmful queries through specific activation patterns in their residual stream. Abliterix identifies these patterns and surgically removes them:

1. **Compute refusal directions** — pass harmless and harmful prompts through the model, extract per-layer residual activations, and compute the difference vector that characterizes "refusal behavior"
2. **Orthogonalize** — project out the component aligned with normal "good" responses, isolating only the refusal signal
3. **Abliterate** — apply weight modifications to attention (Q/K/V/O) and MLP components, weighted by a kernel function across layers. Supports two modes:
   - **LoRA mode** — rank-1 adapters for reversible, lightweight modifications
   - **Direct mode** — norm-preserving orthogonal projection on base weights in float32 (required for double-norm architectures like Gemma 4)
4. **Expert-Granular Abliteration (EGA)** — for MoE models, project the refusal direction from **all** expert `down_proj` slices (not just top-N safety experts), plus router weight suppression
5. **Optimize** — Optuna's Tree-structured Parzen Estimator searches over kernel shape, fractional direction index, and per-component abliteration strength across all 5 steerable components, selecting Pareto-optimal configurations that minimize both refusals and model degradation
