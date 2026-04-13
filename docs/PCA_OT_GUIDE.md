# PCA-OT: Full Affine Transformation for Refusal Ablation

This document describes the PCA-OT (PCA-Optimal Transport) method implemented in Abliterix, based on the paper "Efficient Refusal Ablation in LLM through Optimal Transport" (arxiv:2603.04355, March 2026).

## Overview

PCA-OT is a state-of-the-art abliteration technique that removes safety alignment from language models while preserving capabilities better than existing methods.

**Key advantages over baseline methods:**
- Captures multi-dimensional refusal structure (covariance), not just mean difference
- Layer-selective intervention (1-2 layers at 40-60% depth) vs. all-layer modification
- Up to 11% higher attack success rate with comparable perplexity
- Within 1-6% of baseline on MMLU/TruthfulQA/ARC/GSM8K benchmarks

## How It Works

### Mathematical Foundation

Traditional abliteration methods use a simple direction vector `v` to steer activations:
```
h' = h - α·v
```

PCA-OT uses a full affine transformation that captures both mean and covariance:
```
h' = A_full @ h + b_full
```

Where:
- `A_full` is a (d, d) transformation matrix
- `b_full` is a (d,) bias vector
- The transformation is computed via Gaussian optimal transport in PCA-reduced space

### Algorithm Steps

1. **Compute pooled mean**: `μ_pool = (n_h·μ_H + n_s·μ_S)/(n_h + n_s)`
2. **Center activations**: `Z = [X_H - μ_pool; X_S - μ_pool]`
3. **PCA projection**: SVD on Z, take top k components → P (k, d)
4. **Project to k-dim**: `Y_H = (X_H - μ_pool)P`, `Y_S = (X_S - μ_pool)P`
5. **Gaussian OT in k-dim**: Compute `A_k`, `b_k` via closed-form OT map
6. **Lift to full space**: `A_full = P^T A_k P`, `b_full = μ_S - A_full μ_H`

## Usage

### Basic Example

```python
from abliterix.vectors import compute_pca_ot_transforms
from abliterix.pca_ot_hooks import apply_pca_ot_hooks

# Extract activations (see examples/pca_ot_basic_usage.py)
harmful_states = extract_activations(model, tokenizer, harmful_prompts)
harmless_states = extract_activations(model, tokenizer, harmless_prompts)

# Compute transforms
transforms = compute_pca_ot_transforms(
    harmless_states,
    harmful_states,
    n_components=32,  # Paper recommends 32-64
)

# Apply as hooks (inference-time)
n_layers = len(transforms)
target_layers = list(range(int(0.4 * n_layers), int(0.6 * n_layers)))

with apply_pca_ot_hooks(model, transforms, layer_indices=target_layers):
    outputs = model.generate(...)
```

### Weight Baking (Permanent Modification)

```python
from abliterix.pca_ot_weight_baking import bake_pca_ot_into_model

# Bake into model weights
bake_pca_ot_into_model(
    model,
    transforms,
    layer_indices=target_layers,
    target_module_name="mlp.down_proj",  # Architecture-specific
    rank=16,  # Low-rank approximation
)

# Save modified model
model.save_pretrained("./abliterated-model")
```

### Norm-Preserving Mode

To reduce distribution shift, scale transformed activations to preserve norms:

```python
with apply_pca_ot_hooks(model, transforms, layer_indices=[15], norm_preserving=True):
    outputs = model.generate(...)
```

## Hyperparameters

### n_components (k)
- **Range**: 8-128
- **Paper recommendation**: 32-64
- **Trade-off**: Higher k captures more structure but increases computation
- **Default**: 32

### Layer Selection
- **Paper finding**: 40-60% network depth is optimal
- **PCA-OT1**: Apply to 1 layer (faster, less interference)
- **PCA-OT2**: Apply to 2 layers (higher ASR, more aggressive)
- **Example**: For 32-layer model, try layers 13-19

### Rank (for weight baking)
- **Range**: 8-64
- **Paper recommendation**: 16-32
- **Trade-off**: Lower rank = smaller model size, higher rank = better approximation
- **Default**: 16

## Architecture Support

### Supported Models
Works with any HuggingFace transformer model. The hook-based approach automatically detects common layer structures.

### Target Modules for Weight Baking
When using weight baking, you need to specify which module to modify. This is typically the final projection in the MLP block. Common patterns:

- Most modern transformers: `mlp.down_proj` or `mlp.c_proj`
- Encoder models: `output` or `output.dense`

Inspect your model structure to find the right module name. Look for the linear layer that projects back to hidden_dim after the MLP expansion.

## API Reference

### compute_pca_ot_transforms()

```python
def compute_pca_ot_transforms(
    benign_states: Tensor,
    target_states: Tensor,
    n_components: int = 32,
) -> list[PCAOTTransform]:
    """Compute PCA-OT affine transformations for each layer.

    Parameters
    ----------
    benign_states : Tensor
        Harmless activations, shape (n_benign, n_layers, hidden_dim)
    target_states : Tensor
        Harmful activations, shape (n_harmful, n_layers, hidden_dim)
    n_components : int
        Number of PCA components (paper recommends 32-64)

    Returns
    -------
    list[PCAOTTransform]
        One transform per layer containing A_full, b_full, P, A_k, b_k
    """
```

### PCAOTHookManager

```python
class PCAOTHookManager:
    """Manages forward hooks for PCA-OT affine transformations.

    Parameters
    ----------
    model : nn.Module
        The transformer model to apply hooks to
    transforms : list[PCAOTTransform]
        PCA-OT transforms for each layer
    layer_indices : list[int], optional
        Specific layer indices to apply transforms to
    norm_preserving : bool, default False
        If True, scale transformed activations to preserve norms

    Methods
    -------
    register_hooks()
        Register forward hooks on specified layers
    remove_hooks()
        Remove all registered hooks
    """
```

### bake_pca_ot_into_model()

```python
def bake_pca_ot_into_model(
    model: nn.Module,
    transforms: list[PCAOTTransform],
    layer_indices: list[int],
    target_module_name: str = "mlp.down_proj",
    rank: Optional[int] = None,
) -> None:
    """Bake PCA-OT transformations into model weights in-place.

    Parameters
    ----------
    model : nn.Module
        The transformer model to modify
    transforms : list[PCAOTTransform]
        PCA-OT transforms for each layer
    layer_indices : list[int]
        Layer indices to apply transformations to
    target_module_name : str
        Name of module within each layer to modify
    rank : int, optional
        Rank for low-rank approximation (None = full matrix)
    """
```

## Performance Expectations

Based on the paper (arxiv:2603.04355):

| Metric | PCA-OT1 | PCA-OT2 | Baseline (RFA) |
|--------|---------|---------|----------------|
| ASR (Attack Success Rate) | ~79% | ~82% | ~71% |
| KL Divergence | ~8.4 | ~9.1 | ~7.8 |
| MMLU | -1.2% | -2.1% | -0.8% |
| TruthfulQA | -3.4% | -4.2% | -2.9% |

**Key findings:**
- PCA-OT1 (1 layer): Best balance of effectiveness and capability preservation
- PCA-OT2 (2 layers): Higher ASR but more capability degradation
- Optimal depth: 40-60% of network depth
- n_components=32-64 works best across models

## Comparison with Other Methods

| Method | Type | Pros | Cons |
|--------|------|------|------|
| **PCA-OT** | Affine transform | Multi-dimensional, layer-selective, high ASR | Requires activation extraction |
| MEAN | Direction vector | Simple, fast | Only captures mean difference |
| PCA | Direction vector | Captures variance | Single direction, all layers |
| OPTIMAL_TRANSPORT | Direction vector | Distribution matching | Low-rank (k=2), approximation |
| COSMIC | Direction vector | Cosine-similarity based | Heuristic selection |
| SRA | Spectral cleaning | Concept-guided | Complex, slower |

## Troubleshooting

### "Could not find transformer layers in model"
- Check that your model architecture is supported
- Try manually specifying layer path in `PCAOTHookManager._find_layer_modules()`

### "A_full shape incompatible with linear layer"
- Ensure `target_module_name` points to a module that takes hidden_dim as input
- For Llama: use `mlp.down_proj` (not `mlp.up_proj`)

### Low attack success rate
- Try increasing `n_components` (32 → 64)
- Expand layer range (try 30-70% depth)
- Use PCA-OT2 (2 layers) instead of PCA-OT1
- Increase dataset size (400+ prompts each)

### High capability degradation
- Reduce `n_components` (64 → 32)
- Narrow layer range (50-60% depth)
- Use PCA-OT1 (1 layer) instead of PCA-OT2
- Enable `norm_preserving=True`

## Citation

If you use PCA-OT in your research, please cite:

```bibtex
@article{pcaot2026,
  title={Efficient Refusal Ablation in LLM through Optimal Transport},
  author={[Authors]},
  journal={arXiv preprint arXiv:2603.04355},
  year={2026}
}
```

## References

- Paper: arxiv:2603.04355 "Efficient Refusal Ablation in LLM through Optimal Transport" (March 2026)
- Abliterix: https://github.com/wuwangzhang1216/abliterix
- Heretic: https://github.com/p-e-w/heretic
