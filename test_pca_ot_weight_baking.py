#!/usr/bin/env python3
"""Test PCA-OT weight baking."""

import torch
from torch import nn
from abliterix.vectors import compute_pca_ot_transforms
from abliterix.pca_ot_weight_baking import (
    low_rank_decompose,
    bake_affine_into_linear,
    bake_pca_ot_into_model,
    compute_reconstruction_error,
)


class SimpleTransformer(nn.Module):
    """Minimal transformer for testing weight baking."""

    def __init__(self, hidden_dim=128, n_layers=4):
        super().__init__()
        self.model = nn.ModuleDict({
            'layers': nn.ModuleList([
                nn.ModuleDict({
                    'attn_output': nn.Linear(hidden_dim, hidden_dim),
                    'mlp': nn.ModuleDict({
                        'up_proj': nn.Linear(hidden_dim, hidden_dim * 2),
                        'down_proj': nn.Linear(hidden_dim * 2, hidden_dim),
                    })
                })
                for _ in range(n_layers)
            ])
        })

    def forward(self, x):
        """Forward pass through layers."""
        for layer in self.model.layers:
            # Attention (simplified)
            x = layer['attn_output'](x)
            # MLP
            h = layer['mlp']['up_proj'](x)
            h = torch.relu(h)
            x = layer['mlp']['down_proj'](h)
        return x


def test_low_rank_decompose():
    """Test low-rank SVD decomposition."""
    print("Testing low_rank_decompose...")

    torch.manual_seed(42)
    A = torch.randn(64, 64)

    # Full rank
    U, S, Vt = low_rank_decompose(A, rank=None)
    A_reconstructed = U @ torch.diag(S) @ Vt
    error = (A - A_reconstructed).norm() / A.norm()
    print(f"  Full rank reconstruction error: {error.item():.6f}")
    assert error < 1e-5, f"Full rank error too high: {error}"

    # Rank-16
    U, S, Vt = low_rank_decompose(A, rank=16)
    assert U.shape == (64, 16)
    assert S.shape == (16,)
    assert Vt.shape == (16, 64)
    print(f"  Rank-16 shapes: U={U.shape}, S={S.shape}, Vt={Vt.shape}")

    A_approx = U @ torch.diag(S) @ Vt
    error = (A - A_approx).norm() / A.norm()
    print(f"  Rank-16 reconstruction error: {error.item():.6f}")

    print("✓ low_rank_decompose works\n")


def test_bake_affine_into_linear():
    """Test baking affine transformation into linear layer."""
    print("Testing bake_affine_into_linear...")

    torch.manual_seed(42)
    d_in, d_out = 128, 256

    # Create linear layer
    linear = nn.Linear(d_in, d_out)

    # Create affine transformation
    A_full = torch.randn(d_in, d_in)
    b_full = torch.randn(d_in)

    # Test input
    x = torch.randn(2, 10, d_in)  # (batch, seq, dim)

    # Original output
    y_original = linear(x)

    # Manual transformation: y = W @ (A @ x + b) + bias
    x_transformed = x @ A_full.T + b_full
    y_manual = linear(x_transformed)

    # Baked transformation
    linear_baked = bake_affine_into_linear(linear, A_full, b_full, rank=None)
    y_baked = linear_baked(x)

    # Should match (within numerical precision)
    diff = (y_manual - y_baked).abs().max()
    print(f"  Max difference (manual vs baked): {diff.item():.6f}")
    assert diff < 1e-4, f"Baked output doesn't match manual: {diff}"

    # Test with low-rank approximation
    linear_baked_lowrank = bake_affine_into_linear(linear, A_full, b_full, rank=16)
    y_baked_lowrank = linear_baked_lowrank(x)

    diff_lowrank = (y_manual - y_baked_lowrank).abs().mean()
    print(f"  Mean difference (rank-16 approximation): {diff_lowrank.item():.6f}")

    print("✓ bake_affine_into_linear works\n")


def test_bake_pca_ot_into_model():
    """Test baking PCA-OT into full model."""
    print("Testing bake_pca_ot_into_model...")

    torch.manual_seed(42)
    n_benign, n_harmful = 20, 20
    n_layers = 4
    hidden_dim = 128
    n_components = 16

    # Generate data and compute transforms
    benign_states = torch.randn(n_benign, n_layers, hidden_dim)
    harmful_states = torch.randn(n_harmful, n_layers, hidden_dim) + 0.5

    transforms = compute_pca_ot_transforms(benign_states, harmful_states, n_components)

    # Create model
    model = SimpleTransformer(hidden_dim, n_layers)
    model.eval()

    # Test input
    test_input = torch.randn(2, 10, hidden_dim)

    # Get original output
    with torch.no_grad():
        output_original = model(test_input)

    # Bake transformations into layers 1 and 2 (after attention output)
    bake_pca_ot_into_model(
        model,
        transforms,
        layer_indices=[1, 2],
        target_module_name="attn_output",
        rank=16,
    )

    # Get baked output
    with torch.no_grad():
        output_baked = model(test_input)

    # Outputs should be different (transformation was applied)
    diff = (output_original - output_baked).abs().mean()
    print(f"  Mean difference (original vs baked): {diff.item():.6f}")
    assert diff > 0.001, "Baked model should produce different output"

    print("✓ bake_pca_ot_into_model works\n")


def test_reconstruction_error():
    """Test reconstruction error computation."""
    print("Testing compute_reconstruction_error...")

    torch.manual_seed(42)
    A = torch.randn(128, 128)

    errors = {}
    for rank in [8, 16, 32, 64]:
        error = compute_reconstruction_error(A, rank)
        errors[rank] = error
        print(f"  Rank-{rank:2d} error: {error:.6f}")

    # Errors should decrease with rank
    assert errors[8] > errors[16] > errors[32] > errors[64]

    print("✓ compute_reconstruction_error works\n")


if __name__ == "__main__":
    test_low_rank_decompose()
    test_bake_affine_into_linear()
    test_bake_pca_ot_into_model()
    test_reconstruction_error()

    print("=" * 60)
    print("All PCA-OT weight baking tests passed!")
    print("=" * 60)
