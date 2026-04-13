#!/usr/bin/env python3
"""Quick test of PCA-OT implementation."""

import torch
from src.abliterix.vectors import compute_pca_ot_transforms, _compute_pca_ot_full

def test_pca_ot_basic():
    """Test basic PCA-OT computation on toy data."""
    print("Testing PCA-OT implementation...")

    # Create toy data
    n_benign = 50
    n_harmful = 50
    n_layers = 4
    hidden_dim = 128
    n_components = 16

    # Generate random activations
    torch.manual_seed(42)
    benign_states = torch.randn(n_benign, n_layers, hidden_dim)
    harmful_states = torch.randn(n_harmful, n_layers, hidden_dim) + 0.5  # Shift harmful

    print(f"Benign states shape: {benign_states.shape}")
    print(f"Harmful states shape: {harmful_states.shape}")

    # Compute full PCA-OT transforms
    print(f"\nComputing PCA-OT with {n_components} components...")
    A_full, b_full, P, A_k, b_k = _compute_pca_ot_full(
        benign_states, harmful_states, n_components=n_components
    )

    print(f"A_full shape: {A_full.shape}")  # Should be (n_layers, hidden_dim, hidden_dim)
    print(f"b_full shape: {b_full.shape}")  # Should be (n_layers, hidden_dim)
    print(f"P shape: {P.shape}")  # Should be (n_layers, n_components, hidden_dim)
    print(f"A_k shape: {A_k.shape}")  # Should be (n_layers, n_components, n_components)
    print(f"b_k shape: {b_k.shape}")  # Should be (n_layers, n_components)

    # Verify shapes
    assert A_full.shape == (n_layers, hidden_dim, hidden_dim), f"Wrong A_full shape: {A_full.shape}"
    assert b_full.shape == (n_layers, hidden_dim), f"Wrong b_full shape: {b_full.shape}"
    assert P.shape == (n_layers, n_components, hidden_dim), f"Wrong P shape: {P.shape}"
    assert A_k.shape == (n_layers, n_components, n_components), f"Wrong A_k shape: {A_k.shape}"
    assert b_k.shape == (n_layers, n_components), f"Wrong b_k shape: {b_k.shape}"

    print("\n✓ Shape checks passed!")

    # Test transformation application
    print("\nTesting transformation application...")
    layer_idx = 0
    test_activation = benign_states[0, layer_idx, :]  # (hidden_dim,)

    # Apply affine transform: h' = A_full @ h + b_full
    transformed = A_full[layer_idx] @ test_activation + b_full[layer_idx]

    print(f"Original activation norm: {test_activation.norm().item():.4f}")
    print(f"Transformed activation norm: {transformed.norm().item():.4f}")

    # Verify transform is not identity
    diff = (transformed - test_activation).norm()
    print(f"Difference from identity: {diff.item():.4f}")
    assert diff > 0.01, "Transform appears to be identity!"

    print("\n✓ Transformation application works!")

    # Test public API
    print("\nTesting public API...")
    transforms = compute_pca_ot_transforms(
        benign_states, harmful_states, n_components=n_components
    )

    assert len(transforms) == n_layers, f"Wrong number of transforms: {len(transforms)}"

    for i, t in enumerate(transforms):
        assert t.layer_idx == i, f"Wrong layer index: {t.layer_idx}"
        assert t.A_full.shape == (hidden_dim, hidden_dim)
        assert t.b_full.shape == (hidden_dim,)
        assert t.P.shape == (n_components, hidden_dim)
        assert t.A_k.shape == (n_components, n_components)
        assert t.b_k.shape == (n_components,)

    print(f"✓ Created {len(transforms)} PCAOTTransform objects")

    # Test that A_full = P^T A_k P (approximately)
    print("\nVerifying A_full = P^T A_k P...")
    layer_idx = 0
    t = transforms[layer_idx]
    A_full_reconstructed = t.P.T @ t.A_k @ t.P
    reconstruction_error = (t.A_full - A_full_reconstructed).norm() / t.A_full.norm()
    print(f"Relative reconstruction error: {reconstruction_error.item():.6f}")
    assert reconstruction_error < 0.01, f"Reconstruction error too high: {reconstruction_error}"

    print("\n✓ All tests passed!")
    print("\n" + "="*60)
    print("PCA-OT implementation is working correctly!")
    print("="*60)

if __name__ == "__main__":
    test_pca_ot_basic()
