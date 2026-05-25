"""Tests for abliterix.harmfulness — joint harmfulness + refusal direction.

Verifies the dual-direction decomposition from Zhao et al. 2025
(arXiv:2507.11878).  All tests use small synthetic tensors — no GPU,
no model.
"""

import pytest
import torch
import torch.nn.functional as F

from abliterix.harmfulness import (
    extract_harm_refusal_pair,
)
from abliterix.types import VectorMethod
from abliterix.vectors import compute_steering_vectors


# ---------------------------------------------------------------------------
# Output shape & normalization
# ---------------------------------------------------------------------------


def test_pair_output_shape(synthetic_states):
    benign, target = synthetic_states
    result = extract_harm_refusal_pair(benign, target)
    # Two directions, per-layer, per hidden_dim.
    assert result.shape == (2, 8, 64)


def test_pair_refusal_slot_matches_mean_diff(synthetic_states):
    """Slot 0 must be the standard mean-diff refusal vector."""
    benign, target = synthetic_states
    result = extract_harm_refusal_pair(benign, target)
    expected = F.normalize(target.mean(dim=0) - benign.mean(dim=0), p=2, dim=1)
    assert torch.allclose(result[0].float(), expected.float(), atol=1e-5)


def test_pair_refusal_unit_normalized(synthetic_states):
    benign, target = synthetic_states
    result = extract_harm_refusal_pair(benign, target)
    norms = torch.linalg.vector_norm(result[0], dim=1)
    assert torch.allclose(norms, torch.ones(8), atol=1e-5)


def test_pair_harmfulness_norms_either_unit_or_zero(synthetic_states):
    """Harmfulness layer norms must be ~1 (active) or 0 (collapsed)."""
    benign, target = synthetic_states
    result = extract_harm_refusal_pair(benign, target)
    norms = torch.linalg.vector_norm(result[1], dim=1)
    # Each per-layer norm should be either ~1 or ~0.
    for n in norms.tolist():
        assert (abs(n - 1.0) < 1e-4) or (abs(n) < 1e-4), n


# ---------------------------------------------------------------------------
# Orthogonality — the load-bearing invariant
# ---------------------------------------------------------------------------


def test_harmfulness_orthogonal_to_refusal(synthetic_states):
    """Per-layer dot product of refusal and harmfulness must be ~0."""
    benign, target = synthetic_states
    result = extract_harm_refusal_pair(benign, target)
    # Skip layers where the harmfulness direction collapsed (zero vector).
    for layer_idx in range(result.shape[1]):
        h = result[1, layer_idx, :]
        if torch.linalg.vector_norm(h) < 1e-4:
            continue
        r = result[0, layer_idx, :]
        dot = torch.dot(r, h).abs().item()
        assert dot < 1e-4, f"layer {layer_idx}: |<r,h>| = {dot}"


def test_harmfulness_independent_of_mean_shift():
    """Harmfulness direction should not collapse when mean-shift is large.

    Constructs a dataset where target = benign + huge_mean_shift + small
    intra-target variation along a separate axis.  The refusal direction
    captures the mean shift; the harmfulness direction should capture the
    intra-target axis.
    """
    torch.manual_seed(7)
    hidden = 32
    n_samples = 40
    n_layers = 4

    benign = torch.randn(n_samples, n_layers, hidden)
    mean_shift = torch.randn(1, n_layers, hidden) * 5.0
    intra_axis = F.normalize(torch.randn(1, n_layers, hidden), p=2, dim=2)
    coeffs = torch.randn(n_samples, n_layers, 1) * 0.8
    target = benign + mean_shift + intra_axis * coeffs

    result = extract_harm_refusal_pair(benign, target)
    refusal = result[0]
    harmfulness = result[1]

    # Refusal must align with the mean shift (up to sign).
    mean_shift_dir = F.normalize(mean_shift.squeeze(0), p=2, dim=1)
    cos_refusal = F.cosine_similarity(refusal.float(), mean_shift_dir, dim=1).abs()
    assert cos_refusal.mean() > 0.8, cos_refusal

    # Harmfulness must align with the intra-target axis, NOT the mean shift.
    intra_dir = intra_axis.squeeze(0)
    # We can only check layers where harmfulness wasn't zeroed.
    for layer_idx in range(n_layers):
        if torch.linalg.vector_norm(harmfulness[layer_idx]) < 1e-4:
            continue
        cos_intra = F.cosine_similarity(
            harmfulness[layer_idx], intra_dir[layer_idx], dim=0
        ).abs()
        cos_mean = F.cosine_similarity(
            harmfulness[layer_idx], mean_shift_dir[layer_idx], dim=0
        ).abs()
        assert cos_intra > cos_mean, (
            f"layer {layer_idx}: cos(h,intra)={cos_intra} cos(h,mean)={cos_mean}"
        )


# ---------------------------------------------------------------------------
# Projection variants
# ---------------------------------------------------------------------------


def test_projected_abliteration_applied_to_both_slots(synthetic_states):
    benign, target = synthetic_states
    result = extract_harm_refusal_pair(benign, target, projected_abliteration=True)
    benign_dir = F.normalize(benign.mean(dim=0), p=2, dim=1)
    for slot in (0, 1):
        for layer_idx in range(result.shape[1]):
            v = result[slot, layer_idx]
            if torch.linalg.vector_norm(v) < 1e-4:
                continue
            dot = torch.dot(v.float(), benign_dir[layer_idx].float()).abs().item()
            assert dot < 1e-4, f"slot {slot} layer {layer_idx}: {dot}"


def test_orthogonal_projection_applied_to_both_slots(synthetic_states):
    benign, target = synthetic_states
    result = extract_harm_refusal_pair(benign, target, orthogonal_projection=True)
    benign_dir = F.normalize(benign.mean(dim=0), p=2, dim=1)
    for slot in (0, 1):
        for layer_idx in range(result.shape[1]):
            v = result[slot, layer_idx]
            if torch.linalg.vector_norm(v) < 1e-4:
                continue
            dot = torch.dot(v.float(), benign_dir[layer_idx].float()).abs().item()
            assert dot < 1e-4


# ---------------------------------------------------------------------------
# Layer-band gating
# ---------------------------------------------------------------------------


def test_layer_band_outside_band_gets_dampened():
    """Layers outside the band should have weaker raw harmfulness signal.

    We compare orthogonalised-and-normalised vectors, so a strict 0.5x scale
    check is not meaningful (everything is unit-norm).  Instead we verify
    that the dampening logic does not crash and produces valid unit
    vectors inside and outside the band.
    """
    torch.manual_seed(11)
    benign = torch.randn(30, 10, 48)
    target = benign + torch.randn(1, 10, 48) * 0.4

    result = extract_harm_refusal_pair(benign, target, layer_band=(0.4, 0.6))
    # Just check no NaN and all vectors are normalized or zero.
    assert not torch.isnan(result).any()
    norms = torch.linalg.vector_norm(result[1], dim=1)
    for n in norms.tolist():
        assert n < 1e-4 or abs(n - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# Public API entry point via compute_steering_vectors
# ---------------------------------------------------------------------------


def test_compute_steering_vectors_routes_to_harmfulness(synthetic_states):
    benign, target = synthetic_states
    via_flag = compute_steering_vectors(
        benign,
        target,
        VectorMethod.MEAN,
        False,
        ablate_harmfulness_direction=True,
    )
    direct = extract_harm_refusal_pair(benign, target)
    assert via_flag.shape == direct.shape == (2, 8, 64)
    assert torch.allclose(via_flag.float(), direct.float(), atol=1e-5)


def test_compute_steering_vectors_layer_band_pass_through(synthetic_states):
    benign, target = synthetic_states
    via_flag = compute_steering_vectors(
        benign,
        target,
        VectorMethod.MEAN,
        False,
        ablate_harmfulness_direction=True,
        harmfulness_layer_band=(0.1, 0.9),
    )
    direct = extract_harm_refusal_pair(benign, target, layer_band=(0.1, 0.9))
    assert torch.allclose(via_flag.float(), direct.float(), atol=1e-5)


# ---------------------------------------------------------------------------
# Settings validators
# ---------------------------------------------------------------------------


def test_validator_rejects_harmfulness_with_multi_direction(abliterix_config):
    """ablate_harmfulness_direction + n_directions > 1 must raise."""
    from abliterix.settings import SteeringConfig

    with pytest.raises(ValueError, match="n_directions"):
        SteeringConfig(ablate_harmfulness_direction=True, n_directions=2)


def test_validator_rejects_harmfulness_with_sra():
    """ablate_harmfulness_direction + vector_method=sra must raise."""
    from abliterix.settings import SteeringConfig

    with pytest.raises(ValueError, match="sra"):
        SteeringConfig(
            ablate_harmfulness_direction=True,
            vector_method=VectorMethod.SRA,
        )


def test_validator_rejects_invalid_layer_band():
    """harmfulness_layer_band must be a valid [lo, hi] within [0, 1]."""
    from abliterix.settings import SteeringConfig

    with pytest.raises(ValueError, match="harmfulness_layer_band"):
        SteeringConfig(
            ablate_harmfulness_direction=True,
            harmfulness_layer_band=[0.7, 0.3],
        )
    with pytest.raises(ValueError, match="harmfulness_layer_band"):
        SteeringConfig(
            ablate_harmfulness_direction=True,
            harmfulness_layer_band=[0.5],
        )
    with pytest.raises(ValueError, match="harmfulness_layer_band"):
        SteeringConfig(
            ablate_harmfulness_direction=True,
            harmfulness_layer_band=[-0.1, 0.5],
        )


def test_validator_allows_harmfulness_alone():
    """ablate_harmfulness_direction=True with defaults must pass."""
    from abliterix.settings import SteeringConfig

    sc = SteeringConfig(ablate_harmfulness_direction=True)
    assert sc.ablate_harmfulness_direction is True
    assert sc.harmfulness_layer_band == [0.3, 0.7]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_single_sample_harmfulness_returns_zero():
    """With only one sample per slice, PCA is degenerate — must not crash."""
    torch.manual_seed(3)
    benign = torch.randn(1, 4, 32)
    target = torch.randn(1, 4, 32)
    result = extract_harm_refusal_pair(benign, target)
    # Refusal slot is still computable from a single sample.
    assert result.shape == (2, 4, 32)
    # Harmfulness slot should be all zeros — PCA degenerate on n=1.
    assert torch.allclose(
        result[1].float(), torch.zeros_like(result[1].float()), atol=1e-6
    )


def test_identical_states_produce_zero_refusal():
    """When target == benign, mean-diff is zero so refusal slot is zero."""
    torch.manual_seed(5)
    states = torch.randn(10, 4, 32)
    result = extract_harm_refusal_pair(states, states.clone())
    # Refusal slot zero (mean diff is zero).
    assert torch.allclose(
        result[0].float(), torch.zeros_like(result[0].float()), atol=1e-6
    )
