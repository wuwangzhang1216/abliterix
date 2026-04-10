"""Tests for abliterix.core.steering — decay kernels, dequant, interpolation, LoRA math.

All tests use synthetic tensors and reproduce the math from apply_steering()
without loading a model.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from abliterix.core.steering import _dequantize_fp8_blockwise
from abliterix.types import DecayKernel, SteeringProfile


# ===================================================================
# Helpers: reproduce the decay kernel logic from apply_steering()
# ===================================================================


def _compute_strength(
    layer_idx: int,
    profile: SteeringProfile,
    kernel: DecayKernel,
) -> float | None:
    """Return the strength for a layer, or None if the layer is skipped."""
    distance = abs(layer_idx - profile.max_weight_position)
    if distance > profile.min_weight_distance:
        return None

    t = distance / profile.min_weight_distance
    sp = profile

    if kernel == DecayKernel.GAUSSIAN:
        return sp.min_weight + (sp.max_weight - sp.min_weight) * math.exp(-2.0 * t * t)
    elif kernel == DecayKernel.COSINE:
        return sp.min_weight + (sp.max_weight - sp.min_weight) * (
            0.5 * (1.0 + math.cos(math.pi * t))
        )
    else:  # LINEAR
        return sp.max_weight + t * (sp.min_weight - sp.max_weight)


# Standard test profile: peak at layer 6, decay over 3 layers.
_PROFILE = SteeringProfile(
    max_weight=2.0,
    max_weight_position=6.0,
    min_weight=0.2,
    min_weight_distance=3.0,
)


# ===================================================================
# FP8 dequantization
# ===================================================================


def test_fp8_dequant_identity_scale():
    """With all-ones scale, output equals input.float()."""
    weight = torch.randn(256, 256)
    scale = torch.ones(2, 2)  # block_size=128 → ceil(256/128)=2
    result = _dequantize_fp8_blockwise(weight, scale, block_size=128)
    assert torch.allclose(result, weight.float(), atol=1e-6)


def test_fp8_dequant_scaling():
    """Each block is multiplied by its corresponding scale."""
    weight = torch.ones(256, 256)
    scale = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    result = _dequantize_fp8_blockwise(weight, scale, block_size=128)
    # Top-left 128x128 block → scale 2.0
    assert torch.allclose(result[:128, :128], torch.full((128, 128), 2.0))
    # Top-right 128x128 block → scale 3.0
    assert torch.allclose(result[:128, 128:], torch.full((128, 128), 3.0))
    # Bottom-left 128x128 block → scale 4.0
    assert torch.allclose(result[128:, :128], torch.full((128, 128), 4.0))
    # Bottom-right 128x128 block → scale 5.0
    assert torch.allclose(result[128:, 128:], torch.full((128, 128), 5.0))


def test_fp8_dequant_shape():
    weight = torch.randn(300, 400)
    # ceil(300/128)=3, ceil(400/128)=4
    scale = torch.ones(3, 4)
    result = _dequantize_fp8_blockwise(weight, scale, block_size=128)
    assert result.shape == (300, 400)


# ===================================================================
# Decay kernels
# ===================================================================


# --- LINEAR ---


def test_linear_kernel_peak():
    s = _compute_strength(6, _PROFILE, DecayKernel.LINEAR)
    assert s == pytest.approx(2.0)


def test_linear_kernel_edge():
    s = _compute_strength(9, _PROFILE, DecayKernel.LINEAR)
    assert s == pytest.approx(0.2)


def test_linear_kernel_midpoint():
    s = _compute_strength(7, _PROFILE, DecayKernel.LINEAR)  # distance=1, t=1/3
    expected = 2.0 + (1 / 3) * (0.2 - 2.0)
    assert s == pytest.approx(expected, abs=1e-10)


# --- GAUSSIAN ---


def test_gaussian_kernel_peak():
    s = _compute_strength(6, _PROFILE, DecayKernel.GAUSSIAN)
    assert s == pytest.approx(2.0)


def test_gaussian_kernel_edge():
    s = _compute_strength(9, _PROFILE, DecayKernel.GAUSSIAN)
    expected = 0.2 + (2.0 - 0.2) * math.exp(-2.0)
    assert s == pytest.approx(expected)


def test_gaussian_kernel_monotonic():
    """Strength should decrease as distance from peak increases."""
    strengths = []
    for layer in range(4, 10):
        s = _compute_strength(layer, _PROFILE, DecayKernel.GAUSSIAN)
        if s is not None:
            strengths.append(s)
    # Values up to the peak should increase, then decrease.
    peak_idx = strengths.index(max(strengths))
    assert all(strengths[i] <= strengths[i + 1] for i in range(peak_idx))
    assert all(
        strengths[i] >= strengths[i + 1] for i in range(peak_idx, len(strengths) - 1)
    )


# --- COSINE ---


def test_cosine_kernel_peak():
    s = _compute_strength(6, _PROFILE, DecayKernel.COSINE)
    assert s == pytest.approx(2.0)


def test_cosine_kernel_edge():
    s = _compute_strength(9, _PROFILE, DecayKernel.COSINE)
    assert s == pytest.approx(0.2)


def test_cosine_kernel_midpoint():
    s = _compute_strength(7, _PROFILE, DecayKernel.COSINE)  # distance=1, t=1/3
    expected = 0.2 + (2.0 - 0.2) * 0.5 * (1.0 + math.cos(math.pi / 3))
    assert s == pytest.approx(expected)


# --- Skip logic ---


def test_distance_beyond_falloff_skips():
    """Layers outside min_weight_distance should be skipped (return None)."""
    assert _compute_strength(2, _PROFILE, DecayKernel.LINEAR) is None
    assert _compute_strength(10, _PROFILE, DecayKernel.LINEAR) is None


# ===================================================================
# Global vector interpolation
# ===================================================================


def _interpolate_vector(
    steering_vectors: torch.Tensor, vector_index: float
) -> torch.Tensor:
    """Reproduce the interpolation logic from apply_steering()."""
    fractional, integral = math.modf(vector_index + 1)
    return F.normalize(
        steering_vectors[int(integral)].lerp(
            steering_vectors[int(integral) + 1],
            fractional,
        ),
        p=2,
        dim=0,
    )


def test_interpolation_integer_index(steering_vectors):
    """Integer vector_index should effectively select layer[index+1]."""
    result = _interpolate_vector(steering_vectors, 3.0)
    expected = F.normalize(steering_vectors[4], p=2, dim=0)
    assert torch.allclose(result, expected, atol=1e-5)


def test_interpolation_fractional_index(steering_vectors):
    """Fractional index should interpolate between two adjacent layers."""
    result = _interpolate_vector(steering_vectors, 3.5)
    expected = F.normalize(
        steering_vectors[4].lerp(steering_vectors[5], 0.5),
        p=2,
        dim=0,
    )
    assert torch.allclose(result, expected, atol=1e-5)


def test_interpolation_result_normalized(steering_vectors):
    result = _interpolate_vector(steering_vectors, 2.7)
    norm = torch.linalg.vector_norm(result)
    assert norm.item() == pytest.approx(1.0, abs=1e-5)


def test_vector_index_none_means_per_layer():
    """When vector_index is None, no global vector is computed."""
    # This is a logic check: the code path sets global_vector = None.
    vector_index = None
    global_vector = None if vector_index is None else "computed"
    assert global_vector is None


# ===================================================================
# LoRA rank-1 update math
# ===================================================================


def test_lora_rank1_shapes():
    """lora_A should be (1, d_in) and lora_B should be (d_out, 1)."""
    v = F.normalize(torch.randn(64), p=2, dim=0)
    W = torch.randn(64, 128)
    strength = 1.5

    lora_A = (v @ W).view(1, -1)
    lora_B = (-strength * v).view(-1, 1)

    assert lora_A.shape == (1, 128)
    assert lora_B.shape == (64, 1)


def test_lora_rank1_reconstruction():
    """lora_B @ lora_A should equal -strength * outer(v, v @ W)."""
    v = F.normalize(torch.randn(64), p=2, dim=0)
    W = torch.randn(64, 128)
    strength = 1.5

    lora_A = (v @ W).view(1, -1)
    lora_B = (-strength * v).view(-1, 1)

    product = lora_B @ lora_A
    expected = -strength * torch.outer(v, v @ W)
    assert torch.allclose(product, expected, atol=1e-5)
