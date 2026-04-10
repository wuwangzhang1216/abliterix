# Abliterix — a derivative work of Heretic (https://github.com/p-e-w/heretic)
# Original work Copyright (C) 2025  Philipp Emanuel Weidmann (p-e-w)
# Modified work Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Steering algorithm: modify model weights via LoRA rank-1 updates.

This module implements the core steering (abliteration) procedure as a
standalone function rather than a method on the engine, keeping the algorithm
cleanly separated from model-management concerns.
"""

import math
from typing import cast

import bitsandbytes as bnb
import torch
import torch.linalg as LA
import torch.nn.functional as F
from peft.tuners.lora.layer import Linear
from torch import Tensor

from ..settings import AbliterixConfig
from ..types import DecayKernel, ExpertRoutingConfig, SteeringMode, SteeringProfile, WeightNorm

# Avoid circular import: accept the engine as a duck-typed object rather
# than importing SteeringEngine directly.  The caller is responsible for
# passing a valid engine instance.

_FP8_DTYPES = frozenset()
with __import__("contextlib").suppress(AttributeError):
    _FP8_DTYPES = frozenset({torch.float8_e4m3fn, torch.float8_e5m2})


def _dequantize_fp8_blockwise(
    weight: Tensor,
    weight_scale: Tensor,
) -> Tensor:
    """Block-wise FP8 dequantization: W_real = weight_fp8 * scale_per_block."""
    out_f, in_f = weight.shape
    w = weight.to(torch.float32)
    # Infer block sizes from the ratio of weight/scale dimensions
    # (handles non-square blocks and arbitrary weight shapes).
    block_r = max(1, out_f // weight_scale.shape[0])
    block_c = max(1, in_f // weight_scale.shape[1])
    scale = weight_scale.float().repeat_interleave(block_r, dim=0).repeat_interleave(block_c, dim=1)
    return w * scale[:out_f, :in_f]


def _detect_discriminative_layers(
    steering_vectors: Tensor,
    benign_states: Tensor | None,
    target_states: Tensor | None,
) -> set[int]:
    """Identify layers where harmful/harmless activations project in opposite directions.

    A layer is *discriminative* if the mean projection of harmful activations
    onto the steering vector is positive while the mean projection of harmless
    activations is negative (or vice versa).  Only these layers benefit from
    steering; non-discriminative layers are skipped to avoid coherence damage.

    Based on: Selective Steering (2026) — 5.5× improvement with zero perplexity violations.

    Returns a set of discriminative layer indices (0-based transformer layer indices).
    """
    if benign_states is None or target_states is None:
        # Fall back to all layers if residuals are unavailable.
        return set(range(steering_vectors.shape[0] - 1))

    discriminative: set[int] = set()
    n_layers = min(steering_vectors.shape[0] - 1, benign_states.shape[1] - 1)

    for layer_idx in range(n_layers):
        v = steering_vectors[layer_idx + 1]  # +1 because index 0 is embedding
        b = benign_states[:, layer_idx + 1, :].float()
        t = target_states[:, layer_idx + 1, :].float()

        # Mean scalar projection onto steering direction.
        mu_benign = (b @ v.float()).mean().item()
        mu_target = (t @ v.float()).mean().item()

        # Discriminative = opposite signs.
        if mu_benign * mu_target < 0:
            discriminative.add(layer_idx)

    return discriminative


def _make_angular_hook(
    direction: Tensor,
    angle_degrees: float,
    adaptive: bool = False,
):
    """Create a forward hook that rotates activations within the steering plane.

    Implements Angular Steering (NeurIPS 2025 Spotlight):
        h_steered = h - proj_P(h) + |proj_P(h)| * [b1 b2] R_θ [1 0]^T

    Parameters
    ----------
    direction : Tensor
        Unit-normalised steering direction (hidden_dim,).
    angle_degrees : float
        Rotation angle.  ~200° = compliance, ~20° = refusal.
    adaptive : bool
        If True, only rotate activations positively aligned with the
        direction (Adaptive Angular Steering), reducing interference.
    """
    theta = math.radians(angle_degrees)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    def hook(module, input, output):
        h = output
        if isinstance(h, tuple):
            h = h[0]

        d = direction.to(h.device, dtype=h.dtype)

        # b1 = d (first basis vector of the 2D steering plane).
        # Scalar projection of h onto d.
        proj_scalar = (h @ d).unsqueeze(-1)  # (..., seq, 1)
        proj_on_d = proj_scalar * d           # component along b1

        # b2 = Gram-Schmidt orthogonal complement within the plane.
        residual = h - proj_on_d
        residual_norm = residual.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        b2 = residual / residual_norm

        # The 2D projection has components (proj_scalar, residual_norm).
        # Its magnitude is preserved by rotation.
        # Rotate: new_b1_coeff = cos(θ)*proj_scalar + sin(θ)*residual_norm
        #         new_b2_coeff = -sin(θ)*proj_scalar + cos(θ)*residual_norm
        new_proj_on_d = (cos_t * proj_scalar + sin_t * residual_norm) * d
        new_residual = (-sin_t * proj_scalar + cos_t * residual_norm) * b2

        # Components outside the 2D plane are preserved.
        # h = proj_on_d + residual + h_perp  →  h_perp = h - proj_on_d - residual
        # But residual = residual_norm * b2, so h_perp is everything else.
        # Since we only computed b2 from residual, there's nothing outside;
        # the full h is reconstructed as new_proj_on_d + new_residual.
        h_new = new_proj_on_d + new_residual

        if adaptive:
            mask = (proj_scalar > 0).to(h_new.dtype)
            h_new = mask * h_new + (1 - mask) * h

        if isinstance(output, tuple):
            return (h_new,) + output[1:]
        return h_new

    return hook


def apply_steering(
    engine,  # SteeringEngine
    steering_vectors: Tensor,
    vector_index: float | None,
    profiles: dict[str, SteeringProfile],
    config: AbliterixConfig | None = None,
    safety_experts: dict[int, list[tuple[int, float]]] | None = None,
    routing_config: ExpertRoutingConfig | None = None,
    benign_states: Tensor | None = None,
    target_states: Tensor | None = None,
):
    """Apply rank-1 LoRA steering to every steerable module in the model.

    Parameters
    ----------
    engine : SteeringEngine
        The loaded model wrapper (provides ``transformer_layers``,
        ``steerable_modules``, adapter access, and helper methods).
    steering_vectors : Tensor
        Per-layer vectors of shape ``(layers+1, hidden_dim)``.
    vector_index : float or None
        If not None, interpolate a global vector from two adjacent layers.
        If None, use per-layer vectors.
    profiles : dict
        Component-name → :class:`SteeringProfile` mapping.
    config : AbliterixConfig
        Top-level configuration (kernel choice, normalisation, etc.).
    safety_experts : dict, optional
        MoE profiling results used for expert-level steering.
    routing_config : ExpertRoutingConfig, optional
        Hyper-parameters for MoE expert suppression.
    benign_states : Tensor, optional
        Residual states from benign prompts, used for discriminative layer
        selection.  Shape ``(n, layers+1, hidden_dim)``.
    target_states : Tensor, optional
        Residual states from target prompts.  Shape ``(n, layers+1, hidden_dim)``.
    """
    if config is None:
        config = engine.config

    steering_mode = config.steering.steering_mode

    # --- Discriminative layer selection -----------------------------------
    discriminative_layers: set[int] | None = None
    if config.steering.discriminative_layer_selection:
        discriminative_layers = _detect_discriminative_layers(
            steering_vectors, benign_states, target_states,
        )

    # --- Resolve the global steering vector (if applicable) ---------------
    if vector_index is None:
        global_vector = None
    else:
        fractional, integral = math.modf(vector_index + 1)
        global_vector = F.normalize(
            steering_vectors[int(integral)].lerp(
                steering_vectors[int(integral) + 1],
                fractional,
            ),
            p=2,
            dim=0,
        )

    # --- Direct weight editing (orthogonal projection, no LoRA) -----------
    if steering_mode == SteeringMode.DIRECT:
        _apply_direct_steering(
            engine,
            steering_vectors,
            global_vector,
            profiles,
            config,
            discriminative_layers,
        )
        # Expert-Granular Abliteration: project refusal direction from ALL
        # expert down_proj slices, not just top-N safety experts.  This is
        # critical for MoE models where refusal signal is distributed across
        # all experts (TrevorS EGA method: 3/100 vs 29/100 without).
        if engine.has_expert_routing():
            _apply_ega_steering(
                engine, steering_vectors, global_vector, profiles, config,
                discriminative_layers,
            )
        # Legacy top-N router suppression (complementary to EGA).
        if safety_experts and routing_config:
            _apply_moe_steering(engine, steering_vectors, global_vector, safety_experts, routing_config)
        return

    # --- Angular / Adaptive Angular steering (hook-based) -----------------
    if steering_mode in (SteeringMode.ANGULAR, SteeringMode.ADAPTIVE_ANGULAR):
        _apply_angular_steering(
            engine,
            steering_vectors,
            global_vector,
            profiles,
            config,
            discriminative_layers,
            adaptive=(steering_mode == SteeringMode.ADAPTIVE_ANGULAR),
        )
        # MoE expert steering still uses weight modification.
        if safety_experts and routing_config:
            _apply_moe_steering(engine, steering_vectors, global_vector, safety_experts, routing_config)
        return

    # --- Spherical steering (geodesic rotation on hypersphere) ------------
    if steering_mode == SteeringMode.SPHERICAL:
        _apply_spherical_steering(
            engine,
            steering_vectors,
            global_vector,
            profiles,
            config,
            discriminative_layers,
        )
        if safety_experts and routing_config:
            _apply_moe_steering(engine, steering_vectors, global_vector, safety_experts, routing_config)
        return

    # --- Steering Vector Fields (learned context-dependent directions) ----
    if steering_mode == SteeringMode.VECTOR_FIELD:
        concept_scorers = getattr(engine, "_concept_scorers", None)
        _apply_svf_steering(
            engine,
            steering_vectors,
            global_vector,
            profiles,
            config,
            discriminative_layers,
            concept_scorers=concept_scorers,
        )
        if safety_experts and routing_config:
            _apply_moe_steering(engine, steering_vectors, global_vector, safety_experts, routing_config)
        return

    # --- Pre-cache steering vectors per device ----------------------------
    devices: set[torch.device] = set()
    for idx in range(len(engine.transformer_layers)):
        for mods in engine.steerable_modules(idx).values():
            for mod in mods:
                devices.add(mod.weight.device)

    sv_by_device = {d: steering_vectors.to(d) for d in devices}
    gv_by_device = (
        {d: global_vector.to(d) for d in devices} if global_vector is not None else None
    )

    # --- Per-layer, per-component steering --------------------------------
    kernel = config.steering.decay_kernel

    for layer_idx in range(len(engine.transformer_layers)):
        # Skip non-discriminative layers when the feature is enabled.
        if discriminative_layers is not None and layer_idx not in discriminative_layers:
            continue

        for component, modules in engine.steerable_modules(layer_idx).items():
            sp = profiles[component]

            distance = cast(float, abs(layer_idx - sp.max_weight_position))
            if distance > sp.min_weight_distance:
                continue

            # Compute interpolated weight using the configured decay kernel.
            t = distance / sp.min_weight_distance  # normalised ∈ [0, 1]
            if kernel == DecayKernel.GAUSSIAN:
                strength = sp.min_weight + (sp.max_weight - sp.min_weight) * math.exp(
                    -2.0 * t * t
                )
            elif kernel == DecayKernel.COSINE:
                strength = sp.min_weight + (sp.max_weight - sp.min_weight) * (
                    0.5 * (1.0 + math.cos(math.pi * t))
                )
            else:  # LINEAR
                strength = sp.max_weight + t * (sp.min_weight - sp.max_weight)

            for mod in modules:
                # TODO: The module-interface assumption here is fragile — PEFT
                #       wraps modules differently per quantisation mode.
                mod = cast(Linear, mod)

                device = mod.weight.device
                if global_vector is None:
                    v = sv_by_device[device][layer_idx + 1]
                else:
                    v = gv_by_device[device]  # ty:ignore[non-subscriptable]

                # Obtain the full-precision weight matrix W.
                base_weight = cast(Tensor, mod.base_layer.weight)
                qs = getattr(base_weight, "quant_state", None)
                CB = getattr(base_weight, "CB", None)

                if qs is not None:
                    # 4-bit NF4: use cached dequantised weights when available
                    # to avoid repeated expensive dequantisation.
                    mid = id(mod)
                    if mid in engine._dequant_cache:
                        W = engine._dequant_cache[mid]
                    else:
                        W = cast(
                            Tensor,
                            bnb.functional.dequantize_4bit(  # ty:ignore[possibly-missing-attribute]
                                base_weight.data,
                                qs,
                            ).to(torch.float32),
                        )
                        engine._dequant_cache[mid] = W
                elif CB is not None:
                    # Int8 quantisation: dequantise from CB data and SCB row scales.
                    mid = id(mod)
                    if mid in engine._dequant_cache:
                        W = engine._dequant_cache[mid]
                    else:
                        SCB = base_weight.SCB  # ty:ignore[unresolved-attribute]
                        W = CB.float() * SCB.float().unsqueeze(1) / 127.0
                        engine._dequant_cache[mid] = W
                elif _FP8_DTYPES and base_weight.dtype in _FP8_DTYPES:
                    # FP8 block-quantized: dequantise using per-block weight_scale.
                    mid = id(mod)
                    if mid in engine._dequant_cache:
                        W = engine._dequant_cache[mid]
                    else:
                        weight_scale = getattr(mod.base_layer, "weight_scale", None)
                        if weight_scale is not None:
                            W = _dequantize_fp8_blockwise(
                                base_weight.data, weight_scale
                            )
                        else:
                            W = base_weight.to(torch.float32)
                        engine._dequant_cache[mid] = W
                else:
                    W = base_weight.to(torch.float32)

                W = W.view(W.shape[0], -1)

                # Optional row normalisation before computing the adapter.
                norm_mode = config.steering.weight_normalization
                if norm_mode != WeightNorm.NONE:
                    W_orig = W
                    W_row_norms = LA.vector_norm(W, dim=1, keepdim=True)
                    W = F.normalize(W, p=2, dim=1)

                # Rank-1 steering: project W onto the orthogonal complement of v.
                #   lora_A  =  vᵀ W    (shape 1 × d_in)
                #   lora_B  = -λ v      (shape d_out × 1)
                lora_A = (v @ W).view(1, -1)
                lora_B = (-strength * v).view(-1, 1)

                if norm_mode == WeightNorm.PRE:
                    lora_B = W_row_norms * lora_B
                elif norm_mode == WeightNorm.FULL:
                    # Low-rank SVD approximation that preserves original row
                    # magnitudes after the rank-1 update.
                    W = W + lora_B @ lora_A
                    W = F.normalize(W, p=2, dim=1)
                    W = W * W_row_norms
                    W = W - W_orig
                    r = engine.peft_config.r
                    U, S, Vh = torch.svd_lowrank(W, q=2 * r + 4, niter=6)
                    U = U[:, :r]
                    S = S[:r]
                    Vh = Vh[:, :r].T
                    sqrt_S = torch.sqrt(S)
                    lora_B = U @ torch.diag(sqrt_S)
                    lora_A = torch.diag(sqrt_S) @ Vh

                # Write the adapter weights (PEFT default adapter name).
                wA = cast(Tensor, mod.lora_A["default"].weight)
                wB = cast(Tensor, mod.lora_B["default"].weight)
                wA.data = lora_A.to(wA.dtype)
                wB.data = lora_B.to(wB.dtype)

    # --- MoE expert-level steering ----------------------------------------
    if safety_experts and routing_config:
        _apply_moe_steering(
            engine, steering_vectors, global_vector, safety_experts, routing_config,
            sv_by_device=sv_by_device, gv_by_device=gv_by_device,
        )


# ---------------------------------------------------------------------------
# Direct weight editing (orthogonal projection, bypasses LoRA)
# ---------------------------------------------------------------------------


def _apply_direct_steering(
    engine,
    steering_vectors: Tensor,
    global_vector: Tensor | None,
    profiles: dict[str, SteeringProfile],
    config: AbliterixConfig,
    discriminative_layers: set[int] | None,
):
    """Modify base weights in-place via norm-preserving orthogonal projection.

    Required for architectures like Gemma 4 where double-norm (4 RMSNorm per
    layer) and Per-Layer Embeddings (PLE) suppress LoRA perturbations.

    For each steerable module, projects out the refusal direction from the
    weight matrix while preserving original row norms:

        d = steering_vector (unit-normalised)
        W_new = W - strength * (W @ d) ⊗ d
        W_new = W_new * (||W_row|| / ||W_new_row||)   # norm preservation

    Weight originals are cached on the engine for restore_baseline().
    """
    kernel = config.steering.decay_kernel

    # Cache originals for restore_baseline.
    if not hasattr(engine, "_direct_weight_originals"):
        engine._direct_weight_originals = {}

    for layer_idx in range(len(engine.transformer_layers)):
        if discriminative_layers is not None and layer_idx not in discriminative_layers:
            continue

        for component, modules in engine.steerable_modules(layer_idx).items():
            sp = profiles[component]

            distance = cast(float, abs(layer_idx - sp.max_weight_position))
            if distance > sp.min_weight_distance:
                continue

            # Compute interpolated strength using the configured decay kernel.
            t = distance / sp.min_weight_distance
            if kernel == DecayKernel.GAUSSIAN:
                strength = sp.min_weight + (sp.max_weight - sp.min_weight) * math.exp(
                    -2.0 * t * t
                )
            elif kernel == DecayKernel.COSINE:
                strength = sp.min_weight + (sp.max_weight - sp.min_weight) * (
                    0.5 * (1.0 + math.cos(math.pi * t))
                )
            else:  # LINEAR
                strength = sp.max_weight + t * (sp.min_weight - sp.max_weight)

            for mod in modules:
                # Navigate to the base weight — through PEFT wrapper if present.
                base_mod = mod
                if hasattr(mod, "base_layer"):
                    base_mod = mod.base_layer

                weight = base_mod.weight

                # Cache the original weight for later restoration.
                # Key by the weight tensor itself for O(1) restore.
                if weight not in engine._direct_weight_originals:
                    engine._direct_weight_originals[weight] = weight.data.clone()

                device = weight.device
                if global_vector is None:
                    v = steering_vectors[layer_idx + 1].to(device)
                else:
                    v = global_vector.to(device)

                # Use float32 for projection math to preserve precision
                # (bf16 loses signal in 2816-dim inner products).
                W = weight.data.to(torch.float32)
                vf = v.to(torch.float32)

                # Orthogonal projection: remove the refusal direction from W.
                # W has shape (out_features, in_features).
                # v has shape (hidden_dim,) which may match either dimension.
                out_f, in_f = W.shape

                if vf.shape[0] == out_f:
                    proj = vf @ W
                    W_new = W - strength * vf.unsqueeze(1) * proj.unsqueeze(0)
                elif vf.shape[0] == in_f:
                    proj = W @ vf
                    W_new = W - strength * proj.unsqueeze(1) * vf.unsqueeze(0)
                else:
                    # Dimension mismatch — skip this module.
                    continue

                # Norm-preserving: restore original row magnitudes.
                # Critical for double-norm architectures (Gemma 4) where
                # row norm changes cascade through RMSNorm layers.
                if config.steering.weight_normalization != WeightNorm.NONE:
                    orig_norms = torch.linalg.vector_norm(W, dim=1, keepdim=True)
                    new_norms = torch.linalg.vector_norm(W_new, dim=1, keepdim=True).clamp(min=1e-8)
                    W_new = W_new * (orig_norms / new_norms)

                weight.data = W_new.to(weight.dtype)


# ---------------------------------------------------------------------------
# Expert-Granular Abliteration (EGA)
# ---------------------------------------------------------------------------


def _apply_ega_steering(
    engine,
    steering_vectors: Tensor,
    global_vector: Tensor | None,
    profiles: dict[str, SteeringProfile],
    config: AbliterixConfig,
    discriminative_layers: set[int] | None,
):
    """Project out the refusal direction from ALL expert down_proj slices.

    Unlike ``_apply_moe_steering`` which only targets top-N safety experts
    identified by router profiling, EGA applies norm-preserving orthogonal
    projection to every expert in every MoE layer.  This is necessary because
    refusal signal is distributed across all experts, not concentrated in a
    few (TrevorS EGA method: 3/100 refusals vs 29/100 without EGA on Gemma 4
    26B-A4B).

    The strength for each layer is derived from the ``mlp.down_proj`` profile
    (same component name used for both dense MLP and expert projections).
    """
    kernel = config.steering.decay_kernel
    norm_preserve = config.steering.weight_normalization != WeightNorm.NONE

    if not hasattr(engine, "_direct_weight_originals"):
        engine._direct_weight_originals = {}

    sp = profiles.get("mlp.down_proj")
    if sp is None:
        return

    for layer_idx in range(len(engine.transformer_layers)):
        if discriminative_layers is not None and layer_idx not in discriminative_layers:
            continue

        layer = engine.transformer_layers[layer_idx]
        fused = engine._locate_fused_weights(layer)
        if fused is None:
            continue

        # Compute layer-specific strength from the mlp.down_proj profile.
        distance = cast(float, abs(layer_idx - sp.max_weight_position))
        if distance > sp.min_weight_distance:
            continue

        t = distance / sp.min_weight_distance
        if kernel == DecayKernel.GAUSSIAN:
            strength = sp.min_weight + (sp.max_weight - sp.min_weight) * math.exp(
                -2.0 * t * t
            )
        elif kernel == DecayKernel.COSINE:
            strength = sp.min_weight + (sp.max_weight - sp.min_weight) * (
                0.5 * (1.0 + math.cos(math.pi * t))
            )
        else:
            strength = sp.max_weight + t * (sp.min_weight - sp.max_weight)

        # Pick the steering vector for this layer.
        device = fused.device
        if global_vector is None:
            v = steering_vectors[layer_idx + 1].to(device)
        else:
            v = global_vector.to(device)

        # Cache original for restore_baseline.
        if fused not in engine._direct_weight_originals:
            engine._direct_weight_originals[fused] = fused.data.clone()

        n_experts = fused.shape[0]
        vf = v.to(torch.float32)

        for eid in range(n_experts):
            W = fused.data[eid].to(torch.float32)
            out_f, in_f = W.shape

            if vf.shape[0] == out_f:
                proj = vf @ W
                W_new = W - strength * vf.unsqueeze(1) * proj.unsqueeze(0)
            elif vf.shape[0] == in_f:
                proj = W @ vf
                W_new = W - strength * proj.unsqueeze(1) * vf.unsqueeze(0)
            else:
                continue

            if norm_preserve:
                orig_norms = torch.linalg.vector_norm(W, dim=1, keepdim=True)
                new_norms = torch.linalg.vector_norm(W_new, dim=1, keepdim=True).clamp(min=1e-8)
                W_new = W_new * (orig_norms / new_norms)

            fused.data[eid] = W_new.to(fused.dtype)


# ---------------------------------------------------------------------------
# Angular / Adaptive Angular steering (hook-based)
# ---------------------------------------------------------------------------


def _apply_angular_steering(
    engine,
    steering_vectors: Tensor,
    global_vector: Tensor | None,
    profiles: dict[str, SteeringProfile],
    config: AbliterixConfig,
    discriminative_layers: set[int] | None,
    adaptive: bool = False,
):
    """Register forward hooks that rotate activations toward the compliance arc.

    Each hook implements the Angular Steering rotation in a 2D subspace
    spanned by the steering direction and the activation's component
    orthogonal to it.  The rotation angle is mapped from the steering
    strength computed by the decay kernel.
    """
    kernel = config.steering.decay_kernel

    # Remove any previously registered angular hooks.
    if not hasattr(engine, "_angular_hooks"):
        engine._angular_hooks = []

    for layer_idx in range(len(engine.transformer_layers)):
        if discriminative_layers is not None and layer_idx not in discriminative_layers:
            continue

        layer = engine.transformer_layers[layer_idx]

        # Compute effective strength from profiles (use first component).
        component = next(iter(profiles))
        sp = profiles[component]

        distance = cast(float, abs(layer_idx - sp.max_weight_position))
        if distance > sp.min_weight_distance:
            continue

        t = distance / sp.min_weight_distance
        if kernel == DecayKernel.GAUSSIAN:
            strength = sp.min_weight + (sp.max_weight - sp.min_weight) * math.exp(
                -2.0 * t * t
            )
        elif kernel == DecayKernel.COSINE:
            strength = sp.min_weight + (sp.max_weight - sp.min_weight) * (
                0.5 * (1.0 + math.cos(math.pi * t))
            )
        else:  # LINEAR
            strength = sp.max_weight + t * (sp.min_weight - sp.max_weight)

        # Map strength to rotation angle.  strength=1.0 → 180° (full inversion).
        angle = strength * 180.0

        if global_vector is None:
            v = steering_vectors[layer_idx + 1]
        else:
            v = global_vector

        hook = _make_angular_hook(v, angle, adaptive=adaptive)
        handle = layer.register_forward_hook(hook)
        engine._angular_hooks.append(handle)


# ---------------------------------------------------------------------------
# Spherical steering (geodesic rotation on the activation hypersphere)
# ---------------------------------------------------------------------------


def _make_spherical_hook(
    direction: Tensor,
    angle_degrees: float,
):
    """Create a forward hook that rotates activations along a geodesic.

    Implements Spherical Steering (arxiv:2602.08169):
    Instead of rotating in a 2D plane, this rotates along the great circle
    (geodesic) between the current activation direction and the target
    steering direction on the unit hypersphere, then restores the original
    activation magnitude.

    Parameters
    ----------
    direction : Tensor
        Unit-normalised steering direction (hidden_dim,).
    angle_degrees : float
        Rotation angle along the geodesic.
    """
    theta = math.radians(angle_degrees)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    def hook(module, input, output):
        h = output
        if isinstance(h, tuple):
            h = h[0]

        d = direction.to(h.device, dtype=h.dtype)

        # Preserve original magnitude.
        h_norm = h.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        h_hat = h / h_norm

        # Geodesic angle between h_hat and d.
        cos_alpha = (h_hat @ d).unsqueeze(-1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        sin_alpha = (1.0 - cos_alpha * cos_alpha).clamp(min=1e-14).sqrt()

        # Tangent vector at h_hat pointing toward d on the great circle.
        t = (d - cos_alpha * h_hat) / sin_alpha

        # Rotate h_hat by theta along the geodesic.
        h_hat_new = cos_t * h_hat + sin_t * t

        # Restore original magnitude.
        h_new = h_norm * h_hat_new

        if isinstance(output, tuple):
            return (h_new,) + output[1:]
        return h_new

    return hook


def _apply_spherical_steering(
    engine,
    steering_vectors: Tensor,
    global_vector: Tensor | None,
    profiles: dict[str, SteeringProfile],
    config: AbliterixConfig,
    discriminative_layers: set[int] | None,
):
    """Register forward hooks that rotate activations along geodesics.

    Follows the same decay-kernel pattern as angular steering but uses
    spherical (geodesic) rotation instead of 2D planar rotation.
    """
    kernel = config.steering.decay_kernel

    if not hasattr(engine, "_angular_hooks"):
        engine._angular_hooks = []

    for layer_idx in range(len(engine.transformer_layers)):
        if discriminative_layers is not None and layer_idx not in discriminative_layers:
            continue

        layer = engine.transformer_layers[layer_idx]

        component = next(iter(profiles))
        sp = profiles[component]

        distance = cast(float, abs(layer_idx - sp.max_weight_position))
        if distance > sp.min_weight_distance:
            continue

        t = distance / sp.min_weight_distance
        if kernel == DecayKernel.GAUSSIAN:
            strength = sp.min_weight + (sp.max_weight - sp.min_weight) * math.exp(
                -2.0 * t * t
            )
        elif kernel == DecayKernel.COSINE:
            strength = sp.min_weight + (sp.max_weight - sp.min_weight) * (
                0.5 * (1.0 + math.cos(math.pi * t))
            )
        else:  # LINEAR
            strength = sp.max_weight + t * (sp.min_weight - sp.max_weight)

        angle = strength * 180.0

        if global_vector is None:
            v = steering_vectors[layer_idx + 1]
        else:
            v = global_vector

        hook = _make_spherical_hook(v, angle)
        handle = layer.register_forward_hook(hook)
        engine._angular_hooks.append(handle)


# ---------------------------------------------------------------------------
# Steering Vector Fields (learned context-dependent directions)
# ---------------------------------------------------------------------------


def _make_svf_hook(
    scorer,  # ConceptScorer nn.Module
    direction_fallback: Tensor,
    angle_degrees: float,
):
    """Create a forward hook that steers using learned context-dependent directions.

    Implements Steering Vector Fields (arxiv:2602.01654):
    A trained concept scorer f(h) produces per-token steering directions via
    its gradient ∇_h f, making the intervention context-dependent.  Falls back
    to the static steering direction when the gradient is degenerate.

    Parameters
    ----------
    scorer : ConceptScorer
        Trained concept scoring MLP for this layer.
    direction_fallback : Tensor
        Static steering direction used when the gradient is degenerate.
    angle_degrees : float
        Rotation angle for the steering intervention.
    """
    theta = math.radians(angle_degrees)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    def hook(module, input, output):
        h = output
        if isinstance(h, tuple):
            h = h[0]

        d_fallback = direction_fallback.to(h.device, dtype=h.dtype)

        # Compute context-dependent direction via scorer gradient.
        with torch.enable_grad():
            h_detached = h.detach().requires_grad_(True)
            score = scorer(h_detached)
            grad = torch.autograd.grad(
                score.sum(), h_detached, create_graph=False,
            )[0]

        # Normalise gradient to get per-token steering direction.
        grad_norm = grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        d = grad / grad_norm

        # Fall back to static direction where gradient is degenerate.
        degenerate = (grad_norm.squeeze(-1) < 1e-6)
        if degenerate.any():
            d = torch.where(degenerate.unsqueeze(-1), d_fallback, d)

        # Apply angular rotation in the 2D plane of h and d.
        proj_scalar = (h * d).sum(dim=-1, keepdim=True)
        proj_on_d = proj_scalar * d
        residual = h - proj_on_d
        residual_norm = residual.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        b2 = residual / residual_norm

        new_proj_on_d = (cos_t * proj_scalar + sin_t * residual_norm) * d
        new_residual = (-sin_t * proj_scalar + cos_t * residual_norm) * b2
        h_new = new_proj_on_d + new_residual

        if isinstance(output, tuple):
            return (h_new,) + output[1:]
        return h_new

    return hook


def _apply_svf_steering(
    engine,
    steering_vectors: Tensor,
    global_vector: Tensor | None,
    profiles: dict[str, SteeringProfile],
    config: AbliterixConfig,
    discriminative_layers: set[int] | None,
    concept_scorers: dict | None = None,
):
    """Register forward hooks using learned Steering Vector Fields.

    Falls back to angular steering for layers without a trained concept scorer.
    """
    kernel = config.steering.decay_kernel

    if not hasattr(engine, "_angular_hooks"):
        engine._angular_hooks = []

    for layer_idx in range(len(engine.transformer_layers)):
        if discriminative_layers is not None and layer_idx not in discriminative_layers:
            continue

        layer = engine.transformer_layers[layer_idx]

        component = next(iter(profiles))
        sp = profiles[component]

        distance = cast(float, abs(layer_idx - sp.max_weight_position))
        if distance > sp.min_weight_distance:
            continue

        t = distance / sp.min_weight_distance
        if kernel == DecayKernel.GAUSSIAN:
            strength = sp.min_weight + (sp.max_weight - sp.min_weight) * math.exp(
                -2.0 * t * t
            )
        elif kernel == DecayKernel.COSINE:
            strength = sp.min_weight + (sp.max_weight - sp.min_weight) * (
                0.5 * (1.0 + math.cos(math.pi * t))
            )
        else:  # LINEAR
            strength = sp.max_weight + t * (sp.min_weight - sp.max_weight)

        angle = strength * 180.0

        if global_vector is None:
            v = steering_vectors[layer_idx + 1]
        else:
            v = global_vector

        if concept_scorers is not None and layer_idx in concept_scorers:
            scorer = concept_scorers[layer_idx].to(v.device)
            hook = _make_svf_hook(scorer, v, angle)
        else:
            # Fall back to angular steering for layers without a scorer.
            hook = _make_angular_hook(v, angle, adaptive=False)

        handle = layer.register_forward_hook(hook)
        engine._angular_hooks.append(handle)


# ---------------------------------------------------------------------------
# MoE expert-level steering
# ---------------------------------------------------------------------------


def _apply_moe_steering(
    engine,
    steering_vectors: Tensor,
    global_vector: Tensor | None,
    safety_experts: dict[int, list[tuple[int, float]]],
    routing_config: ExpertRoutingConfig,
    *,
    sv_by_device: dict | None = None,
    gv_by_device: dict | None = None,
):
    """Apply router-weight suppression and fused-expert abliteration."""
    n_suppress = routing_config.n_suppress
    bias_value = routing_config.router_bias
    expert_w = routing_config.expert_ablation_weight

    # Build device caches if not provided.
    if sv_by_device is None:
        devices: set[torch.device] = set()
        for idx in range(len(engine.transformer_layers)):
            for mods in engine.steerable_modules(idx).values():
                for mod in mods:
                    devices.add(mod.weight.device)
        sv_by_device = {d: steering_vectors.to(d) for d in devices}
        gv_by_device = (
            {d: global_vector.to(d) for d in devices} if global_vector is not None else None
        )

    for layer_idx in range(len(engine.transformer_layers)):
        if layer_idx not in safety_experts:
            continue

        layer = engine.transformer_layers[layer_idx]
        top = safety_experts[layer_idx][:n_suppress]
        if not top:
            continue

        # Pick the steering vector for this layer.
        any_device = next(iter(sv_by_device))
        if global_vector is None:
            v = sv_by_device[any_device][layer_idx + 1]
        else:
            v = gv_by_device[any_device]  # ty:ignore[non-subscriptable]

        # (A) Router-weight suppression
        gate = engine._locate_router(layer)
        if gate is not None and bias_value < 0:
            scale = max(0.0, 1.0 + bias_value / 10.0)
            for eid, _ in top:
                engine._router_originals.append(
                    (layer_idx, eid, gate.weight.data[eid].clone())
                )
                gate.weight.data[eid] *= scale

        # (B) Fused-expert down-projection steering
        fused = engine._locate_fused_weights(layer)
        if fused is not None and expert_w > 0:
            v_dev = v.to(fused.device)
            # Pre-fetch FP8 scale for this fused parameter (if applicable).
            fused_scale = None
            if _FP8_DTYPES and fused.dtype in _FP8_DTYPES:
                for attr in ("weight_scale", "scale"):
                    fused_scale = getattr(fused, attr, None)
                    if fused_scale is None:
                        # Try parent modules: mlp.experts (Qwen3), moe.down_proj (Step-3.5).
                        for parent_path in ("mlp.experts", "moe.down_proj", "experts"):
                            with __import__("contextlib").suppress(Exception):
                                parent = layer
                                for part in parent_path.split("."):
                                    parent = getattr(parent, part)
                                fused_scale = getattr(parent, attr, None)
                            if fused_scale is not None:
                                break
                    if fused_scale is not None:
                        break
            for eid, _ in top:
                if fused_scale is not None:
                    W = _dequantize_fp8_blockwise(fused.data[eid], fused_scale)
                else:
                    W = fused.data[eid].to(torch.float32)
                vTW = v_dev.float() @ W
                W -= expert_w * torch.outer(v_dev.float(), vTW)
                fused.data[eid] = W.to(fused.dtype)

                engine._expert_deltas.append(
                    (layer_idx, eid, expert_w, v_dev.float().cpu(), vTW.cpu())
                )
