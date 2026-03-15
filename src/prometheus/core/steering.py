# Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>

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

from ..settings import PrometheusConfig
from ..types import DecayKernel, ExpertRoutingConfig, SteeringProfile, WeightNorm

# Avoid circular import: accept the engine as a duck-typed object rather
# than importing SteeringEngine directly.  The caller is responsible for
# passing a valid engine instance.

_FP8_DTYPES = frozenset()
with __import__("contextlib").suppress(AttributeError):
    _FP8_DTYPES = frozenset({torch.float8_e4m3fn, torch.float8_e5m2})


def _dequantize_fp8_blockwise(
    weight: Tensor,
    weight_scale: Tensor,
    block_size: int = 128,
) -> Tensor:
    """Block-wise FP8 dequantization: W_real = weight_fp8 * scale_per_block."""
    out_f, in_f = weight.shape
    w = weight.to(torch.float32)
    scale = weight_scale.repeat_interleave(block_size, dim=0)[:out_f]
    scale = scale.repeat_interleave(block_size, dim=1)[:, :in_f]
    return w * scale


def apply_steering(
    engine,  # SteeringEngine
    steering_vectors: Tensor,
    vector_index: float | None,
    profiles: dict[str, SteeringProfile],
    config: PrometheusConfig | None = None,
    safety_experts: dict[int, list[tuple[int, float]]] | None = None,
    routing_config: ExpertRoutingConfig | None = None,
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
    config : PrometheusConfig
        Top-level configuration (kernel choice, normalisation, etc.).
    safety_experts : dict, optional
        MoE profiling results used for expert-level steering.
    routing_config : ExpertRoutingConfig, optional
        Hyper-parameters for MoE expert suppression.
    """
    if config is None:
        config = engine.config

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
        n_suppress = routing_config.n_suppress
        bias_value = routing_config.router_bias
        expert_w = routing_config.expert_ablation_weight

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
                            # Try the parent module (mlp.experts).
                            with __import__("contextlib").suppress(Exception):
                                fused_scale = getattr(layer.mlp.experts, attr, None)
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
