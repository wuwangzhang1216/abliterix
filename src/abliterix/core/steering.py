# Abliterix — a derivative work of Heretic (https://github.com/p-e-w/heretic)
# Original work Copyright (C) 2025  Philipp Emanuel Weidmann (p-e-w)
# Modified work Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Steering algorithm: modify model weights via low-rank LoRA updates.

This module implements the core steering (abliteration) procedure as a
standalone function rather than a method on the engine, keeping the algorithm
cleanly separated from model-management concerns.
"""

import math
from typing import cast

import torch
import torch.linalg as LA
import torch.nn.functional as F
from peft.tuners.lora.layer import Linear
from torch import Tensor

from ..settings import AbliterixConfig
from ..util import resolve_seed
from ..types import (
    DecayKernel,
    DirectTransform,
    ExpertRoutingConfig,
    SteeringMode,
    SteeringProfile,
    WeightNorm,
)
from ..weight_transforms import (
    apply_direct_transform,
    apply_ega_projection,
    resolve_ega_axis,
)

try:
    import bitsandbytes as bnb
except ImportError:  # pragma: no cover - exercised on macOS arm64 dev envs.
    bnb = None

# Avoid circular import: accept the engine as a duck-typed object rather
# than importing SteeringEngine directly.  The caller is responsible for
# passing a valid engine instance.

_FP8_DTYPES = frozenset()
with __import__("contextlib").suppress(AttributeError):
    _FP8_DTYPES = frozenset({torch.float8_e4m3fn, torch.float8_e5m2})


def resolve_global_vector(
    steering_vectors: Tensor, vector_index: float | None
) -> Tensor | None:
    """Interpolate the global steering vector from ``vector_index``.

    Returns ``None`` when per-layer vectors should be used (``vector_index is
    None``) or the tensor is a multi-direction subspace (3-D, where the first
    axis is directions, not layers). Shared by :func:`apply_steering` and the
    offline plan recorder (:func:`abliterix.core.fp4_repack.record_steering_plan_from_trial`)
    so both resolve the direction identically.
    """
    if vector_index is None or steering_vectors.ndim == 3:
        return None
    fractional, integral = math.modf(vector_index + 1)
    return F.normalize(
        steering_vectors[int(integral)].lerp(
            steering_vectors[int(integral) + 1],
            fractional,
        ),
        p=2,
        dim=0,
    )


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
    scale = (
        weight_scale.float()
        .repeat_interleave(block_r, dim=0)
        .repeat_interleave(block_c, dim=1)
    )
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
        n_layers = (
            steering_vectors.shape[1] - 1
            if steering_vectors.ndim == 3
            else steering_vectors.shape[0] - 1
        )
        return set(range(n_layers))

    # For multi-direction vectors (n_dirs, layers+1, hidden_dim), use the
    # primary (first) direction for discriminative layer detection.
    if steering_vectors.ndim == 3:
        sv = steering_vectors[0]  # (layers+1, hidden_dim)
    else:
        sv = steering_vectors

    discriminative: set[int] = set()
    n_layers = min(sv.shape[0] - 1, benign_states.shape[1] - 1)

    for layer_idx in range(n_layers):
        v = sv[layer_idx + 1]  # +1 because index 0 is embedding
        b = benign_states[:, layer_idx + 1, :].float()
        t = target_states[:, layer_idx + 1, :].float()

        # Mean scalar projection onto steering direction.
        mu_benign = (b @ v.float()).mean().item()
        mu_target = (t @ v.float()).mean().item()

        # Discriminative = opposite signs.
        if mu_benign * mu_target < 0:
            discriminative.add(layer_idx)

    return discriminative


def _rotate_toward_removal(
    h: Tensor,
    direction: Tensor,
    fraction: float,
) -> Tensor:
    """Geodesically rotate ``h`` toward the equator orthogonal to a direction."""
    if fraction == 0.0:
        return h

    d = direction.to(h.device, dtype=h.dtype)
    if d.norm() == 0:
        return h
    d = F.normalize(d, p=2, dim=0)

    raw_h_norm = h.norm(dim=-1, keepdim=True)
    h_norm = raw_h_norm.clamp(min=1e-8)
    h_hat = h / h_norm
    projection = (h_hat @ d).unsqueeze(-1).clamp(-1.0, 1.0)
    residual = h_hat - projection * d
    residual_norm = residual.norm(dim=-1, keepdim=True)
    removal_tangent = residual / residual_norm.clamp(min=1e-8)

    # Parallel activations do not define a unique great circle.  Pick a
    # deterministic orthogonal tangent by projecting the least-aligned
    # coordinate axis off the steering direction.
    fallback_axis = torch.zeros_like(d)
    fallback_axis[d.abs().argmin()] = 1
    fallback_tangent = F.normalize(
        fallback_axis - (fallback_axis @ d) * d,
        p=2,
        dim=0,
    )
    removal_tangent = torch.where(
        residual_norm <= 1e-6,
        fallback_tangent,
        removal_tangent,
    )

    # h_hat = sign(p) sin(alpha) d + cos(alpha) tangent.  Reducing alpha
    # toward zero removes the directional component without crossing the
    # tangent or inverting the activation.
    alpha = torch.atan2(projection.abs(), residual_norm)
    remaining = (1.0 - fraction) * alpha
    h_hat_new = (
        projection.sign() * torch.sin(remaining) * d
        + torch.cos(remaining) * removal_tangent
    )
    return torch.where(raw_h_norm == 0, h, h_norm * h_hat_new)


def _make_angular_hook(
    direction: Tensor,
    angle_degrees: float,
    adaptive: bool = False,
):
    """Create a forward hook that rotates activations toward removal.

    Abliterix has one direction per layer rather than the paper's second fixed
    plane basis.  It therefore uses the uniquely defined plane spanned by each
    activation and ``direction``, with a bounded rotation toward the
    direction-orthogonal removal tangent.

    Parameters
    ----------
    direction : Tensor
        Steering direction (hidden_dim,).  It is normalised by the hook.
    angle_degrees : float
        Rotation budget clamped to ``[0, 90]`` degrees.  Zero is identity;
        90 degrees is full directional removal.
    adaptive : bool
        If True, only rotate activations positively aligned with the
        direction (Adaptive Angular Steering), reducing interference.
    """
    fraction = min(max(angle_degrees / 90.0, 0.0), 1.0)

    def hook(module, input, output):
        h = output
        if isinstance(h, tuple):
            h = h[0]

        h_new = _rotate_toward_removal(h, direction, fraction)

        if adaptive:
            d = F.normalize(direction.to(h.device, dtype=h.dtype), p=2, dim=0)
            mask = ((h @ d).unsqueeze(-1) > 0).to(h_new.dtype)
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
    """Apply rank-k LoRA steering to every steerable module in the model.

    Parameters
    ----------
    engine : SteeringEngine
        The loaded model wrapper (provides ``transformer_layers``,
        ``steerable_modules``, adapter access, and helper methods).
    steering_vectors : Tensor
        Per-layer vectors of shape ``(layers+1, hidden_dim)``, or a
        multi-direction subspace of shape
        ``(n_directions, layers+1, hidden_dim)``.
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

    if steering_vectors.ndim == 3:
        runtime_hook_modes = {
            SteeringMode.ANGULAR,
            SteeringMode.ADAPTIVE_ANGULAR,
            SteeringMode.SPHERICAL,
            SteeringMode.VECTOR_FIELD,
        }
        if steering_mode in runtime_hook_modes:
            raise ValueError(
                f"Multi-direction steering is not implemented for runtime hook "
                f"mode {steering_mode.value!r}; use LoRA or dense direct mode."
            )
        if steering_mode == SteeringMode.DIRECT and engine.has_expert_routing():
            raise ValueError(
                "Multi-direction direct MoE steering is not yet supported: "
                "the EGA expert path accepts one direction per layer. Use a "
                "single direction or LoRA without expert routing."
            )

    # The legacy HF MoE path below accepts one residual direction per layer.
    # Reject rank-k tensors before either LoRA adapters or router/expert weights
    # are touched; otherwise the later layer lookup indexes the direction axis
    # and can fail after the LoRA update has already been committed.
    if steering_vectors.ndim == 3 and safety_experts and routing_config is not None:
        raise ValueError(
            "Multi-direction steering with HF MoE expert routing is not yet "
            "supported; disable expert routing or use a single direction."
        )

    # --- Discriminative layer selection -----------------------------------
    discriminative_layers: set[int] | None = None
    if config.steering.discriminative_layer_selection:
        discriminative_layers = _detect_discriminative_layers(
            steering_vectors,
            benign_states,
            target_states,
        )

    # --- Resolve the global steering vector (if applicable) ---------------
    global_vector = resolve_global_vector(steering_vectors, vector_index)

    # --- Direct weight editing (orthogonal projection, no LoRA) -----------
    if steering_mode == SteeringMode.DIRECT:
        _apply_direct_steering(
            engine,
            steering_vectors,
            global_vector,
            profiles,
            config,
            discriminative_layers,
            benign_states=benign_states,
        )
        # Expert-Granular Abliteration: project refusal direction from ALL
        # expert down_proj slices, not just top-N safety experts.  This is
        # critical for MoE models where refusal signal is distributed across
        # all experts (TrevorS EGA method: 3/100 vs 29/100 without).
        if engine.has_expert_routing():
            if getattr(config.steering, "frozen_experts", False):
                # Same projection, applied to the expert output instead of the
                # weight, so quantised experts never have to be unpacked.
                _apply_frozen_ega_steering(
                    engine,
                    steering_vectors,
                    global_vector,
                    profiles,
                    config,
                    discriminative_layers,
                )
            else:
                _apply_ega_steering(
                    engine,
                    steering_vectors,
                    global_vector,
                    profiles,
                    config,
                    discriminative_layers,
                )
        # Legacy top-N router suppression (complementary to EGA).
        if safety_experts and routing_config:
            _apply_moe_steering(
                engine, steering_vectors, global_vector, safety_experts, routing_config
            )
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
            _apply_moe_steering(
                engine, steering_vectors, global_vector, safety_experts, routing_config
            )
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
            _apply_moe_steering(
                engine, steering_vectors, global_vector, safety_experts, routing_config
            )
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
            _apply_moe_steering(
                engine, steering_vectors, global_vector, safety_experts, routing_config
            )
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
    adapter_updates: list[tuple[Tensor, Tensor, Tensor, Tensor]] = []

    for layer_idx in range(len(engine.transformer_layers)):
        # Skip non-discriminative layers when the feature is enabled.
        if discriminative_layers is not None and layer_idx not in discriminative_layers:
            continue

        for component, modules in engine.steerable_modules(layer_idx).items():
            # Skip components excluded via disabled_components.
            sp = profiles.get(component)
            if sp is None:
                continue

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

            # A strength of exactly 0 means this component is disabled for this
            # layer (e.g. the optimiser sampled max_weight = 0 via
            # auto_disable_components).  The adapter is already at identity after
            # restore_baseline(), so skip the wasteful dequant + decomposition.
            if strength == 0:
                continue

            for mod in modules:
                # TODO: The module-interface assumption here is fragile — PEFT
                #       wraps modules differently per quantisation mode.
                mod = cast(Linear, mod)

                device = mod.weight.device
                if global_vector is None:
                    if steering_vectors.ndim == 3:
                        # Multi-direction vectors are laid out as
                        # (n_directions, layers + 1, hidden_dim).  Keep the
                        # direction axis intact so each direction occupies one
                        # LoRA rank instead of flattening directions/layers.
                        v = sv_by_device[device][:, layer_idx + 1, :]
                    else:
                        v = sv_by_device[device][layer_idx + 1]
                else:
                    v = gv_by_device[device]  # ty:ignore[non-subscriptable]

                # Obtain the full-precision weight matrix W.
                base_weight = cast(Tensor, mod.base_layer.weight)
                qs = getattr(base_weight, "quant_state", None)
                CB = getattr(base_weight, "CB", None)

                if qs is not None:
                    if bnb is None:
                        raise RuntimeError(
                            "bitsandbytes is required to dequantize 4-bit weights. "
                            "Install abliterix on a supported CUDA platform or use "
                            "an unquantized model for this path."
                        )
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
                        engine._cache_dequant(mid, W)
                elif CB is not None:
                    # Int8 quantisation: dequantise from CB data and SCB row scales.
                    mid = id(mod)
                    if mid in engine._dequant_cache:
                        W = engine._dequant_cache[mid]
                    else:
                        SCB = base_weight.SCB  # ty:ignore[unresolved-attribute]
                        W = CB.float() * SCB.float().unsqueeze(1) / 127.0
                        engine._cache_dequant(mid, W)
                elif _FP8_DTYPES and base_weight.dtype in _FP8_DTYPES:
                    # FP8: dequantise to fp32 via block-wise or per-tensor scale.
                    # Checks `weight_scale_inv` (block-wise; DeepSeek / MiniMax-M2
                    # / Qwen3-FP8) before `weight_scale` (per-tensor; Qwen2-FP8);
                    # the previous code only looked for `weight_scale` and
                    # silently dropped the scale on block-wise models — yielding
                    # the raw FP8 values cast to fp32 (off by the per-block
                    # scale factor, destroying the projection).
                    mid = id(mod)
                    if mid in engine._dequant_cache:
                        W = engine._dequant_cache[mid]
                    else:
                        from . import fp8_utils as _fp8

                        scale_inv = getattr(mod.base_layer, "weight_scale_inv", None)
                        if isinstance(scale_inv, Tensor) and scale_inv.dim() == 2:
                            W = _fp8.dequant_blockwise(
                                base_weight.data,
                                scale_inv,
                                is_inv=True,
                                out_dtype=torch.float32,
                            )
                        else:
                            scale = getattr(mod.base_layer, "weight_scale", None)
                            W = _fp8.dequant_per_tensor(
                                base_weight.data,
                                scale,
                                out_dtype=torch.float32,
                            )
                        engine._cache_dequant(mid, W)
                else:
                    W = base_weight.to(torch.float32)

                W = W.view(W.shape[0], -1)

                # Keep one row per steering direction. Residual-sized outputs
                # use the historical output-side projection. Asymmetric GQA
                # K/V matrices have residual-sized inputs instead, so project
                # their input space rather than silently skipping them.
                V = v.unsqueeze(0) if v.ndim == 1 else v
                hidden_dim = V.shape[-1]
                if W.shape[0] == hidden_dim:
                    projection_side = "output"
                elif W.shape[1] == hidden_dim:
                    projection_side = "input"
                else:
                    continue

                # Optional row normalisation before computing the adapter.
                norm_mode = config.steering.weight_normalization
                if norm_mode != WeightNorm.NONE:
                    W_orig = W
                    W_row_norms = LA.vector_norm(W, dim=1, keepdim=True)
                    W = F.normalize(W, p=2, dim=1)

                # Rank-k steering stacks one update per direction.
                # Output-side (d_out == hidden): ΔW = -λ Vᵀ V W.
                # Input-side  (d_in  == hidden): ΔW = -λ W Vᵀ V.

                # Validate capacity against the requested steering subspace
                # before FULL normalisation can compress the update with SVD.
                # Otherwise a rank-r approximation could silently accept k>r
                # directions and make the configured multi-direction contract
                # impossible to represent.
                wA = cast(Tensor, mod.lora_A["default"].weight)
                wB = cast(Tensor, mod.lora_B["default"].weight)
                direction_rank = V.shape[0]
                if wA.shape[0] < direction_rank or wB.shape[1] < direction_rank:
                    raise ValueError(
                        "LoRA adapter rank is too small for steering subspace: "
                        f"need rank >= {direction_rank}, got A{tuple(wA.shape)} "
                        f"and B{tuple(wB.shape)}"
                    )
                if (
                    wA.shape[0] != wB.shape[1]
                    or wA.shape[1] != W.shape[1]
                    or wB.shape[0] != W.shape[0]
                ):
                    raise ValueError(
                        "LoRA adapter dimensions do not match base weight: "
                        f"adapter A{tuple(wA.shape)}, B{tuple(wB.shape)}; "
                        f"base weight {tuple(W.shape)}"
                    )

                if projection_side == "output":
                    lora_A = V @ W
                    lora_B = -strength * V.T
                else:
                    lora_A = V
                    lora_B = -strength * (W @ V.T)

                if norm_mode == WeightNorm.PRE:
                    lora_B = W_row_norms * lora_B
                elif norm_mode == WeightNorm.FULL:
                    # Low-rank SVD approximation that preserves original row
                    # magnitudes after the rank-k update.
                    W = W + lora_B @ lora_A
                    W = F.normalize(W, p=2, dim=1)
                    W = W * W_row_norms
                    W = W - W_orig
                    r = engine.peft_config.r
                    # svd_lowrank is randomised. Reseed immediately before the
                    # call so a restored/re-evaluated trial reproduces the same
                    # adapter independent of RNG history (otherwise FULL-mode
                    # trials desync the Optuna Pareto front on restore).
                    torch.manual_seed(resolve_seed(config))
                    U, S, Vh = torch.svd_lowrank(W, q=2 * r + 4, niter=6)
                    U = U[:, :r]
                    S = S[:r]
                    Vh = Vh[:, :r].T
                    sqrt_S = torch.sqrt(S)
                    lora_B = U @ torch.diag(sqrt_S)
                    lora_A = torch.diag(sqrt_S) @ Vh

                # Write the adapter weights (PEFT default adapter name).
                required_rank = lora_A.shape[0]
                if wA.shape[0] < required_rank or wB.shape[1] < required_rank:
                    raise ValueError(
                        "LoRA adapter rank is too small for steering subspace: "
                        f"need rank >= {required_rank}, got A{tuple(wA.shape)} "
                        f"and B{tuple(wB.shape)}"
                    )
                if wA.shape[1] != lora_A.shape[1] or wB.shape[0] != lora_B.shape[0]:
                    raise ValueError(
                        "LoRA adapter dimensions do not match steering update: "
                        f"adapter A{tuple(wA.shape)}, B{tuple(wB.shape)}; "
                        f"update A{tuple(lora_A.shape)}, B{tuple(lora_B.shape)}"
                    )

                # Preserve the PEFT Parameter objects and their declared rank.
                # Extra capacity is zero-filled when adapter rank > k.
                new_A = torch.zeros_like(wA)
                new_B = torch.zeros_like(wB)
                new_A[:required_rank].copy_(lora_A.to(wA.dtype))
                new_B[:, :required_rank].copy_(lora_B.to(wB.dtype))
                adapter_updates.append((wA, wB, new_A, new_B))

    # Commit only after every target module has been computed and validated.
    # This keeps a rejected trial from leaving earlier adapters partially
    # steered.  The rollback also covers the unlikely case of a failed device
    # copy during the commit itself.
    originals = [
        (wA, wB, wA.detach().clone(), wB.detach().clone())
        for wA, wB, _new_A, _new_B in adapter_updates
    ]
    try:
        with torch.no_grad():
            for wA, wB, new_A, new_B in adapter_updates:
                wA.copy_(new_A)
                wB.copy_(new_B)
    except Exception:
        with torch.no_grad():
            for wA, wB, original_A, original_B in originals:
                wA.copy_(original_A)
                wB.copy_(original_B)
        raise

    # --- MoE expert-level steering ----------------------------------------
    if safety_experts and routing_config:
        _apply_moe_steering(
            engine,
            steering_vectors,
            global_vector,
            safety_experts,
            routing_config,
            sv_by_device=sv_by_device,
            gv_by_device=gv_by_device,
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
    *,
    benign_states: Tensor | None = None,
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
    direct_transform = config.steering.direct_transform
    preserve_row_norm = config.steering.direct_transform_preserve_row_norm

    # Pre-compute per-layer benign direction (input space) once when an
    # advanced transform that needs the double-GS pre-step is selected.
    # benign_states shape: (n_benign, layers+1, hidden_dim).
    benign_dirs: Tensor | None = None
    if (
        direct_transform in (DirectTransform.ORBA, DirectTransform.HOUSEHOLDER)
        and benign_states is not None
    ):
        benign_mean = benign_states.mean(dim=0).to(torch.float32)  # (layers+1, dim)
        benign_norms = torch.linalg.vector_norm(benign_mean, dim=1, keepdim=True).clamp(
            min=1e-8
        )
        benign_dirs = benign_mean / benign_norms

    # Cache originals for restore_baseline.
    if not hasattr(engine, "_direct_weight_originals"):
        engine._direct_weight_originals = {}

    for layer_idx in range(len(engine.transformer_layers)):
        if discriminative_layers is not None and layer_idx not in discriminative_layers:
            continue

        for component, modules in engine.steerable_modules(layer_idx).items():
            # Skip components excluded via disabled_components.
            sp = profiles.get(component)
            if sp is None:
                continue

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

            # Exactly-0 strength means this component is disabled for this layer
            # (e.g. max_weight clamped to 0 via auto_disable_components). The
            # base weights are untouched, so skip the edit entirely.
            if strength == 0:
                continue

            for mod in modules:
                # Navigate to the base weight — through PEFT wrapper if present.
                base_mod = mod
                if hasattr(mod, "base_layer"):
                    base_mod = mod.base_layer

                weight = base_mod.weight

                # Defense in depth: direct editing needs a writable BF16/full-
                # precision base weight. A bitsandbytes-quantised weight
                # (Params4bit `quant_state`, or int8 `CB`) is packed storage
                # that `.to(float32)` reinterprets rather than dequantises, so
                # an in-place write would corrupt it. The config validator
                # already rejects bnb + direct, but this guards direct
                # programmatic callers too. (FP8 is materialised to BF16 before
                # steering, so it is writable here.)
                if (
                    getattr(weight, "quant_state", None) is not None
                    or getattr(weight, "CB", None) is not None
                ):
                    raise RuntimeError(
                        f"direct steering cannot edit quantised base weight "
                        f"'{component}' (bitsandbytes packed storage is not "
                        "writable in place). Use steering_mode='lora', load "
                        "unquantized, or bake a native-FP4 model offline with "
                        "`abliterix-abliterate-fp4`."
                    )

                # Cache the original weight for later restoration.
                # Key by the weight tensor itself for O(1) restore.
                if weight not in engine._direct_weight_originals:
                    engine._direct_weight_originals[weight] = weight.data.clone()

                device = weight.device

                # Use float32 for projection math to preserve precision
                # (bf16 loses signal in 2816-dim inner products).
                W = weight.data.to(torch.float32)
                out_f, in_f = W.shape

                # Multi-direction subspace projection: when steering_vectors
                # is 3D (n_dirs, layers+1, hidden_dim), project out the full
                # refusal subspace in one shot via QR-based projection.
                if steering_vectors.ndim == 3:
                    # (n_dirs, hidden_dim)
                    V_layer = (
                        steering_vectors[:, layer_idx + 1, :]
                        .to(device)
                        .to(torch.float32)
                    )
                    # Build orthonormal basis via QR.
                    if V_layer.shape[1] == in_f:
                        Q, _ = torch.linalg.qr(V_layer.T)  # (in_f, rank)
                        # Subspace projection: W_new = W - strength * W @ Q @ Q^T
                        W_new = W - strength * (W @ Q) @ Q.T
                    elif V_layer.shape[1] == out_f:
                        Q, _ = torch.linalg.qr(V_layer.T)  # (out_f, rank)
                        W_new = W - strength * Q @ (Q.T @ W)
                    else:
                        continue
                else:
                    if global_vector is None:
                        v = steering_vectors[layer_idx + 1].to(device)
                    else:
                        v = global_vector.to(device)
                    vf = v.to(torch.float32)

                    # Advanced grimjim transforms (ORBA / biprojected /
                    # Householder) accept either input-side or output-side
                    # directions — apply_orba_transform picks the right
                    # branch internally (prefers output-side for square
                    # matrices). Trigger whenever v matches either dim of W
                    # so square modules like attn.o_proj don't silently
                    # fall through to standard, losing ORBA's row-norm
                    # preservation post-step.
                    if direct_transform != DirectTransform.STANDARD and (
                        vf.shape[0] == in_f or vf.shape[0] == out_f
                    ):
                        bdir: Tensor | None = None
                        if (
                            benign_dirs is not None
                            and benign_dirs.shape[1] == vf.shape[0]
                        ):
                            bdir = benign_dirs[layer_idx + 1].to(device)
                        # Householder / ORBA require benign_dir; if we don't
                        # have one (benign_states wasn't kept past the
                        # extraction phase), fall through to standard.
                        if (
                            direct_transform
                            in (DirectTransform.ORBA, DirectTransform.HOUSEHOLDER)
                            and bdir is None
                        ):
                            pass  # fall through to standard path below
                        else:
                            W_new = apply_direct_transform(
                                direct_transform,
                                W,
                                vf,
                                bdir,
                                strength=strength,
                                preserve_row_norm=preserve_row_norm,
                            )
                            weight.data = W_new.to(weight.dtype)
                            continue

                    # Orthogonal projection: remove the refusal direction from W.
                    # W has shape (out_features, in_features).
                    # v has shape (hidden_dim,) which may match either dimension.
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
                    new_norms = torch.linalg.vector_norm(
                        W_new, dim=1, keepdim=True
                    ).clamp(min=1e-8)
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

        # Exactly-0 strength disables EGA for this layer (auto_disable_components).
        if strength == 0:
            continue

        # Pick the steering vector for this layer.
        device = fused.device
        if global_vector is None:
            v = steering_vectors[layer_idx + 1].to(device)
        else:
            v = global_vector.to(device)

        # Cache original for restore_baseline.
        if fused not in engine._direct_weight_originals:
            engine._direct_weight_originals[fused] = fused.data.clone()

        vf = v.to(torch.float32)

        # Layout disambiguation: standard MoE stores fused down_proj as
        # (experts, out=hidden, in=intermediate); gpt-oss stores it
        # transposed as (experts, in=intermediate, out=hidden) and uses
        # `out = act @ W` directly. Shape-based detection is ambiguous when
        # hidden == intermediate (e.g. gpt-oss-20b: both 2880), so we honour
        # the engine's `_fused_down_proj_transposed` flag set at load time.
        transposed = getattr(engine, "_fused_down_proj_transposed", False)

        # Axis + projection are shared with the offline FP4 repack tool via
        # weight_transforms so the abliteration fingerprint is bit-identical.
        axis_is_in = resolve_ega_axis(
            tuple(fused.shape), vf.shape[0], transposed=transposed
        )
        if axis_is_in is None:
            continue

        # Vectorised over the expert dimension: single GPU kernel batch
        # instead of a 128-iter Python loop with per-expert dtype conversions.
        W_new = apply_ega_projection(
            fused.data,
            vf,
            strength=strength,
            axis_is_in=axis_is_in,
            preserve_row_norm=norm_preserve,
        )
        fused.data.copy_(W_new.to(fused.dtype))
        del W_new


def _apply_frozen_ega_steering(
    engine,
    steering_vectors: Tensor,
    global_vector: Tensor | None,
    profiles: dict[str, SteeringProfile],
    config: AbliterixConfig,
    discriminative_layers: set[int] | None,
):
    """EGA applied at forward time, leaving quantised expert weights packed.

    Same per-layer strengths and same projection as :func:`_apply_ega_steering`
    — but installed as hooks on each MoE block rather than written into the
    fused weight, because the rank-1 edit is algebraically identical to
    projecting the direction out of the expert output (see
    :mod:`abliterix.core.frozen_experts`). Nothing is dequantised, so a natively
    4-bit MoE stays at its packed size for the whole search.

    Handles land in ``engine._angular_hooks`` so ``restore_baseline`` removes
    them along with the other runtime-hook modes.
    """
    from .frozen_experts import build_frozen_plan, install_frozen_ega_on_moe_block

    kernel = config.steering.decay_kernel

    sp = profiles.get("mlp.down_proj")
    if sp is None:
        return

    if not hasattr(engine, "_angular_hooks"):
        engine._angular_hooks = []

    installed = 0
    for layer_idx in range(len(engine.transformer_layers)):
        if discriminative_layers is not None and layer_idx not in discriminative_layers:
            continue

        layer = engine.transformer_layers[layer_idx]
        experts = _locate_expert_container(layer)
        if experts is None:
            continue

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
        if strength == 0:
            continue

        v = (
            global_vector
            if global_vector is not None
            else steering_vectors[layer_idx + 1]
        )
        if v.ndim != 1:
            continue

        # No weights are read: preserve_row_norm is rejected by config
        # validation for this mode, so the plan is entirely weight-free.
        plan = build_frozen_plan(
            None,
            v.detach().to(torch.float32),
            float(strength),
            axis_is_in=True,
            preserve_row_norm=False,
        )
        handles = install_frozen_ega_on_moe_block(
            experts,
            plan,
            router_module=engine._locate_router(layer),
            expert_bias=getattr(experts, "down_proj_bias", None),
        )
        engine._angular_hooks.extend(handles)
        installed += 1

    if installed and config.display.print_responses:
        print(f"* frozen EGA: hooked {installed} MoE blocks (experts left packed)")


def _locate_expert_container(layer) -> object | None:
    """Find the module that computes the combined expert output for a layer.

    Deliberately the *container*, not the individual experts: its output is the
    routing-weighted sum, and the projection is linear, so projecting the sum
    equals projecting each expert's contribution.
    """
    for path in (
        "mlp.experts",
        "block_sparse_moe.experts",
        "feed_forward.experts",
        "moe.experts",
        "mixer.experts",
        "ffn.experts",
    ):
        obj = layer
        for attr in path.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "register_forward_hook"):
            return obj
    return None


# ---------------------------------------------------------------------------
# vLLM in-place path: same projection math, dispatched to TP workers.
#
# Mirrors ``_apply_direct_steering`` + ``_apply_ega_steering`` but instead
# of editing an HF model's weights locally, this packages the per-layer
# steering vector + strength into a plan and ships it to every vLLM TP
# worker via ``collective_rpc``. The math is identical to the HF path
# (see :func:`_apply_direct_steering` / :func:`_apply_ega_steering`) so
# the abliteration fingerprint is preserved.
#
# Used only when the VLLMGenerator has ``expert_editor`` + ``attention_editor``
# attached (set up in cli.py when ``[vllm].use_in_place_editing = true``).
# ---------------------------------------------------------------------------


def _save_vec_bytes(v: Tensor) -> bytes:
    """Serialize a 1-D steering vector for collective_rpc transport."""
    import io

    buf = io.BytesIO()
    torch.save(v.detach().to(dtype=torch.float32, device="cpu"), buf)
    return buf.getvalue()


def _interpolate_strength(
    layer_idx: int, sp: SteeringProfile, kernel: DecayKernel
) -> float | None:
    """Replicate the decay-kernel interpolation used by the HF paths.

    Returns ``None`` when the layer falls outside ``[max_pos ± min_dist]`` or
    when the interpolated strength is exactly 0 (component disabled for this
    layer via ``auto_disable_components``), so vLLM callers skip it uniformly.
    """
    distance = cast(float, abs(layer_idx - sp.max_weight_position))
    if distance > sp.min_weight_distance:
        return None
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
    return None if strength == 0 else strength


_ATTN_COMPONENTS: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")


def _apply_direct_steering_vllm(
    vllm_gen,
    steering_vectors: Tensor,
    global_vector: Tensor | None,
    profiles: dict[str, SteeringProfile],
    config: AbliterixConfig,
    n_layers: int,
    discriminative_layers: set[int] | None,
) -> dict:
    """Apply attention q/k/v/o_proj projection via vLLM TP workers.

    Returns the aggregated RPC response from the attention editor.
    """
    kernel = config.steering.decay_kernel
    norm_preserve = config.steering.weight_normalization != WeightNorm.NONE

    plan: list[dict] = []
    for layer_idx in range(n_layers):
        if discriminative_layers is not None and layer_idx not in discriminative_layers:
            continue
        for component in _ATTN_COMPONENTS:
            # Profiles may be keyed as "attn.q_proj" (new) or "q_proj" (legacy).
            sp = profiles.get(f"attn.{component}") or profiles.get(component)
            if sp is None:
                continue

            strength = _interpolate_strength(layer_idx, sp, kernel)
            if strength is None:
                continue

            # Pick steering vector for this layer.
            if global_vector is None:
                v_layer = steering_vectors[layer_idx + 1]
            else:
                v_layer = global_vector

            plan.append(
                {
                    "layer_idx": layer_idx,
                    "component": component,
                    "v": _save_vec_bytes(v_layer),
                    "strength": float(strength),
                }
            )

    if not plan:
        return {"applied": 0, "errors": [], "per_layer": []}
    return vllm_gen.apply_attention_projection(plan, norm_preserve=norm_preserve)


def _apply_ega_steering_vllm(
    vllm_gen,
    steering_vectors: Tensor,
    global_vector: Tensor | None,
    profiles: dict[str, SteeringProfile],
    config: AbliterixConfig,
    n_layers: int,
    hidden_dim: int,
    transposed: bool,
    discriminative_layers: set[int] | None,
) -> dict:
    """Apply EGA on fused expert down_proj via vLLM TP workers."""
    sp = profiles.get("mlp.down_proj")
    if sp is None:
        return {"applied": 0, "errors": [], "per_layer": []}

    kernel = config.steering.decay_kernel
    norm_preserve = config.steering.weight_normalization != WeightNorm.NONE

    plan: list[dict] = []
    for layer_idx in range(n_layers):
        if discriminative_layers is not None and layer_idx not in discriminative_layers:
            continue
        strength = _interpolate_strength(layer_idx, sp, kernel)
        if strength is None:
            continue

        if global_vector is None:
            v_layer = steering_vectors[layer_idx + 1]
        else:
            v_layer = global_vector

        plan.append(
            {
                "layer_idx": layer_idx,
                "v": _save_vec_bytes(v_layer),
                "strength": float(strength),
                "hidden_dim": hidden_dim,
                "transposed": transposed,
            }
        )

    if not plan:
        return {"applied": 0, "errors": [], "per_layer": []}
    return vllm_gen.apply_ega_projection(plan, norm_preserve=norm_preserve)


def apply_steering_vllm_inplace(
    vllm_gen,
    steering_vectors: Tensor,
    vector_index: float | None,
    profiles: dict[str, SteeringProfile],
    config: AbliterixConfig,
    n_layers: int,
    hidden_dim: int,
    transposed: bool = False,
    safety_experts: dict[int, list[tuple[int, float]]] | None = None,
    routing_config: ExpertRoutingConfig | None = None,
) -> dict:
    """End-to-end ``apply_steering`` for the vLLM in-place path.

    Replaces the HF-engine version when vLLM is attached with BOTH
    ``expert_editor`` and ``attention_editor``. Also triggers the existing
    router suppression path (via ``moe_editor``) if safety experts were
    profiled.

    Returns a diagnostic dict summarising how many layers each editor
    touched — useful for a first-trial sanity log.
    """
    # Resolve global vector identically to HF path.
    if vector_index is None or steering_vectors.ndim == 3:
        global_vector = None
    else:
        fractional, integral = math.modf(vector_index + 1)
        global_vector = F.normalize(
            steering_vectors[int(integral)].lerp(
                steering_vectors[int(integral) + 1], fractional
            ),
            p=2,
            dim=0,
        )

    attn_result = _apply_direct_steering_vllm(
        vllm_gen,
        steering_vectors,
        global_vector,
        profiles,
        config,
        n_layers=n_layers,
        discriminative_layers=None,
    )
    ega_result = _apply_ega_steering_vllm(
        vllm_gen,
        steering_vectors,
        global_vector,
        profiles,
        config,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        transposed=transposed,
        discriminative_layers=None,
    )

    router_touched = 0
    if safety_experts and routing_config is not None:
        if routing_config.n_suppress > 0 and routing_config.router_bias < 0:
            router_touched = vllm_gen.apply_router_suppression(
                n_suppress=routing_config.n_suppress,
                bias_value=routing_config.router_bias,
            )

    return {
        "attention": attn_result,
        "ega": ega_result,
        "router_touched": router_touched,
    }


def restore_all_vllm_inplace(vllm_gen) -> dict:
    """Restore every in-place edit applied by :func:`apply_steering_vllm_inplace`.

    Safe to call even if nothing was applied — each editor's ``restore()``
    is a no-op in that case.
    """
    return {
        "attention": vllm_gen.restore_attention_weights(),
        "ega": vllm_gen.restore_expert_weights(),
        "router": vllm_gen.restore_router_suppression(),
    }


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

        # Strength is the fraction of full directional removal.  The hook
        # clamps values above 1.0 at the 90° removal tangent.
        angle = strength * 90.0

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
    """Create a hook that rotates activations toward directional removal.

    Rotation follows the shortest geodesic from the activation toward its
    projection on the hypersphere equator orthogonal to ``direction``.  The
    requested angle is a bounded rotation budget: 0 degrees is identity and
    90 degrees reaches full directional removal without crossing the equator
    or inverting the activation.

    Parameters
    ----------
    direction : Tensor
        Unit-normalised steering direction (hidden_dim,).
    angle_degrees : float
        Rotation budget along the geodesic, clamped to ``[0, 90]`` degrees.
    """
    fraction = min(max(angle_degrees / 90.0, 0.0), 1.0)

    def hook(module, input, output):
        h = output
        if isinstance(h, tuple):
            h = h[0]

        h_new = _rotate_toward_removal(h, direction, fraction)

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

        # Strength has the same bounded removal semantics as angular mode.
        angle = strength * 90.0

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
                score.sum(),
                h_detached,
                create_graph=False,
            )[0]

        # Normalise gradient to get per-token steering direction.
        grad_norm = grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        d = grad / grad_norm

        # Fall back to static direction where gradient is degenerate.
        degenerate = grad_norm.squeeze(-1) < 1e-6
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
            {d: global_vector.to(d) for d in devices}
            if global_vector is not None
            else None
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
            # Resolve the projection axis with the same helper EGA uses so the
            # gpt-oss transposed fused layout (hidden on the last axis) is not
            # silently projected along the wrong axis — and so this edit stays
            # bit-identical to the offline FP4 repack path.
            transposed = getattr(engine, "_fused_down_proj_transposed", False)
            axis_is_in = resolve_ega_axis(
                tuple(fused.shape), v_dev.shape[0], transposed=transposed
            )
            if axis_is_in is None:
                # Direction matches neither axis: nothing meaningful to ablate
                # for this layer, but keep processing the remaining layers.
                continue

            for eid, _ in top:
                if (
                    hasattr(engine, "_direct_weight_originals")
                    and fused in engine._direct_weight_originals
                ):
                    original_slice = (
                        engine._direct_weight_originals[fused][eid].detach().clone()
                    )
                else:
                    original_slice = fused.data[eid].detach().clone()
                if fused_scale is not None:
                    W = _dequantize_fp8_blockwise(fused.data[eid], fused_scale)
                else:
                    W = fused.data[eid].to(torch.float32)
                vf = v_dev.float()
                if axis_is_in:
                    # Transposed (gpt-oss): W is (in, out); out = act @ W.
                    Wv = W @ vf
                    W -= expert_w * torch.outer(Wv, vf)
                else:
                    # Standard: W is (out, in); direction lives on out.
                    vTW = vf @ W
                    W -= expert_w * torch.outer(vf, vTW)
                fused.data[eid] = W.to(fused.dtype)

                # Store the untouched slice so restore_baseline can write it
                # back exactly (idempotent w.r.t. the EGA whole-tensor restore
                # and lossless for fp8/packed storage dtypes).
                engine._expert_deltas.append((layer_idx, eid, original_slice.to("cpu")))
