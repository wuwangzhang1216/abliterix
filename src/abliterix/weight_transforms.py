# Abliterix
# Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Weight-space transforms for direct-mode steering.

Catalogs every direct-mode weight transformation that abliterix can apply
when ``steering_mode = "direct"`` and dispatches between them based on
``SteeringConfig.direct_transform``.

Available transforms
--------------------
``standard``
    The historical abliterix path: ``W ← W - strength · (W · d) ⊗ d`` with
    optional row-norm preservation via :class:`WeightNorm`. Implemented
    directly in :func:`core.steering._apply_direct_steering`; this module
    only exposes the helper :func:`apply_standard_transform` for tests.

``orba``
    *Orthogonal Reflection Bounded Ablation* — `grimjim, 2025
    <https://huggingface.co/blog/grimjim/orthogonal-reflection-bounded-ablation>`_.
    Applies the standard rank-1 ablation after a **double Gram-Schmidt**
    orthogonalisation of the refusal direction against the benign-mean
    direction (the "twice is enough" stability pass). Optionally enforces
    row-Frobenius-norm preservation in a separate post-step.

``biprojected``
    *Norm-Preserving Biprojected Abliteration* — `grimjim, 2025
    <https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration>`_.
    Decomposes ``W = M · Ŵ`` (per-row magnitudes × per-row unit directions),
    ablates the refusal direction on ``Ŵ`` only, re-normalises each row of
    ``Ŵ`` to unit length, then recombines ``W_new = M · Ŵ_new``. The row
    norm of each output neuron is exactly preserved.

``householder``
    Exact isometric reflection ``W ← H W`` where ``H = I − 2 û ûᵀ`` is the
    Householder reflector aligned with the (orthogonalised) refusal
    direction. Norm-preserving by construction, but grimjim reports token-
    level glitches in practice — included for completeness; defaults
    off-bounds in the auto search.

All transforms are pure functions of ``(W, refusal, benign_dir, strength)``
plus their own knobs; they return a *new* weight tensor and never mutate
input. The caller is responsible for caching originals (the in-place
mutation is :func:`core.steering._apply_direct_steering`'s job).
"""

from __future__ import annotations

from enum import Enum

import torch
from torch import Tensor


class DirectTransform(str, Enum):
    STANDARD = "standard"
    ORBA = "orba"
    BIPROJECTED = "biprojected"
    HOUSEHOLDER = "householder"


# ---------------------------------------------------------------------------
# Direction-side helpers
# ---------------------------------------------------------------------------


def _normalise(v: Tensor, eps: float = 1e-8) -> Tensor:
    n = torch.linalg.vector_norm(v).clamp(min=eps)
    return v / n


def double_gram_schmidt(
    refusal: Tensor,
    benign_dir: Tensor,
) -> Tensor:
    """Orthogonalise ``refusal`` against ``benign_dir`` twice.

    Modified Gram-Schmidt iteration ("twice is enough"): two passes of
    projection-and-subtract restore numerical orthogonality when the
    initial vectors are nearly aligned. The benign direction is assumed
    unit-normalised — pass ``F.normalize(benign_mean, dim=-1)``.

    Returns the unit-normalised orthogonalised refusal direction.
    """
    h = benign_dir.to(dtype=torch.float32)
    h = _normalise(h)
    u = refusal.to(dtype=torch.float32)
    # First pass
    u = u - torch.dot(u, h) * h
    # Second pass (twice is enough)
    u = u - torch.dot(u, h) * h
    return _normalise(u)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


def _rescale_norms(
    W_orig: Tensor,
    W_new: Tensor,
    *,
    axis_is_in: bool,
    preserve_row_norm: bool,
    preserve_projected_norm: bool,
) -> Tensor:
    """Restore per-slice L2 norms after a rank-1 or subspace projection.

    When ``preserve_projected_norm`` is True, rescales along the projected axis
    (``dim=-1`` for input-side projections, ``dim=-2`` for output-side
    projections), which multiplies each projected vector by a positive scalar
    and preserves exact orthogonality ``v^T W_new == 0`` (at strength 1).
    Legacy ``preserve_row_norm`` rescales ``dim=-1`` regardless of side.
    """
    if preserve_projected_norm:
        dim = -1 if axis_is_in else -2
    elif preserve_row_norm:
        dim = -1
    else:
        return W_new
    orig_norms = torch.linalg.vector_norm(W_orig, dim=dim, keepdim=True)
    new_norms = torch.linalg.vector_norm(W_new, dim=dim, keepdim=True).clamp(min=1e-8)
    return W_new * (orig_norms / new_norms)


def apply_standard_transform(
    W: Tensor,
    refusal: Tensor,
    *,
    strength: float = 1.0,
    preserve_row_norm: bool = False,
    preserve_projected_norm: bool = False,
) -> Tensor:
    """The plain rank-1 ablation used by the historical direct path.

    ``W_new = W - strength · (W · d) ⊗ d`` where ``d`` is assumed
    unit-normalised.

    ``preserve_row_norm`` reproduces the engine's post-step (see
    :func:`core.steering._apply_direct_steering`): every output row (dim 1,
    the ``out_features`` axis) is rescaled to its original L2 norm. Setting
    ``preserve_projected_norm=True`` rescales along the projected axis instead
    (columns for output-side, rows for input-side) so exact orthogonality is
    preserved.
    """
    d = _normalise(refusal.to(dtype=torch.float32))
    W32 = W.to(dtype=torch.float32)
    if W32.shape[1] == d.shape[0]:
        # Input-side ablation: W (out, in), direction is (in,).
        axis_is_in = True
        new_W = W32 - strength * (W32 @ d).unsqueeze(1) * d.unsqueeze(0)
    elif W32.shape[0] == d.shape[0]:
        # Output-side ablation: W (out, in), direction is (out,).
        axis_is_in = False
        new_W = W32 - strength * d.unsqueeze(1) * (d @ W32).unsqueeze(0)
    else:
        raise ValueError(
            f"Refusal direction shape {d.shape} does not match either axis "
            f"of W shape {W32.shape}."
        )
    new_W = _rescale_norms(
        W32,
        new_W,
        axis_is_in=axis_is_in,
        preserve_row_norm=preserve_row_norm,
        preserve_projected_norm=preserve_projected_norm,
    )
    return new_W.to(W.dtype)


# ---------------------------------------------------------------------------
# Expert-Granular Abliteration (EGA) — per-expert rank-1 projection on a
# fused 3-D expert weight. Single source of truth shared by the in-engine HF
# path (core.steering._apply_ega_steering) and the offline repack tool
# (core.fp4_repack), so the abliteration fingerprint is bit-identical across
# both. Mirrors the vLLM TP-worker kernel in core.vllm_moe_editor.
# ---------------------------------------------------------------------------


def resolve_ega_axis(
    fused_shape: tuple[int, ...],
    hidden_dim: int,
    *,
    transposed: bool = False,
) -> bool | None:
    """Decide which axis of a fused expert tensor the refusal direction lives on.

    ``fused_shape`` is ``(E, d0, d1)``. Returns:
        ``True``  — project the last/input axis (``d1``); direction has len d1
        ``False`` — project the ``d0``/output axis; direction has len d0
        ``None``  — ``hidden_dim`` matches neither axis; caller should skip

    ``transposed=True`` forces the gpt-oss layout where hidden is the input
    (last) axis. For the ambiguous square case (``d0 == d1 == hidden``) the
    standard (output-axis) convention is chosen, matching both the HF and
    vLLM EGA kernels.
    """
    if len(fused_shape) != 3:
        return None
    _, d0, d1 = fused_shape
    if transposed:
        return True if d1 == hidden_dim else None
    if d0 == hidden_dim and d1 != hidden_dim:
        return False
    if d1 == hidden_dim and d0 != hidden_dim:
        return True
    if d0 == hidden_dim:
        return False  # ambiguous square → prefer standard/output axis
    return None


def apply_ega_projection(
    W: Tensor,
    refusal: Tensor,
    *,
    strength: float,
    axis_is_in: bool,
    preserve_row_norm: bool = True,
    preserve_projected_norm: bool = False,
) -> Tensor:
    """Per-expert rank-1 orthogonal projection on a fused 3-D expert weight.

    ``W`` has shape ``(E, d0, d1)``; ``refusal`` is a 1-D direction whose
    length matches the projected axis (``d1`` when ``axis_is_in`` else
    ``d0``). Vectorised over the expert dimension. Returns a new float32
    tensor; the caller casts back to the storage dtype.

    This is the exact math of ``core.steering._apply_ega_steering`` and the
    ``_worker_apply_ega_batch`` vLLM kernel — do not diverge it.
    """
    W_all = W.to(torch.float32)
    vf = refusal.to(device=W_all.device, dtype=torch.float32)
    if axis_is_in:
        proj = torch.matmul(W_all, vf)  # (E, d0)
        W_new = W_all - strength * (proj.unsqueeze(-1) * vf.view(1, 1, -1))
    else:
        proj = torch.einsum("o,eoi->ei", vf, W_all)  # (E, d1)
        W_new = W_all - strength * (vf.view(1, -1, 1) * proj.unsqueeze(1))
    W_new = _rescale_norms(
        W_all,
        W_new,
        axis_is_in=axis_is_in,
        preserve_row_norm=preserve_row_norm,
        preserve_projected_norm=preserve_projected_norm,
    )
    return W_new


def apply_orba_transform(
    W: Tensor,
    refusal: Tensor,
    benign_dir: Tensor,
    *,
    strength: float = 1.0,
    preserve_row_norm: bool = True,
    preserve_projected_norm: bool = False,
) -> Tensor:
    """ORBA: double-GS orthogonalisation + rank-1 ablation + optional norm preserve.

    For modules that write to the residual stream (e.g. attn.o_proj,
    mlp.down_proj) the canonical abliteration removes the refusal direction
    from the OUTPUT side:

        W_new = W - strength · û ⊗ (û · W)        # output-side

    For modules that read from the residual stream (e.g. attn.q/k/v_proj —
    rarely the abliteration target on its own) the input-side variant is:

        W_new = W - strength · (W · û) ⊗ û        # input-side

    When the refusal direction matches both dims (square matrix such as a
    standard-attention o_proj), prefer the OUTPUT-side variant so the kernel
    matches what the LoRA path and ``apply_standard_transform`` produce —
    the historical input-side-only behaviour was a bug that produced very
    different deltas on square matrices vs. the LoRA / standard paths,
    visible as a ~8x weaker ablation on Granite 4.1 8B's o_proj layers.

    If ``preserve_projected_norm`` is True, rescales along the projected axis
    (columns for output-side, rows for input-side) so exact orthogonality is
    preserved; else if ``preserve_row_norm`` is True, rescales every output
    row (grimjim's legacy default).
    """
    u_hat = double_gram_schmidt(refusal, benign_dir)
    W32 = W.to(dtype=torch.float32)
    out_f, in_f = W32.shape

    if u_hat.shape[0] == out_f:
        # Output-side: prefered when v lives in the residual / output space.
        axis_is_in = False
        new_W = W32 - strength * u_hat.unsqueeze(1) * (u_hat @ W32).unsqueeze(0)
    elif u_hat.shape[0] == in_f:
        # Input-side fallback for asymmetric matrices where v only matches
        # the input dim (rare for abliteration target modules).
        axis_is_in = True
        new_W = W32 - strength * (W32 @ u_hat).unsqueeze(1) * u_hat.unsqueeze(0)
    else:
        raise ValueError(
            f"ORBA direction shape {u_hat.shape} does not match either axis "
            f"of W shape {W32.shape}."
        )

    new_W = _rescale_norms(
        W32,
        new_W,
        axis_is_in=axis_is_in,
        preserve_row_norm=preserve_row_norm,
        preserve_projected_norm=preserve_projected_norm,
    )

    return new_W.to(W.dtype)


def apply_biprojected_transform(
    W: Tensor,
    refusal: Tensor,
    *,
    strength: float = 1.0,
) -> Tensor:
    """Norm-Preserving Biprojected Abliteration (grimjim).

    Picks input-side or output-side decomposition based on which axis of W
    the refusal direction matches; prefers output-side when both match
    (square modules) so the semantics line up with LoRA / standard / ORBA.

    Output-side (``refusal.shape[0] == W.shape[0]``): column-magnitude
    decomposition ``W = Ŵ · diag(N)`` where Nⱼ is the L2 norm of column j
    and Ŵ has unit-norm columns. Ablate refusal from the unit columns,
    re-normalise, recombine. Preserves each column's L2 norm exactly.

        p     = û · Ŵ                              # (in,)
        Ŵ_abl = Ŵ − strength · û ⊗ p
        Ŵ_new = normalize_cols(Ŵ_abl)
        W_new = Ŵ_new · diag(N)

    Input-side (``refusal.shape[0] == W.shape[1]``): the original row-mag
    decomposition.

        p     = Ŵ · d                              # (out,)
        Ŵ_abl = Ŵ − strength · p ⊗ d
        Ŵ_new = normalize_rows(Ŵ_abl)
        W_new = diag(M) · Ŵ_new
    """
    d = _normalise(refusal.to(dtype=torch.float32))
    W32 = W.to(dtype=torch.float32)
    out_f, in_f = W32.shape

    if d.shape[0] == out_f:
        # Output-side: column-wise decomposition.
        N = torch.linalg.vector_norm(W32, dim=0, keepdim=True).clamp(min=1e-8)
        W_hat = W32 / N  # (out, in), per-column unit
        p = d @ W_hat  # (in,)
        W_hat_ablated = W_hat - strength * d.unsqueeze(1) * p.unsqueeze(0)
        new_norms = torch.linalg.vector_norm(W_hat_ablated, dim=0, keepdim=True).clamp(
            min=1e-8
        )
        W_hat_new = W_hat_ablated / new_norms
        W_new = W_hat_new * N
    elif d.shape[0] == in_f:
        # Input-side: row-wise decomposition (original behaviour).
        M = torch.linalg.vector_norm(W32, dim=1, keepdim=True).clamp(min=1e-8)
        W_hat = W32 / M  # (out, in), per-row unit
        p = W_hat @ d  # (out,)
        W_hat_ablated = W_hat - strength * p.unsqueeze(1) * d.unsqueeze(0)
        new_norms = torch.linalg.vector_norm(W_hat_ablated, dim=1, keepdim=True).clamp(
            min=1e-8
        )
        W_hat_new = W_hat_ablated / new_norms
        W_new = M * W_hat_new
    else:
        raise ValueError(
            f"Biprojected direction shape {d.shape} does not match either "
            f"axis of W shape {W32.shape}."
        )

    return W_new.to(W.dtype)


def apply_householder_transform(
    W: Tensor,
    refusal: Tensor,
    benign_dir: Tensor,
    *,
    strength: float = 1.0,
) -> Tensor:
    """Exact isometric reflection along the (orthogonalised) refusal direction.

    Two variants depending on which axis of W the direction matches:

    Output-side (preferred for residual-stream-write modules where v matches
    ``W.shape[0]``):

        H = I − 2 û ûᵀ                      (acts on output space)
        W_new = (I − 2 û ûᵀ) W = W − 2 û (û · W)

    Input-side (when v only matches ``W.shape[1]``):

        H = I − 2 û ûᵀ                      (acts on input space)
        W_new = W (I − 2 û ûᵀ) = W − 2 (W û) ûᵀ

    With ``strength = 1.0`` either is an isometry (norm-preserving by
    construction). With ``strength < 1.0`` the reflection is interpolated
    toward the identity, useful for partial ablation. grimjim observed
    token-level glitches at full strength on some checkpoints — keep it as
    an opt-in knob, not the default.

    Prefer output-side when v matches both axes (square modules) to match
    the LoRA / standard / ORBA semantics.
    """
    u_hat = double_gram_schmidt(refusal, benign_dir)
    W32 = W.to(dtype=torch.float32)
    out_f, in_f = W32.shape

    if u_hat.shape[0] == out_f:
        new_W = W32 - 2.0 * strength * u_hat.unsqueeze(1) * (u_hat @ W32).unsqueeze(0)
    elif u_hat.shape[0] == in_f:
        new_W = W32 - 2.0 * strength * (W32 @ u_hat).unsqueeze(1) * u_hat.unsqueeze(0)
    else:
        raise ValueError(
            f"Householder direction shape {u_hat.shape} does not match "
            f"either axis of W shape {W32.shape}."
        )
    return new_W.to(W.dtype)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def apply_direct_transform(
    transform: DirectTransform | str,
    W: Tensor,
    refusal: Tensor,
    benign_dir: Tensor | None,
    *,
    strength: float = 1.0,
    preserve_row_norm: bool = True,
    preserve_projected_norm: bool = False,
) -> Tensor:
    """Route ``W`` through the requested transform.

    ``benign_dir`` is required for ORBA and Householder (they use it for
    the orthogonalisation pre-step) and ignored for STANDARD / BIPROJECTED.
    """
    t = DirectTransform(transform) if isinstance(transform, str) else transform

    if t == DirectTransform.STANDARD:
        return apply_standard_transform(
            W,
            refusal,
            strength=strength,
            preserve_projected_norm=preserve_projected_norm,
        )
    if t == DirectTransform.ORBA:
        if benign_dir is None:
            raise ValueError(
                "ORBA requires benign_dir for double-GS orthogonalisation."
            )
        return apply_orba_transform(
            W,
            refusal,
            benign_dir,
            strength=strength,
            preserve_row_norm=preserve_row_norm,
            preserve_projected_norm=preserve_projected_norm,
        )
    if t == DirectTransform.BIPROJECTED:
        return apply_biprojected_transform(W, refusal, strength=strength)
    if t == DirectTransform.HOUSEHOLDER:
        if benign_dir is None:
            raise ValueError(
                "Householder requires benign_dir for double-GS orthogonalisation."
            )
        return apply_householder_transform(W, refusal, benign_dir, strength=strength)
    raise ValueError(f"Unknown direct transform: {t!r}")
