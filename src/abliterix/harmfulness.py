# Abliterix
# Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Joint harmfulness + refusal direction extraction.

Implements the dual-direction decomposition from
`Zhao et al., 2025 <https://arxiv.org/abs/2507.11878>`_ — *LLMs Encode
Harmfulness and Refusal Separately*.

The standard mean-difference vector `mean(target) - mean(benign)` conflates
two distinct circuits:

* **Refusal direction** — controls whether the model voices a refusal.
  Dominant in *late* layers, closer to the output.
* **Harmfulness direction** — controls the model's internal judgment that
  the prompt is harmful.  Dominant in *mid* layers (the model decides
  before it speaks).

Ablating only the refusal direction can leave the harmfulness signal intact,
producing models that comply but still hedge ("I will answer this even
though it is harmful..."). Joint ablation of both directions removes the
hedging by also flattening the internal judgment.

This module is opt-in (see ``SteeringConfig.ablate_harmfulness_direction``)
and reuses the existing multi-direction stacking infrastructure: the output
shape is ``(2, layers+1, hidden_dim)`` so downstream code paths that already
handle ``n_directions > 1`` work without modification.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def _refusal_direction(
    benign_states: Tensor,
    target_states: Tensor,
) -> Tensor:
    """Standard mean-diff refusal direction, per layer.

    Returns
    -------
    Tensor
        Shape ``(layers+1, hidden_dim)``, unit-normalised per layer.
    """
    diff = target_states.mean(dim=0) - benign_states.mean(dim=0)
    return F.normalize(diff.float(), p=2, dim=1)


def _harmfulness_direction(
    benign_states: Tensor,
    target_states: Tensor,
    refusal: Tensor,
    layer_band: tuple[float, float] = (0.3, 0.7),
) -> Tensor:
    """Extract the harmfulness direction orthogonal to the refusal direction.

    Strategy
    --------
    1. Identify a mid-layer band ``[layer_band[0] * L, layer_band[1] * L]``
       where Zhao et al. show the harmfulness judgment crystallises.
    2. At each layer, compute the dominant axis of variation among
       *harmful* prompts (PCA-1 of the centred ``target_states`` slice).
       This captures "how harmful is this prompt" rather than "do I refuse".
    3. Orthogonalise that direction against the refusal direction at the
       same layer, so the two vectors span complementary subspaces.
    4. For layers outside the mid band, fall back to the per-layer PCA-1
       (still orthogonalised against refusal) so every layer has a usable
       harmfulness direction — the band only changes *which layers carry
       the strongest signal*, not which layers are editable.

    Parameters
    ----------
    benign_states, target_states : Tensor
        Shape ``(n, layers+1, hidden_dim)``.
    refusal : Tensor
        Shape ``(layers+1, hidden_dim)``, unit-normalised — output of
        :func:`_refusal_direction`.
    layer_band : tuple[float, float]
        Fractional range of layers in which the harmfulness signal is
        strongest.  Default ``(0.3, 0.7)`` matches the mid-layer band
        identified by Zhao et al. for Llama-3 / Qwen-2 class models.

    Returns
    -------
    Tensor
        Shape ``(layers+1, hidden_dim)``, unit-normalised per layer.
        Layers where the orthogonalisation collapses the vector below a
        tiny epsilon are returned as zeros (and skipped downstream).
    """
    n_layers = target_states.shape[1]
    lo = max(0, int(layer_band[0] * n_layers))
    hi = min(n_layers, max(lo + 1, int(layer_band[1] * n_layers)))

    per_layer = []
    for layer_idx in range(n_layers):
        t = target_states[:, layer_idx, :].float()
        # Centre target activations so PCA picks up the intra-target variation,
        # not the mean shift (which is already the refusal direction).
        t_centred = t - t.mean(dim=0, keepdim=True)

        if t_centred.shape[0] < 2:
            # Not enough samples — fall back to a zero vector (handled below).
            per_layer.append(torch.zeros_like(refusal[layer_idx]))
            continue

        # PCA-1 of centred target = dominant intra-harmful variation.
        try:
            _, _, Vh = torch.linalg.svd(t_centred, full_matrices=False)
            v = Vh[0]
        except RuntimeError:
            # SVD can fail on degenerate inputs (e.g. all-zero rows after
            # heavy quantisation). Fall back to zero so the layer is skipped.
            per_layer.append(torch.zeros_like(refusal[layer_idx]))
            continue

        # Outside the mid-band: dampen this layer's harmfulness signal so
        # the optimiser concentrates its budget on mid-layer steering.
        if not (lo <= layer_idx < hi):
            v = v * 0.5

        # Orthogonalise against the layer's refusal direction so the two
        # spans cover complementary subspaces.
        r = refusal[layer_idx]
        v = v - torch.dot(v, r) * r

        # If orthogonalisation collapsed the vector (refusal already
        # absorbed all intra-harmful variation at this layer), zero it
        # out so downstream code skips it.
        norm = torch.linalg.vector_norm(v)
        if norm < 1e-6:
            per_layer.append(torch.zeros_like(r))
        else:
            per_layer.append(v / norm)

    return torch.stack(per_layer, dim=0)


def extract_harm_refusal_pair(
    benign_states: Tensor,
    target_states: Tensor,
    *,
    layer_band: tuple[float, float] = (0.3, 0.7),
    orthogonal_projection: bool = False,
    projected_abliteration: bool = False,
) -> Tensor:
    """Build the stacked (refusal, harmfulness) direction pair.

    The output slots in directly to the existing multi-direction code path
    in :func:`abliterix.vectors.compute_steering_vectors` — shape
    ``(2, layers+1, hidden_dim)`` with ``vectors[0]`` being refusal and
    ``vectors[1]`` being the orthogonalised harmfulness component.

    Parameters
    ----------
    benign_states, target_states : Tensor
        Shape ``(n, layers+1, hidden_dim)``.
    layer_band : tuple[float, float]
        Fractional layer range where the harmfulness direction is strongest.
    orthogonal_projection : bool
        If True, after extraction also project both directions against the
        benign mean (standard ortho-projection behaviour).
    projected_abliteration : bool
        If True, apply grimjim's helpfulness-preserving projection to both
        directions.  Takes precedence over ``orthogonal_projection``.

    Returns
    -------
    Tensor
        Shape ``(2, layers+1, hidden_dim)``.  ``[0]`` is the refusal
        direction, ``[1]`` is the harmfulness direction orthogonal to it.
    """
    refusal = _refusal_direction(benign_states, target_states)
    harmfulness = _harmfulness_direction(
        benign_states, target_states, refusal, layer_band=layer_band
    )

    directions = torch.stack([refusal, harmfulness], dim=0)

    if projected_abliteration or orthogonal_projection:
        benign_mean = benign_states.mean(dim=0)
        benign_dir = F.normalize(benign_mean.float(), p=2, dim=1)
        for i in range(directions.shape[0]):
            v = directions[i]
            proj_scalar = torch.sum(v * benign_dir, dim=1, keepdim=True)
            v = v - proj_scalar * benign_dir
            directions[i] = F.normalize(v, p=2, dim=1)

    return directions.to(benign_states.dtype)
