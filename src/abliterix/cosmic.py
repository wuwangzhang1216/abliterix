# Abliterix — a derivative work of Heretic (https://github.com/p-e-w/heretic)
# Modified work Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""COSMIC: Cosine Similarity Metrics for Inversion of Concepts.

Automated steering-direction and target-layer selection using activation-space
cosine similarity, independent of output text analysis.

Based on: Siu et al., "COSMIC: Generalized Refusal Direction Identification
in LLM Activations", ACL 2025 Findings.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from .util import print


def _extract_candidate_directions(
    benign_states: Tensor,
    target_states: Tensor,
    n_token_positions: int = 5,
) -> tuple[Tensor, list[tuple[int, int]]]:
    """Extract candidate refusal directions from multiple token positions.

    For each layer, computes the mean-difference direction at each of the
    last ``n_token_positions`` token positions (simulated by splitting the
    sample dimension, since our residuals are already position-aggregated).

    For our architecture where residuals are per-prompt (not per-token),
    we generate candidates by sub-sampling groups.

    Returns
    -------
    candidates : Tensor
        Shape ``(n_candidates, hidden_dim)`` — unit-normalised directions.
    indices : list of (layer_idx, group_idx) tuples
        Maps each candidate to its source layer and sub-group.
    """
    n_layers = benign_states.shape[1]
    # torch.chunk raises an opaque "chunks must be greater than 0" for an
    # empty prompt set; surface a clear error instead.
    if benign_states.shape[0] == 0 or target_states.shape[0] == 0:
        raise ValueError(
            "COSMIC direction selection requires at least one benign and one "
            "target prompt."
        )
    n_groups = min(n_token_positions, benign_states.shape[0])
    b_chunks = torch.chunk(benign_states, n_groups, dim=0)
    t_chunks = torch.chunk(target_states, n_groups, dim=0)

    candidates = []
    indices = []

    for layer_idx in range(n_layers):
        for g, (bc, tc) in enumerate(zip(b_chunks, t_chunks)):
            direction = tc[:, layer_idx, :].mean(dim=0) - bc[:, layer_idx, :].mean(
                dim=0
            )
            direction = F.normalize(direction.float(), p=2, dim=0)
            candidates.append(direction)
            indices.append((layer_idx, g))

    return torch.stack(candidates, dim=0), indices


def _compute_layer_discriminability(
    benign_states: Tensor,
    target_states: Tensor,
) -> Tensor:
    """Compute per-layer cosine similarity between harmful/harmless mean activations.

    Layers with *lower* cosine similarity encode stronger refusal-specific
    representations.  Returns shape ``(n_layers,)``.
    """
    b_mean = benign_states.mean(dim=0).float()  # (layers, hidden)
    t_mean = target_states.mean(dim=0).float()
    cos_sim = F.cosine_similarity(b_mean, t_mean, dim=1)  # (layers,)
    return cos_sim


def _score_candidate(
    candidate: Tensor,
    benign_acts: Tensor,
    target_acts: Tensor,
) -> float:
    """Score a candidate direction using COSMIC's cosine similarity metric.

    The score combines two signals:
    1. S_refuse: similarity between (harmless + induced refusal) and (naturally refused)
    2. S_comply: similarity between (harmful + ablated) and (naturally compliant)

    Approximated using projections onto the candidate direction.

    Parameters
    ----------
    candidate : Tensor
        Unit-normalised direction, shape ``(hidden_dim,)``.
    benign_acts, target_acts : Tensor
        Mean activations at the evaluation layers, shape ``(hidden_dim,)``.
    """
    c = candidate.float()
    b = benign_acts.float()
    t = target_acts.float()

    # Simulate directional ablation: remove candidate component from target.
    t_ablated = t - (t @ c) * c
    # Simulate activation addition: add candidate component to benign.
    b_induced = b + (t @ c) * c

    # S_refuse: induced-refusal benign ↔ natural target.
    s_refuse = F.cosine_similarity(b_induced.unsqueeze(0), t.unsqueeze(0)).item()
    # S_comply: ablated target ↔ natural benign.
    s_comply = F.cosine_similarity(t_ablated.unsqueeze(0), b.unsqueeze(0)).item()

    return s_refuse + s_comply


def select_cosmic_direction(
    benign_states: Tensor,
    target_states: Tensor,
    bottom_pct: float = 0.10,
) -> tuple[Tensor, list[int]]:
    """Select optimal steering directions and target layers using COSMIC.

    Parameters
    ----------
    benign_states : Tensor
        Shape ``(n_benign, layers+1, hidden_dim)``.
    target_states : Tensor
        Shape ``(n_target, layers+1, hidden_dim)``.
    bottom_pct : float
        Fraction of layers (by cosine similarity) to use as evaluation set.

    Returns
    -------
    vectors : Tensor
        Per-layer unit-normalised steering vectors, shape ``(layers+1, hidden_dim)``.
        For non-selected layers, falls back to the best global direction.
    target_layers : list of int
        Indices of the most discriminative layers (bottom_pct by cosine similarity).
    """
    n_layers = benign_states.shape[1]

    # Step 1: Identify evaluation layers (lowest cosine similarity = strongest refusal encoding).
    cos_sim = _compute_layer_discriminability(benign_states, target_states)
    n_eval = max(1, int(n_layers * bottom_pct))
    eval_layer_indices = torch.topk(cos_sim, n_eval, largest=False).indices.tolist()

    print(
        f"* COSMIC: {n_eval} evaluation layers selected (indices: {eval_layer_indices})"
    )

    # Step 2: Extract candidate directions.
    candidates, indices = _extract_candidate_directions(benign_states, target_states)

    print(f"* COSMIC: scoring {len(candidates)} candidate directions...")

    # Step 3: Score each candidate on the evaluation layers.
    best_score = float("-inf")
    best_idx = 0

    for i, (candidate, (layer_idx, _)) in enumerate(zip(candidates, indices)):
        score = 0.0
        for eval_layer in eval_layer_indices:
            b_mean = benign_states[:, eval_layer, :].mean(dim=0)
            t_mean = target_states[:, eval_layer, :].mean(dim=0)
            score += _score_candidate(candidate, b_mean, t_mean)
        score /= len(eval_layer_indices)

        if score > best_score:
            best_score = score
            best_idx = i

    best_layer, best_group = indices[best_idx]
    best_direction = candidates[best_idx]

    print(
        f"* COSMIC: best direction from layer {best_layer}, "
        f"group {best_group} (score: {best_score:.4f})"
    )

    # Step 4: Build per-layer vectors.
    # Use the best global direction as default, but also compute per-layer
    # directions for layers in the evaluation set.
    vectors = best_direction.unsqueeze(0).expand(n_layers, -1).clone()

    # For evaluation layers, compute layer-specific directions and pick
    # the best candidate per layer.
    for eval_layer in eval_layer_indices:
        layer_best_score = float("-inf")
        layer_best_dir = best_direction

        for i, (candidate, (layer_idx, _)) in enumerate(zip(candidates, indices)):
            if layer_idx != eval_layer:
                continue
            b_mean = benign_states[:, eval_layer, :].mean(dim=0)
            t_mean = target_states[:, eval_layer, :].mean(dim=0)
            s = _score_candidate(candidate, b_mean, t_mean)
            if s > layer_best_score:
                layer_best_score = s
                layer_best_dir = candidate

        vectors[eval_layer] = layer_best_dir

    vectors = F.normalize(vectors, p=2, dim=1)

    return vectors, eval_layer_indices
