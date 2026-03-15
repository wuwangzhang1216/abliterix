# Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>

"""Steering-vector computation from paired residual streams.

Supports three derivation strategies (mean difference, median-of-means,
PCA) and optional orthogonal projection against the benign-direction
component.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from .types import VectorMethod


def compute_steering_vectors(
    benign_states: Tensor,
    target_states: Tensor,
    method: VectorMethod,
    orthogonal_projection: bool,
) -> Tensor:
    """Derive per-layer steering vectors from benign and target residuals.

    Parameters
    ----------
    benign_states : Tensor
        Shape ``(n_benign, layers+1, hidden_dim)`` — residuals from benign prompts.
    target_states : Tensor
        Shape ``(n_target, layers+1, hidden_dim)`` — residuals from target prompts.
    method : VectorMethod
        Strategy for combining per-prompt differences into a single vector.
    orthogonal_projection : bool
        If True, remove the benign-mean component so that only the
        safety-specific signal remains.

    Returns
    -------
    Tensor
        Unit-normalised steering vectors, shape ``(layers+1, hidden_dim)``.
    """

    if method == VectorMethod.MEDIAN_OF_MEANS:
        n_groups = 5
        b_chunks = torch.chunk(benign_states, n_groups, dim=0)
        t_chunks = torch.chunk(target_states, n_groups, dim=0)
        group_dirs = torch.stack(
            [
                F.normalize(tc.mean(dim=0) - bc.mean(dim=0), p=2, dim=1)
                for bc, tc in zip(b_chunks, t_chunks)
            ],
            dim=0,
        )
        vectors = F.normalize(group_dirs.median(dim=0).values, p=2, dim=1)

    elif method == VectorMethod.PCA:
        diff = target_states - benign_states.mean(dim=0, keepdim=True)
        n_layers = diff.shape[1]
        per_layer = []
        for layer_idx in range(n_layers):
            d = diff[:, layer_idx, :]
            d = d - d.mean(dim=0, keepdim=True)
            _, _, Vh = torch.linalg.svd(d, full_matrices=False)
            per_layer.append(Vh[0])
        vectors = F.normalize(torch.stack(per_layer, dim=0), p=2, dim=1)

    else:  # MEAN (default)
        vectors = F.normalize(
            target_states.mean(dim=0) - benign_states.mean(dim=0),
            p=2,
            dim=1,
        )

    if orthogonal_projection:
        benign_dir = F.normalize(benign_states.mean(dim=0), p=2, dim=1)
        proj = torch.sum(vectors * benign_dir, dim=1)
        vectors = vectors - proj.unsqueeze(1) * benign_dir
        vectors = F.normalize(vectors, p=2, dim=1)

    return vectors
