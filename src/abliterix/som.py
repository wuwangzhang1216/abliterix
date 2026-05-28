# Abliterix
# Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Self-Organising Map (SOM) refusal directions.

Implements the multi-direction extraction from `Piras et al., AAAI 2026
<https://arxiv.org/abs/2511.08379>`_ — *SOM Directions Are Better Than One*.

The standard ``n_directions`` mode in abliterix extracts top-k SVD
components of the harmful-vs-benign difference matrix and orthogonalises
them via Gram-Schmidt. That forces orthogonality the data may not
actually have. SOM trains a small Kohonen grid on harmful representations
and uses each node's centroid (minus the benign mean) as a candidate
refusal direction. The resulting directions are **correlated**, not
orthogonal — capturing the low-dimensional manifold structure the paper
identifies.

Algorithm
---------
1. Run a per-layer Kohonen SOM (default 3×3 = 9 nodes) on the harmful
   activations. Each node converges to a region of the harmful manifold.
2. For each SOM node, compute its centroid as the mean of harmful samples
   that map to it (best-matching-unit assignments).
3. Refusal direction per node = unit-normalise(node centroid − benign mean).
4. Optionally apply orthogonal / projected abliteration projection against
   the benign mean to each direction (same post-step as the other vector
   methods).

The output shape is ``(n_dirs, layers+1, hidden_dim)`` matching the
existing multi-direction conventions, so all downstream LoRA / direct
paths work without changes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Kohonen SOM
# ---------------------------------------------------------------------------


def _train_som(
    data: Tensor,
    grid_h: int,
    grid_w: int,
    n_iters: int,
    initial_lr: float,
    initial_sigma: float,
    seed: int,
) -> Tensor:
    """Train a 2-D Kohonen SOM on ``data`` and return the trained codebook.

    Parameters
    ----------
    data : Tensor
        Shape ``(n_samples, hidden_dim)``, float32.
    grid_h, grid_w : int
        Topology of the SOM grid. Total nodes = ``grid_h * grid_w``.
    n_iters : int
        Total training iterations. Each iter picks one random sample.
    initial_lr : float
        Initial learning rate, decayed exponentially toward ``lr * 0.01``.
    initial_sigma : float
        Initial Gaussian neighbourhood radius (in grid units), decayed
        exponentially toward ``sigma * 0.5``.
    seed : int
        RNG seed for reproducibility (init + sample order).

    Returns
    -------
    Tensor
        Codebook of shape ``(grid_h * grid_w, hidden_dim)``.
    """
    n_samples, hidden_dim = data.shape
    n_nodes = grid_h * grid_w
    gen = torch.Generator(device="cpu").manual_seed(seed)

    # Initialise codebook from random samples (PCA init would be better,
    # but adds a torch.linalg.svd cost per layer we don't need here).
    perm = torch.randperm(n_samples, generator=gen)[:n_nodes]
    codebook = data[perm].clone().to(torch.float32)
    # Pad if data has fewer rows than nodes.
    if codebook.shape[0] < n_nodes:
        repeat = (n_nodes // codebook.shape[0]) + 1
        codebook = codebook.repeat(repeat, 1)[:n_nodes].clone()

    # Pre-compute grid coordinates: (n_nodes, 2).
    rows = torch.arange(grid_h).repeat_interleave(grid_w)
    cols = torch.arange(grid_w).repeat(grid_h)
    grid_coords = torch.stack([rows, cols], dim=1).to(
        device=data.device, dtype=torch.float32
    )

    lr_floor = initial_lr * 0.01
    sigma_floor = max(initial_sigma * 0.5, 0.5)

    for step in range(n_iters):
        t = step / max(1, n_iters - 1)
        lr = initial_lr * (lr_floor / initial_lr) ** t
        sigma = initial_sigma * (sigma_floor / initial_sigma) ** t

        # Sample a random data point.
        idx = torch.randint(0, n_samples, (1,), generator=gen).item()
        x = data[idx]

        # BMU = best-matching unit (closest node in L2).
        dists = torch.linalg.vector_norm(codebook - x, dim=1)
        bmu = int(torch.argmin(dists).item())

        # Gaussian neighbourhood around the BMU on the grid.
        bmu_coord = grid_coords[bmu]
        grid_dist_sq = ((grid_coords - bmu_coord) ** 2).sum(dim=1)
        h = torch.exp(-grid_dist_sq / (2.0 * sigma * sigma))  # (n_nodes,)

        codebook = codebook + lr * h.unsqueeze(1) * (x.unsqueeze(0) - codebook)

    return codebook


def _bmu_assignments(data: Tensor, codebook: Tensor) -> Tensor:
    """For each sample, return the BMU node index. Shape ``(n_samples,)``."""
    # Pairwise squared distances via expansion (fine for the small sizes here).
    diffs = data.unsqueeze(1) - codebook.unsqueeze(0)  # (n, n_nodes, dim)
    d2 = (diffs * diffs).sum(dim=2)
    return torch.argmin(d2, dim=1)


def _node_centroids(data: Tensor, codebook: Tensor) -> Tensor:
    """Compute the per-node centroid of samples assigned to that node.

    Falls back to the codebook entry for any node with no assignments —
    keeps the output shape constant even in pathological cases.

    Returns shape ``(n_nodes, hidden_dim)``.
    """
    n_nodes = codebook.shape[0]
    assigns = _bmu_assignments(data, codebook)
    centroids = torch.zeros_like(codebook)
    counts = torch.zeros(n_nodes, dtype=torch.long, device=data.device)
    centroids.index_add_(0, assigns, data)
    counts.index_add_(0, assigns, torch.ones_like(assigns, dtype=torch.long))

    # Replace empty-bucket centroids with the codebook entry.
    safe_counts = counts.clamp(min=1).unsqueeze(1).to(centroids.dtype)
    centroids = centroids / safe_counts
    empty_mask = counts == 0
    if empty_mask.any():
        centroids[empty_mask] = codebook[empty_mask]
    return centroids


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_som_directions(
    benign_states: Tensor,
    target_states: Tensor,
    *,
    grid_h: int = 3,
    grid_w: int = 3,
    n_iters: int = 500,
    initial_lr: float = 0.5,
    initial_sigma: float | None = None,
    seed: int = 0,
    orthogonal_projection: bool = False,
    projected_abliteration: bool = False,
) -> Tensor:
    """Compute SOM-based per-node refusal directions for every layer.

    Parameters
    ----------
    benign_states, target_states : Tensor
        Shape ``(n, layers+1, hidden_dim)`` — paired residual streams.
    grid_h, grid_w : int
        SOM topology. Total directions = ``grid_h * grid_w``.
    n_iters : int
        Training iterations per layer.
    initial_lr : float
        Initial Kohonen learning rate.
    initial_sigma : float, optional
        Initial neighbourhood radius. Defaults to ``max(grid_h, grid_w) / 2``.
    seed : int
        RNG seed for SOM init / sample order. The same seed is reused for
        every layer; combined with layer index for deterministic per-layer
        decorrelation.
    orthogonal_projection : bool
        If True, project the benign-mean direction out of every SOM
        direction post-extraction. Standard ortho behaviour.
    projected_abliteration : bool
        If True, apply grimjim's helpfulness-preserving projection (kept
        for parity with the other vector methods). Takes precedence over
        ``orthogonal_projection``.

    Returns
    -------
    Tensor
        Shape ``(n_dirs, layers+1, hidden_dim)`` where
        ``n_dirs = grid_h * grid_w``. Each per-layer slice is unit-norm.
    """
    if benign_states.shape[1:] != target_states.shape[1:]:
        raise ValueError(
            "benign_states and target_states must have the same "
            f"(layers, hidden_dim) shape, got {benign_states.shape} vs "
            f"{target_states.shape}."
        )

    n_layers = target_states.shape[1]
    hidden = target_states.shape[2]
    n_dirs = grid_h * grid_w
    if initial_sigma is None:
        initial_sigma = max(grid_h, grid_w) / 2.0

    benign_mean = benign_states.mean(dim=0).to(torch.float32)  # (layers+1, hidden)
    benign_dir = F.normalize(benign_mean, p=2, dim=1)

    out = torch.zeros(
        n_dirs, n_layers, hidden, dtype=torch.float32, device=target_states.device
    )
    for layer_idx in range(n_layers):
        target_layer = target_states[:, layer_idx, :].to(torch.float32)
        # Skip degenerate layers (e.g. all-zero rows) — fall back to mean-diff.
        if target_layer.shape[0] < n_dirs:
            diff = target_layer.mean(dim=0) - benign_mean[layer_idx]
            out[:, layer_idx, :] = F.normalize(diff, p=2, dim=0).unsqueeze(0)
            continue

        codebook = _train_som(
            target_layer,
            grid_h=grid_h,
            grid_w=grid_w,
            n_iters=n_iters,
            initial_lr=initial_lr,
            initial_sigma=initial_sigma,
            seed=seed + layer_idx,
        )
        centroids = _node_centroids(target_layer, codebook)
        # Direction = node centroid − benign mean (per the paper).
        directions = centroids - benign_mean[layer_idx].unsqueeze(0)
        out[:, layer_idx, :] = F.normalize(directions, p=2, dim=1)

    # Optional projection against the benign direction (mirrors the other
    # vector methods so the SOM-mode obeys the same projection knobs).
    if projected_abliteration or orthogonal_projection:
        for i in range(n_dirs):
            v = out[i]
            proj_scalar = torch.sum(v * benign_dir, dim=1, keepdim=True)
            v = v - proj_scalar * benign_dir
            out[i] = F.normalize(v, p=2, dim=1)

    return out.to(benign_states.dtype)
