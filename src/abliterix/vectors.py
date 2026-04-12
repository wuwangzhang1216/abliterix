# Abliterix — a derivative work of Heretic (https://github.com/p-e-w/heretic)
# Original work Copyright (C) 2025  Philipp Emanuel Weidmann (p-e-w)
# Modified work Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Steering-vector computation from paired residual streams.

Supports multiple derivation strategies (mean difference, median-of-means,
PCA, optimal transport) and optional orthogonal projection against the
benign-direction component, including the improved projected abliteration
variant that preserves helpfulness-aligned components.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from .types import VectorMethod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _winsorize(vector: Tensor, quantile: float = 0.995) -> Tensor:
    """Symmetric magnitude winsorization (grimjim's method).

    Clips per-element magnitudes at the given quantile of ``|vector|`` to
    reduce the influence of extreme activations on the steering direction.
    """
    abs_v = torch.abs(vector.float())
    threshold = torch.quantile(abs_v, quantile)
    return torch.clamp(vector.float(), min=-threshold, max=threshold).to(vector.dtype)


def _compute_ot_transform(
    benign_states: Tensor,
    target_states: Tensor,
    n_components: int = 2,
) -> Tensor:
    """Compute PCA–Gaussian Optimal Transport steering vectors.

    Instead of a simple mean-difference direction, this method computes a
    per-layer affine transformation ``T(x) = Ax + b`` that maps the harmful
    activation distribution onto the harmless one, capturing both mean *and*
    covariance structure differences.

    Parameters
    ----------
    benign_states, target_states : Tensor
        Shape ``(n, layers+1, hidden_dim)``.
    n_components : int
        PCA dimensionality for the reduced space (typically 1–2).

    Returns
    -------
    Tensor
        Unit-normalised direction vectors, shape ``(layers+1, hidden_dim)``.
    """
    n_layers = benign_states.shape[1]
    benign_states.shape[2]
    per_layer = []

    for layer_idx in range(n_layers):
        b = benign_states[:, layer_idx, :].float()
        t = target_states[:, layer_idx, :].float()

        # Mean difference in full space (used as primary direction).
        mu_b = b.mean(dim=0)
        mu_t = t.mean(dim=0)
        diff = mu_t - mu_b

        # PCA on the concatenated centred activations to find the dominant
        # subspace shared by both distributions.
        combined = torch.cat([b - mu_b, t - mu_t], dim=0)
        _, _, Vh = torch.linalg.svd(combined, full_matrices=False)
        P = Vh[:n_components]  # (k, d)

        # Project into reduced space.
        b_proj = (b - mu_b) @ P.T  # (n_b, k)
        t_proj = (t - mu_t) @ P.T  # (n_t, k)

        # Covariance in reduced space.
        cov_b = (b_proj.T @ b_proj) / max(b_proj.shape[0] - 1, 1)
        cov_t = (t_proj.T @ t_proj) / max(t_proj.shape[0] - 1, 1)

        # Regularise for numerical stability.
        eye = torch.eye(n_components, device=b.device, dtype=b.dtype) * 1e-6
        cov_b = cov_b + eye
        cov_t = cov_t + eye

        # Closed-form Gaussian OT: T(x) = A x + b where
        #   A = Σ_b^{-1/2} (Σ_b^{1/2} Σ_t Σ_b^{1/2})^{1/2} Σ_b^{-1/2}
        # The direction in full space combines the mean shift with the
        # covariance-corrected displacement.
        Lb = torch.linalg.cholesky(cov_b)
        M = Lb @ cov_t @ Lb.T
        eigvals, eigvecs = torch.linalg.eigh(M)
        M_sqrt = eigvecs @ torch.diag(torch.sqrt(eigvals.clamp(min=1e-8))) @ eigvecs.T
        Lb_inv = torch.linalg.inv(Lb)
        A_reduced = Lb_inv @ M_sqrt @ Lb_inv  # (k, k)

        # Lift direction back to full space: the steering direction is the
        # mean shift plus a covariance-alignment correction.
        covariance_correction = (
            A_reduced - torch.eye(n_components, device=b.device, dtype=b.dtype)
        ) @ P  # (k, d)
        # The overall direction in full space.
        direction = diff + covariance_correction.mean(dim=0)
        per_layer.append(direction)

    return F.normalize(torch.stack(per_layer, dim=0), p=2, dim=1)


def _extract_multi_directions(
    benign_states: Tensor,
    target_states: Tensor,
    n_directions: int = 3,
) -> Tensor:
    """Extract top-k independent refusal directions via SVD.

    Instead of a single mean-difference vector, decomposes the refusal
    subspace into multiple orthogonal components.  The returned tensor has
    shape ``(n_directions, layers+1, hidden_dim)`` — each slice is a
    unit-normalised direction.
    """
    n_layers = benign_states.shape[1]
    all_dirs = []

    for k in range(n_directions):
        per_layer = []
        for layer_idx in range(n_layers):
            diff = (
                target_states[:, layer_idx, :].float()
                - benign_states[:, layer_idx, :].float()
            )
            diff = diff - diff.mean(dim=0, keepdim=True)
            _, _, Vh = torch.linalg.svd(diff, full_matrices=False)
            per_layer.append(Vh[k] if k < Vh.shape[0] else torch.zeros_like(Vh[0]))
        all_dirs.append(F.normalize(torch.stack(per_layer, dim=0), p=2, dim=1))

    return torch.stack(all_dirs, dim=0)


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------


def compute_steering_vectors(
    benign_states: Tensor,
    target_states: Tensor,
    method: VectorMethod,
    orthogonal_projection: bool,
    *,
    winsorize: bool = False,
    winsorize_quantile: float = 0.995,
    projected_abliteration: bool = False,
    ot_components: int = 2,
    n_directions: int = 1,
    token_position_split: bool = False,
    sra_base_method: VectorMethod | None = None,
    sra_n_atoms: int = 8,
    sra_ridge_alpha: float = 0.01,
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
    winsorize : bool
        If True, apply symmetric magnitude winsorization to the raw
        direction vectors before projection, reducing outlier influence.
    winsorize_quantile : float
        Quantile for winsorization (default 0.995 per grimjim's method).
    projected_abliteration : bool
        If True, use the improved projected-abliteration technique that
        only removes the component of the refusal direction orthogonal to
        the harmless mean, preserving helpfulness-aligned signals.
        Overrides ``orthogonal_projection`` when enabled.
    ot_components : int
        Number of PCA components for optimal-transport method (default 2).
    n_directions : int
        Number of independent refusal directions to extract (>1 enables
        multi-direction mode, returning shape ``(n_dirs, layers+1, dim)``).
    token_position_split : bool
        Reserved for future harmfulness/refusal token-position separation.

    Returns
    -------
    Tensor
        Unit-normalised steering vectors.  Shape ``(layers+1, hidden_dim)``
        when ``n_directions == 1``, otherwise ``(n_dirs, layers+1, dim)``.
    """

    # --- Multi-direction mode ---
    if n_directions > 1:
        directions = _extract_multi_directions(
            benign_states, target_states, n_directions
        )
        # Apply projection to each direction independently.
        if projected_abliteration or orthogonal_projection:
            benign_mean = benign_states.mean(dim=0)
            benign_dir = F.normalize(benign_mean, p=2, dim=1)
            projected = []
            for i in range(n_directions):
                v = directions[i]
                if projected_abliteration:
                    proj_scalar = torch.sum(v * benign_dir, dim=1, keepdim=True)
                    v = v - proj_scalar * benign_dir
                else:
                    proj_scalar = torch.sum(v * benign_dir, dim=1, keepdim=True)
                    v = v - proj_scalar * benign_dir
                projected.append(F.normalize(v, p=2, dim=1))
            return torch.stack(projected, dim=0)
        return directions

    # --- Single-direction methods ---

    if method == VectorMethod.COSMIC:
        from .cosmic import select_cosmic_direction

        vectors, _target_layers = select_cosmic_direction(
            benign_states,
            target_states,
        )
        # COSMIC already returns normalised vectors; skip to projection.
        if projected_abliteration:
            benign_mean = benign_states.mean(dim=0)
            benign_dir = F.normalize(benign_mean.float(), p=2, dim=1)
            proj_scalar = torch.sum(vectors.float() * benign_dir, dim=1, keepdim=True)
            vectors = vectors.float() - proj_scalar * benign_dir
            vectors = F.normalize(vectors, p=2, dim=1)
        elif orthogonal_projection:
            benign_dir = F.normalize(benign_states.mean(dim=0), p=2, dim=1)
            proj = torch.sum(vectors * benign_dir, dim=1)
            vectors = vectors - proj.unsqueeze(1) * benign_dir
            vectors = F.normalize(vectors, p=2, dim=1)
        return vectors

    if method == VectorMethod.SRA:
        from .sra import compute_sra_vectors

        vectors = compute_sra_vectors(
            benign_states,
            target_states,
            base_method=sra_base_method or VectorMethod.MEAN,
            n_atoms=sra_n_atoms,
            ridge_alpha=sra_ridge_alpha,
            orthogonal_projection=orthogonal_projection,
            projected_abliteration=projected_abliteration,
            winsorize=winsorize,
            winsorize_quantile=winsorize_quantile,
            ot_components=ot_components,
        )
        # SRA already returns cleaned, normalised vectors.
        return vectors

    if method == VectorMethod.OPTIMAL_TRANSPORT:
        vectors = _compute_ot_transform(
            benign_states, target_states, n_components=ot_components
        )

    elif method == VectorMethod.MEDIAN_OF_MEANS:
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

    # --- Winsorization ---
    if winsorize:
        vectors = _winsorize(vectors, quantile=winsorize_quantile)
        vectors = F.normalize(vectors, p=2, dim=1)

    # --- Projection ---
    if projected_abliteration:
        # Improved method (grimjim): only remove the orthogonal component
        # of the refusal direction relative to the harmless mean.  The
        # parallel component represents helpfulness variation and is kept.
        #
        #   r = target_mean - benign_mean
        #   r_parallel = (r · μ̂_A) μ̂_A
        #   r_projected = r - r_parallel   (the orthogonal residual)
        #
        # We ablate r_projected (not the full r) to preserve helpfulness.
        benign_mean = benign_states.mean(dim=0)
        benign_dir = F.normalize(benign_mean.float(), p=2, dim=1)
        proj_scalar = torch.sum(vectors.float() * benign_dir, dim=1, keepdim=True)
        vectors = vectors.float() - proj_scalar * benign_dir
        vectors = F.normalize(vectors, p=2, dim=1)

    elif orthogonal_projection:
        benign_dir = F.normalize(benign_states.mean(dim=0), p=2, dim=1)
        proj = torch.sum(vectors * benign_dir, dim=1)
        vectors = vectors - proj.unsqueeze(1) * benign_dir
        vectors = F.normalize(vectors, p=2, dim=1)

    return vectors


# ---------------------------------------------------------------------------
# Iterative abliteration helpers
# ---------------------------------------------------------------------------


def orthogonalize_against(
    new_dirs: Tensor,
    previous_dirs: list[Tensor],
    norm_threshold: float = 0.1,
) -> Tensor:
    """Gram-Schmidt orthogonalise *new_dirs* against all *previous_dirs* per layer.

    Directions whose residual L2 norm (relative to the first direction ever
    extracted) falls below *norm_threshold* are zeroed out — they carry no
    new information.

    Parameters
    ----------
    new_dirs : Tensor
        Shape ``(k, layers+1, hidden_dim)`` — freshly extracted directions.
    previous_dirs : list[Tensor]
        Each element has shape ``(k_i, layers+1, hidden_dim)``.
    norm_threshold : float
        Minimum relative norm to keep a direction.

    Returns
    -------
    Tensor
        Orthogonalised directions, same shape as *new_dirs*.  Some slices may
        be all-zeros if they fell below the threshold.
    """
    if not previous_dirs:
        return new_dirs

    # Stack all previous directions: (N_prev, layers+1, hidden_dim)
    prev = torch.cat(previous_dirs, dim=0)
    n_layers = new_dirs.shape[1]
    result = new_dirs.clone().float()

    for layer_idx in range(n_layers):
        P = prev[:, layer_idx, :].float()  # (N_prev, d)
        for k in range(result.shape[0]):
            v = result[k, layer_idx, :]  # (d,)
            # Project out each previous direction.
            for j in range(P.shape[0]):
                p = P[j]
                p_norm_sq = torch.dot(p, p)
                if p_norm_sq > 1e-12:
                    v = v - (torch.dot(v, p) / p_norm_sq) * p
            result[k, layer_idx, :] = v

    # Check norms and zero out weak directions.
    for k in range(result.shape[0]):
        layer_norms = result[k].norm(p=2, dim=1)  # (layers+1,)
        if layer_norms.mean().item() < norm_threshold:
            result[k] = 0.0
        else:
            result[k] = F.normalize(result[k], p=2, dim=1)

    return result.to(new_dirs.dtype)


def build_subspace_basis(directions: list[Tensor]) -> Tensor:
    """Combine directions from multiple iterations into an orthonormal basis.

    Uses per-layer QR decomposition to produce a minimal set of orthogonal
    directions spanning the full refusal subspace discovered across all
    iterations.

    Parameters
    ----------
    directions : list[Tensor]
        Each element has shape ``(k_i, layers+1, hidden_dim)``.

    Returns
    -------
    Tensor
        Orthonormal basis, shape ``(rank, layers+1, hidden_dim)``.
    """
    stacked = torch.cat(directions, dim=0).float()  # (N, layers+1, d)
    n_dirs, n_layers, hidden_dim = stacked.shape

    # Remove all-zero directions.
    active_mask = stacked.view(n_dirs, -1).norm(p=2, dim=1) > 1e-8
    stacked = stacked[active_mask]
    if stacked.shape[0] == 0:
        return stacked

    # Per-layer QR to find the rank of the refusal subspace.
    per_layer_basis = []
    max_rank = 0
    for layer_idx in range(n_layers):
        V = stacked[:, layer_idx, :].T  # (d, N)
        Q, R = torch.linalg.qr(V)
        # Keep columns with non-negligible R diagonal.
        diag = torch.abs(torch.diag(R))
        rank = int((diag > 1e-6).sum().item())
        max_rank = max(max_rank, rank)
        per_layer_basis.append(Q[:, :rank])  # (d, rank_l)

    # Pad to uniform rank and assemble.
    result = torch.zeros(max_rank, n_layers, hidden_dim, dtype=stacked.dtype, device=stacked.device)
    for layer_idx, Q in enumerate(per_layer_basis):
        r = Q.shape[1]
        result[:r, layer_idx, :] = Q[:, :r].T  # (r, d)

    # Re-normalise each direction.
    norms = result.norm(p=2, dim=2, keepdim=True).clamp(min=1e-8)
    result = result / norms

    return result.to(directions[0].dtype)
