# Abliterix — a derivative work of Heretic (https://github.com/p-e-w/heretic)
# Original work Copyright (C) 2025  Philipp Emanuel Weidmann (p-e-w)
# Modified work Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Weight baking for PCA-OT affine transformations.

This module provides utilities to permanently bake PCA-OT transformations
into model weights using low-rank decomposition, avoiding inference-time hooks.
"""

from typing import Optional

import torch
from torch import Tensor, nn

from .types import PCAOTTransform


def low_rank_decompose(
    A: Tensor, rank: Optional[int] = None
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute low-rank SVD decomposition of a matrix.

    Parameters
    ----------
    A : Tensor
        Matrix to decompose, shape (m, n).
    rank : int, optional
        Target rank for decomposition. If None, uses full rank.

    Returns
    -------
    U : Tensor
        Left singular vectors, shape (m, rank).
    S : Tensor
        Singular values, shape (rank,).
    Vt : Tensor
        Right singular vectors (transposed), shape (rank, n).

    Notes
    -----
    A ≈ U @ diag(S) @ Vt
    """
    U, S, Vt = torch.linalg.svd(A, full_matrices=False)

    if rank is not None:
        U = U[:, :rank]
        S = S[:rank]
        Vt = Vt[:rank, :]

    return U, S, Vt


def bake_affine_into_linear(
    linear: nn.Linear,
    A_full: Tensor,
    b_full: Tensor,
    rank: Optional[int] = None,
) -> nn.Linear:
    """Bake affine transformation into a linear layer's weights.

    Given transformation h' = A_full @ h + b_full and linear layer y = W @ h + b,
    computes modified layer y = W' @ h + b' where:
    - W' = W @ A_full (using rank-r approximation if rank is specified)
    - b' = b + W @ b_full

    Parameters
    ----------
    linear : nn.Linear
        Linear layer to modify (in_features must match A_full.shape[1]).
    A_full : Tensor
        Affine transformation matrix, shape (d, d).
    b_full : Tensor
        Affine bias vector, shape (d,).
    rank : int, optional
        Rank for low-rank approximation of A_full. If None, uses full matrix.

    Returns
    -------
    nn.Linear
        New linear layer with baked transformation.

    Example
    -------
    >>> linear = nn.Linear(128, 256)
    >>> transform = transforms[layer_idx]
    >>> linear_baked = bake_affine_into_linear(linear, transform.A_full, transform.b_full, rank=16)
    """
    device = linear.weight.device
    dtype = linear.weight.dtype

    A_full = A_full.to(device=device, dtype=dtype)
    b_full = b_full.to(device=device, dtype=dtype)

    # Verify dimensions
    d_in = linear.in_features
    d_out = linear.out_features

    if A_full.shape != (d_in, d_in):
        raise ValueError(
            f"A_full shape {A_full.shape} incompatible with linear layer "
            f"in_features={d_in}"
        )

    if b_full.shape != (d_in,):
        raise ValueError(
            f"b_full shape {b_full.shape} incompatible with linear layer "
            f"in_features={d_in}"
        )

    # Compute W' = W @ A_full (with optional low-rank approximation)
    if rank is not None:
        U, S, Vt = low_rank_decompose(A_full, rank=rank)
        # W @ A_full ≈ W @ (U @ diag(S) @ Vt) = (W @ U) @ diag(S) @ Vt
        W_new = linear.weight @ U @ torch.diag(S) @ Vt
    else:
        W_new = linear.weight @ A_full

    # Compute b' = b + W @ b_full
    if linear.bias is not None:
        b_new = linear.bias + linear.weight @ b_full
    else:
        b_new = linear.weight @ b_full

    # Create new linear layer
    linear_new = nn.Linear(d_in, d_out, bias=True, device=device, dtype=dtype)
    linear_new.weight.data = W_new
    linear_new.bias.data = b_new

    return linear_new


def bake_pca_ot_into_model(
    model: nn.Module,
    transforms: list[PCAOTTransform],
    layer_indices: list[int],
    target_module_name: str = "mlp.down_proj",
    rank: Optional[int] = None,
) -> None:
    """Bake PCA-OT transformations into model weights in-place.

    This modifies the model by replacing specified linear layers with versions
    that have the affine transformation baked in.

    Parameters
    ----------
    model : nn.Module
        The transformer model to modify.
    transforms : list[PCAOTTransform]
        PCA-OT transforms for each layer.
    layer_indices : list[int]
        Layer indices to apply transformations to.
    target_module_name : str, default "mlp.down_proj"
        Name of the module within each layer to modify. Common choices:
        - "mlp.down_proj" (Llama, Mistral) - after MLP
        - "mlp.c_proj" (GPT-2) - after MLP
        - "output" (BERT) - layer output
    rank : int, optional
        Rank for low-rank approximation. If None, uses full matrix.
        Paper recommends rank=16-32 for efficiency.

    Example
    -------
    >>> from abliterix.vectors import compute_pca_ot_transforms
    >>> transforms = compute_pca_ot_transforms(benign_states, harmful_states)
    >>> bake_pca_ot_into_model(model, transforms, layer_indices=[15, 16], rank=16)
    >>> # Model now has transformations permanently baked in
    """
    # Find layer modules
    layer_paths = [
        "model.layers",  # Llama, Mistral, Qwen
        "transformer.h",  # GPT-2
        "transformer.layers",  # Generic
        "encoder.layer",  # BERT-style
    ]

    layers = None
    for path in layer_paths:
        try:
            parts = path.split(".")
            module = model
            for part in parts:
                module = getattr(module, part)
            layers = module
            break
        except AttributeError:
            continue

    if layers is None:
        raise ValueError(
            "Could not find transformer layers in model. "
            "Supported architectures: Llama, Mistral, Qwen, GPT-2, BERT."
        )

    # Build transform lookup
    transform_dict = {t.layer_idx: t for t in transforms}

    # Apply transformations
    modified_count = 0
    for layer_idx in layer_indices:
        if layer_idx not in transform_dict:
            raise ValueError(
                f"No transform available for layer {layer_idx}. "
                f"Available layers: {sorted(transform_dict.keys())}"
            )

        if layer_idx >= len(layers):
            raise ValueError(
                f"Layer {layer_idx} not found in model. Model has {len(layers)} layers."
            )

        transform = transform_dict[layer_idx]
        layer = layers[layer_idx]

        # Navigate to target module
        try:
            parts = target_module_name.split(".")
            parent = layer
            for part in parts[:-1]:
                parent = getattr(parent, part)
            module_name = parts[-1]
            target_module = getattr(parent, module_name)
        except AttributeError:
            raise ValueError(
                f"Could not find module '{target_module_name}' in layer {layer_idx}. "
                f"Check that target_module_name is correct for this architecture."
            )

        if not isinstance(target_module, nn.Linear):
            raise ValueError(
                f"Target module '{target_module_name}' in layer {layer_idx} "
                f"is not nn.Linear (got {type(target_module)})"
            )

        # Bake transformation
        modified_module = bake_affine_into_linear(
            target_module,
            transform.A_full,
            transform.b_full,
            rank=rank,
        )

        # Replace module
        setattr(parent, module_name, modified_module)
        modified_count += 1

    print(
        f"✓ Baked PCA-OT into {modified_count} layers: {layer_indices} "
        f"(rank={rank if rank else 'full'})"
    )


def compute_reconstruction_error(
    A_full: Tensor,
    rank: int,
) -> float:
    """Compute relative reconstruction error for low-rank approximation.

    Parameters
    ----------
    A_full : Tensor
        Full matrix, shape (d, d).
    rank : int
        Target rank for approximation.

    Returns
    -------
    float
        Relative Frobenius norm error: ||A - A_approx||_F / ||A||_F

    Example
    -------
    >>> error = compute_reconstruction_error(transform.A_full, rank=16)
    >>> print(f"Rank-16 approximation error: {error:.4f}")
    """
    U, S, Vt = low_rank_decompose(A_full, rank=rank)
    A_approx = U @ torch.diag(S) @ Vt

    error = (A_full - A_approx).norm() / A_full.norm()
    return error.item()
