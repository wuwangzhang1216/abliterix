# Abliterix
# Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Surgical Refusal Ablation (SRA) — concept-guided spectral cleaning.

Implements the core idea from arxiv:2601.08489: the raw refusal vector is
polysemantic, entangling the refusal signal with syntax, formatting, and
capability circuits (math, code, reasoning).  SRA constructs a registry of
independent *Concept Atoms* representing protected capabilities, then uses
ridge-regularised spectral residualisation to orthogonalise the refusal
vector against these directions.

The result is a *clean* refusal vector that ablates safety alignment without
collateral damage to model capabilities.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from .types import VectorMethod


# ---------------------------------------------------------------------------
# Concept Atom extraction
# ---------------------------------------------------------------------------


def _build_concept_atoms(
    benign_states: Tensor,
    n_atoms: int = 8,
) -> Tensor:
    """Extract concept atom directions from benign activations.

    Each concept atom captures an independent capability direction.  We
    cluster benign residuals via k-means-style PCA partitioning:

    1. Run PCA on the benign residual matrix to extract the top ``n_atoms``
       principal components.
    2. Each component represents a distinct capability direction (math,
       code, reasoning, style, etc.) in activation space.

    Parameters
    ----------
    benign_states : Tensor
        Shape ``(n, layers+1, hidden_dim)``.
    n_atoms : int
        Number of concept atoms to extract per layer.

    Returns
    -------
    Tensor
        Shape ``(n_atoms, layers+1, hidden_dim)`` — unit-normalised concept
        atom directions per layer.
    """
    n_layers = benign_states.shape[1]
    atoms_per_layer = []

    for layer_idx in range(n_layers):
        b = benign_states[:, layer_idx, :].float()
        b_centred = b - b.mean(dim=0, keepdim=True)

        # PCA via truncated SVD — top-n_atoms directions.
        k = min(n_atoms, b_centred.shape[0], b_centred.shape[1])
        _, _, Vh = torch.linalg.svd(b_centred, full_matrices=False)
        atoms = Vh[:k]  # (k, hidden_dim)

        # Pad with zeros if fewer components than requested.
        if k < n_atoms:
            pad = torch.zeros(
                n_atoms - k,
                atoms.shape[1],
                device=atoms.device,
                dtype=atoms.dtype,
            )
            atoms = torch.cat([atoms, pad], dim=0)

        atoms_per_layer.append(F.normalize(atoms, p=2, dim=1))

    # Stack: (n_layers, n_atoms, hidden_dim) → transpose to (n_atoms, n_layers, hidden_dim)
    stacked = torch.stack(atoms_per_layer, dim=0)  # (n_layers, n_atoms, hidden_dim)
    return stacked.permute(1, 0, 2)  # (n_atoms, n_layers, hidden_dim)


# ---------------------------------------------------------------------------
# Spectral residualisation
# ---------------------------------------------------------------------------


def _spectral_residualize(
    refusal_vector: Tensor,
    concept_atoms: Tensor,
    ridge_alpha: float = 0.01,
) -> Tensor:
    """Orthogonalise the refusal vector against concept atom directions.

    For each layer, project the refusal vector onto the subspace spanned by
    concept atoms and subtract the projection (with ridge regularisation):

        C = concept_atom_matrix   (n_atoms × hidden_dim)
        v_clean = v − C^T (C C^T + α I)^{-1} C v

    Ridge regularisation prevents over-cleaning: with ``α → 0`` the
    projection removes the maximal component along concept atoms; with
    ``α → ∞`` the original vector is preserved.

    Parameters
    ----------
    refusal_vector : Tensor
        Shape ``(layers+1, hidden_dim)``.
    concept_atoms : Tensor
        Shape ``(n_atoms, layers+1, hidden_dim)``.
    ridge_alpha : float
        Ridge regularisation coefficient.

    Returns
    -------
    Tensor
        Cleaned refusal vector, shape ``(layers+1, hidden_dim)``.
    """
    n_layers = refusal_vector.shape[0]
    cleaned = []

    for layer_idx in range(n_layers):
        v = refusal_vector[layer_idx].float()  # (hidden_dim,)
        C = concept_atoms[:, layer_idx, :].float()  # (n_atoms, hidden_dim)

        # Skip layers where concept atoms are degenerate.
        atom_norms = C.norm(dim=1)
        valid = atom_norms > 1e-8
        if valid.sum() == 0:
            cleaned.append(v)
            continue

        C = C[valid]  # (k, hidden_dim) where k ≤ n_atoms

        # Ridge-regularised projection: v_proj = C^T (C C^T + α I)^{-1} C v
        CCT = C @ C.T  # (k, k)
        eye = torch.eye(CCT.shape[0], device=v.device, dtype=v.dtype)
        Cv = C @ v  # (k,)
        coeffs = torch.linalg.solve(CCT + ridge_alpha * eye, Cv)  # (k,)
        v_proj = C.T @ coeffs  # (hidden_dim,)

        cleaned.append(v - v_proj)

    return torch.stack(cleaned, dim=0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_sra_vectors(
    benign_states: Tensor,
    target_states: Tensor,
    base_method: VectorMethod = VectorMethod.MEAN,
    n_atoms: int = 8,
    ridge_alpha: float = 0.01,
    *,
    orthogonal_projection: bool = False,
    projected_abliteration: bool = False,
    winsorize: bool = False,
    winsorize_quantile: float = 0.995,
    ot_components: int = 2,
) -> Tensor:
    """Compute SRA-cleaned steering vectors.

    1. Derive an initial refusal direction using ``base_method``.
    2. Extract concept atoms from benign activations.
    3. Residualise the refusal vector against concept atoms.
    4. Normalise and return.

    Parameters
    ----------
    benign_states, target_states : Tensor
        Shape ``(n, layers+1, hidden_dim)``.
    base_method : VectorMethod
        Method used to compute the initial refusal direction (before SRA
        cleaning).  Must not be SRA itself.
    n_atoms : int
        Number of concept atoms (protected capability clusters).
    ridge_alpha : float
        Ridge regularisation coefficient.
    orthogonal_projection, projected_abliteration, winsorize, etc.
        Passed through to the base method for initial vector computation.

    Returns
    -------
    Tensor
        Unit-normalised cleaned steering vectors, shape ``(layers+1, hidden_dim)``.
    """
    # Import here to avoid circular dependency (vectors.py imports us).
    from .vectors import compute_steering_vectors

    # Step 1: Compute the base refusal direction.
    base_vectors = compute_steering_vectors(
        benign_states,
        target_states,
        method=base_method,
        orthogonal_projection=orthogonal_projection,
        winsorize=winsorize,
        winsorize_quantile=winsorize_quantile,
        projected_abliteration=projected_abliteration,
        ot_components=ot_components,
        n_directions=1,
    )

    # Step 2: Build concept atoms from benign activations.
    concept_atoms = _build_concept_atoms(benign_states, n_atoms=n_atoms)

    # Step 3: Residualise the refusal vector against concept atoms.
    cleaned = _spectral_residualize(
        base_vectors, concept_atoms, ridge_alpha=ridge_alpha
    )

    # Step 4: Normalise.
    return F.normalize(cleaned, p=2, dim=1)
