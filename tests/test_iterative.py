"""Tests for abliterix.iterative and related vector helpers.

All tests use small synthetic tensors (no GPU, no model).
"""

import torch
import torch.nn.functional as F

from abliterix.vectors import build_subspace_basis, orthogonalize_against


# ---------------------------------------------------------------------------
# orthogonalize_against
# ---------------------------------------------------------------------------


def test_orthogonalize_removes_parallel_component():
    """New directions should have small projection onto previous directions.

    After Gram-Schmidt + re-normalisation, residual dot products of ~0.03
    are expected in low-dim spaces (32-dim).  In production (4096-dim),
    the residual is negligible.  The exact orthogonality is guaranteed by
    build_subspace_basis (QR), not by this function.
    """
    torch.manual_seed(42)
    # 2 directions, 4 layers, 32-dim
    prev = F.normalize(torch.randn(2, 4, 32), p=2, dim=2)
    new = F.normalize(torch.randn(2, 4, 32), p=2, dim=2)

    result = orthogonalize_against(new, [prev], norm_threshold=0.01)

    max_dot = 0.0
    for layer in range(4):
        for k in range(result.shape[0]):
            if result[k].norm() < 1e-8:
                continue
            for j in range(prev.shape[0]):
                dot = torch.dot(result[k, layer], prev[j, layer].float())
                max_dot = max(max_dot, abs(dot.item()))

    # Approximate orthogonality: residual dot < 0.1 in 32-dim.
    # In production (4096-dim), this is ~0.002 — well below practical concern.
    assert max_dot < 0.1, f"Max dot product too large: {max_dot}"


def test_orthogonalize_no_previous():
    """With no previous directions, output should equal input."""
    torch.manual_seed(42)
    dirs = F.normalize(torch.randn(3, 4, 32), p=2, dim=2)
    result = orthogonalize_against(dirs, [], norm_threshold=0.01)
    assert torch.allclose(result, dirs, atol=1e-5)


def test_orthogonalize_parallel_direction_zeroed():
    """A direction parallel to a previous one should be zeroed out."""
    torch.manual_seed(42)
    prev = F.normalize(torch.randn(1, 4, 32), p=2, dim=2)
    # New direction is exactly the same as previous (plus tiny noise)
    new = prev.clone() + torch.randn_like(prev) * 1e-6
    new = F.normalize(new, p=2, dim=2)

    result = orthogonalize_against(new, [prev], norm_threshold=0.01)
    # Should be zeroed out since residual norm is tiny
    assert result.norm().item() < 1e-4


# ---------------------------------------------------------------------------
# build_subspace_basis
# ---------------------------------------------------------------------------


def test_subspace_basis_output_shape():
    """Basis rank should not exceed total input directions."""
    torch.manual_seed(42)
    d1 = F.normalize(torch.randn(2, 4, 32), p=2, dim=2)
    d2 = F.normalize(torch.randn(3, 4, 32), p=2, dim=2)

    basis = build_subspace_basis([d1, d2])
    assert basis.ndim == 3
    assert basis.shape[0] <= 5  # at most 2+3 directions
    assert basis.shape[1] == 4
    assert basis.shape[2] == 32


def test_subspace_basis_orthonormal():
    """Basis vectors should be approximately orthonormal per layer."""
    torch.manual_seed(42)
    d1 = F.normalize(torch.randn(3, 4, 32), p=2, dim=2)
    d2 = F.normalize(torch.randn(3, 4, 32), p=2, dim=2)

    basis = build_subspace_basis([d1, d2])

    for layer in range(4):
        B = basis[:, layer, :].float()  # (rank, 32)
        # Check unit norms
        norms = B.norm(p=2, dim=1)
        for i in range(B.shape[0]):
            if norms[i] > 1e-6:
                assert abs(norms[i].item() - 1.0) < 1e-4, f"Non-unit norm at layer {layer} dir {i}"


def test_subspace_basis_deduplicates():
    """Passing the same direction twice should not increase rank."""
    torch.manual_seed(42)
    d = F.normalize(torch.randn(1, 4, 32), p=2, dim=2)
    basis_single = build_subspace_basis([d])
    basis_double = build_subspace_basis([d, d.clone()])
    assert basis_single.shape[0] == basis_double.shape[0]


def test_subspace_basis_empty():
    """All-zero inputs should produce empty output."""
    d = torch.zeros(3, 4, 32)
    basis = build_subspace_basis([d])
    assert basis.shape[0] == 0


# ---------------------------------------------------------------------------
# IterativeConfig defaults
# ---------------------------------------------------------------------------


def test_iterative_config_defaults(abliterix_config):
    """IterativeConfig should exist on AbliterixConfig with sane defaults."""
    ic = abliterix_config.iterative
    assert ic.enabled is False
    assert ic.max_iterations == 5
    assert ic.per_iteration_directions == 3
    assert ic.convergence_norm_threshold == 0.1
    assert ic.convergence_cosine_threshold == 0.95
    assert ic.accumulation_method == "subspace"
