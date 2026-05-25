"""Tests for abliterix.weight_transforms — ORBA / biprojected / Householder.

Verifies the grimjim weight-space transforms ported from:
* https://huggingface.co/blog/grimjim/orthogonal-reflection-bounded-ablation
* https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration

All tests use small synthetic tensors — no GPU, no model.
"""

import pytest
import torch
import torch.nn.functional as F

from abliterix.weight_transforms import (
    DirectTransform,
    apply_biprojected_transform,
    apply_direct_transform,
    apply_householder_transform,
    apply_orba_transform,
    apply_standard_transform,
    double_gram_schmidt,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rand_W(out_f: int = 16, in_f: int = 32, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(out_f, in_f)


def _rand_unit(dim: int, seed: int = 1) -> torch.Tensor:
    torch.manual_seed(seed)
    return F.normalize(torch.randn(dim), p=2, dim=0)


# ---------------------------------------------------------------------------
# double_gram_schmidt
# ---------------------------------------------------------------------------


def test_double_gs_is_unit_norm():
    refusal = _rand_unit(16, seed=2)
    benign = _rand_unit(16, seed=3)
    out = double_gram_schmidt(refusal, benign)
    assert abs(out.norm().item() - 1.0) < 1e-5


def test_double_gs_orthogonal_to_benign():
    refusal = _rand_unit(16, seed=4)
    benign = _rand_unit(16, seed=5)
    out = double_gram_schmidt(refusal, benign)
    assert abs(torch.dot(out, benign).item()) < 1e-5


def test_double_gs_idempotent_on_orthogonal_input():
    """If refusal is already orthogonal to benign, output should equal refusal."""
    benign = _rand_unit(16, seed=6)
    # Build an orthogonal refusal direction by Gram-Schmidt once manually.
    raw = torch.randn(16)
    refusal = F.normalize(raw - torch.dot(raw, benign) * benign, p=2, dim=0)
    out = double_gram_schmidt(refusal, benign)
    assert torch.allclose(out, refusal, atol=1e-5)


# ---------------------------------------------------------------------------
# Standard transform — sanity check it matches the old math
# ---------------------------------------------------------------------------


def test_standard_input_side():
    W = _rand_W(out_f=8, in_f=16)
    d = _rand_unit(16, seed=10)
    out = apply_standard_transform(W, d, strength=1.0)
    expected = W.float() - (W.float() @ d).unsqueeze(1) * d.unsqueeze(0)
    assert torch.allclose(out.float(), expected, atol=1e-5)


def test_standard_output_side():
    W = _rand_W(out_f=8, in_f=16)
    d = _rand_unit(8, seed=11)
    out = apply_standard_transform(W, d, strength=1.0)
    expected = W.float() - d.unsqueeze(1) * (d @ W.float()).unsqueeze(0)
    assert torch.allclose(out.float(), expected, atol=1e-5)


def test_standard_zero_strength_is_identity():
    W = _rand_W()
    d = _rand_unit(W.shape[1])
    out = apply_standard_transform(W, d, strength=0.0)
    assert torch.allclose(out, W, atol=1e-6)


def test_standard_shape_mismatch_raises():
    W = _rand_W(out_f=4, in_f=8)
    d = torch.randn(7)  # neither 4 nor 8
    with pytest.raises(ValueError, match="does not match"):
        apply_standard_transform(W, d)


# ---------------------------------------------------------------------------
# ORBA
# ---------------------------------------------------------------------------


def test_orba_orthogonalises_before_ablation():
    """ORBA should ablate the orthogonalised direction, not the raw one.

    Construct refusal = αbenign + βorthogonal. The plain ablation removes
    the whole refusal vector (including the benign-aligned component);
    ORBA's double-GS step strips the benign component first, so only the
    orthogonal residual is ablated.
    """
    in_f = 32
    benign = _rand_unit(in_f, seed=20)
    orth = torch.randn(in_f)
    orth = F.normalize(orth - torch.dot(orth, benign) * benign, p=2, dim=0)
    refusal = F.normalize(0.7 * benign + 0.3 * orth, p=2, dim=0)

    W = _rand_W(out_f=8, in_f=in_f, seed=21)
    out_orba = apply_orba_transform(
        W, refusal, benign, strength=1.0, preserve_row_norm=False
    )
    # Compare against ablating the manually-orthogonalised direction.
    expected = apply_standard_transform(W, orth, strength=1.0)
    assert torch.allclose(out_orba.float(), expected.float(), atol=1e-4)


def test_orba_preserves_row_norm_when_requested():
    W = _rand_W(out_f=8, in_f=16, seed=30)
    refusal = _rand_unit(16, seed=31)
    benign = _rand_unit(16, seed=32)
    orig_norms = torch.linalg.vector_norm(W, dim=1)
    out = apply_orba_transform(W, refusal, benign, strength=0.9, preserve_row_norm=True)
    new_norms = torch.linalg.vector_norm(out, dim=1)
    assert torch.allclose(new_norms, orig_norms, atol=1e-4)


def test_orba_no_norm_preserve_changes_row_norms():
    W = _rand_W(out_f=8, in_f=16, seed=40)
    refusal = _rand_unit(16, seed=41)
    benign = _rand_unit(16, seed=42)
    orig_norms = torch.linalg.vector_norm(W, dim=1)
    out = apply_orba_transform(
        W, refusal, benign, strength=0.9, preserve_row_norm=False
    )
    new_norms = torch.linalg.vector_norm(out, dim=1)
    # At least one row should have noticeably different norm.
    assert not torch.allclose(new_norms, orig_norms, atol=1e-3)


def test_orba_zero_strength_with_norm_preserve_is_identity():
    W = _rand_W(seed=50)
    out = apply_orba_transform(
        W,
        _rand_unit(W.shape[1]),
        _rand_unit(W.shape[1], seed=51),
        strength=0.0,
        preserve_row_norm=True,
    )
    assert torch.allclose(out, W, atol=1e-5)


def test_orba_input_dim_mismatch_raises():
    W = _rand_W(out_f=4, in_f=8)
    with pytest.raises(ValueError, match="input-side"):
        apply_orba_transform(W, torch.randn(10), torch.randn(10))


# ---------------------------------------------------------------------------
# Biprojected
# ---------------------------------------------------------------------------


def test_biprojected_preserves_row_norm_exactly():
    """Biprojected must preserve each row L2 norm exactly (not approximately)."""
    W = _rand_W(out_f=8, in_f=16, seed=60)
    refusal = _rand_unit(16, seed=61)
    orig_norms = torch.linalg.vector_norm(W, dim=1)
    out = apply_biprojected_transform(W, refusal, strength=1.0)
    new_norms = torch.linalg.vector_norm(out, dim=1)
    assert torch.allclose(new_norms, orig_norms, atol=1e-5)


def test_biprojected_removes_direction_after_norm_ablate():
    """After biprojected, the ablated weight should have reduced projection on d."""
    W = _rand_W(out_f=8, in_f=16, seed=70)
    refusal = _rand_unit(16, seed=71)
    before = (W.float() @ refusal).abs().mean().item()
    out = apply_biprojected_transform(W, refusal, strength=1.0)
    after = (out.float() @ refusal).abs().mean().item()
    assert after < before


def test_biprojected_zero_strength_with_renorm_returns_input():
    """At strength=0, biprojected is W = M · normalize(Ŵ) = original W (since Ŵ
    is already unit-norm). So output equals input within float tolerance."""
    W = _rand_W(seed=80)
    out = apply_biprojected_transform(W, _rand_unit(W.shape[1]), strength=0.0)
    assert torch.allclose(out, W, atol=1e-5)


def test_biprojected_input_dim_mismatch_raises():
    W = _rand_W(out_f=4, in_f=8)
    with pytest.raises(ValueError, match="input-side"):
        apply_biprojected_transform(W, torch.randn(7))


# ---------------------------------------------------------------------------
# Householder
# ---------------------------------------------------------------------------


def test_householder_is_isometry_at_full_strength():
    """At strength=1.0 the row L2 norms must be preserved exactly (reflection)."""
    W = _rand_W(out_f=8, in_f=16, seed=90)
    refusal = _rand_unit(16, seed=91)
    benign = _rand_unit(16, seed=92)
    orig_norms = torch.linalg.vector_norm(W, dim=1)
    out = apply_householder_transform(W, refusal, benign, strength=1.0)
    new_norms = torch.linalg.vector_norm(out, dim=1)
    assert torch.allclose(new_norms, orig_norms, atol=1e-4)


def test_householder_input_dim_mismatch_raises():
    W = _rand_W(out_f=4, in_f=8)
    with pytest.raises(ValueError, match="input-side"):
        apply_householder_transform(W, torch.randn(10), torch.randn(10))


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def test_dispatcher_routes_standard():
    W = _rand_W()
    d = _rand_unit(W.shape[1])
    out = apply_direct_transform(DirectTransform.STANDARD, W, d, None, strength=0.7)
    expected = apply_standard_transform(W, d, strength=0.7)
    assert torch.allclose(out, expected, atol=1e-6)


def test_dispatcher_routes_orba():
    W = _rand_W()
    d = _rand_unit(W.shape[1])
    b = _rand_unit(W.shape[1], seed=99)
    out = apply_direct_transform("orba", W, d, b, strength=0.8, preserve_row_norm=True)
    expected = apply_orba_transform(W, d, b, strength=0.8, preserve_row_norm=True)
    assert torch.allclose(out, expected, atol=1e-6)


def test_dispatcher_routes_biprojected():
    W = _rand_W()
    d = _rand_unit(W.shape[1])
    out = apply_direct_transform("biprojected", W, d, None, strength=1.0)
    expected = apply_biprojected_transform(W, d, strength=1.0)
    assert torch.allclose(out, expected, atol=1e-6)


def test_dispatcher_routes_householder():
    W = _rand_W()
    d = _rand_unit(W.shape[1])
    b = _rand_unit(W.shape[1], seed=88)
    out = apply_direct_transform("householder", W, d, b, strength=1.0)
    expected = apply_householder_transform(W, d, b, strength=1.0)
    assert torch.allclose(out, expected, atol=1e-6)


def test_dispatcher_requires_benign_dir_for_orba():
    W = _rand_W()
    d = _rand_unit(W.shape[1])
    with pytest.raises(ValueError, match="ORBA requires"):
        apply_direct_transform("orba", W, d, None)


def test_dispatcher_requires_benign_dir_for_householder():
    W = _rand_W()
    d = _rand_unit(W.shape[1])
    with pytest.raises(ValueError, match="Householder requires"):
        apply_direct_transform("householder", W, d, None)


def test_dispatcher_rejects_unknown_transform():
    W = _rand_W()
    d = _rand_unit(W.shape[1])
    with pytest.raises(ValueError):
        apply_direct_transform("does_not_exist", W, d, None)
