"""Tests for abliterix.cliff_head — attention-head ablation for reasoning models.

Verifies the static weight-alignment heuristic and its storage contract.
These tests do not establish causal head identity or behavioral effects.

All tests use a synthetic mock engine with a tiny multi-head attention
shape — no real model, no GPU.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from abliterix.cliff_head import (
    HeadScore,
    apply_cliff_head_ablation,
    identify_safety_heads,
    restore_cliff_head_ablation,
    run_cliff_head_ablation,
)


# ---------------------------------------------------------------------------
# Synthetic engine helper
# ---------------------------------------------------------------------------


def _make_engine(
    n_layers: int = 4,
    num_heads: int = 8,
    head_dim: int = 16,
    hidden: int | None = None,
    seed: int = 0,
):
    """Build a SimpleNamespace mock engine matching SteeringEngine's contract.

    Exposes the minimum surface area cliff_head uses:
    * ``engine.model.config`` — has ``num_attention_heads``, ``hidden_size``
    * ``engine.get_n_layers()``
    * ``engine.steerable_modules(layer_idx)`` — returns ``{"attn.o_proj": [Linear]}``
    """
    torch.manual_seed(seed)
    if hidden is None:
        hidden = num_heads * head_dim

    o_projs = [
        nn.Linear(num_heads * head_dim, hidden, bias=False) for _ in range(n_layers)
    ]

    config = SimpleNamespace(
        num_attention_heads=num_heads,
        hidden_size=hidden,
    )
    model = SimpleNamespace(config=config)

    def get_n_layers():
        return n_layers

    def steerable_modules(layer_idx):
        return {"attn.o_proj": [o_projs[layer_idx]]}

    engine = SimpleNamespace(
        model=model,
        get_n_layers=get_n_layers,
        steerable_modules=steerable_modules,
        _o_projs=o_projs,  # test handle
    )
    return engine


# ---------------------------------------------------------------------------
# Identification
# ---------------------------------------------------------------------------


def test_identify_returns_top_k_fraction():
    engine = _make_engine(n_layers=4, num_heads=8, head_dim=16)
    # Random refusal vector (per-layer plus embedding slot at 0).
    refusal = F.normalize(torch.randn(5, 128), p=2, dim=1)
    heads = identify_safety_heads(engine, refusal, top_k_frac=0.25)
    # Total heads = 4 layers × 8 heads = 32. 25% = 8.
    assert len(heads) == 8


def test_identify_sorted_descending():
    engine = _make_engine()
    refusal = F.normalize(torch.randn(5, 128), p=2, dim=1)
    heads = identify_safety_heads(engine, refusal, top_k_frac=0.5)
    scores = [h.score for h in heads]
    assert scores == sorted(scores, reverse=True)


def test_identify_min_heads_floor():
    """When top_k_frac is tiny, at least min_heads must be returned."""
    engine = _make_engine(n_layers=2, num_heads=2, head_dim=8)  # 4 heads total
    refusal = F.normalize(torch.randn(3, 16), p=2, dim=1)
    heads = identify_safety_heads(engine, refusal, top_k_frac=0.01, min_heads=2)
    assert len(heads) == 2


def test_identify_handles_multi_direction_input():
    """Multi-direction shape (n_dirs, layers+1, hidden) must work."""
    engine = _make_engine()
    refusal = F.normalize(torch.randn(2, 5, 128), p=2, dim=2)
    heads = identify_safety_heads(engine, refusal, top_k_frac=0.1)
    assert len(heads) >= 1


def test_identify_skips_layers_with_size_mismatch():
    """If a layer's o_proj does not match num_heads × head_dim, skip it."""
    engine = _make_engine(n_layers=3, num_heads=4, head_dim=8)
    # Replace layer 1's o_proj with a mismatched shape.
    engine._o_projs[1] = nn.Linear(40, 32, bias=False)  # 40 != 4 * 8 = 32
    refusal = F.normalize(torch.randn(4, 32), p=2, dim=1)
    heads = identify_safety_heads(engine, refusal, top_k_frac=0.5)
    # No head from layer 1 in the output.
    assert all(h.layer != 1 for h in heads)


def test_heterogeneous_head_dimensions_are_resolved_per_layer():
    """Head discovery and ablation use each layer's heterogeneous head_dim."""

    class AmbiguousGlobalPerLayerAttributeError(RuntimeError):
        pass

    class HeterogeneousConfig:
        num_attention_heads = 8
        hidden_size = 64
        per_layer_config = [
            SimpleNamespace(num_attention_heads=8, head_dim=4),
            SimpleNamespace(num_attention_heads=8, head_dim=8),
        ]

        def __getattribute__(self, name):
            if name == "head_dim":
                raise AmbiguousGlobalPerLayerAttributeError(
                    "head_dim varies across layers; use per_layer_config"
                )
            return object.__getattribute__(self, name)

    config = HeterogeneousConfig()

    with pytest.raises(AmbiguousGlobalPerLayerAttributeError, match="head_dim"):
        _ = config.head_dim

    o_projs = [
        nn.Linear(32, 64, bias=False),
        nn.Linear(64, 64, bias=False),
    ]
    engine = SimpleNamespace(
        model=SimpleNamespace(config=config),
        get_n_layers=lambda: len(o_projs),
        steerable_modules=lambda layer_idx: {"attn.o_proj": [o_projs[layer_idx]]},
    )
    refusal = F.normalize(torch.randn(3, 64), p=2, dim=1)

    heads = identify_safety_heads(engine, refusal, top_k_frac=1.0)

    assert len(heads) == 16
    assert sum(head.layer == 0 for head in heads) == 8
    assert sum(head.layer == 1 for head in heads) == 8

    modified = apply_cliff_head_ablation(engine, heads)

    assert modified == 16
    for entry in heads:
        head_dim = 4 if entry.layer == 0 else 8
        start = entry.head * head_dim
        assert (
            torch.count_nonzero(
                o_projs[entry.layer].weight[:, start : start + head_dim]
            )
            == 0
        )


def test_identify_finds_planted_head():
    """If we plant a strong refusal-aligned head, it must be at the top.

    Constructs an engine where head (1, 3) is the only one whose o_proj
    columns are strongly aligned with the layer-1 refusal direction; that
    head should be ranked first.
    """
    n_layers, num_heads, head_dim = 4, 8, 16
    hidden = num_heads * head_dim
    engine = _make_engine(n_layers=n_layers, num_heads=num_heads, head_dim=head_dim)

    # Layer 0, head 5: plant a direction in the o_proj columns aligned
    # with a planted layer-0 refusal vector.
    planted_layer = 0
    planted_head = 5
    planted_dir = F.normalize(torch.randn(hidden), p=2, dim=0)

    o_proj = engine._o_projs[planted_layer]
    with torch.no_grad():
        o_proj.weight.zero_()
        # All hidden_out rows of this head's column block point along the
        # planted direction with magnitude 10.
        block = planted_dir.unsqueeze(1).expand(-1, head_dim) * 10.0
        o_proj.weight[:, planted_head * head_dim : (planted_head + 1) * head_dim] = (
            block
        )

    # Refusal vector: planted at layer planted_layer (residual index +1),
    # weak random everywhere else.
    refusal = torch.randn(n_layers + 1, hidden) * 0.01
    refusal[planted_layer + 1] = planted_dir

    heads = identify_safety_heads(engine, refusal, top_k_frac=0.1, min_heads=1)
    assert heads, "expected at least one head returned"
    assert heads[0].layer == planted_layer
    assert heads[0].head == planted_head


# ---------------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------------


def test_apply_ablation_zeroes_selected_columns():
    engine = _make_engine(n_layers=2, num_heads=4, head_dim=8)

    # Manually craft a head list.
    heads = [
        HeadScore(layer=0, head=1, score=1.0),
        HeadScore(layer=1, head=3, score=0.9),
    ]
    n_modified = apply_cliff_head_ablation(engine, heads, strength=1.0)
    assert n_modified == 2

    w0 = engine._o_projs[0].weight
    w1 = engine._o_projs[1].weight
    assert torch.allclose(w0[:, 8:16], torch.zeros_like(w0[:, 8:16]))
    assert torch.allclose(w1[:, 24:32], torch.zeros_like(w1[:, 24:32]))
    # Other heads untouched.
    assert not torch.allclose(w0[:, 0:8], torch.zeros_like(w0[:, 0:8]))


def test_apply_ablation_partial_strength():
    engine = _make_engine(n_layers=1, num_heads=4, head_dim=8)
    original = engine._o_projs[0].weight.detach().clone()

    heads = [HeadScore(layer=0, head=2, score=0.5)]
    apply_cliff_head_ablation(engine, heads, strength=0.5)

    after = engine._o_projs[0].weight
    block = after[:, 16:24]
    expected = original[:, 16:24] * 0.5
    assert torch.allclose(block, expected, atol=1e-5)


def test_apply_ablation_strength_zero_is_noop():
    engine = _make_engine()
    original = engine._o_projs[0].weight.detach().clone()
    heads = [HeadScore(layer=0, head=0, score=1.0)]
    n = apply_cliff_head_ablation(engine, heads, strength=0.0)
    assert n == 0
    assert torch.allclose(engine._o_projs[0].weight, original)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_restore_roundtrip(dtype):
    engine = _make_engine(n_layers=2, num_heads=4, head_dim=8)
    for module in engine._o_projs:
        module.to(dtype=dtype)
    snapshots = [w.weight.detach().clone() for w in engine._o_projs]

    heads = [
        HeadScore(layer=0, head=1, score=0.9),
        HeadScore(layer=1, head=2, score=0.8),
    ]
    apply_cliff_head_ablation(engine, heads, strength=1.0)
    # Confirm modification.
    assert not torch.allclose(engine._o_projs[0].weight, snapshots[0])

    n_restored = restore_cliff_head_ablation(engine)
    assert n_restored == 2
    for layer_idx, original in enumerate(snapshots):
        assert torch.allclose(engine._o_projs[layer_idx].weight, original)
    # Cache cleared.
    assert engine._cliff_head_originals == {}


def test_restore_no_cache_returns_zero():
    engine = _make_engine()
    assert restore_cliff_head_ablation(engine) == 0


def test_restore_idempotent_after_full_clear():
    engine = _make_engine()
    heads = [HeadScore(layer=0, head=0, score=1.0)]
    apply_cliff_head_ablation(engine, heads, strength=1.0)
    restore_cliff_head_ablation(engine)
    # Second restore is a no-op.
    assert restore_cliff_head_ablation(engine) == 0


# ---------------------------------------------------------------------------
# End-to-end orchestration
# ---------------------------------------------------------------------------


def test_run_orchestration_returns_count_and_heads():
    engine = _make_engine(n_layers=4, num_heads=8, head_dim=16)
    refusal = F.normalize(torch.randn(5, 128), p=2, dim=1)
    n_modified, heads = run_cliff_head_ablation(
        engine, refusal, top_k_frac=0.05, strength=0.7
    )
    assert n_modified == len(heads) > 0
    # All ablation strengths consistent.
    # Spot-check that an arbitrary cached slice was scaled.
    weight, head, original = next(iter(engine._cliff_head_originals.values()))
    head_dim = original.shape[1]
    current = weight.data[:, head * head_dim : (head + 1) * head_dim]
    expected = original.to(weight.dtype) * (1.0 - 0.7)
    assert torch.allclose(current, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_missing_model_raises():
    engine = SimpleNamespace(
        model=None,
        get_n_layers=lambda: 1,
        steerable_modules=lambda i: {},
    )
    with pytest.raises(RuntimeError, match="no loaded HF model"):
        identify_safety_heads(engine, torch.randn(2, 16), top_k_frac=0.1)


def test_missing_num_heads_raises():
    config = SimpleNamespace(hidden_size=128)  # no num_attention_heads
    engine = SimpleNamespace(
        model=SimpleNamespace(config=config),
        get_n_layers=lambda: 1,
        steerable_modules=lambda i: {},
    )
    with pytest.raises(RuntimeError, match="num_attention_heads"):
        identify_safety_heads(engine, torch.randn(2, 128), top_k_frac=0.1)


@pytest.mark.parametrize(
    "storage", ["int8", "fp8", "meta", "bnb4_lazy", "bnb8_lazy", "offload"]
)
def test_unsupported_storage_rejects_scoring_and_preserves_prior_edits(storage):
    """A later unsupported projection must not partly edit an earlier layer."""
    engine = _make_engine(n_layers=2, num_heads=2, head_dim=2)
    apply_cliff_head_ablation(engine, [HeadScore(0, 1, 1.0)], strength=0.25)
    before = engine._o_projs[0].weight.detach().clone()
    cache = engine._cliff_head_originals
    cached_items = dict(cache)
    bad = engine._o_projs[1]
    if storage == "int8":
        bad.weight = nn.Parameter(
            torch.ones(4, 4, dtype=torch.int8), requires_grad=False
        )
    elif storage == "fp8":
        bad.weight = nn.Parameter(
            torch.ones(4, 4).to(torch.float8_e4m3fn), requires_grad=False
        )
    elif storage == "meta":
        bad.to("meta")
    elif storage in {"bnb4_lazy", "bnb8_lazy"}:
        bnb = pytest.importorskip("bitsandbytes")
        bad = (
            bnb.nn.Linear4bit(4, 4, bias=False)
            if storage == "bnb4_lazy"
            else bnb.nn.Linear8bitLt(4, 4, bias=False, has_fp16_weights=False)
        )
        engine._o_projs[1] = bad
        # Before first use, both packed formats can look like ordinary floats.
        assert bad.weight.is_floating_point()
    else:
        bad._hf_hook = SimpleNamespace(hooks=[SimpleNamespace(offload=True)])

    with pytest.raises(RuntimeError, match="resident dense floating-point.*layer 1"):
        identify_safety_heads(engine, torch.ones(3, 4), top_k_frac=1.0)
    with pytest.raises(RuntimeError, match="resident dense floating-point.*layer 1"):
        apply_cliff_head_ablation(
            engine, [HeadScore(0, 0, 1.0), HeadScore(1, 0, 1.0)], strength=0.5
        )
    torch.testing.assert_close(engine._o_projs[0].weight, before, rtol=0, atol=0)
    assert engine._cliff_head_originals is cache
    assert cache.keys() == cached_items.keys()
    assert all(cache[key] is item for key, item in cached_items.items())


@pytest.mark.parametrize("layer,head", [(1, -1), (1, 2), (-1, 0), (2, 0)])
def test_invalid_head_address_does_not_edit_weights_or_create_restore_cache(
    layer, head
):
    engine = _make_engine(n_layers=2, num_heads=2, head_dim=2)
    before = [module.weight.detach().clone() for module in engine._o_projs]
    with pytest.raises(ValueError, match="Invalid cliff-head"):
        apply_cliff_head_ablation(
            engine, [HeadScore(0, 0, 1.0), HeadScore(layer, head, 1.0)], strength=0.5
        )
    assert not hasattr(engine, "_cliff_head_originals")
    for module, original in zip(engine._o_projs, before):
        torch.testing.assert_close(module.weight, original, rtol=0, atol=0)


def test_selected_projection_shape_mismatch_is_atomic():
    engine = _make_engine(n_layers=2, num_heads=2, head_dim=2)
    engine._o_projs[1] = nn.Linear(5, 4, bias=False)
    before = engine._o_projs[0].weight.detach().clone()
    with pytest.raises(RuntimeError, match="projection shape at layer 1"):
        apply_cliff_head_ablation(
            engine, [HeadScore(0, 0, 1.0), HeadScore(1, 0, 1.0)], strength=0.5
        )
    assert not hasattr(engine, "_cliff_head_originals")
    torch.testing.assert_close(engine._o_projs[0].weight, before, rtol=0, atol=0)


def test_duplicate_head_selection_scales_once():
    engine = _make_engine(n_layers=1, num_heads=2, head_dim=2)
    before = engine._o_projs[0].weight.detach().clone()
    head = HeadScore(0, 1, 1.0)
    assert apply_cliff_head_ablation(engine, [head, head], strength=0.5) == 1
    torch.testing.assert_close(engine._o_projs[0].weight[:, 2:], before[:, 2:] * 0.5)


@pytest.mark.parametrize("strength", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_public_api_strength_fails_before_mutation(strength):
    engine = _make_engine(n_layers=1, num_heads=2, head_dim=2)
    before = engine._o_projs[0].weight.detach().clone()
    with pytest.raises(ValueError, match="finite and in"):
        apply_cliff_head_ablation(engine, [HeadScore(0, 0, 1.0)], strength=strength)
    assert not hasattr(engine, "_cliff_head_originals")
    torch.testing.assert_close(engine._o_projs[0].weight, before, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Settings validators
# ---------------------------------------------------------------------------


def test_validator_rejects_invalid_top_k_frac():
    from abliterix.settings import SteeringConfig

    with pytest.raises(ValueError, match="cliff_head_top_k_frac"):
        SteeringConfig(cliff_head_ablation=True, cliff_head_top_k_frac=0.0)
    with pytest.raises(ValueError, match="cliff_head_top_k_frac"):
        SteeringConfig(cliff_head_ablation=True, cliff_head_top_k_frac=1.5)


def test_validator_rejects_invalid_strength():
    from abliterix.settings import SteeringConfig

    with pytest.raises(ValueError, match="cliff_head_strength"):
        SteeringConfig(cliff_head_ablation=True, cliff_head_strength=1.5)
    with pytest.raises(ValueError, match="cliff_head_strength"):
        SteeringConfig(cliff_head_ablation=True, cliff_head_strength=-0.1)


def test_validator_allows_cliff_head_alone():
    from abliterix.settings import SteeringConfig

    sc = SteeringConfig(cliff_head_ablation=True)
    assert sc.cliff_head_top_k_frac == 0.03
    assert sc.cliff_head_strength == 1.0
