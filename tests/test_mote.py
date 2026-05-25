"""Tests for abliterix.mote — Mixture of Tunable Experts inference-time gains."""

from types import SimpleNamespace

import torch
import torch.nn as nn

from abliterix.mote import (
    MoTEHandle,
    gains_from_safety_experts,
    install_mote,
    remove_mote,
)


# ---------------------------------------------------------------------------
# Mock engine with per-expert modules
# ---------------------------------------------------------------------------


def _make_engine(n_layers: int = 2, n_experts: int = 4, in_dim: int = 8):
    """Build a mock engine whose layers expose `n_experts` Linear modules."""
    layers = []
    expert_lists = []
    for _ in range(n_layers):
        experts = [nn.Linear(in_dim, in_dim, bias=False) for _ in range(n_experts)]
        # Initialise each expert's weight to identity for predictable arithmetic.
        for e in experts:
            with torch.no_grad():
                e.weight.copy_(torch.eye(in_dim))
        layer = SimpleNamespace(experts=experts)
        layers.append(layer)
        expert_lists.append(experts)

    def steerable_modules(idx):
        return {"mlp.down_proj": list(layers[idx].experts)}

    engine = SimpleNamespace(
        transformer_layers=layers,
        steerable_modules=steerable_modules,
        get_n_layers=lambda: n_layers,
        _expert_lists=expert_lists,  # test handle
    )
    return engine


# ---------------------------------------------------------------------------
# install_mote
# ---------------------------------------------------------------------------


def test_install_mote_registers_hooks():
    engine = _make_engine(n_layers=2, n_experts=3)
    gains = {0: {1: 0.0, 2: 0.5}, 1: {0: 0.25}}
    handle = install_mote(engine, gains)
    assert isinstance(handle, MoTEHandle)
    assert handle.n_hooked == 3  # three (layer, expert) pairs


def test_install_mote_skips_identity_gain():
    """Gains very close to 1.0 must be no-ops (no hook installed)."""
    engine = _make_engine()
    gains = {0: {0: 1.0, 1: 1.0 + 1e-12, 2: 0.5}}
    handle = install_mote(engine, gains)
    # Only expert 2 should be hooked.
    assert handle.n_hooked == 1


def test_install_mote_skips_out_of_range_layers():
    engine = _make_engine(n_layers=2, n_experts=2)
    gains = {99: {0: 0.0}, 0: {0: 0.0}}
    handle = install_mote(engine, gains)
    # Only the in-range layer is touched.
    assert handle.n_hooked == 1


def test_install_mote_skips_out_of_range_experts():
    engine = _make_engine(n_layers=1, n_experts=2)
    gains = {0: {0: 0.0, 5: 0.0}}
    handle = install_mote(engine, gains)
    assert handle.n_hooked == 1


def test_install_mote_no_per_expert_modules():
    """Layers without mlp.down_proj entries are silently skipped."""
    engine = SimpleNamespace(
        transformer_layers=[SimpleNamespace()],
        steerable_modules=lambda i: {"attn.q_proj": [nn.Linear(4, 4)]},
        get_n_layers=lambda: 1,
    )
    handle = install_mote(engine, {0: {0: 0.0}})
    assert handle.n_hooked == 0


# ---------------------------------------------------------------------------
# Hook effect — actual output scaling
# ---------------------------------------------------------------------------


def test_install_mote_scales_expert_output_by_gain():
    engine = _make_engine(n_layers=1, n_experts=3, in_dim=8)
    expert_1 = engine._expert_lists[0][1]
    x = torch.randn(2, 8)
    baseline = expert_1(x)

    install_mote(engine, {0: {1: 0.0}})
    after = expert_1(x)
    assert torch.allclose(after, torch.zeros_like(after))

    # Other experts untouched.
    expert_0 = engine._expert_lists[0][0]
    assert torch.allclose(expert_0(x), x, atol=1e-5)
    # And expert 1's underlying weight is unchanged.
    assert torch.allclose(baseline, x, atol=1e-5)


def test_install_mote_partial_gain():
    engine = _make_engine(n_layers=1, n_experts=2, in_dim=4)
    expert = engine._expert_lists[0][0]
    x = torch.randn(3, 4)
    install_mote(engine, {0: {0: 0.3}})
    out = expert(x)
    assert torch.allclose(out, x * 0.3, atol=1e-5)


# ---------------------------------------------------------------------------
# remove_mote
# ---------------------------------------------------------------------------


def test_remove_mote_uninstalls_hooks_and_restores_behaviour():
    engine = _make_engine(n_layers=1, n_experts=2, in_dim=4)
    expert = engine._expert_lists[0][0]
    x = torch.randn(2, 4)
    handle = install_mote(engine, {0: {0: 0.0}})
    assert torch.allclose(expert(x), torch.zeros_like(x))

    n_removed = remove_mote(handle)
    assert n_removed == 1
    assert handle.n_hooked == 0
    assert torch.allclose(expert(x), x, atol=1e-5)


def test_remove_mote_idempotent():
    engine = _make_engine()
    handle = install_mote(engine, {0: {0: 0.0}})
    assert remove_mote(handle) == 1
    assert remove_mote(handle) == 0


# ---------------------------------------------------------------------------
# gains_from_safety_experts
# ---------------------------------------------------------------------------


def test_gains_from_safety_experts_builds_top_n():
    safety = {
        0: [(3, 0.9), (1, 0.8), (4, 0.7), (2, 0.5)],
        1: [(2, 0.95), (0, 0.6)],
    }
    gains = gains_from_safety_experts(safety, n_suppress=2)
    assert gains == {0: {3: 0.0, 1: 0.0}, 1: {2: 0.0, 0: 0.0}}


def test_gains_from_safety_experts_respects_suppress_gain():
    safety = {0: [(1, 0.9)]}
    gains = gains_from_safety_experts(safety, n_suppress=1, suppress_gain=0.1)
    assert gains == {0: {1: 0.1}}


def test_gains_from_safety_experts_empty_layer_skipped():
    safety = {0: [], 1: [(0, 0.9)]}
    gains = gains_from_safety_experts(safety, n_suppress=1)
    assert 0 not in gains
    assert gains[1] == {0: 0.0}


# ---------------------------------------------------------------------------
# Tuple output handling
# ---------------------------------------------------------------------------


def test_mote_hook_handles_tuple_output():
    """If a module returns (Tensor, ...), only the first element is scaled."""

    class TupleExpert(nn.Module):
        def forward(self, x):
            return (x * 2.0, "metadata")

    layer = SimpleNamespace(experts=[TupleExpert()])
    engine = SimpleNamespace(
        transformer_layers=[layer],
        steerable_modules=lambda i: {"mlp.down_proj": [layer.experts[0]]},
        get_n_layers=lambda: 1,
    )
    install_mote(engine, {0: {0: 0.5}})
    out = layer.experts[0](torch.tensor([1.0, 2.0]))
    assert torch.allclose(out[0], torch.tensor([1.0, 2.0]) * 2.0 * 0.5)
    assert out[1] == "metadata"
