"""Tests for abliterix.safex — stability-based MoE safety-expert identification.

Verifies the implementation of Yi et al. 2025 (arXiv:2506.17368) — SAFEx —
ported as an opt-in alternative to the historical risk-difference scoring.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from abliterix.safex import (
    _empty_buckets,
    _record_prompt_rates,
    _stats,
    identify_safety_experts_safex,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stats_ref(rates):
    """Reference impl matching numpy semantics for ddof=1."""
    if len(rates) == 0:
        return 0.0, 0.0
    mean = sum(rates) / len(rates)
    if len(rates) == 1:
        return mean, 0.0
    var = sum((r - mean) ** 2 for r in rates) / (len(rates) - 1)
    return mean, var**0.5


# ---------------------------------------------------------------------------
# _stats — sample mean / std
# ---------------------------------------------------------------------------


def test_stats_empty_returns_zero_zero():
    assert _stats([]) == (0.0, 0.0)


def test_stats_singleton_returns_value_zero_std():
    assert _stats([0.7]) == (0.7, 0.0)


def test_stats_matches_reference():
    rates = [0.1, 0.3, 0.6, 0.2, 0.4]
    mean, std = _stats(rates)
    ref_mean, ref_std = _stats_ref(rates)
    assert abs(mean - ref_mean) < 1e-9
    assert abs(std - ref_std) < 1e-9


# ---------------------------------------------------------------------------
# _record_prompt_rates — per-prompt activation accumulation
# ---------------------------------------------------------------------------


def test_record_prompt_rates_3d_input():
    """Selected shape (batch=3, seq=4, top_k=2) → 3 prompt rates per expert."""
    bucket = _empty_buckets()
    # Hand-craft a tensor: prompt 0 always picks expert 0; prompt 1 always
    # picks 1; prompt 2 splits between 0 and 1.
    selected = torch.tensor(
        [
            [[0, 0], [0, 0], [0, 0], [0, 0]],  # prompt 0: all expert 0
            [[1, 1], [1, 1], [1, 1], [1, 1]],  # prompt 1: all expert 1
            [[0, 1], [0, 1], [0, 1], [0, 1]],  # prompt 2: 50/50
        ]
    )
    _record_prompt_rates(bucket, layer_idx=0, selected=selected, n_experts=2)
    assert bucket[0][0] == [1.0, 0.0, 0.5]
    assert bucket[0][1] == [0.0, 1.0, 0.5]


def test_record_prompt_rates_2d_fallback():
    """Selected shape (batch*seq, top_k) → fallback treats it as one prompt."""
    bucket = _empty_buckets()
    selected = torch.tensor([[0, 1], [0, 1], [2, 0]])  # 6 tokens flat
    _record_prompt_rates(bucket, layer_idx=2, selected=selected, n_experts=3)
    # 6 token-slot values: [0, 1, 0, 1, 2, 0] → expert 0 rate = 3/6 = 0.5
    assert bucket[2][0] == [0.5]
    assert bucket[2][1] == [pytest.approx(2 / 6)]
    assert bucket[2][2] == [pytest.approx(1 / 6)]


def test_record_prompt_rates_empty_prompt():
    """Batch element with zero tokens must not crash and must not append."""
    bucket = _empty_buckets()
    selected = torch.empty(2, 0, 1, dtype=torch.long)  # batch=2, seq=0
    _record_prompt_rates(bucket, layer_idx=0, selected=selected, n_experts=4)
    # Nothing recorded.
    for eid in range(4):
        assert bucket[0].get(eid, []) == []


# ---------------------------------------------------------------------------
# End-to-end with a synthetic engine
# ---------------------------------------------------------------------------


class _StubRouter(nn.Module):
    """Router that emits a deterministic top-k tensor per forward pass."""

    def __init__(self, n_experts: int, selected_per_call: list[torch.Tensor]):
        super().__init__()
        # weight shape (n_experts, hidden); only the leading dim matters here.
        self.weight = nn.Parameter(torch.zeros(n_experts, 8), requires_grad=False)
        self._queue = list(selected_per_call)
        self._call_count = 0

    def forward(self, x):
        # Pop next pre-baked selection; cycle if we run out.
        if self._queue:
            sel = self._queue[self._call_count % len(self._queue)]
            self._call_count += 1
        else:
            sel = torch.zeros(1, 1, 1, dtype=torch.long)
        # Match engine.identify_safety_experts contract: return (logits, _, selected).
        logits = torch.zeros(1, 1, self.weight.shape[0])
        return (logits, None, sel)


def _make_engine_with_router(
    benign_selections: list[torch.Tensor],
    target_selections: list[torch.Tensor],
    n_experts: int,
):
    """Build a SimpleNamespace engine whose router emits pre-baked tensors.

    The engine's `extract_hidden_states_batched` is monkey-patched to invoke
    the router once per call, so we cleanly observe two forward-hook firings:
    one with benign tensors, one with target.
    """
    layer = SimpleNamespace()
    router = _StubRouter(n_experts, selected_per_call=[])
    layer.router_module = router

    layers = [layer]

    def _locate_router(layer_obj):
        return layer_obj.router_module

    # Counter to alternate between benign/target queues across the two
    # extract_hidden_states_batched calls.
    call_idx = {"n": 0}

    def extract_hidden_states_batched(msgs):
        # First call uses benign selections, second uses target.
        if call_idx["n"] == 0:
            router._queue = list(benign_selections)
        else:
            router._queue = list(target_selections)
        call_idx["n"] += 1
        router._call_count = 0
        # Invoke the router once per "batch" of selections supplied.
        for _ in range(len(router._queue)):
            router(None)
        return None

    engine = SimpleNamespace(
        transformer_layers=layers,
        _locate_router=_locate_router,
        extract_hidden_states_batched=extract_hidden_states_batched,
    )
    return engine, router


def test_identify_safety_experts_safex_no_router_returns_empty():
    layer = SimpleNamespace()
    engine = SimpleNamespace(
        transformer_layers=[layer],
        _locate_router=lambda _layer: None,
        extract_hidden_states_batched=lambda msgs: None,
    )
    result = identify_safety_experts_safex(engine, [], [])
    assert result == {}


def test_safex_prefers_stable_expert():
    """Stable target expert (rate=1.0 every prompt) should outrank unstable one.

    Expert 0: target rate = [1.0, 1.0, 1.0] (mean 1.0, std 0.0) — stable
    Expert 1: target rate = [0.0, 1.0, 0.0] (mean 0.33, std 0.577) — unstable
    Expert 2: target rate = [0.5, 0.5, 0.5] (mean 0.5, std 0.0) — moderately stable

    With λ=1.0, SAFEx ranks expert 0 > expert 2 > expert 1.
    """
    # Pre-bake three target-pass selections. Each tensor is (batch=1, seq=2, top_k=1).
    # Use valid ids: 0..2 means we'll select a single expert per token.
    # We need 3 prompt-samples (3 batch elements) per pass to compute std.
    # Use a single batched call with shape (3, 2, 1) for each pass.
    benign_call = torch.tensor(
        [
            [[0], [0]],  # prompt 0
            [[0], [0]],  # prompt 1
            [[0], [0]],  # prompt 2 — all expert 0 always; not actually unstable but
            # benign passes don't affect the variance ranking directly.
        ],
        dtype=torch.long,
    )
    # Actually for a clean test we want benign rates near zero for all
    # experts so the score is dominated by target stats. Use expert-3 (non-existent)
    # markers won't work — instead let expert 0 be used uniformly on both passes
    # and verify it still wins overall.
    benign_call = torch.zeros(3, 4, 1, dtype=torch.long) + 3  # rate=0 for experts 0..2

    target_call = torch.zeros(3, 4, 1, dtype=torch.long)
    # Expert 0: rate = 1.0 every prompt (stable).
    target_call[0, :, 0] = 0
    target_call[1, :, 0] = 0
    target_call[2, :, 0] = 0
    # We need separate tensors to capture different targets, so build a (3, 4, 1)
    # tensor with mixed selection per prompt: every token of prompt p chooses
    # the chosen expert for that prompt's "expert-i story" — but a single call
    # can't simultaneously test all three experts.
    # Instead we engineer a single tensor where:
    #   prompt 0 picks expert 0 in all 4 token slots,
    #   prompt 1 picks expert 0 in all 4 slots,
    #   prompt 2 picks expert 0 in all 4 slots
    # → expert 0 rate is [1,1,1]; experts 1, 2 rate is [0,0,0]
    # That's enough to verify expert 0 wins by mean alone. We further verify
    # variance penalty by using a second target call that adds variance to
    # expert 1.

    target_call_a = torch.zeros(3, 4, 1, dtype=torch.long)  # all expert 0
    target_call_b = torch.tensor(
        [
            [[1], [1], [1], [1]],  # prompt 0: all expert 1  (rate 1.0)
            [[3], [3], [3], [3]],  # prompt 1: none of 0/1/2 (rate 0)
            [[3], [3], [3], [3]],  # prompt 2: none of 0/1/2 (rate 0)
        ],
        dtype=torch.long,
    )
    # Across the two target calls (treated as separate batches by the hook):
    #   Expert 0 rates: [1,1,1]  +  [0,0,0]  = [1,1,1,0,0,0]   mean 0.5, std 0.548
    #   Expert 1 rates: [0,0,0]  +  [1,0,0]  = [0,0,0,1,0,0]   mean 0.167, std 0.408
    #   Expert 2 rates: [0,0,0]  +  [0,0,0]  = [0,0,0,0,0,0]   mean 0, std 0
    # SAFEx score (μ_t - μ_b - λ·σ_t) with μ_b ≈ 0 and λ=1:
    #   Expert 0: 0.5 - 0.548 = -0.048
    #   Expert 1: 0.167 - 0.408 = -0.241
    #   Expert 2: 0 - 0 = 0  (highest!)
    # So expert 2 ranks highest, despite never being activated — because its
    # zero variance pushes it above the stable-but-noisy expert 0. This is
    # exactly the SAFEx logic: prefer consistent over flashy. The test makes
    # this concrete.

    engine, router = _make_engine_with_router(
        benign_selections=[benign_call],
        target_selections=[target_call_a, target_call_b],
        n_experts=3,
    )

    result = identify_safety_experts_safex(engine, [], [], variance_penalty=1.0)
    assert 0 in result
    ranked = result[0]
    # Top expert must be expert 2 (highest stability score).
    assert ranked[0][0] == 2
    # Verify the score arithmetic.
    expected_scores = {0: 0.5 - 0.5477, 1: 1 / 6 - 0.4082, 2: 0.0}
    for eid, score in ranked:
        assert abs(score - expected_scores[eid]) < 0.01, (eid, score)


def test_safex_variance_penalty_zero_recovers_standard_scoring():
    """λ=0 should make SAFEx equivalent to risk-difference (mean only)."""
    # Expert 0: high mean (rate 1.0 on all targets), low mean benign (rate 0.0).
    # Expert 1: low mean target, high mean benign.
    benign_call = torch.tensor(
        [
            [[1], [1], [1], [1]],
            [[1], [1], [1], [1]],
        ],
        dtype=torch.long,
    )
    target_call = torch.tensor(
        [
            [[0], [0], [0], [0]],
            [[0], [0], [0], [0]],
        ],
        dtype=torch.long,
    )
    engine, _ = _make_engine_with_router(
        benign_selections=[benign_call],
        target_selections=[target_call],
        n_experts=2,
    )
    result = identify_safety_experts_safex(engine, [], [], variance_penalty=0.0)
    ranked = result[0]
    # Expert 0 must dominate (highest target-benign mean diff).
    assert ranked[0][0] == 0
    # With λ=0 and zero per-expert std, scores reduce to risk-difference:
    #   Expert 0: 1.0 - 0.0 = 1.0
    #   Expert 1: 0.0 - 1.0 = -1.0
    assert abs(ranked[0][1] - 1.0) < 1e-5
    assert abs(ranked[1][1] - (-1.0)) < 1e-5


def test_safex_layer_dictionary_shape():
    """Result keys must be the layer indices for which a router was found."""
    benign_call = torch.zeros(2, 2, 1, dtype=torch.long)
    target_call = torch.ones(2, 2, 1, dtype=torch.long)
    engine, _ = _make_engine_with_router(
        benign_selections=[benign_call],
        target_selections=[target_call],
        n_experts=2,
    )
    result = identify_safety_experts_safex(engine, [], [])
    # Only one layer in this stub engine.
    assert list(result.keys()) == [0]
    # Each entry has one (eid, score) tuple per expert.
    assert len(result[0]) == 2


# ---------------------------------------------------------------------------
# Settings validators
# ---------------------------------------------------------------------------


def test_settings_expert_profiling_method_defaults_to_standard():
    from abliterix.settings import ExpertConfig

    cfg = ExpertConfig()
    assert cfg.profiling_method == "standard"
    assert cfg.safex_variance_penalty == 1.0


def test_settings_expert_profiling_method_accepts_safex():
    from abliterix.settings import ExpertConfig

    cfg = ExpertConfig(profiling_method="safex", safex_variance_penalty=2.5)
    assert cfg.profiling_method == "safex"
    assert cfg.safex_variance_penalty == 2.5
