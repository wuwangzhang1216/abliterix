# Abliterix
# Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""SAFEx-style stability-based MoE safety-expert identification.

Implements the stability-aware scoring from `Yi et al., 2025
<https://arxiv.org/abs/2506.17368>`_ — *SAFEx: Identifying Safety-Critical
Experts in Mixture-of-Experts LLMs*.

The historical abliterix profiler (``SteeringEngine.identify_safety_experts``)
scores each expert by ``mean(target_activation_rate) − mean(benign_rate)``.
That picks up experts that fire *on average* more for harmful prompts —
but doesn't distinguish a *stable* safety expert (fires on ~every harmful
prompt) from an unstable one (fires sporadically). Ablating unstable
experts wastes budget and risks capability damage; ablating stable ones
yields the precise reduction Yi et al. report (12 experts → 22 % drop in
refusal rate).

SAFEx scoring
-------------
For each expert ``e`` in each MoE layer:

* ``μ_t`` = mean per-prompt activation rate on harmful prompts
* ``σ_t`` = standard deviation of per-prompt rate on harmful prompts
* ``μ_b`` = mean per-prompt activation rate on benign prompts

``score(e) = (μ_t − μ_b) − λ · σ_t``

The variance penalty ``λ`` (``safex_variance_penalty``, default 1.0)
demotes experts whose harmful-prompt activation is noisy. Experts with
high mean and low std are preferred — these are the "stable" detection
or control experts the paper identifies.

Implementation
--------------
Reuses the engine's router hooks but accumulates per-batch-element
activation rates instead of pooled token counts. Each batch element
becomes a "prompt sample" for the variance estimate. Works with any MoE
architecture for which ``engine._locate_router`` finds a router module.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import torch
from torch import Tensor
from torch.nn import Module


def _empty_buckets() -> defaultdict[int, defaultdict[int, list[float]]]:
    """Per-layer → per-expert → list of per-prompt activation rates."""
    return defaultdict(lambda: defaultdict(list))


def _record_prompt_rates(
    bucket: defaultdict[int, defaultdict[int, list[float]]],
    layer_idx: int,
    selected: Tensor,
    n_experts: int,
) -> None:
    """Accumulate per-prompt expert activation rates for one batch.

    ``selected`` shape: ``(batch, seq, top_k)`` — top-k expert ids per
    token, as produced by the router forward hook. For each batch
    element we compute ``(count_of_expert_e / total_tokens)`` and append
    to the per-(layer, expert) list.
    """
    if selected.dim() == 2:
        # Some routers emit (batch*seq, top_k); reshape back is impossible
        # without knowing seq, so treat the whole batch as one prompt.
        flat = selected.reshape(-1)
        for eid in range(n_experts):
            rate = (flat == eid).float().mean().item()
            bucket[layer_idx][eid].append(rate)
        return

    batch_size = selected.shape[0]
    flat_per_prompt = selected.reshape(batch_size, -1)
    for b in range(batch_size):
        prompt_tokens = flat_per_prompt[b]
        if prompt_tokens.numel() == 0:
            continue
        for eid in range(n_experts):
            rate = (prompt_tokens == eid).float().mean().item()
            bucket[layer_idx][eid].append(rate)


def _stats(rates: list[float]) -> tuple[float, float]:
    """Return ``(mean, std)`` of a list of activation rates.

    Uses sample std (``ddof=1``) when more than one sample is available;
    falls back to 0 std on a singleton.
    """
    n = len(rates)
    if n == 0:
        return 0.0, 0.0
    mean = sum(rates) / n
    if n == 1:
        return mean, 0.0
    var = sum((r - mean) ** 2 for r in rates) / (n - 1)
    return mean, math.sqrt(max(var, 0.0))


def identify_safety_experts_safex(
    engine,
    benign_msgs: list[Any],
    target_msgs: list[Any],
    *,
    variance_penalty: float = 1.0,
) -> dict[int, list[tuple[int, float]]]:
    """Per-layer ranked expert list using SAFEx stability-based scoring.

    Parameters
    ----------
    engine : SteeringEngine
        Must have a loaded HF model with a discoverable router per layer
        (``engine._locate_router``).
    benign_msgs, target_msgs : list[Any]
        Chat-message lists already prepared by the caller (same format
        ``extract_hidden_states_batched`` expects).
    variance_penalty : float
        Multiplier on the harmful-prompt activation std. Higher = harder
        on unstable experts. Default 1.0 matches the paper's recipe.

    Returns
    -------
    dict[int, list[tuple[int, float]]]
        ``{layer_idx: [(expert_idx, score), ...]}`` sorted descending.
        Same shape as ``identify_safety_experts`` so the optimiser slots
        it in transparently.
    """
    layers = engine.transformer_layers
    gates: dict[int, Module] = {}
    for idx in range(len(layers)):
        g = engine._locate_router(layers[idx])
        if g is not None:
            gates[idx] = g

    if not gates:
        return {}

    benign_bucket = _empty_buckets()
    target_bucket = _empty_buckets()
    active = [benign_bucket]
    handles = []

    def _make_hook(layer_idx: int, n_experts: int):
        def hook(module: Module, inp: Any, out: Any):
            with torch.no_grad():
                # Router output shapes vary by family — extract the
                # top-k selection tensor in the same order the engine's
                # canonical hook does (see engine.identify_safety_experts).
                if isinstance(out, tuple) and len(out) >= 3:
                    selected = out[2]
                elif isinstance(out, tuple) and len(out) == 2:
                    selected = out[1]
                else:
                    logits = out if not isinstance(out, tuple) else out[0]
                    k = getattr(module, "top_k", 8)
                    _, selected = logits.topk(k, dim=-1)

                _record_prompt_rates(active[0], layer_idx, selected, n_experts)

        return hook

    # Inspect router weight to learn n_experts per layer (most routers store
    # weights as (n_experts, hidden) — same convention the engine relies on).
    n_experts_per_layer = {idx: gate.weight.shape[0] for idx, gate in gates.items()}  # type: ignore[union-attr]

    for idx, gate in gates.items():
        handles.append(
            gate.register_forward_hook(_make_hook(idx, n_experts_per_layer[idx]))
        )

    print("  [SAFEx] Profiling benign prompts (per-prompt stability stats)...")
    active[0] = benign_bucket
    with torch.no_grad():
        engine.extract_hidden_states_batched(benign_msgs)

    print("  [SAFEx] Profiling target prompts...")
    active[0] = target_bucket
    with torch.no_grad():
        engine.extract_hidden_states_batched(target_msgs)

    for h in handles:
        h.remove()

    # Compute SAFEx score per (layer, expert).
    safety: dict[int, list[tuple[int, float]]] = {}
    for idx in gates.keys():
        n_experts = n_experts_per_layer[idx]
        scores: list[tuple[int, float]] = []
        for eid in range(n_experts):
            b_mean, _b_std = _stats(benign_bucket[idx].get(eid, []))
            t_mean, t_std = _stats(target_bucket[idx].get(eid, []))
            score = (t_mean - b_mean) - variance_penalty * t_std
            scores.append((eid, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        safety[idx] = scores

    n_layers = len(safety)
    top_scores = [safety[i][0][1] for i in sorted(safety) if safety[i]]
    avg = sum(top_scores) / len(top_scores) if top_scores else 0
    print(
        f"  [SAFEx] Profiled {n_layers} MoE layers (λ={variance_penalty:.2f}), "
        f"avg top stability score: {avg:.4f}"
    )

    return safety
