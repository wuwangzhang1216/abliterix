# Abliterix
# Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Mixture of Tunable Experts (MoTE) — inference-time expert-gain modulation.

Implements `Bai et al., 2025 <https://arxiv.org/abs/2502.11096>`_ —
*Mixture of Tunable Experts: Behavior Modification of DeepSeek-R1 at
Inference Time*.

Where abliterix's existing router-suppression path edits the router's
weight matrix (so a particular expert routes less often), MoTE leaves
the router untouched and instead **scales each expert's output**
multiplicatively via a forward hook:

    expert_output ← gain[layer, expert] · expert_output

Setting ``gain = 0.0`` is equivalent to disabling the expert without
changing the routing; ``gain < 1`` partially suppresses; ``gain > 1``
amplifies. Because the hook runs at inference time the change is
**fully reversible** — no weights mutated, no cache invalidation, no
re-load. This is the right primitive when:

* You want to A/B test which expert removal helps before committing to
  a weight edit;
* The MoE checkpoint is FP8 / MXFP4 / frozen (no editable BF16
  Parameters);
* You're running under vLLM TP and the router-suppression collective_rpc
  surface is too heavy to install per-trial.

The implementation hooks every per-expert module that abliterix's
``steerable_modules(layer_idx)`` returns under ``mlp.down_proj`` — that
key covers Mixtral, Qwen3-MoE (non-fused), DeepSeek-V2/V3, Phi-3.5-MoE,
Granite-MoE, etc. Fused-3-D MoE containers (gpt-oss MXFP4,
GptOssExperts) are *not* covered — for those, abliterix's existing
EGA / router-suppression paths still apply.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from torch import Tensor
from torch.nn import Module


# ---------------------------------------------------------------------------
# Hook factory
# ---------------------------------------------------------------------------


def _make_gain_hook(gain: float):
    """Forward-hook closure that scales ``output`` by ``gain``.

    Handles both Tensor outputs and ``tuple[Tensor, ...]`` outputs (some
    expert modules return additional metadata; we scale the first element).
    """

    def hook(_module: Module, _inp: Any, output: Any):
        if isinstance(output, Tensor):
            return output * gain
        if isinstance(output, tuple) and output and isinstance(output[0], Tensor):
            return (output[0] * gain, *output[1:])
        return output

    return hook


# ---------------------------------------------------------------------------
# Install / remove
# ---------------------------------------------------------------------------


@dataclass
class MoTEHandle:
    """Handle returned by :func:`install_mote` to be passed to :func:`remove_mote`."""

    handles: list[Any]
    layer_expert_gains: Mapping[int, Mapping[int, float]]

    @property
    def n_hooked(self) -> int:
        return len(self.handles)


def install_mote(
    engine,
    layer_expert_gains: Mapping[int, Mapping[int, float]],
) -> MoTEHandle:
    """Install per-expert output-scaling hooks on a loaded HF model.

    Parameters
    ----------
    engine : SteeringEngine
        Must have an HF model loaded; the function uses
        ``engine.transformer_layers`` and ``engine.steerable_modules``.
    layer_expert_gains : Mapping[int, Mapping[int, float]]
        ``{layer_idx: {expert_idx: gain}}``.  Missing layers or experts
        are left at their natural gain of 1.0 (no hook installed).

    Returns
    -------
    MoTEHandle
        Pass to :func:`remove_mote` to uninstall.

    Notes
    -----
    Experts are hooked in the order ``steerable_modules`` lists them for
    the ``"mlp.down_proj"`` key — this matches the per-expert ordering
    used by abliterix's EGA and router-suppression paths. Fused MoE
    containers (e.g. gpt-oss MXFP4) do not register per-expert Modules
    under that key, so this function silently skips them; use the
    existing weight-edit path for those.
    """
    handles: list[Any] = []
    n_layers = (
        engine.get_n_layers()
        if hasattr(engine, "get_n_layers")
        else len(engine.transformer_layers)
    )

    for layer_idx, expert_gains in layer_expert_gains.items():
        if not 0 <= layer_idx < n_layers:
            continue
        if not expert_gains:
            continue

        modules = engine.steerable_modules(layer_idx)
        expert_modules = modules.get("mlp.down_proj", [])
        if not expert_modules:
            continue

        for expert_idx, gain in expert_gains.items():
            # Floats close to 1.0 are no-ops — skip to keep the hook list short.
            if abs(gain - 1.0) < 1e-9:
                continue
            if not 0 <= expert_idx < len(expert_modules):
                continue
            module = expert_modules[expert_idx]
            handle = module.register_forward_hook(_make_gain_hook(float(gain)))
            handles.append(handle)

    return MoTEHandle(handles=handles, layer_expert_gains=layer_expert_gains)


def remove_mote(handle: MoTEHandle) -> int:
    """Uninstall every hook registered by :func:`install_mote`.

    Returns the number of hooks removed. Safe to call multiple times —
    subsequent calls return 0.
    """
    n = 0
    for h in handle.handles:
        try:
            h.remove()
            n += 1
        except Exception:
            pass
    handle.handles.clear()
    return n


# ---------------------------------------------------------------------------
# Convenience: build a gain map from a safety-expert ranking
# ---------------------------------------------------------------------------


def gains_from_safety_experts(
    safety_experts: Mapping[int, list[tuple[int, float]]],
    *,
    n_suppress: int,
    suppress_gain: float = 0.0,
) -> dict[int, dict[int, float]]:
    """Build a ``layer_expert_gains`` map that zeroes the top-N safety experts.

    Parameters
    ----------
    safety_experts : Mapping[int, list[tuple[int, float]]]
        Output of ``engine.identify_safety_experts`` or
        :func:`abliterix.safex.identify_safety_experts_safex`:
        ``{layer_idx: [(expert_idx, score), ...]}`` already sorted
        descending.
    n_suppress : int
        How many top-scoring experts per layer to suppress.
    suppress_gain : float
        Gain applied to those experts.  ``0.0`` (default) is full
        suppression; ``0.1`` keeps a small residual signal.

    Returns
    -------
    dict[int, dict[int, float]]
        Ready to pass to :func:`install_mote`.
    """
    gains: dict[int, dict[int, float]] = {}
    for layer_idx, ranked in safety_experts.items():
        per_layer = {eid: float(suppress_gain) for eid, _score in ranked[:n_suppress]}
        if per_layer:
            gains[layer_idx] = per_layer
    return gains
