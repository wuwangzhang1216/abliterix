# Abliterix — vLLM 0.20.x compilation_config builder
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Single-purpose deep module that constructs the dict passed to vLLM's
``compilation_config`` LLM kwarg.

Why this exists: vLLM 0.20.x's compilation_config schema is a moving target
(``mode``, ``cudagraph_mode``, ``splitting_ops``, ``static_all_moe_layers``,
``fast_moe_cold_start``, ...).  Centralising the dict construction here lets
the rest of abliterix express intent ("eager", "MoE-eager rest-graph", etc.)
without touching schema details.  When the schema changes again in v0.21
only this module needs an update.
"""

from __future__ import annotations

from typing import Any, Literal

CompileMode = Literal["eager", "moe_eager_rest_compile", "full_compile"]

# MoE op names that the v0.20.x compiler recognises as splitting points.
# When listed in ``splitting_ops``, these ops are NOT captured into a CUDA
# graph, so PyTorch forward hooks attached to MoE modules continue to fire.
# vLLM accepts either bare names (``"fused_moe"``) or fully-qualified
# torch.ops names (``"vllm.fused_moe"``); the bare form is forward-compatible
# with the upstream registry helpers.
_MOE_SPLITTING_OPS: tuple[str, ...] = (
    "fused_moe",
    "moe_forward",
    "fused_experts",
    "moe_align_block_size",
)


def build(
    mode: CompileMode = "eager",
    *,
    moe_layer_indices: list[int] | None = None,
) -> dict[str, Any]:
    """Build the dict to pass as ``LLM(compilation_config=...)``.

    Parameters
    ----------
    mode
        High-level intent.  Maps to:

        * ``"eager"`` — equivalent to ``enforce_eager=True``.  All CUDA
          graphs off; every forward hook fires.  Use when MoE editing or
          attention editing is active.
        * ``"moe_eager_rest_compile"`` — capture CUDA graphs for non-MoE
          layers; keep MoE layers eager so router-suppression and
          expert-edit hooks survive.  Adds ``splitting_ops`` for fused
          MoE kernels and populates ``static_all_moe_layers`` so vLLM
          treats those layer indices as static-shape.
        * ``"full_compile"`` — full vLLM compile + CUDA graph capture
          everywhere.  No MoE editing supported.  Dense-only.

    moe_layer_indices
        Layer indices to mark static for the ``moe_eager_rest_compile``
        path.  Required when ``mode == "moe_eager_rest_compile"``.
        Ignored otherwise.

    Returns
    -------
    dict
        Ready to pass directly to ``LLM(compilation_config=...)``.

    Raises
    ------
    ValueError
        If ``mode`` is unknown or ``moe_layer_indices`` is missing for the
        MoE-aware path.
    """
    if mode == "eager":
        # CompilationMode.NONE + CUDAGraphMode.NONE — same effect as
        # enforce_eager=True at the LLM() level. Kept as a separate path
        # so callers can express the choice through compilation_config
        # uniformly without juggling enforce_eager.
        return {"mode": 0, "cudagraph_mode": 0}

    if mode == "moe_eager_rest_compile":
        if not moe_layer_indices:
            raise ValueError(
                "moe_layer_indices is required for "
                "mode='moe_eager_rest_compile'; pass the list of MoE layer "
                "indices that must remain eager."
            )
        return {
            # CompilationMode.VLLM_COMPILE — vLLM's standard compile path.
            "mode": 3,
            # CUDAGraphMode.PIECEWISE — capture per-piece graphs and skip
            # the splitting_ops entries.
            "cudagraph_mode": 1,
            "splitting_ops": list(_MOE_SPLITTING_OPS),
            "static_all_moe_layers": list(moe_layer_indices),
        }

    if mode == "full_compile":
        # CompilationMode.VLLM_COMPILE + CUDAGraphMode.FULL.  No splitting,
        # no static MoE list — the model is compiled end-to-end.  Forward
        # hooks attached after warmup are silently dropped under this mode
        # (PyTorch issue #117758) so abliterix's MoE editor cannot run.
        return {"mode": 3, "cudagraph_mode": 2}

    raise ValueError(
        f"Unknown compile mode: {mode!r}. "
        "Expected one of: 'eager', 'moe_eager_rest_compile', 'full_compile'."
    )
