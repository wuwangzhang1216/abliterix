"""Unit tests for abliterix.core.vllm_compilation_config — dict shape only.

We assert on the keys and values returned by ``build()`` rather than
importing vLLM. The constants 0/1/2/3 used for ``mode`` and
``cudagraph_mode`` are fixed in vLLM 0.20.x; if upstream renumbers them
this test will break loudly.
"""

from __future__ import annotations

import pytest

from abliterix.core.vllm_compilation_config import _MOE_SPLITTING_OPS, build


def test_build_eager_mode():
    cfg = build("eager")
    assert cfg == {"mode": 0, "cudagraph_mode": 0}


def test_build_eager_is_default():
    assert build() == build("eager")


def test_build_full_compile_mode():
    cfg = build("full_compile")
    assert cfg["mode"] == 3
    assert cfg["cudagraph_mode"] == 2
    # Full compile must NOT add splitting_ops or static_all_moe_layers —
    # those only apply to the MoE-aware path.
    assert "splitting_ops" not in cfg
    assert "static_all_moe_layers" not in cfg


def test_build_moe_eager_rest_compile_requires_layer_indices():
    with pytest.raises(ValueError, match="moe_layer_indices is required"):
        build("moe_eager_rest_compile")


def test_build_moe_eager_rest_compile_populates_indices():
    indices = [0, 2, 4, 6]
    cfg = build("moe_eager_rest_compile", moe_layer_indices=indices)
    assert cfg["mode"] == 3
    assert cfg["cudagraph_mode"] == 1
    assert cfg["splitting_ops"] == list(_MOE_SPLITTING_OPS)
    assert cfg["static_all_moe_layers"] == indices


def test_build_moe_eager_rest_compile_copies_indices():
    """The dict must contain a fresh list, not the caller's mutable input."""
    indices = [0, 2, 4]
    cfg = build("moe_eager_rest_compile", moe_layer_indices=indices)
    indices.append(99)
    assert cfg["static_all_moe_layers"] == [0, 2, 4]


def test_build_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown compile mode"):
        build("not_a_real_mode")  # type: ignore[arg-type]


def test_build_eager_ignores_layer_indices():
    """Passing moe_layer_indices to 'eager' is harmless — eager mode does
    not need them and they should not appear in the returned dict."""
    cfg = build("eager", moe_layer_indices=[1, 2, 3])
    assert "static_all_moe_layers" not in cfg
    assert "splitting_ops" not in cfg


def test_moe_splitting_ops_includes_fused_moe():
    """Sanity check on the splitting-op list — the ``fused_moe`` op is
    the one vLLM uses on every MoE forward, so missing it would defeat
    the purpose of the moe_eager_rest_compile mode."""
    assert "fused_moe" in _MOE_SPLITTING_OPS
