"""Unit tests for abliterix.core.vllm_compat — version gate + env auto-set.

These tests do not import vLLM. They mock ``importlib.metadata.version``
and an in-memory env dict to exercise the compat helpers in isolation, so
they run on CPU-only CI.
"""

from __future__ import annotations

import warnings
from unittest.mock import patch

import pytest

from abliterix.core.vllm_compat import (
    _MAX_VLLM_EXCLUSIVE,
    _MIN_VLLM,
    _parse_version,
    check_vllm_version,
    ensure_vllm_env,
)


# ---------------------------------------------------------------------------
# _parse_version
# ---------------------------------------------------------------------------


def test_parse_version_handles_simple():
    assert _parse_version("0.20.1") == (0, 20, 1)
    assert _parse_version("0.18.0") == (0, 18, 0)
    assert _parse_version("1.0.0") == (1, 0, 0)


def test_parse_version_handles_dev_suffix():
    # vLLM commonly publishes pre-releases like 0.20.1rc1 or 0.21.0.dev42.
    # The parser stops at the first non-numeric character of each chunk,
    # which keeps the version comparable.
    assert _parse_version("0.20.1rc1") == (0, 20, 1)
    assert _parse_version("0.21.0.dev42") == (0, 21, 0)
    assert _parse_version("0.18.0+cu130") == (0, 18, 0)


def test_parse_version_strips_v_prefix():
    """PR #21 review item 6: ``v0.20.1`` should parse to (0, 20, 1).
    Although ``importlib.metadata.version`` never returns the prefix,
    callers may pass it directly; we should accept it."""
    assert _parse_version("v0.20.1") == (0, 20, 1)
    assert _parse_version("v0.18.0") == (0, 18, 0)
    # Without lstrip("v") this used to return (0,) — the parser would
    # see no leading digits and bail.
    assert _parse_version("v0.20.1") != (0,)


def test_check_vllm_version_accepts_v_prefix():
    """check_vllm_version should also tolerate the ``v`` prefix end-to-end."""
    assert check_vllm_version("v0.20.1") == (0, 20, 1)


def test_parse_version_handles_short():
    assert _parse_version("1.0") == (1, 0)
    assert _parse_version("2") == (2,)


# ---------------------------------------------------------------------------
# check_vllm_version
# ---------------------------------------------------------------------------


def test_check_vllm_version_accepts_floor():
    floor = ".".join(str(p) for p in _MIN_VLLM)
    assert check_vllm_version(floor) == _MIN_VLLM


def test_check_vllm_version_accepts_in_range():
    assert check_vllm_version("0.20.1") == (0, 20, 1)
    assert check_vllm_version("0.19.0") == (0, 19, 0)


def test_check_vllm_version_rejects_below_floor():
    with pytest.raises(ImportError, match="abliterix requires"):
        check_vllm_version("0.11.0")


def test_check_vllm_version_rejects_very_old():
    with pytest.raises(ImportError):
        check_vllm_version("0.8.0")


def test_check_vllm_version_warns_above_ceiling():
    above_ceiling = ".".join(str(p) for p in _MAX_VLLM_EXCLUSIVE)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_vllm_version(above_ceiling)
    assert any("newer than the highest" in str(w.message) for w in caught), (
        f"expected upper-bound warning, got: {[str(w.message) for w in caught]}"
    )


def test_check_vllm_version_raises_when_uninstalled():
    # Simulate vllm not being importable by making
    # importlib.metadata.version raise PackageNotFoundError. The import
    # is local to check_vllm_version, so patch the metadata module itself.
    from importlib.metadata import PackageNotFoundError

    with patch(
        "importlib.metadata.version",
        side_effect=PackageNotFoundError("vllm"),
    ):
        with pytest.raises(ImportError, match="not installed"):
            check_vllm_version()


# ---------------------------------------------------------------------------
# ensure_vllm_env
# ---------------------------------------------------------------------------


def test_ensure_vllm_env_sets_base_only_by_default():
    env: dict[str, str] = {}
    written = ensure_vllm_env(env=env)
    assert "FLASHINFER_DISABLE_VERSION_CHECK" in written
    assert env["FLASHINFER_DISABLE_VERSION_CHECK"] == "1"
    # Without needs_collective_rpc, the insecure-serialization flag stays unset.
    assert "VLLM_ALLOW_INSECURE_SERIALIZATION" not in written
    assert "VLLM_ALLOW_INSECURE_SERIALIZATION" not in env


def test_ensure_vllm_env_adds_rpc_flag_when_requested():
    env: dict[str, str] = {}
    written = ensure_vllm_env(needs_collective_rpc=True, env=env)
    assert "FLASHINFER_DISABLE_VERSION_CHECK" in written
    assert "VLLM_ALLOW_INSECURE_SERIALIZATION" in written
    assert env["VLLM_ALLOW_INSECURE_SERIALIZATION"] == "1"


def test_ensure_vllm_env_never_overwrites_user_value():
    env = {
        "FLASHINFER_DISABLE_VERSION_CHECK": "0",  # user explicitly disabled
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "0",
    }
    written = ensure_vllm_env(needs_collective_rpc=True, env=env)
    # Nothing got written because every key was already present.
    assert written == {}
    # User values preserved.
    assert env["FLASHINFER_DISABLE_VERSION_CHECK"] == "0"
    assert env["VLLM_ALLOW_INSECURE_SERIALIZATION"] == "0"


def test_ensure_vllm_env_is_idempotent():
    env: dict[str, str] = {}
    first = ensure_vllm_env(needs_collective_rpc=True, env=env)
    second = ensure_vllm_env(needs_collective_rpc=True, env=env)
    # Second call is a no-op because everything is set.
    assert first
    assert second == {}


def test_ensure_vllm_env_does_not_set_deprecated_var():
    """The old ``VLLM_FUSED_MOE_UNQUANTIZED_BACKEND`` must NOT be set —
    vLLM 0.20.x logs an "Unknown vLLM environment variable" warning when
    it sees this name. Backend selection is now config-driven via
    ``ModelConfig.moe_backend``."""
    env: dict[str, str] = {}
    ensure_vllm_env(needs_collective_rpc=True, env=env)
    assert "VLLM_FUSED_MOE_UNQUANTIZED_BACKEND" not in env
