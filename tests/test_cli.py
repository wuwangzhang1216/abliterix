"""Integration tests for abliterix.cli — config loading, CLI aliases, and error handling."""

import sys
import os
import tempfile

import pytest


# ---------------------------------------------------------------------------
# CLI alias expansion (replicating run()'s argv rewriting)
# ---------------------------------------------------------------------------


def test_model_alias_expansion():
    """--model should be rewritten to --model.model-id before config parsing."""
    old_argv = sys.argv
    try:
        sys.argv = ["test", "--model", "org/my-model"]

        # Replicate the alias expansion from cli.run()
        _cli_aliases = {"--model": "--model.model-id"}
        for short, full in _cli_aliases.items():
            for i, arg in enumerate(sys.argv):
                if arg == short:
                    sys.argv[i] = full

        from abliterix.settings import AbliterixConfig

        config = AbliterixConfig()
        assert config.model.model_id == "org/my-model"
    finally:
        sys.argv = old_argv


def test_positional_model_id():
    """A trailing non-flag argument should be treated as the model ID."""
    old_argv = sys.argv
    try:
        sys.argv = ["test", "org/positional-model"]

        # Replicate positional-arg inference from cli.run()
        if (
            len(sys.argv) > 1
            and "--model.model-id" not in sys.argv
            and not sys.argv[-1].startswith("-")
        ):
            sys.argv.insert(-1, "--model.model-id")

        from abliterix.settings import AbliterixConfig

        config = AbliterixConfig()
        assert config.model.model_id == "org/positional-model"
    finally:
        sys.argv = old_argv


# ---------------------------------------------------------------------------
# TOML config loading
# ---------------------------------------------------------------------------


def test_custom_toml_config():
    """AbliterixConfig should load settings from a custom TOML file."""
    old_argv = sys.argv

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write('[model]\n')
        f.write('quant_method = "bnb_4bit"\n')
        f.write('[optimization]\n')
        f.write('num_trials = 42\n')
        f.write('[kl]\n')
        f.write('scale = 2.5\n')
        toml_path = f.name

    try:
        sys.argv = [
            "test",
            "--model.model-id",
            "test/toml-model",
            "--config",
            toml_path,
        ]
        from abliterix.settings import AbliterixConfig
        from abliterix.types import QuantMode

        config = AbliterixConfig()
        assert config.model.model_id == "test/toml-model"
        assert config.model.quant_method == QuantMode.BNB_4BIT
        assert config.optimization.num_trials == 42
        assert config.kl.scale == 2.5
    finally:
        sys.argv = old_argv
        os.unlink(toml_path)


# ---------------------------------------------------------------------------
# CLI override precedence
# ---------------------------------------------------------------------------


def test_cli_overrides_toml():
    """CLI flags should take precedence over TOML values."""
    old_argv = sys.argv

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write('[optimization]\n')
        f.write('num_trials = 100\n')
        toml_path = f.name

    try:
        sys.argv = [
            "test",
            "--model.model-id",
            "test/precedence",
            "--config",
            toml_path,
            "--optimization.num-trials",
            "999",
        ]
        from abliterix.settings import AbliterixConfig

        config = AbliterixConfig()
        assert config.optimization.num_trials == 999
    finally:
        sys.argv = old_argv
        os.unlink(toml_path)


# ---------------------------------------------------------------------------
# Validation error handling
# ---------------------------------------------------------------------------


def test_invalid_quant_method_in_cli():
    """Invalid enum values via CLI should raise an error."""
    old_argv = sys.argv
    try:
        sys.argv = [
            "test",
            "--model.model-id",
            "test/model",
            "--model.quant-method",
            "not_a_real_mode",
        ]
        from abliterix.settings import AbliterixConfig
        from pydantic import ValidationError

        with pytest.raises((ValidationError, SystemExit, Exception)):
            AbliterixConfig()
    finally:
        sys.argv = old_argv


# ---------------------------------------------------------------------------
# Environment variable support
# ---------------------------------------------------------------------------


def test_env_var_override():
    """AX_ prefixed environment variables should configure settings."""
    old_argv = sys.argv
    old_env = os.environ.get("AX_NON_INTERACTIVE")
    try:
        sys.argv = ["test", "--model.model-id", "test/env-model"]
        os.environ["AX_NON_INTERACTIVE"] = "true"

        from abliterix.settings import AbliterixConfig

        config = AbliterixConfig()
        assert config.non_interactive is True
    finally:
        sys.argv = old_argv
        if old_env is not None:
            os.environ["AX_NON_INTERACTIVE"] = old_env
        else:
            os.environ.pop("AX_NON_INTERACTIVE", None)


# ---------------------------------------------------------------------------
# Startup helpers (no GPU required)
# ---------------------------------------------------------------------------


def test_configure_libraries_disables_grad():
    """_configure_libraries should disable gradient tracking."""
    import torch
    from abliterix.cli import _configure_libraries

    _configure_libraries()
    assert not torch.is_grad_enabled()
