"""Shared fixtures for Abliterix tests."""

import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


# AbliterixConfig requires --model via CLI.  Inject a dummy value once.
sys.argv = ["test", "--model.model-id", "test/model-001"]

from abliterix.settings import AbliterixConfig  # noqa: E402


@pytest.fixture
def abliterix_config():
    """AbliterixConfig with dummy model ID (no TOML file needed)."""
    return AbliterixConfig()


@pytest.fixture
def synthetic_states():
    """Paired (benign, target) residual tensors: shape (20, 8, 64) each.

    Target states have a systematic offset so steering vectors are non-trivial.
    """
    torch.manual_seed(42)
    benign = torch.randn(20, 8, 64)
    offset = torch.randn(1, 8, 64) * 0.5
    target = benign + offset + torch.randn(20, 8, 64) * 0.1
    return benign, target


@pytest.fixture
def steering_vectors():
    """Unit-normalized vectors: shape (8, 64)."""
    torch.manual_seed(42)
    v = torch.randn(8, 64)
    return F.normalize(v, p=2, dim=1)


@pytest.fixture
def sample_trial_attrs():
    """Dict mimicking trial.user_attrs for extract_trial_params / format_trial_params."""
    return {
        "vector_index": 5.5,
        "parameters": {
            "q_proj": {
                "max_weight": 1.2,
                "max_weight_position": 6.0,
                "min_weight": 0.1,
                "min_weight_distance": 3.0,
            },
            "v_proj": {
                "max_weight": 0.8,
                "max_weight_position": 5.0,
                "min_weight": 0.05,
                "min_weight_distance": 2.0,
            },
        },
        "kl_divergence": 0.0123,
        "refusals": 3,
    }


@pytest.fixture
def mock_trial(sample_trial_attrs):
    """SimpleNamespace with .user_attrs, usable where an Optuna Trial is expected."""
    return SimpleNamespace(user_attrs=sample_trial_attrs)
