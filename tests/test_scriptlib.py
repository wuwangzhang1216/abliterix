"""Tests for abliterix.scriptlib — trial parameter extraction.

Uses mock trial objects (no checkpoint files needed).
"""

from types import SimpleNamespace

from abliterix.scriptlib import extract_trial_params
from abliterix.types import ExpertRoutingConfig, SteeringProfile


def _trial_with(attrs: dict) -> SimpleNamespace:
    return SimpleNamespace(user_attrs=attrs)


# ---------------------------------------------------------------------------
# extract_trial_params — dense models
# ---------------------------------------------------------------------------


def test_extract_dense_model(sample_trial_attrs):
    trial = _trial_with(sample_trial_attrs)
    vi, profiles, routing = extract_trial_params(trial)
    assert vi == 5.5
    assert isinstance(profiles, dict)
    assert routing is None


def test_profiles_are_steering_profile(sample_trial_attrs):
    trial = _trial_with(sample_trial_attrs)
    _, profiles, _ = extract_trial_params(trial)
    for v in profiles.values():
        assert isinstance(v, SteeringProfile)


def test_extract_per_layer():
    attrs = {
        "vector_index": None,
        "parameters": {
            "q_proj": {
                "max_weight": 1.0,
                "max_weight_position": 5.0,
                "min_weight": 0.1,
                "min_weight_distance": 2.0,
            }
        },
    }
    vi, _, _ = extract_trial_params(_trial_with(attrs))
    assert vi is None


# ---------------------------------------------------------------------------
# extract_trial_params — MoE models
# ---------------------------------------------------------------------------


def test_extract_moe_model(sample_trial_attrs):
    attrs = {
        **sample_trial_attrs,
        "moe_parameters": {
            "n_suppress": 5,
            "router_bias": -3.0,
            "expert_ablation_weight": 1.5,
        },
    }
    _, _, routing = extract_trial_params(_trial_with(attrs))
    assert isinstance(routing, ExpertRoutingConfig)
    assert routing.n_suppress == 5
    assert routing.router_bias == -3.0
    assert routing.expert_ablation_weight == 1.5


def test_missing_moe_parameters(sample_trial_attrs):
    """Without moe_parameters key, routing should be None."""
    trial = _trial_with(sample_trial_attrs)
    _, _, routing = extract_trial_params(trial)
    assert routing is None
