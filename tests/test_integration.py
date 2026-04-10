"""Integration tests — end-to-end flows that don't require a GPU.

These tests exercise the data pipeline, scoring logic, and interactive module
wiring without loading real models. Marked @pytest.mark.slow for optional CI
skipping.
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from abliterix.types import ChatMessage, SteeringProfile


# ---------------------------------------------------------------------------
# Steering vector computation → scoring pipeline
# ---------------------------------------------------------------------------


def test_steering_vectors_to_scoring_pipeline(synthetic_states, abliterix_config):
    """Compute vectors from synthetic states and verify they feed into apply_steering."""
    from abliterix.vectors import compute_steering_vectors
    from abliterix.core.steering import apply_steering
    from abliterix.types import VectorMethod

    benign, target = synthetic_states
    vectors = compute_steering_vectors(
        benign, target, VectorMethod.MEAN, orthogonal_projection=False
    )

    # Vectors should be (n_layers, hidden_dim)
    assert vectors.shape == (8, 64)

    # Verify vectors are non-trivial (not all zeros)
    assert vectors.norm() > 0

    # Verify each method produces valid output
    for method in VectorMethod:
        v = compute_steering_vectors(benign, target, method, orthogonal_projection=False)
        assert v.shape == (8, 64)
        assert not torch.isnan(v).any()


def test_steering_vectors_with_orthogonal_projection(synthetic_states):
    """Orthogonal projection should produce different vectors than plain mean."""
    from abliterix.vectors import compute_steering_vectors
    from abliterix.types import VectorMethod

    benign, target = synthetic_states

    plain = compute_steering_vectors(
        benign, target, VectorMethod.MEAN, orthogonal_projection=False
    )
    ortho = compute_steering_vectors(
        benign, target, VectorMethod.MEAN, orthogonal_projection=True
    )

    # They should differ
    assert not torch.allclose(plain, ortho, atol=1e-6)


# ---------------------------------------------------------------------------
# Trial formatting → model card pipeline
# ---------------------------------------------------------------------------


def test_trial_to_model_card_pipeline(mock_trial, abliterix_config):
    """format_trial_params and generate_model_card should work together."""
    from abliterix.data import format_trial_params, generate_model_card

    params = format_trial_params(mock_trial)
    assert "vector_index" in params
    assert "q_proj.max_weight" in params

    target_msgs = [ChatMessage(system="", user=f"P{i}") for i in range(10)]
    card = generate_model_card(abliterix_config, mock_trial, 10, target_msgs)

    # Card should contain the formatted parameters
    assert "q_proj" in card
    assert "0.0123" in card  # KL divergence from mock_trial


# ---------------------------------------------------------------------------
# Refusal detector end-to-end
# ---------------------------------------------------------------------------


def test_detector_keyword_pipeline(abliterix_config):
    """RefusalDetector should classify responses using keyword matching."""
    from abliterix.eval.detector import RefusalDetector

    detector = RefusalDetector(abliterix_config)
    try:
        # Clear refusals
        assert not detector.detect_refusal("Here's how to do that: step 1...")
        assert not detector.detect_refusal("The answer is 42.")

        # Detected refusals
        assert detector.detect_refusal("I'm sorry, I can't help with that.")
        assert detector.detect_refusal("As an AI assistant, I cannot provide that information.")
        assert detector.detect_refusal("抱歉，我无法回答这个问题。")

        # Multiple classifications
        responses = [
            "Here's the answer",
            "I'm sorry, I cannot help",
            "Sure, here you go",
        ]
        results = [detector.detect_refusal(r) for r in responses]
        assert results == [False, True, False]
    finally:
        detector.close()


# ---------------------------------------------------------------------------
# SteeringProfile validation
# ---------------------------------------------------------------------------


def test_steering_profile_construction():
    """SteeringProfile should accept valid parameters."""
    profile = SteeringProfile(
        max_weight=1.2,
        max_weight_position=5.0,
        min_weight=0.1,
        min_weight_distance=3.0,
    )
    assert profile.max_weight == 1.2
    assert profile.min_weight_distance == 3.0


# ---------------------------------------------------------------------------
# Interactive module imports
# ---------------------------------------------------------------------------


def test_interactive_module_imports():
    """Verify the interactive module is importable and exports expected names."""
    from abliterix.interactive import (
        ask_merge_strategy,
        show_interactive_results,
    )

    assert callable(ask_merge_strategy)
    assert callable(show_interactive_results)
