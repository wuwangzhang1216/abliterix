"""Tests for abliterix.eval.scorer — multi-objective scoring math.

Tests _compute_objectives() which is pure arithmetic.
The TrialScorer is constructed by bypassing __init__ (which requires a model).
"""

import sys

import pytest

sys.argv = ["test", "--model.model-id", "dummy/model"]

from abliterix.eval.scorer import TrialScorer  # noqa: E402
from abliterix.settings import AbliterixConfig  # noqa: E402


def _make_scorer(
    kl_scale: float = 1.0,
    kl_target: float = 0.01,
    baseline_refusals: int = 200,
) -> TrialScorer:
    """Create a TrialScorer with injected attributes, bypassing __init__."""
    scorer = object.__new__(TrialScorer)
    config = AbliterixConfig()
    config.kl.scale = kl_scale
    config.kl.target = kl_target
    scorer.config = config
    scorer.baseline_refusal_count = baseline_refusals
    return scorer


# ---------------------------------------------------------------------------
# Compliance objective
# ---------------------------------------------------------------------------


def test_zero_refusals_zero_compliance():
    scorer = _make_scorer()
    _, compliance = scorer._compute_objectives(kl_divergence=0.05, detected_refusals=0)
    assert compliance == 0.0


def test_all_refusals_compliance_one():
    scorer = _make_scorer(baseline_refusals=200)
    _, compliance = scorer._compute_objectives(
        kl_divergence=0.05, detected_refusals=200
    )
    assert compliance == pytest.approx(1.0)


def test_half_refusals():
    scorer = _make_scorer(baseline_refusals=200)
    _, compliance = scorer._compute_objectives(
        kl_divergence=0.05, detected_refusals=100
    )
    assert compliance == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Divergence objective
# ---------------------------------------------------------------------------


def test_kl_above_target():
    """When kl >= target, divergence = kl / scale."""
    scorer = _make_scorer(kl_scale=2.0, kl_target=0.01)
    divergence, _ = scorer._compute_objectives(kl_divergence=0.05, detected_refusals=50)
    assert divergence == pytest.approx(0.05 / 2.0)


def test_kl_below_target():
    """When kl < target, divergence = compliance * target / scale."""
    scorer = _make_scorer(kl_scale=2.0, kl_target=0.01, baseline_refusals=200)
    divergence, compliance = scorer._compute_objectives(
        kl_divergence=0.005, detected_refusals=50
    )
    assert compliance == pytest.approx(0.25)
    assert divergence == pytest.approx(0.25 * 0.01 / 2.0)


def test_kl_exactly_at_target():
    """kl == target should take the >= branch."""
    scorer = _make_scorer(kl_scale=1.0, kl_target=0.05)
    divergence, _ = scorer._compute_objectives(
        kl_divergence=0.05, detected_refusals=100
    )
    assert divergence == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Length deviation penalty
# ---------------------------------------------------------------------------


def test_length_deviation_no_penalty():
    """Deviation <= 2.0 should not inflate divergence."""
    scorer = _make_scorer(kl_scale=1.0, kl_target=0.01)
    div_no_dev, _ = scorer._compute_objectives(
        kl_divergence=0.05, detected_refusals=50, length_deviation=0.0
    )
    div_at_two, _ = scorer._compute_objectives(
        kl_divergence=0.05, detected_refusals=50, length_deviation=2.0
    )
    assert div_no_dev == pytest.approx(div_at_two)


def test_length_deviation_penalty():
    """Deviation > 2.0 should multiply divergence by (1 + 0.1*(dev-2))."""
    scorer = _make_scorer(kl_scale=1.0, kl_target=0.01)
    base_div, _ = scorer._compute_objectives(
        kl_divergence=0.05, detected_refusals=50, length_deviation=0.0
    )
    penalised_div, _ = scorer._compute_objectives(
        kl_divergence=0.05, detected_refusals=50, length_deviation=5.0
    )
    expected = base_div * (1.0 + 0.1 * (5.0 - 2.0))
    assert penalised_div == pytest.approx(expected)
