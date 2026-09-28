"""Tests for abliterix.eval.scorer — multi-objective scoring math.

Tests _compute_objectives() which is pure arithmetic.
The TrialScorer is constructed by bypassing __init__ (which requires a model).
"""

from types import SimpleNamespace

import pytest
import torch


from abliterix.eval.scorer import (  # noqa: E402
    TrialScorer,
    _safe_kl_divergence,
)
from abliterix.eval.metrics import ComplianceResult, MetricResult  # noqa: E402
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


def test_zero_baseline_uses_absolute_refusal_fraction():
    scorer = _make_scorer(baseline_refusals=0)
    scorer.target_msgs = ["target-a", "target-b", "target-c", "target-d"]

    _, compliance = scorer._compute_objectives(
        kl_divergence=0.05,
        detected_refusals=1,
    )

    assert compliance == pytest.approx(0.25)


def test_unknown_labels_use_paired_known_denominators():
    scorer = _make_scorer(baseline_refusals=2)

    _, compliance = scorer._compute_objectives(
        kl_divergence=0.05,
        detected_refusals=1,
        evaluated_count=2,
        baseline_refusals=1,
        baseline_evaluated_count=2,
    )

    assert compliance == pytest.approx(1.0)


def test_no_known_labels_cannot_produce_compliance_objective():
    scorer = _make_scorer(baseline_refusals=0)

    with pytest.raises(ValueError, match="at least one known label"):
        scorer._compute_objectives(
            kl_divergence=0.05,
            detected_refusals=0,
            evaluated_count=0,
            baseline_evaluated_count=0,
        )


def test_measure_compliance_compares_shared_known_samples_only():
    scorer = _make_scorer(baseline_refusals=2)
    scorer.target_msgs = ["a", "b", "c", "d"]
    scorer.baseline_compliance_result = ComplianceResult(
        labels=(True, True, False, None),
        evaluator="judge",
        protocol_version="test",
    )
    scorer.detector = SimpleNamespace(
        evaluate_compliance_result=lambda *_args: ComplianceResult(
            labels=(True, None, False, False),
            evaluator="judge",
            protocol_version="test",
        )
    )

    detected, objective = scorer.measure_compliance_objective(SimpleNamespace())

    assert detected == 1
    # Shared known indices are 0 and 2: both baseline and trial are 1/2.
    assert objective == pytest.approx(1.0)


def test_measure_compliance_rejects_different_sample_counts():
    scorer = _make_scorer(baseline_refusals=1)
    scorer.target_msgs = ["a", "b"]
    scorer.baseline_compliance_result = ComplianceResult(
        labels=(True, False),
        evaluator="judge",
        protocol_version="test",
    )
    scorer.detector = SimpleNamespace(
        evaluate_compliance_result=lambda *_args: ComplianceResult(
            labels=(True,),
            evaluator="judge",
            protocol_version="test",
        )
    )

    with pytest.raises(RuntimeError, match="different sample counts"):
        scorer.measure_compliance_objective(SimpleNamespace())


def test_measure_compliance_rejects_different_evaluator_identity():
    scorer = _make_scorer(baseline_refusals=1)
    scorer.target_msgs = ["a"]
    scorer.baseline_compliance_result = ComplianceResult(
        labels=(True,),
        evaluator="judge-a",
        protocol_version="prompt-v1",
    )
    scorer.detector = SimpleNamespace(
        evaluate_compliance_result=lambda *_args: ComplianceResult(
            labels=(False,),
            evaluator="judge-b",
            protocol_version="prompt-v1",
        )
    )

    with pytest.raises(RuntimeError, match="different evaluator identities"):
        scorer.measure_compliance_objective(SimpleNamespace())


# ---------------------------------------------------------------------------
# Divergence objective
# ---------------------------------------------------------------------------


def test_kl_above_target():
    """When kl >= target, divergence = kl / scale."""
    scorer = _make_scorer(kl_scale=2.0, kl_target=0.01)
    divergence, _ = scorer._compute_objectives(kl_divergence=0.05, detected_refusals=50)
    assert divergence == pytest.approx(0.05 / 2.0)


def test_kl_below_target():
    """When kl < target, divergence = kl / scale (same as above-target)."""
    scorer = _make_scorer(kl_scale=2.0, kl_target=0.01, baseline_refusals=200)
    divergence, compliance = scorer._compute_objectives(
        kl_divergence=0.005, detected_refusals=50
    )
    assert compliance == pytest.approx(0.25)
    assert divergence == pytest.approx(0.005 / 2.0)


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


def test_safe_kl_divergence_handles_nonfinite_logprobs():
    current = torch.tensor([[float("nan"), -2.0, float("-inf")]])
    baseline = torch.log_softmax(torch.tensor([[1.0, 0.0, -1.0]]), dim=-1)

    kl = _safe_kl_divergence(current, baseline)

    assert kl >= 0
    assert torch.isfinite(torch.tensor(kl))


def test_hf_damage_metric_identifies_full_distribution_kl():
    scorer = _make_scorer()
    scorer.benign_msgs = ["prompt"]
    scorer.baseline_logprobs = torch.log_softmax(
        torch.tensor([[2.0, 1.0]]),
        dim=-1,
    )
    current = torch.log_softmax(torch.tensor([[1.0, 2.0]]), dim=-1)

    class FakeHFEngine:
        _vllm_gen = None

        def compute_logprobs_batched(self, messages):
            assert messages == scorer.benign_msgs
            return current

    result = scorer.measure_damage(FakeHFEngine())

    assert result == MetricResult(
        name="full_distribution_kl",
        estimator="mean_per_token_kl",
        units="nats/token",
        value=pytest.approx(_safe_kl_divergence(current, scorer.baseline_logprobs)),
    )


def test_vllm_damage_metric_identifies_fixed_continuation_nll_mae(capsys):
    scorer = _make_scorer()
    scorer.config.model.use_in_place_editing = True
    scorer.benign_msgs = ["prompt-a", "prompt-b"]
    scorer.baseline_continuations = ["base-a", "base-b"]
    scorer.baseline_continuation_nll = torch.tensor([1.0, 2.0])

    class FakeVLLM:
        def score_continuations_nll(self, messages, continuations, adapter_path=None):
            assert messages == scorer.benign_msgs
            assert continuations == scorer.baseline_continuations
            assert adapter_path == "/tmp/adapter"
            return torch.tensor([1.25, 1.50])

    engine = SimpleNamespace(
        _vllm_gen=FakeVLLM(),
        _current_adapter_path="/tmp/adapter",
    )

    result = scorer.measure_damage(engine)

    assert result == MetricResult(
        name="continuation_nll_drift",
        estimator="fixed_continuation_mean_absolute_error",
        units="nats/token",
        value=pytest.approx(0.375),
    )
    output = capsys.readouterr().out
    assert "continuation_nll_drift" in output
    assert "KL divergence" not in output


def test_legacy_numeric_damage_api_does_not_mislabel_nll_as_kl(capsys):
    scorer = _make_scorer()
    scorer.config.model.use_in_place_editing = True
    scorer.benign_msgs = ["prompt"]
    scorer.baseline_continuations = ["base"]
    scorer.baseline_continuation_nll = torch.tensor([1.0])

    class FakeVLLM:
        def score_continuations_nll(self, *args, **kwargs):
            return torch.tensor([1.25])

    engine = SimpleNamespace(
        _vllm_gen=FakeVLLM(),
        _current_adapter_path=None,
    )

    value = scorer.measure_kl_divergence(engine)

    assert value == pytest.approx(0.25)
    output = capsys.readouterr().out
    assert "continuation_nll_drift" in output
    assert "KL divergence" not in output


def test_vllm_continuation_kl_uses_nll_drift():
    scorer = _make_scorer()
    scorer.config.model.use_in_place_editing = True
    scorer.benign_msgs = ["prompt-a", "prompt-b"]
    scorer.baseline_continuations = ["base-a", "base-b"]
    scorer.baseline_continuation_nll = torch.tensor([1.0, 2.0])

    class FakeVLLM:
        def score_continuations_nll(self, messages, continuations, adapter_path=None):
            assert messages == scorer.benign_msgs
            assert continuations == scorer.baseline_continuations
            assert adapter_path == "/tmp/adapter"
            return torch.tensor([1.25, 1.50])

    kl = scorer._measure_vllm_continuation_kl(FakeVLLM(), "/tmp/adapter")

    assert kl == pytest.approx(0.375)


def test_in_place_coherence_does_not_collect_discarded_sampler_logprobs():
    scorer = _make_scorer()
    scorer.config.model.use_in_place_editing = True
    scorer.benign_msgs = ["prompt-a", "prompt-b"]
    scorer.baseline_continuations = ["base-a", "base-b"]
    scorer.baseline_continuation_nll = torch.tensor([1.0, 2.0])
    scorer.baseline_mean_length = 1.0
    scorer.baseline_stdev_length = 1.0

    class FakeVLLM:
        def generate_and_score_batched(self, *args, **kwargs):
            raise AssertionError("in-place mode must not collect sampler logprobs")

        def generate_text_batched(self, *args, **kwargs):
            return ["one", "two"]

        def score_continuations_nll(self, *args, **kwargs):
            return torch.tensor([1.25, 1.50])

    engine = SimpleNamespace(
        _vllm_gen=FakeVLLM(),
        _current_adapter_path="/tmp/adapter",
    )

    kl, deviation = scorer.measure_kl_and_coherence(engine)

    assert kl == pytest.approx(0.375)
    assert deviation == pytest.approx(0.0)


def test_in_place_combined_measurement_preserves_nll_metric_identity(capsys):
    scorer = _make_scorer()
    scorer.config.model.use_in_place_editing = True
    scorer.benign_msgs = ["prompt-a", "prompt-b"]
    scorer.baseline_continuations = ["base-a", "base-b"]
    scorer.baseline_continuation_nll = torch.tensor([1.0, 2.0])
    scorer.baseline_mean_length = 1.0
    scorer.baseline_stdev_length = 1.0

    class FakeVLLM:
        def generate_text_batched(self, *args, **kwargs):
            return ["one", "two"]

        def score_continuations_nll(self, *args, **kwargs):
            return torch.tensor([1.25, 1.50])

    engine = SimpleNamespace(
        _vllm_gen=FakeVLLM(),
        _current_adapter_path="/tmp/adapter",
    )

    metric, deviation = scorer.measure_damage_and_coherence(engine)

    assert metric.name == "continuation_nll_drift"
    assert metric.estimator == "fixed_continuation_mean_absolute_error"
    assert metric.units == "nats/token"
    assert metric.value == pytest.approx(0.375)
    assert deviation == pytest.approx(0.0)
    output = capsys.readouterr().out
    assert "continuation_nll_drift" in output
    assert "KL divergence" not in output


def test_in_place_baseline_skips_unused_sampler_logprobs(capsys):
    scorer = _make_scorer()
    scorer.config.model.use_in_place_editing = True
    scorer.benign_msgs = ["prompt-a", "prompt-b"]
    scorer.target_msgs = []
    scorer.detector = SimpleNamespace(evaluate_compliance=lambda *_args: 0)

    class FakeVLLM:
        def generate_and_score_batched(self, *args, **kwargs):
            raise AssertionError("in-place baseline must not collect sampler logprobs")

        def generate_text_batched(self, *args, **kwargs):
            return ["base a", "base b"]

        def score_continuations_nll(self, *args, **kwargs):
            return torch.tensor([1.0, 2.0])

    engine = SimpleNamespace(_vllm_gen=FakeVLLM())

    scorer._capture_baseline(engine)

    assert scorer.baseline_logprobs is None
    assert scorer.baseline_continuations == ["base a", "base b"]
    torch.testing.assert_close(
        scorer.baseline_continuation_nll, torch.tensor([1.0, 2.0])
    )
    output = capsys.readouterr().out
    assert "fixed-continuation NLL" in output
    assert "in-place KL" not in output


def test_hf_multitoken_kl_scores_baseline_and_trial_on_shared_continuation():
    scorer = _make_scorer()
    scorer.config.kl.token_count = 2
    scorer.config.inference.min_gen_tokens = None
    scorer.config.inference.max_gen_tokens = 8
    scorer.benign_msgs = ["prompt-a", "prompt-b"]
    scorer.target_msgs = []
    scorer.detector = SimpleNamespace(evaluate_compliance=lambda *_args: 0)

    baseline_continuations = ["shared-a second-a", "shared-b second-b"]
    trial_responses = ["trial-a diverged-a", "trial-b diverged-b"]
    baseline = torch.log_softmax(
        torch.tensor(
            [
                [[3.0, 1.0], [1.0, 2.0]],
                [[2.0, 0.0], [0.0, 3.0]],
            ]
        ),
        dim=-1,
    )
    current = torch.log_softmax(
        torch.tensor(
            [
                [[2.0, 1.0], [2.0, 1.0]],
                [[1.0, 0.0], [1.0, 2.0]],
            ]
        ),
        dim=-1,
    )

    class FakeHFEngine:
        _vllm_gen = None

        def __init__(self):
            self.generation_count = 0
            self.scored_continuations = []

        def generate_and_score_batched(self, *args, **kwargs):
            raise AssertionError("multi-token HF KL must use fixed continuations")

        def generate_text_batched(self, messages, **kwargs):
            assert messages == scorer.benign_msgs
            self.generation_count += 1
            if self.generation_count == 1:
                assert kwargs["min_new_tokens"] >= 2
                return baseline_continuations
            return trial_responses

        def score_continuation_logprobs_batched(
            self, messages, continuations, token_count
        ):
            assert messages == scorer.benign_msgs
            assert token_count == 2
            self.scored_continuations.append(list(continuations))
            return baseline if len(self.scored_continuations) == 1 else current

    engine = FakeHFEngine()
    scorer._capture_baseline(engine)
    kl, deviation = scorer.measure_kl_and_coherence(engine)

    assert scorer.baseline_continuations == baseline_continuations
    assert engine.scored_continuations == [
        baseline_continuations,
        baseline_continuations,
    ]
    assert kl == pytest.approx(_safe_kl_divergence(current, baseline))
    assert deviation == pytest.approx(0.0)


def test_hf_single_token_kl_retains_single_pass_generation_fast_path():
    scorer = _make_scorer()
    scorer.config.kl.token_count = 1
    scorer.benign_msgs = ["prompt"]
    scorer.target_msgs = []
    scorer.detector = SimpleNamespace(evaluate_compliance=lambda *_args: 0)

    class FakeHFEngine:
        _vllm_gen = None

        def __init__(self):
            self.calls = 0

        def generate_and_score_batched(self, *_args, **kwargs):
            self.calls += 1
            assert kwargs["kl_token_count"] == 1
            return ["same response"], torch.log_softmax(
                torch.tensor([[2.0, 1.0]]), dim=-1
            )

        def generate_text_batched(self, *_args, **_kwargs):
            raise AssertionError("single-token fast path must stay single-pass")

        def score_continuation_logprobs_batched(self, *_args, **_kwargs):
            raise AssertionError("single-token fast path must stay single-pass")

    engine = FakeHFEngine()
    scorer._capture_baseline(engine)
    kl, deviation = scorer.measure_kl_and_coherence(engine)

    assert engine.calls == 2
    assert kl == pytest.approx(0.0)
    assert deviation == pytest.approx(0.0)


def test_tp_multitoken_kl_scores_shared_continuation_with_current_adapter():
    scorer = _make_scorer()
    scorer.config.kl.token_count = 2
    scorer.config.inference.min_gen_tokens = None
    scorer.benign_msgs = ["prompt"]
    scorer.target_msgs = []
    scorer.detector = SimpleNamespace(evaluate_compliance=lambda *_args: 0)

    baseline = torch.log_softmax(torch.tensor([[[2.0, 1.0], [1.0, 2.0]]]), dim=-1)
    current = torch.log_softmax(torch.tensor([[[1.0, 2.0], [2.0, 1.0]]]), dim=-1)

    class FakeTPBackend:
        def __init__(self):
            self.generation_count = 0
            self.score_adapters = []
            self.score_continuations = []

        def generate_and_score_batched(self, *_args, **_kwargs):
            raise AssertionError("multi-token TP KL must use fixed continuations")

        def generate_text_batched(self, messages, **kwargs):
            assert messages == scorer.benign_msgs
            self.generation_count += 1
            if self.generation_count == 1:
                assert kwargs["adapter_path"] is None
                assert kwargs["min_new_tokens"] >= 2
                return ["shared second"]
            assert kwargs["adapter_path"] == "/tmp/adapter"
            return ["trial diverged"]

        def score_continuation_logprobs_batched(
            self, messages, continuations, token_count, adapter_path=None
        ):
            assert messages == scorer.benign_msgs
            assert token_count == 2
            self.score_adapters.append(adapter_path)
            self.score_continuations.append(list(continuations))
            return baseline if adapter_path is None else current

    backend = FakeTPBackend()
    engine = SimpleNamespace(
        _vllm_gen=backend,
        _current_adapter_path="/tmp/adapter",
    )

    scorer._capture_baseline(engine)
    kl, deviation = scorer.measure_kl_and_coherence(engine)

    assert scorer.baseline_continuations == ["shared second"]
    assert backend.score_adapters == [None, "/tmp/adapter"]
    assert backend.score_continuations == [["shared second"], ["shared second"]]
    assert kl == pytest.approx(_safe_kl_divergence(current, baseline))
    assert deviation == pytest.approx(0.0)
