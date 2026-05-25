"""Tests for abliterix.external_eval — JALMBench / MTJ / TamperBench / GSM8K wrappers."""

from abliterix.external_eval import (
    ExternalEvalReport,
    _answers_match,
    _normalise_answer,
    compute_tamper_resistance,
    evaluate_gsm8k,
    evaluate_jailbreak,
    evaluate_multi_turn,
    run_external_evals,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generator_from(canned: list[str]):
    """Return a generator that yields the canned response for every prompt."""

    def gen(prompts):
        return [canned[i % len(canned)] for i in range(len(prompts))]

    return gen


class _Detector:
    """Marks any response containing 'cannot' or 'refuse' as a refusal."""

    def classify_batch(self, responses):
        return [("cannot" in r.lower() or "refuse" in r.lower()) for r in responses]


# ---------------------------------------------------------------------------
# evaluate_jailbreak
# ---------------------------------------------------------------------------


def test_evaluate_jailbreak_counts_non_refusals_as_successes():
    prompts = ["p1", "p2", "p3"]
    gen = _generator_from(["I cannot help", "Sure, here is", "Sure"])
    result = evaluate_jailbreak(gen, _Detector(), prompts, attack_name="test")
    assert result.attack_name == "test"
    assert result.n_prompts == 3
    assert result.n_succeeded == 2  # two non-refusals
    assert abs(result.success_rate - 2 / 3) < 1e-9


def test_evaluate_jailbreak_empty_prompt_list():
    result = evaluate_jailbreak(_generator_from(["x"]), _Detector(), [])
    assert result.n_prompts == 0
    assert result.success_rate == 0.0


def test_evaluate_jailbreak_sample_responses_cap():
    prompts = ["p"] * 10
    gen = _generator_from(["Sure"] * 10)
    result = evaluate_jailbreak(gen, _Detector(), prompts, sample_responses=3)
    assert len(result.sample_responses) == 3


# ---------------------------------------------------------------------------
# evaluate_multi_turn
# ---------------------------------------------------------------------------


def test_evaluate_multi_turn_succeeds_at_first_non_refusal():
    """Chain of 3 turns; jailbreak occurs at turn 2."""
    chains = [["t1", "t2", "t3"]]

    def gen(history):
        # Turn 1 → "I cannot"; turn 2 → "Sure, here is"; turn 3 → "Sure".
        if len(history) == 1:
            return ["I cannot help"]
        return ["Sure, here is the answer"]

    result = evaluate_multi_turn(gen, _Detector(), chains)
    assert result.n_chains == 1
    assert result.n_succeeded == 1
    assert result.mean_turns_to_jailbreak == 2.0


def test_evaluate_multi_turn_chain_that_never_jailbreaks():
    chains = [["t1", "t2", "t3"]]

    def gen(history):
        return ["I cannot help with that"]

    result = evaluate_multi_turn(gen, _Detector(), chains)
    assert result.n_succeeded == 0
    assert result.success_rate == 0.0
    # Mean turns when nothing jailbreaks: 0.0 (division by zero guarded).
    assert result.mean_turns_to_jailbreak == 0.0


def test_evaluate_multi_turn_empty_chains():
    result = evaluate_multi_turn(_generator_from(["x"]), _Detector(), [])
    assert result.n_chains == 0
    assert result.success_rate == 0.0


def test_evaluate_multi_turn_aggregates_across_multiple_chains():
    chains = [
        ["t1"],  # jailbreaks at turn 1
        ["t1", "t2"],  # jailbreaks at turn 2
        ["t1", "t2"],  # never jailbreaks
    ]

    def gen(history):
        # Chain 1: single turn → "Sure".
        # Chain 2: turn 1 refusal, turn 2 success.
        # Chain 3: both refusals.
        # We distinguish chains by the prompt strings.
        if len(history) == 1 and history[0] == "t1":
            # Could be chain 1, chain 2 turn 1, or chain 3 turn 1.
            # Chain 1 only has one turn; this branch can't tell which chain.
            # We'll use a side state to track call count.
            return [_choose(history)]
        return [_choose(history)]

    call_state = {
        "n": 0,
        "responses": iter(
            [
                "Sure here is",  # chain 1, turn 1 → success
                "I cannot",  # chain 2, turn 1 → refuse
                "Sure here is the way",  # chain 2, turn 2 → success
                "I cannot",  # chain 3, turn 1 → refuse
                "I refuse",  # chain 3, turn 2 → refuse
            ]
        ),
    }

    def _choose(_history):
        call_state["n"] += 1
        return next(call_state["responses"])

    result = evaluate_multi_turn(gen, _Detector(), chains)
    assert result.n_chains == 3
    assert result.n_succeeded == 2
    assert abs(result.success_rate - 2 / 3) < 1e-9
    # Chain 1 jailbreak at turn 1; chain 2 at turn 2 → mean = 1.5.
    assert abs(result.mean_turns_to_jailbreak - 1.5) < 1e-9


# ---------------------------------------------------------------------------
# compute_tamper_resistance
# ---------------------------------------------------------------------------


def test_tamper_resistance_zero_when_recovery_fully_restored():
    result = compute_tamper_resistance(refusal_rate_pre=0.05, refusal_rate_post=0.5)
    # rate_pre/rate_post = 0.1 → resistance = 1.0 - 0.1 = 0.9
    assert abs(result.tamper_resistance - 0.9) < 1e-9


def test_tamper_resistance_full_when_post_zero():
    """Post-recovery refusal of zero means resistance is full."""
    result = compute_tamper_resistance(0.05, 0.0)
    assert result.tamper_resistance == 1.0


def test_tamper_resistance_clamped_to_unit_interval():
    """rate_pre > rate_post should yield 0 (not negative)."""
    result = compute_tamper_resistance(0.8, 0.4)
    # ratio = 2 → 1 - 2 = -1, clipped to 0.
    assert result.tamper_resistance == 0.0


# ---------------------------------------------------------------------------
# evaluate_gsm8k
# ---------------------------------------------------------------------------


def test_normalise_answer_handles_gsm8k_marker():
    assert _normalise_answer("Reasoning chain ... #### 42") == "42"


def test_normalise_answer_handles_trailing_number():
    assert _normalise_answer("After computing, the answer is 7.") == "7"


def test_normalise_answer_handles_comma_separated_thousands():
    assert _normalise_answer("#### 1,234") == "1234"


def test_normalise_answer_returns_none_for_no_number():
    assert _normalise_answer("no numbers here at all") is None


def test_answers_match_numeric_tolerance():
    assert _answers_match("42", "42")
    assert _answers_match("42", "42.0")
    assert not _answers_match("42", "43")


def test_evaluate_gsm8k_counts_correct_answers():
    problems = [
        {"question": "Q1", "answer": "42"},
        {"question": "Q2", "answer": "100"},
        {"question": "Q3", "answer": "#### 7"},
    ]

    def gen(qs):
        return [
            "The answer is 42.",
            "I would guess 99.",
            "After working through this... #### 7",
        ]

    result = evaluate_gsm8k(gen, problems)
    assert result.n_problems == 3
    assert result.n_correct == 2
    assert abs(result.accuracy - 2 / 3) < 1e-9


def test_evaluate_gsm8k_empty():
    result = evaluate_gsm8k(_generator_from(["x"]), [])
    assert result.n_problems == 0
    assert result.accuracy == 0.0


# ---------------------------------------------------------------------------
# run_external_evals — aggregator
# ---------------------------------------------------------------------------


def test_run_external_evals_runs_only_what_was_supplied():
    gen = _generator_from(["Sure"])
    detector = _Detector()
    report = run_external_evals(
        generator=gen,
        detector=detector,
        jailbreak_suites={"direct": ["p1", "p2"]},
        multi_turn_chains=None,
        gsm8k_problems=None,
        tamper_pre_post=None,
    )
    assert len(report.jailbreak) == 1
    assert report.multi_turn is None
    assert report.gsm8k is None
    assert report.tamper is None


def test_run_external_evals_packs_all_results():
    gen = _generator_from(["Sure, the answer is 42."])
    detector = _Detector()
    report = run_external_evals(
        generator=gen,
        detector=detector,
        jailbreak_suites={"a": ["p"]},
        multi_turn_chains=[["p"]],
        gsm8k_problems=[{"question": "q", "answer": "42"}],
        tamper_pre_post=(0.05, 0.10, 1),
    )
    assert isinstance(report, ExternalEvalReport)
    assert len(report.jailbreak) == 1
    assert report.multi_turn is not None and report.multi_turn.n_chains == 1
    assert report.gsm8k is not None and report.gsm8k.accuracy == 1.0
    assert report.tamper is not None
