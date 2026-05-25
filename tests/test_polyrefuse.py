"""Tests for abliterix.polyrefuse — cross-lingual refusal evaluation harness."""

from abliterix.polyrefuse import (
    LanguageResult,
    evaluate_per_language,
    format_report,
    summarise_transfer,
)


# ---------------------------------------------------------------------------
# Stub generator + detector for end-to-end tests
# ---------------------------------------------------------------------------


def _stub_generator(canned: dict[tuple, list[str]]):
    """Return a generator that maps tuple(prompts) to canned responses."""

    def gen(msgs):
        return canned[tuple(msgs)]

    return gen


class _StubDetector:
    """Detector that flags responses containing 'I cannot' or 'I will not'."""

    def classify_batch(self, responses):
        return [
            ("i cannot" in r.lower() or "i will not" in r.lower()) for r in responses
        ]


# ---------------------------------------------------------------------------
# evaluate_per_language
# ---------------------------------------------------------------------------


def test_evaluate_per_language_basic():
    prompts = {
        "en": ["prompt_en_1", "prompt_en_2"],
        "zh": ["prompt_zh_1", "prompt_zh_2"],
    }
    gen = _stub_generator(
        {
            ("prompt_en_1", "prompt_en_2"): [
                "Here you go.",
                "I cannot help with that.",
            ],
            ("prompt_zh_1", "prompt_zh_2"): [
                "I will not assist.",
                "I will not comply.",
            ],
        }
    )
    results = evaluate_per_language(gen, _StubDetector(), prompts, sample_responses=2)
    assert results["en"].n_total == 2
    assert results["en"].n_refused == 1
    assert results["en"].refusal_rate == 0.5
    assert results["zh"].refusal_rate == 1.0


def test_evaluate_per_language_empty_group():
    """Empty prompt list must yield zero counts, not raise."""
    prompts = {"fr": []}
    gen = _stub_generator({})
    results = evaluate_per_language(gen, _StubDetector(), prompts)
    assert results["fr"].n_total == 0
    assert results["fr"].refusal_rate == 0.0
    assert results["fr"].sample_responses == []


def test_evaluate_per_language_sample_responses_cap():
    prompts = {"en": ["a", "b", "c", "d", "e"]}
    gen = _stub_generator({("a", "b", "c", "d", "e"): ["ok", "ok", "ok", "ok", "ok"]})
    results = evaluate_per_language(gen, _StubDetector(), prompts, sample_responses=2)
    assert len(results["en"].sample_responses) == 2


def test_evaluate_per_language_uses_is_refusal_fallback():
    """If detector lacks classify_batch, fall back to is_refusal."""

    class _IsRefusalOnly:
        def is_refusal(self, r):
            return "no" in r.lower()

    prompts = {"en": ["x", "y"]}
    gen = _stub_generator({("x", "y"): ["yes please", "no way"]})
    results = evaluate_per_language(gen, _IsRefusalOnly(), prompts)
    assert results["en"].n_refused == 1


# ---------------------------------------------------------------------------
# format_report
# ---------------------------------------------------------------------------


def test_format_report_no_results_returns_placeholder():
    assert format_report({}) == "(no languages evaluated)"


def test_format_report_includes_all_languages():
    results = {
        "en": LanguageResult("en", 100, 6, 0.06, []),
        "zh": LanguageResult("zh", 100, 9, 0.09, []),
        "es": LanguageResult("es", 50, 2, 0.04, []),
    }
    report = format_report(results)
    for lang in ("en", "zh", "es"):
        assert lang in report
    assert "6.0%" in report
    assert "9.0%" in report
    assert "4.0%" in report


# ---------------------------------------------------------------------------
# summarise_transfer
# ---------------------------------------------------------------------------


def test_summarise_transfer_empty_returns_empty_dict():
    assert summarise_transfer({}) == {}


def test_summarise_transfer_no_evaluated_languages_returns_empty():
    results = {"en": LanguageResult("en", 0, 0, 0.0, [])}
    assert summarise_transfer(results) == {}


def test_summarise_transfer_computes_aggregates():
    results = {
        "en": LanguageResult("en", 100, 5, 0.05, []),
        "zh": LanguageResult("zh", 100, 9, 0.09, []),
        "es": LanguageResult("es", 100, 7, 0.07, []),
    }
    summary = summarise_transfer(results)
    assert abs(summary["english_refusal_rate"] - 0.05) < 1e-9
    assert abs(summary["max_refusal_rate"] - 0.09) < 1e-9
    assert abs(summary["min_refusal_rate"] - 0.05) < 1e-9
    assert abs(summary["mean_refusal_rate"] - (0.05 + 0.09 + 0.07) / 3) < 1e-9
    assert abs(summary["transfer_gap"] - (0.09 - 0.05)) < 1e-9


def test_summarise_transfer_handles_missing_english():
    """When 'en' is absent, english_refusal_rate is NaN and transfer_gap NaN."""
    import math

    results = {
        "zh": LanguageResult("zh", 100, 9, 0.09, []),
        "es": LanguageResult("es", 100, 7, 0.07, []),
    }
    summary = summarise_transfer(results)
    assert math.isnan(summary["english_refusal_rate"])
    assert math.isnan(summary["transfer_gap"])


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def test_polyrefuse_config_defaults():
    from abliterix.settings import PolyRefuseConfig

    cfg = PolyRefuseConfig()
    assert cfg.enabled is False
    assert cfg.languages == {}
    assert cfg.sample_responses == 3


def test_polyrefuse_config_accepts_languages():
    from abliterix.settings import PolyRefuseConfig
    from abliterix.types import PromptSource

    src = PromptSource(
        dataset="local/path",
        split="test",
        column="text",
    )
    cfg = PolyRefuseConfig(
        enabled=True,
        languages={"en": src, "zh": src},
    )
    assert cfg.enabled is True
    assert set(cfg.languages.keys()) == {"en", "zh"}


def test_abliterix_config_polyrefuse_default_off():
    from abliterix.settings import AbliterixConfig

    cfg = AbliterixConfig()
    assert cfg.polyrefuse.enabled is False
