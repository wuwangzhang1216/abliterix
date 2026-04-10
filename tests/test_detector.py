"""Tests for abliterix.eval.detector — keyword-based refusal detection.

These tests exercise the keyword matching logic without requiring a GPU,
model, or external API.
"""

import sys

# TrialScorer init requires a real model; we only test RefusalDetector's
# detect_refusal / is_obvious_refusal methods which are purely textual.

# Provide a minimal CLI argv so AbliterixConfig doesn't fail on missing --model
sys.argv = ["test", "--model.model-id", "dummy/model"]

from abliterix.eval.detector import RefusalDetector
from abliterix.settings import AbliterixConfig


def _make_detector() -> RefusalDetector:
    config = AbliterixConfig()
    return RefusalDetector(config)


# ---------------------------------------------------------------------------
# detect_refusal
# ---------------------------------------------------------------------------


def test_detect_empty_response():
    d = _make_detector()
    assert d.detect_refusal("") is True
    assert d.detect_refusal("   ") is True


def test_detect_obvious_refusal_english():
    d = _make_detector()
    assert d.detect_refusal("I'm sorry, I cannot help with that.") is True
    assert d.detect_refusal("As an AI assistant, I must decline.") is True


def test_detect_obvious_refusal_chinese():
    d = _make_detector()
    assert d.detect_refusal("抱歉，我无法提供这方面的帮助。") is True
    assert d.detect_refusal("作为AI助手，我不能回答这个问题。") is True


def test_detect_compliant_response():
    d = _make_detector()
    assert (
        d.detect_refusal("Here is how you can do it: first, gather the materials...")
        is False
    )


def test_detect_compliant_chinese():
    d = _make_detector()
    assert d.detect_refusal("你可以按照以下步骤操作：首先准备材料...") is False


# ---------------------------------------------------------------------------
# is_obvious_refusal
# ---------------------------------------------------------------------------


def test_obvious_refusal_prefix():
    d = _make_detector()
    assert d.is_obvious_refusal("I can't help with that request.") is True
    assert d.is_obvious_refusal("I cannot assist you with this.") is True


def test_obvious_refusal_chinese_prefix():
    d = _make_detector()
    assert d.is_obvious_refusal("抱歉,我无法提供这类信息。") is True


def test_obvious_refusal_compliant():
    d = _make_detector()
    assert d.is_obvious_refusal("Sure, here's how to do it...") is False


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_detect_refusal_case_insensitive():
    d = _make_detector()
    assert d.detect_refusal("I'M SORRY, I CANNOT DO THAT.") is True


def test_detect_refusal_with_markdown():
    d = _make_detector()
    assert d.detect_refusal("**I'm sorry**, I *cannot* help.") is True


# ---------------------------------------------------------------------------
# _normalise — text normalisation
# ---------------------------------------------------------------------------


def test_normalise_lowercase():
    assert RefusalDetector._normalise("HELLO World") == "hello world"


def test_normalise_strips_markdown():
    assert RefusalDetector._normalise("**bold** and *italic*") == "bold and italic"


def test_normalise_fullwidth_punctuation():
    result = RefusalDetector._normalise("你好，世界。测试！问题？")
    assert "," in result
    assert "." in result
    assert "!" in result
    assert "?" in result
    # Full-width forms should be gone.
    assert "\uff0c" not in result
    assert "\u3002" not in result


def test_normalise_curly_quotes():
    result = RefusalDetector._normalise("it\u2019s a \u2018test\u2019")
    assert "\u2019" not in result
    assert "\u2018" not in result
    assert "'" in result


def test_normalise_collapses_whitespace():
    assert RefusalDetector._normalise("  hello   \n\t world  ") == "hello world"


def test_normalise_empty():
    assert RefusalDetector._normalise("") == ""


# ---------------------------------------------------------------------------
# detect_refusal — additional edge cases
# ---------------------------------------------------------------------------


def test_sorry_with_for_not_refusal():
    """'sorry for' in the positional window should NOT trigger a refusal."""
    d = _make_detector()
    assert d.detect_refusal("I'm sorry for the delay, here is the answer.") is False


def test_sorry_positional_window():
    """'sorry' without 'for' in the first 8 words should be a refusal."""
    d = _make_detector()
    assert d.detect_refusal("I'm sorry, I really cannot do this.") is True


def test_extra_pattern_chinese_safety():
    d = _make_detector()
    assert d.detect_refusal("出于安全考虑，我不会回答这个问题。") is True
    assert d.detect_refusal("作为一个负责任的AI，我需要遵守相关规定。") is True


def test_extra_pattern_english():
    d = _make_detector()
    assert d.detect_refusal("I'm not comfortable answering that question.") is True
    assert d.detect_refusal("That goes against my guidelines.") is True
