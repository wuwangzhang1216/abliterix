"""Tests for MiniMax LLM judge provider support.

Verifies that the detector correctly resolves API keys, constructs requests,
and handles provider-specific behaviour (no response_format, temperature=1.0)
when ``llm_judge_base_url`` points to the MiniMax API.
"""

import json
import sys
from unittest.mock import patch

# Provide a minimal CLI argv so AbliterixConfig doesn't fail on missing --model
sys.argv = ["test", "--model.model-id", "dummy/model"]

from abliterix.eval.detector import RefusalDetector, _resolve_judge_api_key
from abliterix.settings import AbliterixConfig


# ---------------------------------------------------------------------------
# _resolve_judge_api_key
# ---------------------------------------------------------------------------


def test_resolve_key_openrouter_default(monkeypatch):
    """No base URL → OpenRouter path uses OPENROUTER_API_KEY."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    config = AbliterixConfig()
    assert config.detection.llm_judge_base_url is None
    assert _resolve_judge_api_key(config) == "or-test-key"


def test_resolve_key_minimax(monkeypatch):
    """Custom base URL → MiniMax path uses MINIMAX_API_KEY."""
    monkeypatch.setenv("MINIMAX_API_KEY", "mm-test-key")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    config = AbliterixConfig(
        detection={"llm_judge_base_url": "https://api.minimax.io/v1"}
    )
    assert _resolve_judge_api_key(config) == "mm-test-key"


def test_resolve_key_missing_returns_empty(monkeypatch):
    """Missing API key returns an empty string (signals 'not configured')."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    config = AbliterixConfig()
    assert _resolve_judge_api_key(config) == ""


# ---------------------------------------------------------------------------
# DetectionConfig defaults
# ---------------------------------------------------------------------------


def test_llm_judge_base_url_defaults_to_none():
    config = AbliterixConfig()
    assert config.detection.llm_judge_base_url is None


def test_llm_judge_base_url_minimax():
    config = AbliterixConfig(
        detection={"llm_judge_base_url": "https://api.minimax.io/v1"}
    )
    assert config.detection.llm_judge_base_url == "https://api.minimax.io/v1"


def test_llm_judge_model_minimax():
    config = AbliterixConfig(
        detection={
            "llm_judge_base_url": "https://api.minimax.io/v1",
            "llm_judge_model": "MiniMax-M2.7",
        }
    )
    assert config.detection.llm_judge_model == "MiniMax-M2.7"


def test_llm_judge_model_minimax_highspeed():
    config = AbliterixConfig(
        detection={
            "llm_judge_base_url": "https://api.minimax.io/v1",
            "llm_judge_model": "MiniMax-M2.7-highspeed",
        }
    )
    assert config.detection.llm_judge_model == "MiniMax-M2.7-highspeed"


# ---------------------------------------------------------------------------
# _query_judge_api — MiniMax request construction
# ---------------------------------------------------------------------------


def _make_judge_response(labels: list[str]) -> bytes:
    """Build a fake OpenAI-compatible chat completion response."""
    content = json.dumps({"labels": labels})
    payload = {"choices": [{"message": {"content": content}}]}
    return json.dumps(payload).encode("utf-8")


def _make_minimax_detector(monkeypatch) -> RefusalDetector:
    """Create a RefusalDetector configured for MiniMax with the cache disabled."""
    monkeypatch.setenv("MINIMAX_API_KEY", "mm-test-key")
    config = AbliterixConfig(
        detection={
            "llm_judge_base_url": "https://api.minimax.io/v1",
            "llm_judge_model": "MiniMax-M2.7",
        }
    )
    detector = RefusalDetector(config)
    # Disable the persistent cache so that every test unconditionally reaches
    # the urlopen call; otherwise a cached result from a previous run would
    # short-circuit the request and leave `captured` empty.
    detector._cache = None
    return detector


def _make_openrouter_detector(monkeypatch) -> RefusalDetector:
    """Create a RefusalDetector configured for OpenRouter with the cache disabled."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    config = AbliterixConfig()
    detector = RefusalDetector(config)
    detector._cache = None
    return detector


def test_minimax_request_omits_response_format(monkeypatch):
    """MiniMax path must NOT include response_format in the request body."""
    detector = _make_minimax_detector(monkeypatch)
    captured: dict = {}

    class FakeResponse:
        def read(self):
            return _make_judge_response(["R"])

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    def fake_urlopen(req, timeout=30):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return FakeResponse()

    with patch("urllib.request.urlopen", fake_urlopen):
        detector._query_judge_api([("harmful question", "I'll help you with that.")])

    assert "body" in captured, "urlopen was never called — check the patch path"
    assert "response_format" not in captured["body"], (
        "MiniMax does not support response_format — it must be omitted"
    )


def test_minimax_request_uses_temperature_one(monkeypatch):
    """MiniMax path must use temperature=1.0 (not 0, which is invalid)."""
    detector = _make_minimax_detector(monkeypatch)
    captured: dict = {}

    class FakeResponse:
        def read(self):
            return _make_judge_response(["C"])

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    def fake_urlopen(req, timeout=30):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return FakeResponse()

    with patch("urllib.request.urlopen", fake_urlopen):
        detector._query_judge_api([("q", "compliant answer")])

    assert "body" in captured, "urlopen was never called"
    assert captured["body"].get("temperature") == 1.0, (
        "MiniMax requires temperature in (0.0, 1.0]; must not be 0"
    )


def test_minimax_request_targets_minimax_endpoint(monkeypatch):
    """MiniMax path must POST to the configured base URL, not OpenRouter."""
    detector = _make_minimax_detector(monkeypatch)
    captured: dict = {}

    class FakeResponse:
        def read(self):
            return _make_judge_response(["R"])

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    def fake_urlopen(req, timeout=30):
        captured["url"] = req.full_url
        return FakeResponse()

    with patch("urllib.request.urlopen", fake_urlopen):
        detector._query_judge_api([("q", "r")])

    assert "url" in captured, "urlopen was never called"
    assert "minimax.io" in captured["url"], (
        f"Expected MiniMax endpoint, got: {captured['url']}"
    )
    assert "openrouter" not in captured["url"]


def test_openrouter_request_uses_response_format(monkeypatch):
    """Default OpenRouter path must include response_format for structured JSON."""
    detector = _make_openrouter_detector(monkeypatch)
    captured: dict = {}

    class FakeResponse:
        def read(self):
            return _make_judge_response(["R"])

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    def fake_urlopen(req, timeout=30):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return FakeResponse()

    with patch("urllib.request.urlopen", fake_urlopen):
        detector._query_judge_api([("q", "r")])

    assert "body" in captured, "urlopen was never called"
    assert "response_format" in captured["body"]
    assert captured["body"]["temperature"] == 0


def test_minimax_think_tags_stripped(monkeypatch):
    """MiniMax <think>…</think> reasoning blocks must be stripped before JSON parsing."""
    detector = _make_minimax_detector(monkeypatch)

    think_response = json.dumps(
        {
            "choices": [
                {
                    "message": {
                        "content": (
                            "<think>\nInternal reasoning here.\n</think>\n\n"
                            '{"labels": ["R"]}'
                        )
                    }
                }
            ]
        }
    ).encode("utf-8")

    class FakeResponse:
        def read(self):
            return think_response

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    with patch("urllib.request.urlopen", lambda req, timeout=30: FakeResponse()):
        result = detector._query_judge_api([("harmful question", "I cannot do that.")])

    assert result == [True], "Response with <think> tag should parse as refusal (R)"


def test_minimax_max_tokens_includes_think_budget(monkeypatch):
    """MiniMax requests must reserve extra tokens for the <think> reasoning block."""
    detector = _make_minimax_detector(monkeypatch)
    captured: dict = {}

    class FakeResponse:
        def read(self):
            return _make_judge_response(["C"])

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    def fake_urlopen(req, timeout=30):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return FakeResponse()

    # Batch of 1 item: base max_tokens = 1*5+50 = 55; MiniMax adds 512 → 567
    with patch("urllib.request.urlopen", fake_urlopen):
        detector._query_judge_api([("q", "compliant answer")])

    assert "body" in captured, "urlopen was never called"
    assert captured["body"]["max_tokens"] > 55, (
        "MiniMax max_tokens must include a budget for the <think> reasoning block"
    )
