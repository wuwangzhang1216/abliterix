"""Tests for the generic OpenAI-compatible LLM-judge path.

Verifies that:
  - ``llm_judge_base_url=None`` keeps the OpenRouter default path intact
    (OPENROUTER_API_KEY, attribution headers, structured response_format).
  - Any other value routes to a generic OpenAI-compatible endpoint whose
    provider-specific quirks (temperature, response_format, api-key env var,
    reasoning budget) are fully driven by explicit config fields — no URL
    sniffing, no hardcoded provider names.

MiniMax is exercised as one representative concrete case, but the same
knobs cover vLLM, Together, DeepInfra, local llama.cpp / LM Studio / Ollama.
"""

import json
import sys
from unittest.mock import patch

# Provide a minimal CLI argv so AbliterixConfig doesn't fail on missing --model
sys.argv = ["test", "--model.model-id", "dummy/model"]

from abliterix.eval.detector import (
    RefusalDetector,
    _judge_api_key_env,
    _resolve_judge_api_key,
)
from abliterix.settings import AbliterixConfig


# ---------------------------------------------------------------------------
# API-key env-var resolution
# ---------------------------------------------------------------------------


def test_key_env_defaults_to_openrouter_for_default_path():
    """No base_url → OPENROUTER_API_KEY."""
    config = AbliterixConfig()
    assert config.detection.llm_judge_base_url is None
    assert _judge_api_key_env(config) == "OPENROUTER_API_KEY"


def test_key_env_defaults_to_llm_judge_for_custom_url():
    """Custom base_url with no explicit env override → LLM_JUDGE_API_KEY."""
    config = AbliterixConfig(
        detection={"llm_judge_base_url": "http://localhost:8000/v1"}
    )
    assert _judge_api_key_env(config) == "LLM_JUDGE_API_KEY"


def test_key_env_honors_explicit_override():
    """Explicit llm_judge_api_key_env wins over both defaults."""
    config = AbliterixConfig(
        detection={
            "llm_judge_base_url": "https://api.minimax.io/v1",
            "llm_judge_api_key_env": "MINIMAX_API_KEY",
        }
    )
    assert _judge_api_key_env(config) == "MINIMAX_API_KEY"


def test_resolve_key_reads_configured_env(monkeypatch):
    monkeypatch.setenv("TOGETHER_API_KEY", "tg-test-key")
    config = AbliterixConfig(
        detection={
            "llm_judge_base_url": "https://api.together.xyz/v1",
            "llm_judge_api_key_env": "TOGETHER_API_KEY",
        }
    )
    assert _resolve_judge_api_key(config) == "tg-test-key"


def test_resolve_key_missing_returns_empty(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("LLM_JUDGE_API_KEY", raising=False)
    config = AbliterixConfig()
    assert _resolve_judge_api_key(config) == ""


def test_resolve_key_strips_whitespace(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "  or-key-with-spaces \n")
    config = AbliterixConfig()
    assert _resolve_judge_api_key(config) == "or-key-with-spaces"


# ---------------------------------------------------------------------------
# DetectionConfig field defaults
# ---------------------------------------------------------------------------


def test_new_fields_have_stable_defaults():
    """Defaults must preserve the legacy OpenRouter behaviour."""
    config = AbliterixConfig()
    d = config.detection
    assert d.llm_judge_base_url is None
    assert d.llm_judge_api_key_env is None
    assert d.llm_judge_temperature == 0.0
    assert d.llm_judge_use_response_format is True
    assert d.llm_judge_max_tokens_field == "max_tokens"
    assert d.llm_judge_reasoning_budget is None
    assert d.llm_judge_auth_header == "Authorization"
    assert d.llm_judge_auth_prefix == "Bearer "


def test_config_accepts_openai_preset():
    """OpenAI's newer models require max_completion_tokens instead of max_tokens."""
    config = AbliterixConfig(
        detection={
            "llm_judge_base_url": "https://api.openai.com/v1",
            "llm_judge_model": "gpt-5.4-mini-2026-03-17",
            "llm_judge_api_key_env": "OPENAI_API_KEY",
            "llm_judge_max_tokens_field": "max_completion_tokens",
        }
    )
    d = config.detection
    assert d.llm_judge_max_tokens_field == "max_completion_tokens"


def test_config_accepts_minimax_style_preset():
    """A MiniMax-style preset should round-trip through config."""
    config = AbliterixConfig(
        detection={
            "llm_judge_base_url": "https://api.minimax.io/v1",
            "llm_judge_model": "MiniMax-M2.7",
            "llm_judge_api_key_env": "MINIMAX_API_KEY",
            "llm_judge_temperature": 1.0,
            "llm_judge_use_response_format": False,
        }
    )
    d = config.detection
    assert d.llm_judge_base_url == "https://api.minimax.io/v1"
    assert d.llm_judge_model == "MiniMax-M2.7"
    assert d.llm_judge_api_key_env == "MINIMAX_API_KEY"
    assert d.llm_judge_temperature == 1.0
    assert d.llm_judge_use_response_format is False


def test_config_accepts_local_vllm_preset():
    """A local vLLM-style preset should round-trip through config."""
    config = AbliterixConfig(
        detection={
            "llm_judge_base_url": "http://localhost:8000/v1",
            "llm_judge_model": "Qwen/Qwen3.6-35B-A3B",
            "llm_judge_api_key_env": "LLM_JUDGE_API_KEY",
            "llm_judge_reasoning_budget": 0,
        }
    )
    d = config.detection
    assert d.llm_judge_base_url == "http://localhost:8000/v1"
    assert d.llm_judge_temperature == 0.0
    assert d.llm_judge_use_response_format is True
    assert d.llm_judge_reasoning_budget == 0


# ---------------------------------------------------------------------------
# _query_judge_api — request construction
# ---------------------------------------------------------------------------


def _judge_response(labels: list[str]) -> bytes:
    """Build a fake OpenAI-compatible chat completion response."""
    content = json.dumps({"labels": labels})
    payload = {"choices": [{"message": {"content": content}}]}
    return json.dumps(payload).encode("utf-8")


class _FakeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


def _patch_urlopen(body: bytes, captured: dict):
    def fake_urlopen(req, timeout=30):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.headers)
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse(body)

    return patch("urllib.request.urlopen", fake_urlopen)


def _openrouter_detector(monkeypatch) -> RefusalDetector:
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
    monkeypatch.delenv("LLM_JUDGE_API_KEY", raising=False)
    detector = RefusalDetector(AbliterixConfig())
    detector._cache = None
    return detector


def _minimax_detector(monkeypatch) -> RefusalDetector:
    monkeypatch.setenv("MINIMAX_API_KEY", "mm-test-key")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    config = AbliterixConfig(
        detection={
            "llm_judge_base_url": "https://api.minimax.io/v1",
            "llm_judge_model": "MiniMax-M2.7",
            "llm_judge_api_key_env": "MINIMAX_API_KEY",
            "llm_judge_temperature": 1.0,
            "llm_judge_use_response_format": False,
        }
    )
    detector = RefusalDetector(config)
    detector._cache = None
    return detector


def _vllm_detector(monkeypatch) -> RefusalDetector:
    monkeypatch.setenv("LLM_JUDGE_API_KEY", "dummy-local-key")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    config = AbliterixConfig(
        detection={
            "llm_judge_base_url": "http://localhost:8000/v1",
            "llm_judge_model": "Qwen/Qwen3.6-35B-A3B",
            "llm_judge_reasoning_budget": 0,
        }
    )
    detector = RefusalDetector(config)
    detector._cache = None
    return detector


# --- OpenRouter (default) --------------------------------------------------


def test_openrouter_request_targets_openrouter(monkeypatch):
    detector = _openrouter_detector(monkeypatch)
    captured: dict = {}
    with _patch_urlopen(_judge_response(["R"]), captured):
        detector._query_judge_api([("q", "r")])
    assert "openrouter.ai" in captured["url"]


def test_openrouter_request_includes_attribution_headers(monkeypatch):
    detector = _openrouter_detector(monkeypatch)
    captured: dict = {}
    with _patch_urlopen(_judge_response(["R"]), captured):
        detector._query_judge_api([("q", "r")])
    headers = {k.lower(): v for k, v in captured["headers"].items()}
    assert "http-referer" in headers
    assert "x-title" in headers


def test_openrouter_request_uses_response_format_and_temp_zero(monkeypatch):
    detector = _openrouter_detector(monkeypatch)
    captured: dict = {}
    with _patch_urlopen(_judge_response(["R"]), captured):
        detector._query_judge_api([("q", "r")])
    assert "response_format" in captured["body"]
    assert captured["body"]["temperature"] == 0.0


def test_openrouter_request_skips_reasoning_budget(monkeypatch):
    """Default OpenRouter model is non-reasoning → no extra token budget."""
    detector = _openrouter_detector(monkeypatch)
    captured: dict = {}
    with _patch_urlopen(_judge_response(["R"]), captured):
        detector._query_judge_api([("q", "r")])
    assert captured["body"]["max_tokens"] == 55


# --- MiniMax preset --------------------------------------------------------


def test_minimax_request_targets_minimax(monkeypatch):
    detector = _minimax_detector(monkeypatch)
    captured: dict = {}
    with _patch_urlopen(_judge_response(["C"]), captured):
        detector._query_judge_api([("q", "r")])
    assert "minimax.io" in captured["url"]
    assert "openrouter" not in captured["url"]


def test_minimax_request_omits_response_format(monkeypatch):
    detector = _minimax_detector(monkeypatch)
    captured: dict = {}
    with _patch_urlopen(_judge_response(["R"]), captured):
        detector._query_judge_api([("q", "r")])
    assert "response_format" not in captured["body"]


def test_minimax_request_uses_configured_temperature(monkeypatch):
    detector = _minimax_detector(monkeypatch)
    captured: dict = {}
    with _patch_urlopen(_judge_response(["R"]), captured):
        detector._query_judge_api([("q", "r")])
    assert captured["body"]["temperature"] == 1.0


def test_minimax_request_omits_attribution_headers(monkeypatch):
    """Only OpenRouter should receive abliterix attribution headers."""
    detector = _minimax_detector(monkeypatch)
    captured: dict = {}
    with _patch_urlopen(_judge_response(["R"]), captured):
        detector._query_judge_api([("q", "r")])
    headers = {k.lower(): v for k, v in captured["headers"].items()}
    assert "http-referer" not in headers
    assert "x-title" not in headers


def test_minimax_think_tags_stripped(monkeypatch):
    detector = _minimax_detector(monkeypatch)
    think_reply = json.dumps(
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
    with patch(
        "urllib.request.urlopen",
        lambda req, timeout=30: _FakeResponse(think_reply),
    ):
        result = detector._query_judge_api([("harmful q", "I cannot do that.")])
    assert result == [True]


def test_minimax_reasoning_budget_scales_with_batch(monkeypatch):
    """Auto-scaled budget (default None) must grow with batch size."""
    detector = _minimax_detector(monkeypatch)
    captured: dict = {}
    batch = [("q" + str(i), "r" + str(i)) for i in range(5)]
    response = _judge_response(["C"] * 5)
    with _patch_urlopen(response, captured):
        detector._query_judge_api(batch)
    # base = 5*5 + 50 = 75; reasoning = 256 + 32*5 = 416 → 491
    assert captured["body"]["max_tokens"] == 75 + 256 + 32 * 5


def test_explicit_reasoning_budget_overrides_default(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "mm-test-key")
    config = AbliterixConfig(
        detection={
            "llm_judge_base_url": "https://api.minimax.io/v1",
            "llm_judge_model": "MiniMax-M2.7",
            "llm_judge_api_key_env": "MINIMAX_API_KEY",
            "llm_judge_temperature": 1.0,
            "llm_judge_use_response_format": False,
            "llm_judge_reasoning_budget": 2048,
        }
    )
    detector = RefusalDetector(config)
    detector._cache = None
    captured: dict = {}
    with _patch_urlopen(_judge_response(["R"]), captured):
        detector._query_judge_api([("q", "r")])
    assert captured["body"]["max_tokens"] == 55 + 2048


# --- Generic vLLM / local preset -------------------------------------------


def test_vllm_request_targets_local_endpoint(monkeypatch):
    detector = _vllm_detector(monkeypatch)
    captured: dict = {}
    with _patch_urlopen(_judge_response(["R"]), captured):
        detector._query_judge_api([("q", "r")])
    assert captured["url"] == "http://localhost:8000/v1/chat/completions"


def test_vllm_preset_uses_response_format_by_default(monkeypatch):
    """Generic custom endpoints inherit response_format=True by default."""
    detector = _vllm_detector(monkeypatch)
    captured: dict = {}
    with _patch_urlopen(_judge_response(["R"]), captured):
        detector._query_judge_api([("q", "r")])
    assert "response_format" in captured["body"]
    assert captured["body"]["temperature"] == 0.0


def test_vllm_reasoning_budget_zero_disables_extra_tokens(monkeypatch):
    """reasoning_budget=0 on a non-reasoning custom endpoint opts out."""
    detector = _vllm_detector(monkeypatch)
    captured: dict = {}
    with _patch_urlopen(_judge_response(["R"]), captured):
        detector._query_judge_api([("q", "r")])
    assert captured["body"]["max_tokens"] == 55


def test_openai_preset_uses_max_completion_tokens(monkeypatch):
    """OpenAI (newer models) require 'max_completion_tokens' instead of 'max_tokens'."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    config = AbliterixConfig(
        detection={
            "llm_judge_base_url": "https://api.openai.com/v1",
            "llm_judge_model": "gpt-5.4-mini-2026-03-17",
            "llm_judge_api_key_env": "OPENAI_API_KEY",
            "llm_judge_max_tokens_field": "max_completion_tokens",
            "llm_judge_reasoning_budget": 0,
        }
    )
    detector = RefusalDetector(config)
    detector._cache = None
    captured: dict = {}
    with _patch_urlopen(_judge_response(["R"]), captured):
        detector._query_judge_api([("q", "r")])
    assert "max_completion_tokens" in captured["body"]
    assert "max_tokens" not in captured["body"]
    assert captured["body"]["max_completion_tokens"] == 55


def test_default_request_uses_max_tokens(monkeypatch):
    """Legacy 'max_tokens' must remain the default for every other provider."""
    detector = _openrouter_detector(monkeypatch)
    captured: dict = {}
    with _patch_urlopen(_judge_response(["R"]), captured):
        detector._query_judge_api([("q", "r")])
    assert "max_tokens" in captured["body"]
    assert "max_completion_tokens" not in captured["body"]


def test_vllm_request_reads_llm_judge_api_key(monkeypatch):
    detector = _vllm_detector(monkeypatch)
    captured: dict = {}
    with _patch_urlopen(_judge_response(["R"]), captured):
        detector._query_judge_api([("q", "r")])
    headers = {k.lower(): v for k, v in captured["headers"].items()}
    assert headers.get("authorization") == "Bearer dummy-local-key"


# ---------------------------------------------------------------------------
# Per-provider preset contracts
#
# These tests pin the request shape each concrete provider expects so the
# generic path stays honest — if anyone ever regresses the header, URL, or
# body construction for a specific provider, the test will say which one.
# ---------------------------------------------------------------------------


def _build_detector(
    monkeypatch, env: dict[str, str], detection: dict
) -> RefusalDetector:
    for k in list(env.keys()):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    detector = RefusalDetector(AbliterixConfig(detection=detection))
    detector._cache = None
    return detector


def test_preset_azure_openai(monkeypatch):
    """Azure rejects Bearer auth — key goes into `api-key` header with no prefix."""
    detector = _build_detector(
        monkeypatch,
        {"AZURE_OPENAI_API_KEY": "az-key"},
        {
            "llm_judge_base_url": "https://contoso.openai.azure.com/openai/deployments/gpt-5-3-instant",
            "llm_judge_model": "gpt-5.3-instant",
            "llm_judge_api_key_env": "AZURE_OPENAI_API_KEY",
            "llm_judge_auth_header": "api-key",
            "llm_judge_auth_prefix": "",
            "llm_judge_reasoning_budget": 0,
        },
    )
    captured: dict = {}
    with _patch_urlopen(_judge_response(["R"]), captured):
        detector._query_judge_api([("q", "r")])
    headers = {k.lower(): v for k, v in captured["headers"].items()}
    assert headers["api-key"] == "az-key"
    assert "authorization" not in headers
    assert (
        captured["url"]
        == "https://contoso.openai.azure.com/openai/deployments/gpt-5-3-instant/chat/completions"
    )


def test_preset_deepseek_chat(monkeypatch):
    """DeepSeek's json_schema is unsupported — must go out without response_format."""
    detector = _build_detector(
        monkeypatch,
        {"DEEPSEEK_API_KEY": "ds-key"},
        {
            "llm_judge_base_url": "https://api.deepseek.com/v1",
            "llm_judge_model": "deepseek-chat",
            "llm_judge_api_key_env": "DEEPSEEK_API_KEY",
            "llm_judge_use_response_format": False,
            "llm_judge_reasoning_budget": 0,
        },
    )
    captured: dict = {}
    with _patch_urlopen(_judge_response(["R"]), captured):
        detector._query_judge_api([("q", "r")])
    assert "response_format" not in captured["body"]
    # DeepSeek relies on the prompt containing "json" — our prompt does.
    prompt = captured["body"]["messages"][0]["content"].lower()
    assert "json" in prompt
    assert captured["url"] == "https://api.deepseek.com/v1/chat/completions"


def test_preset_ollama_local(monkeypatch):
    """Ollama's /v1/chat/completions doesn't honour response_format; keep it off."""
    detector = _build_detector(
        monkeypatch,
        {"LLM_JUDGE_API_KEY": "ollama"},
        {
            "llm_judge_base_url": "http://localhost:11434/v1",
            "llm_judge_model": "qwen3:8b",
            "llm_judge_use_response_format": False,
            "llm_judge_reasoning_budget": 0,
        },
    )
    captured: dict = {}
    with _patch_urlopen(_judge_response(["R"]), captured):
        detector._query_judge_api([("q", "r")])
    assert "response_format" not in captured["body"]
    assert captured["url"] == "http://localhost:11434/v1/chat/completions"


def test_preset_gemini(monkeypatch):
    """Gemini's OpenAI-compat endpoint URL ends in /openai/ — keep it intact."""
    detector = _build_detector(
        monkeypatch,
        {"GEMINI_API_KEY": "gm-key"},
        {
            "llm_judge_base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "llm_judge_model": "gemini-3.1-flash-lite",
            "llm_judge_api_key_env": "GEMINI_API_KEY",
            "llm_judge_reasoning_budget": 0,
        },
    )
    captured: dict = {}
    with _patch_urlopen(_judge_response(["R"]), captured):
        detector._query_judge_api([("q", "r")])
    assert (
        captured["url"]
        == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    )
    headers = {k.lower(): v for k, v in captured["headers"].items()}
    assert headers["authorization"] == "Bearer gm-key"


def test_preset_perplexity_no_v1_suffix(monkeypatch):
    """Perplexity's base URL is bare — no /v1 — trailing-slash hygiene still holds."""
    detector = _build_detector(
        monkeypatch,
        {"PERPLEXITY_API_KEY": "pplx-key"},
        {
            "llm_judge_base_url": "https://api.perplexity.ai",
            "llm_judge_model": "sonar-pro",
            "llm_judge_api_key_env": "PERPLEXITY_API_KEY",
            "llm_judge_reasoning_budget": 0,
        },
    )
    captured: dict = {}
    with _patch_urlopen(_judge_response(["R"]), captured):
        detector._query_judge_api([("q", "r")])
    assert captured["url"] == "https://api.perplexity.ai/chat/completions"


def test_preset_xai_grok(monkeypatch):
    """xAI Grok is a plain OpenAI-compatible endpoint — all defaults work."""
    detector = _build_detector(
        monkeypatch,
        {"XAI_API_KEY": "xai-key"},
        {
            "llm_judge_base_url": "https://api.x.ai/v1",
            "llm_judge_model": "grok-4.20-non-reasoning",
            "llm_judge_api_key_env": "XAI_API_KEY",
            "llm_judge_reasoning_budget": 0,
        },
    )
    captured: dict = {}
    with _patch_urlopen(_judge_response(["R"]), captured):
        detector._query_judge_api([("q", "r")])
    assert captured["url"] == "https://api.x.ai/v1/chat/completions"
    assert captured["body"]["temperature"] == 0.0
    assert "response_format" in captured["body"]
    headers = {k.lower(): v for k, v in captured["headers"].items()}
    assert headers["authorization"] == "Bearer xai-key"


def test_preset_anthropic_openai_compat(monkeypatch):
    """Anthropic's OpenAI-compat layer works with the generic path + ANTHROPIC_API_KEY."""
    detector = _build_detector(
        monkeypatch,
        {"ANTHROPIC_API_KEY": "ant-key"},
        {
            "llm_judge_base_url": "https://api.anthropic.com/v1",
            "llm_judge_model": "claude-haiku-4.5",
            "llm_judge_api_key_env": "ANTHROPIC_API_KEY",
            "llm_judge_reasoning_budget": 0,
        },
    )
    captured: dict = {}
    with _patch_urlopen(_judge_response(["R"]), captured):
        detector._query_judge_api([("q", "r")])
    assert captured["url"] == "https://api.anthropic.com/v1/chat/completions"
    headers = {k.lower(): v for k, v in captured["headers"].items()}
    assert headers["authorization"] == "Bearer ant-key"
