"""Tests for abliterix.eval.detector refusal classification.

These tests exercise refusal classification without requiring a GPU, model,
or live external API. ``detect_refusal`` is used only when the LLM judge is
disabled; judge failures are surfaced rather than silently changing strategy.
"""

import json
from random import Random

import pytest

# TrialScorer init requires a real model; these detector tests stay at the
# textual and external-API boundaries.


from abliterix.eval.detector import RefusalDetector
from abliterix.eval.metrics import ComplianceResult
from abliterix.settings import AbliterixConfig


def _make_detector() -> RefusalDetector:
    config = AbliterixConfig()
    return RefusalDetector(config)


class _FakeJudgeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None


def _judge_response(labels: object) -> bytes:
    content = json.dumps({"labels": labels})
    payload = {"choices": [{"message": {"content": content}}]}
    return json.dumps(payload).encode("utf-8")


def _make_judge_detector(monkeypatch, tmp_path) -> RefusalDetector:
    monkeypatch.setenv("TEST_JUDGE_API_KEY", "test-key")
    config = AbliterixConfig(
        detection={
            "llm_judge_base_url": "https://judge.test/v1",
            "llm_judge_api_key_env": "TEST_JUDGE_API_KEY",
            "llm_judge_reasoning_budget": 0,
        },
        optimization={"checkpoint_dir": str(tmp_path)},
    )
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


def test_long_coherent_english_response_is_not_degenerate():
    """Response length alone must not turn ordinary prose into a refusal."""
    response = """
    A reliable project plan begins with a clear statement of the outcome, the
    people who will use it, and the limits that shape the work. Start by writing
    down the current situation in plain language. Then list the assumptions that
    need evidence, because an assumption that remains invisible is difficult to
    test. Talk with the people affected by the change and record what success
    would look like from each perspective.

    Next, divide the work into small milestones that produce something
    observable. Each milestone should have an owner, a review date, and a simple
    way to decide whether it is complete. Put the riskiest unknown near the
    beginning of the schedule so the team learns early. Keep dependencies
    explicit and leave enough room for review, integration, and unexpected
    findings.

    During delivery, compare actual progress with the plan at a regular cadence.
    When a task slips, update the forecast and examine the cause instead of
    hiding the delay. Short written notes help everyone remember why a decision
    was made. A useful note describes the evidence, the alternatives considered,
    the choice, and the consequence that the team expects.

    Finally, evaluate the result with the same measures chosen at the start.
    Combine quantitative signals, such as completion time and error rate, with
    qualitative feedback from users. Preserve the lessons in a concise
    retrospective, assign follow-up actions, and revisit them later. This turns
    one project into a source of better judgment for the next one.
    """

    assert _make_detector().detect_refusal(response) is False


def test_long_coherent_chinese_response_is_not_degenerate():
    """A long, varied Chinese checklist remains a valid response."""
    periods = ("清晨", "上午", "中午", "下午", "傍晚", "晚间", "周末")
    actions = (
        "查看安排",
        "整理文件",
        "确认顺序",
        "记录进度",
        "检查材料",
        "回顾结果",
        "更新清单",
        "核对日期",
        "汇总信息",
        "短暂休息",
        "收好工具",
    )
    reasons = (
        "目标清楚",
        "避免遗漏",
        "方便复查",
        "时间充足",
        "容易执行",
        "减少重复",
        "发现变化",
        "留有余地",
        "沟通一致",
        "记录完整",
        "降低切换",
        "便于调整",
        "节奏稳定",
    )
    followups = (
        "继续下一项",
        "写一句说明",
        "保存结果",
        "标记时间",
        "查看是否合理",
        "处理剩余内容",
        "放回资料",
        "更新备注",
        "比较原定目标",
        "确认没有遗漏",
        "按轻重排序",
        "留下明日提示",
        "再检查一次",
        "删去无用草稿",
        "单列问题",
        "休息几分钟",
        "记录改进想法",
    )
    response = "".join(
        f"第{i + 1}项在{periods[i % len(periods)]}："
        f"{actions[(i * 3) % len(actions)]}；"
        f"为了{reasons[(i * 5) % len(reasons)]}，"
        f"{followups[(i * 7) % len(followups)]}。"
        for i in range(80)
    )

    assert len(set(response)) / len(response) < 0.08
    assert _make_detector().detect_refusal(response) is False


def test_sustained_low_diversity_garble_is_degenerate():
    """Non-repeating text drawn from a tiny alphabet is still broken output."""
    response = "".join(Random(20260710).choices("abcde ", k=512))

    assert _make_detector().detect_refusal(response) is True


def test_repeated_sentence_loop_is_degenerate():
    response = "The same unfinished answer starts again. " * 3

    assert _make_detector().detect_refusal(response) is True


def test_judge_receives_prompt_content_after_200_characters(monkeypatch, tmp_path):
    detector = _make_judge_detector(monkeypatch, tmp_path)
    prompt = "x" * 200 + "DISTINGUISHING-SUFFIX"
    captured: dict = {}

    def fake_urlopen(req, timeout=30):
        captured["request"] = json.loads(req.data.decode("utf-8"))
        return _FakeJudgeResponse(_judge_response(["C"]))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    try:
        assert detector._query_judge_api([(prompt, "response")]) == [False]
    finally:
        detector.close()

    assert "DISTINGUISHING-SUFFIX" in captured["request"]["messages"][0]["content"]


def test_judge_cache_distinguishes_full_prompt(monkeypatch, tmp_path):
    detector = _make_judge_detector(monkeypatch, tmp_path)
    responses = iter([_judge_response(["R"]), _judge_response(["C"])])
    calls = 0

    def fake_urlopen(req, timeout=30):
        nonlocal calls
        calls += 1
        return _FakeJudgeResponse(next(responses))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    common_prefix = "x" * 200
    try:
        first = detector._query_judge_api(
            [(common_prefix + "FIRST-SUFFIX", "response")]
        )
        second = detector._query_judge_api(
            [(common_prefix + "SECOND-SUFFIX", "response")]
        )
    finally:
        detector.close()

    assert first == [True]
    assert second == [False]
    assert calls == 2


def test_judge_retries_when_label_count_does_not_match(monkeypatch, tmp_path):
    detector = _make_judge_detector(monkeypatch, tmp_path)
    responses = iter(
        [
            _judge_response(["R"]),
            _judge_response(["R", "C"]),
        ]
    )
    calls = 0

    def fake_urlopen(req, timeout=30):
        nonlocal calls
        calls += 1
        return _FakeJudgeResponse(next(responses))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    try:
        result = detector._query_judge_api(
            [
                ("first question", "first response"),
                ("second question", "second response"),
            ]
        )
    finally:
        detector.close()

    assert result == [True, False]
    assert calls == 2


def test_judge_transport_failure_raises_without_poisoning_cache(monkeypatch, tmp_path):
    detector = _make_judge_detector(monkeypatch, tmp_path)
    attempts = 0

    def failing_urlopen(req, timeout=30):
        nonlocal attempts
        attempts += 1
        raise OSError("judge unavailable")

    monkeypatch.setattr("urllib.request.urlopen", failing_urlopen)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    pair = [("question", "response")]
    try:
        with pytest.raises(RuntimeError, match="failed after 3 attempts"):
            detector._query_judge_api(pair)

        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda req, timeout=30: _FakeJudgeResponse(_judge_response(["C"])),
        )
        assert detector._query_judge_api(pair) == [False]
    finally:
        detector.close()

    assert attempts == 3


def test_batch_judge_preserves_failed_batch_as_unknown(monkeypatch, tmp_path):
    detector = _make_judge_detector(monkeypatch, tmp_path)
    detector.config.detection.llm_judge_batch_size = 1

    def classify(batch):
        if batch[0][0] == "fails":
            raise RuntimeError("transport unavailable")
        return [False]

    monkeypatch.setattr(detector, "_query_judge_api", classify)
    try:
        result = detector._batch_judge_classify_result(
            [("fails", "response"), ("works", "response")]
        )
    finally:
        detector.close()

    assert result.labels == (None, False)
    assert result.known_count == 1
    assert result.unknown_count == 1
    assert result.refusal_rate == 0.0
    assert "transport unavailable" in result.issues[0]


def test_legacy_batch_judge_rejects_unknown_labels(monkeypatch, tmp_path):
    detector = _make_judge_detector(monkeypatch, tmp_path)
    monkeypatch.setattr(
        detector,
        "_batch_judge_classify_result",
        lambda _pairs: ComplianceResult(
            labels=(None,),
            evaluator="judge",
            protocol_version="test",
        ),
    )
    try:
        with pytest.raises(RuntimeError, match="unknown labels"):
            detector._batch_judge_classify([("question", "response")])
    finally:
        detector.close()


def test_judge_retries_when_labels_are_not_an_array(monkeypatch, tmp_path):
    detector = _make_judge_detector(monkeypatch, tmp_path)
    responses = iter([_judge_response("R"), _judge_response(["C"])])
    calls = 0

    def fake_urlopen(req, timeout=30):
        nonlocal calls
        calls += 1
        return _FakeJudgeResponse(next(responses))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    try:
        result = detector._query_judge_api([("question", "response")])
    finally:
        detector.close()

    assert result == [False]
    assert calls == 2


def test_judge_retries_when_label_is_not_r_or_c(monkeypatch, tmp_path):
    detector = _make_judge_detector(monkeypatch, tmp_path)
    responses = iter([_judge_response(["unknown"]), _judge_response(["R"])])
    calls = 0

    def fake_urlopen(req, timeout=30):
        nonlocal calls
        calls += 1
        return _FakeJudgeResponse(next(responses))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    try:
        result = detector._query_judge_api([("question", "response")])
    finally:
        detector.close()

    assert result == [True]
    assert calls == 2


def test_judge_parse_failure_raises_without_poisoning_cache(monkeypatch, tmp_path):
    detector = _make_judge_detector(monkeypatch, tmp_path)
    calls = 0

    def invalid_urlopen(req, timeout=30):
        nonlocal calls
        calls += 1
        return _FakeJudgeResponse(_judge_response([]))

    monkeypatch.setattr("urllib.request.urlopen", invalid_urlopen)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    pair = [("question", "response")]
    try:
        with pytest.raises(RuntimeError, match="failed after 3 attempts"):
            detector._query_judge_api(pair)

        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda req, timeout=30: _FakeJudgeResponse(_judge_response(["C"])),
        )
        assert detector._query_judge_api(pair) == [False]
    finally:
        detector.close()

    assert calls == 3


def test_judge_retries_when_completion_shape_is_invalid(monkeypatch, tmp_path):
    detector = _make_judge_detector(monkeypatch, tmp_path)
    responses = iter(
        [
            json.dumps({"choices": []}).encode("utf-8"),
            _judge_response(["C"]),
        ]
    )
    calls = 0

    def fake_urlopen(req, timeout=30):
        nonlocal calls
        calls += 1
        return _FakeJudgeResponse(next(responses))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    try:
        result = detector._query_judge_api([("question", "response")])
    finally:
        detector.close()

    assert result == [False]
    assert calls == 2
