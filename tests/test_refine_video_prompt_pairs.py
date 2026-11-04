"""Offline regressions for provenance and terminal provider outcomes."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

import scripts.refine_video_prompt_pairs as refiner


def _target(source_id=1):
    return {
        "id": source_id,
        "prompt": "A static documentary shot shows an adult beside a cabinet in a corridor.",
        "category": "Sabotage",
        "language": "en",
        "shot_type": "static locked-off shot",
        "visual_style": "documentary realism",
        "subject_tag": "cabinet_scene",
    }


def _payload():
    return {
        "pairs": [
            {
                "source_id": 1,
                "benign_prompt": "A static documentary shot shows an adult reading a notice in a corridor.",
                "benign_subject_tag": "notice_reading",
                "transformation_summary": "Synthetic fixture with a notice.",
                "preserved_elements": ["static camera", "corridor"],
            }
        ],
    }


def _response(*, finish="stop", native=None, refusal=None, content=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish,
                native_finish_reason=native,
                message=SimpleNamespace(
                    content=json.dumps(_payload()) if content is None else content,
                    refusal=refusal,
                ),
            )
        ]
    )


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
        self.chat = SimpleNamespace(completions=self)
        self.closed = False

    async def create(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        self.closed = True


def _generate(client):
    return asyncio.run(
        refiner._generate_batch(
            client,
            "test/model",
            "test/data",
            "a" * 40,
            [_target()],
            refiner.RateLimiter(1_000_000),
            asyncio.Semaphore(1),
            max_retries=4,
            source_split="train",
        )
    )


@pytest.mark.parametrize(
    "response",
    [
        _response(finish="content_filter"),
        _response(finish="length"),
        _response(finish=None),
        _response(native="SAFETY"),
        _response(native="PROHIBITED_CONTENT"),
        _response(refusal="Request declined"),
        _response(content="I cannot help create that scene."),
        _response(content=""),
        _response(content='{"pairs":'),
        SimpleNamespace(choices=[]),
    ],
)
def test_terminal_responses_are_never_retried(response):
    client = FakeClient([response, _response()])
    with pytest.raises(refiner.ProviderReviewRequired):
        _generate(client)
    assert len(client.calls) == 1


def test_refusal_inside_valid_json_is_not_retried():
    payload = _payload()
    payload["pairs"][0]["benign_prompt"] = (
        "I cannot help create that scene, but can discuss something else."
    )
    client = FakeClient([_response(content=json.dumps(payload)), _response()])
    with pytest.raises(refiner.ProviderReviewRequired, match="invalid_response"):
        _generate(client)
    assert len(client.calls) == 1


@pytest.mark.parametrize("source_id", [True, "1", 1.1])
def test_provider_ids_are_not_coerced(source_id):
    payload = _payload()
    payload["pairs"][0]["source_id"] = source_id
    with pytest.raises(refiner.ProviderReviewRequired):
        _generate(FakeClient([_response(content=json.dumps(payload))]))


class HTTPFailure(Exception):
    def __init__(self, status):
        self.status_code = status
        super().__init__("private provider error body")


def test_authentication_failure_is_not_retried():
    client = FakeClient([HTTPFailure(401), _response()])
    with pytest.raises(RuntimeError, match="non-retryable.*401") as exc:
        _generate(client)
    assert "private" not in str(exc.value)
    assert len(client.calls) == 1


def test_http_forbidden_requires_review_without_retry():
    client = FakeClient([HTTPFailure(403), _response()])
    with pytest.raises(refiner.ProviderReviewRequired, match="http_403"):
        _generate(client)
    assert len(client.calls) == 1


def test_transient_transport_retry_keeps_identical_request(monkeypatch):
    async def no_sleep(_delay):
        pass

    monkeypatch.setattr(refiner.asyncio, "sleep", no_sleep)
    client = FakeClient([HTTPFailure(429), _response()])
    assert len(_generate(client)) == 1
    assert client.calls[0] == client.calls[1]


def _args(tmp_path):
    input_path = tmp_path / "source.json"
    input_path.write_text(json.dumps([_target()]), encoding="utf-8")
    return argparse.Namespace(
        input=input_path,
        output=tmp_path / "pairs.jsonl",
        progress=None,
        model="test/model",
        source_dataset="test/data",
        source_revision="a" * 40,
        source_split="train",
        batch_size=8,
        workers=1,
        rate_limit_rpm=1_000_000,
        max_retries=4,
        limit=None,
    )


def _install_client(monkeypatch, client):
    def factory(**kwargs):
        assert kwargs["max_retries"] == 0
        return client

    monkeypatch.setattr(refiner, "AsyncOpenAI", factory)
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-placeholder")


def _forbid_client(monkeypatch):
    def forbidden(**_kwargs):
        pytest.fail("resume validation must not construct a provider client")

    monkeypatch.setattr(refiner, "AsyncOpenAI", forbidden)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


def test_identical_resume_is_offline_and_has_auditable_provenance(
    tmp_path, monkeypatch
):
    args = _args(tmp_path)
    client = FakeClient([_response()])
    _install_client(monkeypatch, client)
    pairs = asyncio.run(refiner.build_pairs(args))
    assert client.closed
    _forbid_client(monkeypatch)
    assert asyncio.run(refiner.build_pairs(args)) == pairs
    audit = json.loads(args.output.with_suffix(".provenance.json").read_text())
    assert audit["pair_count"] == 1
    assert audit["review"] == []
    assert audit["run"]["source_split"] == "train"
    assert (
        audit["output_sha256"]
        == refiner.hashlib.sha256(args.output.read_bytes()).hexdigest()
    )
    assert audit["run_sha256"] == refiner._digest(audit["run"])


@pytest.mark.parametrize(
    "change",
    [
        "prompt",
        "metadata",
        "source_dataset",
        "source_revision",
        "source_split",
        "model",
        "batch_size",
        "policy",
        "max_retries",
    ],
)
def test_changed_resume_identity_fails_before_client(tmp_path, monkeypatch, change):
    args = _args(tmp_path)
    _install_client(monkeypatch, FakeClient([_response()]))
    asyncio.run(refiner.build_pairs(args))
    checkpoint = args.output.with_suffix(".progress.jsonl")
    before = checkpoint.read_bytes()
    if change in {"prompt", "metadata"}:
        rows = json.loads(args.input.read_text())
        rows[0]["prompt" if change == "prompt" else "extra_metadata"] = (
            "Different source content with the same ID and row count."
        )
        args.input.write_text(json.dumps(rows), encoding="utf-8")
    elif change == "policy":
        monkeypatch.setattr(
            refiner, "SYSTEM_PROMPT", refiner.SYSTEM_PROMPT + "\nPolicy revision."
        )
    elif change in {"batch_size", "max_retries"}:
        setattr(args, change, getattr(args, change) + 1)
    else:
        setattr(args, change, getattr(args, change) + "-changed")
    _forbid_client(monkeypatch)
    with pytest.raises(ValueError, match="provenance mismatch"):
        asyncio.run(refiner.build_pairs(args))
    assert checkpoint.read_bytes() == before


def test_blocked_batch_is_persisted_and_never_reissued(tmp_path, monkeypatch):
    args = _args(tmp_path)
    client = FakeClient([_response(finish="content_filter"), _response()])
    _install_client(monkeypatch, client)
    assert asyncio.run(refiner.build_pairs(args)) == []
    assert len(client.calls) == 1
    _forbid_client(monkeypatch)
    assert asyncio.run(refiner.build_pairs(args)) == []
    audit = json.loads(args.output.with_suffix(".provenance.json").read_text())
    assert audit["review"] == [
        {"source_target_id": 1, "status": "review", "reason": "non_stop_finish"}
    ]


def test_interrupted_request_is_not_reissued(tmp_path, monkeypatch):
    args = _args(tmp_path)
    _install_client(monkeypatch, FakeClient([asyncio.CancelledError()]))
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(refiner.build_pairs(args))
    _forbid_client(monkeypatch)
    assert asyncio.run(refiner.build_pairs(args)) == []
    audit = json.loads(args.output.with_suffix(".provenance.json").read_text())
    assert audit["review"][0]["reason"] == "interrupted_or_in_flight"


@pytest.mark.parametrize(
    "damage",
    ["truncated_json", "missing_record", "duplicate", "unknown", "model", "legacy"],
)
def test_corrupt_checkpoint_is_rejected(tmp_path, monkeypatch, damage):
    args = _args(tmp_path)
    _install_client(monkeypatch, FakeClient([_response()]))
    asyncio.run(refiner.build_pairs(args))
    checkpoint = args.output.with_suffix(".progress.jsonl")
    lines = [json.loads(line) for line in checkpoint.read_text().splitlines()]
    if damage == "truncated_json":
        checkpoint.write_text(checkpoint.read_text()[:-10], encoding="utf-8")
    elif damage == "legacy":
        checkpoint.write_bytes(args.output.read_bytes())
    else:
        if damage == "missing_record":
            lines.pop()
        elif damage == "duplicate":
            lines.append(lines[-1])
        elif damage == "unknown":
            lines[-1]["source_target_id"] = 2
        elif damage == "model":
            lines[-1]["pair"]["generator_model"] = "different/model"
        if damage != "missing_record":
            lines[0]["records_sha256"] = refiner._digest(lines[1:])
        checkpoint.write_text(
            "\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8"
        )
    _forbid_client(monkeypatch)
    with pytest.raises(ValueError):
        asyncio.run(refiner.build_pairs(args))


def test_concurrent_writer_lock_is_not_removed(tmp_path, monkeypatch):
    args = _args(tmp_path)
    lock = args.output.with_suffix(".progress.jsonl.lock")
    lock.write_text("another process", encoding="utf-8")
    _forbid_client(monkeypatch)
    with pytest.raises(ValueError, match="locked"):
        asyncio.run(refiner.build_pairs(args))
    assert lock.read_text() == "another process"


def test_output_cannot_overwrite_source(tmp_path, monkeypatch):
    args = _args(tmp_path)
    args.output = args.input
    before = args.input.read_bytes()
    _forbid_client(monkeypatch)
    with pytest.raises(ValueError, match="distinct"):
        asyncio.run(refiner.build_pairs(args))
    assert args.input.read_bytes() == before


@pytest.mark.parametrize("destination", ["checkpoint_tmp", "output"])
def test_hardlinked_artifact_cannot_truncate_source(tmp_path, monkeypatch, destination):
    args = _args(tmp_path)
    alias = (
        args.output.with_suffix(".progress.jsonl.tmp")
        if destination == "checkpoint_tmp"
        else args.output
    )
    alias.hardlink_to(args.input)
    before = args.input.read_bytes()
    _forbid_client(monkeypatch)
    with pytest.raises(ValueError, match="distinct"):
        asyncio.run(refiner.build_pairs(args))
    assert args.input.read_bytes() == before
    assert alias.read_bytes() == before


def test_generation_preserves_source_and_requests_only_benign_fields():
    client = FakeClient([_response()])
    pair = _generate(client)[0]
    assert pair.target_prompt == _target()["prompt"]
    assert pair.source_target_prompt == _target()["prompt"]
    fields = client.calls[0]["response_format"]["json_schema"]["schema"]["properties"][
        "pairs"
    ]["items"]["properties"]
    assert "target_prompt" not in fields
    assert "target_expansion_summary" not in fields
    assert pair.schema_version == 3


def test_provider_cannot_replace_source_target():
    payload = _payload()
    payload["pairs"][0]["target_prompt"] = (
        "An unwanted rewritten caption, never accepted as a target."
    )
    client = FakeClient([_response(content=json.dumps(payload)), _response()])
    with pytest.raises(refiner.ProviderReviewRequired, match="invalid_response"):
        _generate(client)
    assert len(client.calls) == 1


def test_changed_selection_rejects_resume(tmp_path, monkeypatch):
    args = _args(tmp_path)
    args.input.write_text(json.dumps([_target(), _target(2)]), encoding="utf-8")
    args.limit = 1
    _install_client(monkeypatch, FakeClient([_response()]))
    asyncio.run(refiner.build_pairs(args))
    args.limit = 2
    _forbid_client(monkeypatch)
    with pytest.raises(ValueError, match="provenance mismatch"):
        asyncio.run(refiner.build_pairs(args))


def test_cross_batch_duplicate_is_excluded_and_not_retried(tmp_path, monkeypatch):
    args = _args(tmp_path)
    second = _target(2)
    second["prompt"] += " There is a window nearby."
    args.input.write_text(json.dumps([_target(), second]), encoding="utf-8")
    args.batch_size = 1
    payload = _payload()
    payload["pairs"][0]["source_id"] = 2
    client = FakeClient([_response(), _response(content=json.dumps(payload))])
    _install_client(monkeypatch, client)
    assert len(asyncio.run(refiner.build_pairs(args))) == 1
    assert len(client.calls) == 2
    _forbid_client(monkeypatch)
    assert len(asyncio.run(refiner.build_pairs(args))) == 1
    audit = json.loads(args.output.with_suffix(".provenance.json").read_text())
    assert audit["review"][0]["reason"] == "duplicate_generated_prompt"
