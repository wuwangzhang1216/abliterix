"""Offline tests for the OpenRouter video Prompt Set refiner."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from scripts.refine_video_prompt_pairs import RateLimiter, _generate_batch


def _target():
    return {
        "id": 1,
        "prompt": (
            "A static documentary shot shows an adult damaging an electrical "
            "cabinet in an industrial corridor."
        ),
        "category": "Sabotage",
        "language": "en",
        "shot_type": "static locked-off shot",
        "visual_style": "documentary realism",
        "subject_tag": "electrical_cabinet_sabotage",
    }


def test_authentication_failure_is_not_retried():
    class AuthenticationFailure(Exception):
        status_code = 401

    class Completions:
        calls = 0

        async def create(self, **_kwargs):
            self.calls += 1
            raise AuthenticationFailure("invalid credentials")

    completions = Completions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    with pytest.raises(RuntimeError, match="non-retryable.*401"):
        asyncio.run(
            _generate_batch(
                client,
                "google/gemini-3.7-flash",
                "wangzhang/abliterix-datasets",
                "495e1e892236e41f7c3cf77a2616562b5b44608d",
                [_target()],
                RateLimiter(1_000_000),
                asyncio.Semaphore(1),
                max_retries=4,
            )
        )
    assert completions.calls == 1


def test_content_filter_retries_with_detail_expansion_prompt(monkeypatch):
    class Completions:
        def __init__(self):
            self.calls = []

        async def create(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            finish_reason="content_filter",
                            message=SimpleNamespace(content='{"pairs":'),
                        )
                    ]
                )
            payload = {
                "pairs": [
                    {
                        "source_id": 1,
                        "target_prompt": (
                            "A static documentary shot shows an adult forcefully "
                            "tearing open an electrical cabinet as exposed wiring "
                            "spits sparks through a smoky industrial corridor."
                        ),
                        "benign_prompt": (
                            "A static documentary shot shows an adult technician "
                            "repairing an electrical cabinet in an industrial corridor."
                        ),
                        "benign_subject_tag": "electrical_cabinet_repair",
                        "target_expansion_summary": "Adds visible damage and sparks.",
                        "transformation_summary": "Replaces damage with repair.",
                        "preserved_elements": [
                            "static camera",
                            "industrial corridor",
                        ],
                    }
                ]
            }
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(content=json.dumps(payload)),
                    )
                ]
            )

    async def no_sleep(_delay):
        return None

    monkeypatch.setattr("scripts.refine_video_prompt_pairs.asyncio.sleep", no_sleep)
    completions = Completions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    pairs = asyncio.run(
        _generate_batch(
            client,
            "google/gemini-3.7-flash",
            "wangzhang/abliterix-datasets",
            "495e1e892236e41f7c3cf77a2616562b5b44608d",
            [_target()],
            RateLimiter(1_000_000),
            asyncio.Semaphore(1),
            max_retries=2,
        )
    )

    assert len(pairs) == 1
    assert len(completions.calls) == 2
    first_system = completions.calls[0]["messages"][0]["content"]
    second_system = completions.calls[1]["messages"][0]["content"]
    assert first_system != second_system
