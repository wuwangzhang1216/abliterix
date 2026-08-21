"""Offline tests for the OpenRouter video Prompt Set refiner."""

from __future__ import annotations

import asyncio
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
