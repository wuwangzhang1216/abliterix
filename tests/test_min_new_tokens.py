"""Verify that ``min_new_tokens`` is plumbed through AxEngine generation.

The HonestAbliterationBench (benchmarks/SPEC.md) requires that every
generation call uses ``min_new_tokens=100, max_new_tokens=150`` so that
delayed-refusal patterns become visible to the judge. The plumbing
already exists in ``engine.generate_text`` /
``engine.generate_text_batched``; this test exists so a future refactor
can't silently drop the kwarg.

Mocks the underlying ``model.generate`` so the test doesn't require a
real HF model or a GPU.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from abliterix.core.engine import SteeringEngine
from abliterix.types import ChatMessage


def _make_mock_engine() -> SteeringEngine:
    """Build a barebones SteeringEngine without loading a real model."""
    engine = object.__new__(SteeringEngine)  # bypass __init__

    captured: dict[str, object] = {}

    # Stub tokenizer: only batch_decode + pad_token_id are touched after we
    # also stub _tokenize below.
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.batch_decode.return_value = ["resp_a", "resp_b"]

    # Fake model: records every kwarg passed to .generate() and returns a
    # tensor whose shape >= input_ids so the post-slice in generate_text()
    # doesn't blow up.
    def fake_generate(**kwargs):
        captured.clear()
        captured.update(kwargs)
        return torch.zeros((2, 3 + 5), dtype=torch.long)

    model = SimpleNamespace(generate=fake_generate)

    engine.tokenizer = tokenizer
    engine.model = model  # ty:ignore[invalid-assignment]
    engine.response_prefix = ""
    engine.config = SimpleNamespace(
        inference=SimpleNamespace(max_gen_tokens=999, batch_size=2),
    )
    # _reset_position_cache is called from _generate; stub it out.
    engine._reset_position_cache = lambda: None  # type: ignore[method-assign]

    # Replace _tokenize so we don't need a real tokenizer / model.device.
    # _generate then unpacks **inputs into model.generate(); we strip the
    # input_ids on assertion via the captured dict but model.generate
    # tolerates extra kwargs because we control the fake_generate signature.
    fake_inputs = {"input_ids": torch.zeros((2, 3), dtype=torch.long)}
    engine._tokenize = lambda _msgs: fake_inputs  # type: ignore[method-assign]

    engine._captured_generate_kwargs = captured  # type: ignore[attr-defined]
    return engine


def test_generate_text_passes_min_new_tokens():
    engine = _make_mock_engine()
    msgs = [ChatMessage(system="", user="hi"), ChatMessage(system="", user="hi")]

    engine.generate_text(msgs, max_new_tokens=150, min_new_tokens=100)

    captured = engine._captured_generate_kwargs  # type: ignore[attr-defined]
    assert captured.get("max_new_tokens") == 150
    assert captured.get("min_new_tokens") == 100


def test_generate_text_omits_min_new_tokens_when_not_set():
    engine = _make_mock_engine()
    msgs = [ChatMessage(system="", user="hi"), ChatMessage(system="", user="hi")]

    engine.generate_text(msgs, max_new_tokens=42)

    captured = engine._captured_generate_kwargs  # type: ignore[attr-defined]
    assert captured.get("max_new_tokens") == 42
    assert "min_new_tokens" not in captured


def test_generate_text_batched_passes_min_new_tokens():
    engine = _make_mock_engine()
    msgs = [ChatMessage(system="", user="hi"), ChatMessage(system="", user="hi")]

    engine.generate_text_batched(msgs, max_new_tokens=150, min_new_tokens=100)

    captured = engine._captured_generate_kwargs  # type: ignore[attr-defined]
    assert captured.get("max_new_tokens") == 150
    assert captured.get("min_new_tokens") == 100
