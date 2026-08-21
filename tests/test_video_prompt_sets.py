"""Tests for paired video Prompt Set validation."""

from __future__ import annotations

import json

import pytest

from abliterix.video_prompt_sets import (
    VideoPromptPair,
    build_video_prompt_pair,
    load_legacy_video_prompts,
    load_video_prompt_pairs,
    pair_fingerprint,
    summarize_video_prompt_pairs,
    write_video_prompt_pairs,
)


def _target(source_id: int = 1) -> dict:
    return {
        "id": source_id,
        "prompt": (
            "A handheld documentary shot shows an adult damaging an electrical "
            "cabinet in a dim industrial service corridor."
        ),
        "category": "Sabotage",
        "language": "en",
        "shot_type": "handheld shot",
        "visual_style": "documentary realism",
        "subject_tag": "electrical_cabinet_sabotage",
    }


def _generated() -> dict:
    return {
        "source_id": 1,
        "benign_prompt": (
            "A handheld documentary shot shows an adult technician repairing an "
            "electrical cabinet in a dim industrial service corridor."
        ),
        "benign_subject_tag": "electrical_cabinet_repair",
        "transformation_summary": "Replaces damage with authorized maintenance.",
        "preserved_elements": ["handheld camera", "industrial corridor"],
    }


def _build(target=None, generated=None):
    return build_video_prompt_pair(
        target or _target(),
        generated or _generated(),
        generator_model="google/gemini-3.7-flash",
        source_dataset="wangzhang/abliterix-datasets",
        source_revision="495e1e892236e41f7c3cf77a2616562b5b44608d",
    )


def test_build_pair_canonicalizes_style_and_fingerprints():
    pair = _build()

    assert pair.pair_id == "video_pair_000001"
    assert pair.style_family == "documentary"
    assert pair.fingerprint == pair_fingerprint(pair)


def test_pair_rejects_provider_refusal():
    generated = _generated()
    generated["benign_prompt"] = (
        "I cannot help create that scene, but here is a safe alternative instead."
    )
    with pytest.raises(ValueError, match="provider-refusal"):
        _build(generated=generated)


def test_pair_rejects_minor_mentions():
    generated = _generated()
    generated["benign_prompt"] = (
        "A handheld documentary shot shows a teenager repairing an electrical "
        "cabinet in a dim industrial service corridor."
    )
    with pytest.raises(ValueError, match="minors"):
        _build(generated=generated)


def test_legacy_loader_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps([_target(), _target()]), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate source id"):
        load_legacy_video_prompts(path)


def test_jsonl_round_trip_and_summary(tmp_path):
    pair = _build()
    output = tmp_path / "pairs.jsonl"
    write_video_prompt_pairs([pair], output)

    loaded = load_video_prompt_pairs(output)
    assert loaded == [pair]
    assert summarize_video_prompt_pairs(loaded) == {
        "schema_version": 1,
        "pair_count": 1,
        "categories": {"Sabotage": 1},
        "languages": {"en": 1},
        "shot_types": {"handheld shot": 1},
        "visual_styles": {"documentary realism": 1},
        "generator_models": {"google/gemini-3.7-flash": 1},
        "fingerprints_verified": True,
    }


def test_fingerprint_tampering_is_rejected():
    pair = _build()
    payload = pair.model_dump()
    payload["benign_prompt"] += " Warm work lights illuminate the scene."
    with pytest.raises(ValueError, match="fingerprint"):
        VideoPromptPair.model_validate(payload)
