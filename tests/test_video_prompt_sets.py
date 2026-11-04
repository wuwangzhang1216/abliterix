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
            "A handheld documentary shot shows an adult standing beside an electrical "
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
        source_split="train",
    )


def test_build_pair_canonicalizes_style_and_fingerprints():
    pair = _build()

    assert pair.pair_id == "video_pair_000001"
    assert pair.source_target_prompt == _target()["prompt"]
    assert pair.target_prompt == _target()["prompt"]
    assert pair.schema_version == 3
    assert pair.generation_policy == "preserve_source_benign_counterfactual_v1"
    assert pair.style_family == "documentary"
    assert pair.fingerprint == pair_fingerprint(pair)


def test_pair_rejects_provider_refusal():
    generated = _generated()
    generated["benign_prompt"] = (
        "I cannot help create that scene, but here is a safe alternative instead."
    )
    with pytest.raises(ValueError, match="provider-refusal"):
        _build(generated=generated)


def test_pair_allows_minor_mentions():
    generated = _generated()
    generated["benign_prompt"] = (
        "A handheld documentary shot shows a teenager repairing an electrical "
        "cabinet in a dim industrial service corridor."
    )
    pair = _build(generated=generated)

    assert "teenager" in pair.benign_prompt


def test_pair_rejects_provider_target_rewriting():
    generated = _generated()
    generated["target_prompt"] = _target()["prompt"]

    with pytest.raises(ValueError, match="must not rewrite"):
        _build(generated=generated)


def test_legacy_loader_allows_minor_mentions(tmp_path):
    target = _target()
    target["prompt"] = (
        "A handheld documentary shot shows a teenager standing beside an electrical "
        "cabinet in a dim industrial service corridor."
    )
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps([target]), encoding="utf-8")

    assert load_legacy_video_prompts(path)[0]["prompt"] == target["prompt"]


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
        "schema_version": 3,
        "pair_schema_versions": {3: 1},
        "pair_count": 1,
        "categories": {"Sabotage": 1},
        "languages": {"en": 1},
        "shot_types": {"handheld shot": 1},
        "visual_styles": {"documentary realism": 1},
        "generator_models": {"google/gemini-3.7-flash": 1},
        "source_preservation": {
            "verbatim_target_count": 1,
            "mean_source_chars": len(pair.source_target_prompt),
            "mean_target_chars": len(pair.target_prompt),
            "mean_growth_ratio": round(
                len(pair.target_prompt) / len(pair.source_target_prompt), 6
            ),
            "min_growth_ratio": round(
                len(pair.target_prompt) / len(pair.source_target_prompt), 6
            ),
            "max_growth_ratio": round(
                len(pair.target_prompt) / len(pair.source_target_prompt), 6
            ),
        },
        "fingerprints_verified": True,
    }


def test_fingerprint_tampering_is_rejected():
    pair = _build()
    payload = pair.model_dump()
    payload["benign_prompt"] += " Warm work lights illuminate the scene."
    with pytest.raises(ValueError, match="fingerprint"):
        VideoPromptPair.model_validate(payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", True),
        ("id", 1.5),
        ("id", "1"),
        ("id", 0),
        ("id", 1000000),
        ("prompt", None),
        ("prompt", "short"),
        ("shot_type", ""),
        ("subject_tag", "not a tag"),
        ("category", []),
    ],
)
def test_legacy_loader_validates_before_provider_work(tmp_path, field, value):
    target = _target()
    target[field] = value
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps([target]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_legacy_video_prompts(path)


def test_pair_rejects_inconsistent_id_and_unknown_schema():
    pair = _build()
    for update in ({"pair_id": "video_pair_000002"}, {"schema_version": 99}):
        with pytest.raises(ValueError):
            VideoPromptPair.model_validate({**pair.model_dump(), **update})


def test_v3_preserves_source_verbatim_including_whitespace_and_case():
    target = _target()
    target["prompt"] = "  A Static shot beside a cabinet.\nThe corridor is Quiet.  "
    pair = _build(target=target)
    assert pair.target_prompt == target["prompt"]
    assert pair.source_target_prompt == target["prompt"]
    assert pair.target_expansion_summary is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("generator_model", "another/model"),
        ("source_split", "validation"),
        ("transformation_summary", "A different explanation of the replacement."),
    ],
)
def test_v3_fingerprint_covers_provenance_and_annotations(field, value):
    payload = _build().model_dump()
    payload[field] = value
    with pytest.raises(ValueError, match="fingerprint"):
        VideoPromptPair.model_validate(payload)


def test_v2_import_preserves_historical_fingerprint_contract(tmp_path):
    payload = _build().model_dump()
    payload.update(
        schema_version=2,
        source_split=None,
        generation_policy=None,
        fingerprint=None,
        target_prompt=payload["source_target_prompt"]
        + " Warm daylight fills the hallway.",
        target_expansion_summary="Historical synthetic fixture adds neutral lighting.",
    )
    legacy = VideoPromptPair.model_validate(payload)
    serialized = legacy.model_dump(exclude={"source_split", "generation_policy"})
    output = tmp_path / "legacy-v2.jsonl"
    output.write_text(json.dumps(serialized) + "\n", encoding="utf-8")
    imported = load_video_prompt_pairs(output)[0]
    assert imported.schema_version == 2
    assert imported.fingerprint == legacy.fingerprint
    assert imported.generation_policy is None
