"""Tests for MiniMax-H3 manifest validation and execution planning."""

from __future__ import annotations

import json

import pytest

from abliterix.h3_training import (
    H3TrainingConfig,
    H3TrainingPlan,
    load_h3_manifest,
    write_h3_plan,
)


def _write_manifest(tmp_path, *, task="fl2va"):
    video = tmp_path / "target.mp4"
    audio = tmp_path / "target.wav"
    video.write_bytes(b"video")
    audio.write_bytes(b"audio")
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "sample-1",
                "task": task,
                "caption": "A camera tracks an adult walking through a gallery.",
                "target_video": str(video),
                "target_audio": str(audio),
                "reference_images": [],
                "reference_videos": [],
                "reference_audios": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def _config(tmp_path, manifest, **overrides):
    model = tmp_path / "MiniMax-H3"
    trainer = tmp_path / "trainer"
    model.mkdir(exist_ok=True)
    trainer.mkdir(exist_ok=True)
    (trainer / "prepare_cache.py").write_text("", encoding="utf-8")
    (trainer / "train.py").write_text("", encoding="utf-8")
    payload = {
        "model_path": model,
        "trainer_path": trainer,
        "manifest_path": manifest,
        "cache_path": tmp_path / "cache",
        "output_path": tmp_path / "run",
        "frames": 430,
        "num_gpus": 8,
    }
    payload.update(overrides)
    return H3TrainingConfig.model_validate(payload)


def test_h3_dimensions_and_frames_are_validated(tmp_path):
    manifest = _write_manifest(tmp_path)
    with pytest.raises(ValueError, match="divisible by 32"):
        _config(tmp_path, manifest, width=767)
    with pytest.raises(ValueError, match="frames % 17 == 5"):
        _config(tmp_path, manifest, frames=431)


def test_prompt_only_manifest_is_rejected(tmp_path):
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(
        json.dumps({"id": "prompt-only", "caption": "A quiet park."}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="target_video"):
        load_h3_manifest(manifest)


def test_plan_emits_argument_vectors(tmp_path):
    manifest = _write_manifest(tmp_path)
    config = _config(tmp_path, manifest)
    plan = H3TrainingPlan.from_config(config)

    assert plan.cache_command()[-1] == "--encode-audio"
    assert plan.train_command()[:3] == ["deepspeed", "--num_gpus", "8"]
    assert "--trainable" in plan.train_command()
    assert "lora" in plan.train_command()

    output = tmp_path / "plan.json"
    write_h3_plan(plan, output)
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["sample_count"] == 1
    assert isinstance(artifact["train_command"], list)


def test_reference_media_require_ref2va(tmp_path):
    video = tmp_path / "target.mp4"
    image = tmp_path / "reference.png"
    video.write_bytes(b"video")
    image.write_bytes(b"image")
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "bad-task",
                "task": "fl2va",
                "caption": "Use the reference image.",
                "target_video": str(video),
                "reference_images": [str(image)],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="reference media"):
        load_h3_manifest(manifest)


def test_plan_rejects_missing_media(tmp_path):
    manifest = _write_manifest(tmp_path)
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    rows[0]["target_video"] = str(tmp_path / "missing.mp4")
    manifest.write_text(json.dumps(rows[0]) + "\n")
    config = _config(tmp_path, manifest)
    with pytest.raises(FileNotFoundError, match="missing media"):
        H3TrainingPlan.from_config(config)


def test_manifest_resolves_media_relative_to_its_directory(tmp_path):
    media = tmp_path / "media"
    media.mkdir()
    (media / "target.mp4").write_bytes(b"video")
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "relative",
                "caption": "A quiet gallery scene.",
                "target_video": "media/target.mp4",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    sample = load_h3_manifest(manifest)[0]
    assert sample.target_video == (media / "target.mp4").resolve()
