"""Tests for MiniMax-H3 manifest validation and execution planning."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from abliterix import h3_training
from abliterix.h3_training import (
    H3TrainingConfig,
    H3TrainingPlan,
    load_h3_manifest,
    write_h3_plan,
)


def _write_manifest(tmp_path, *, task="fl2va"):
    video = tmp_path / "target.mp4"
    video.write_bytes(b"video")
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "sample-1",
                "task": task,
                "caption": "A camera tracks an adult walking through a gallery.",
                "target_video": str(video),
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

    assert plan.cache_command(tmp_path / "normalized.jsonl")[-1] == "--encode-audio"
    assert plan.train_command()[:3] == ["deepspeed", "--num_gpus", "8"]
    assert "--trainable" in plan.train_command()
    assert "lora" in plan.train_command()

    output = tmp_path / "plan.json"
    write_h3_plan(plan, output)
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["sample_count"] == 1
    assert isinstance(artifact["train_command"], list)
    assert artifact["trainer"]["local_commit"] is None
    assert artifact["trainer"]["compatibility"] == "unvalidated"
    assert artifact["trainer"]["runtime_validation"] == "not_run"
    assert artifact["trainer"]["reviewed_upstream_commit"] == (
        "74d535b2cae5cd56f9565059e6f2b6c9ddb39df4"
    )
    assert (
        artifact["trainer"]["script_sha256"]["train.py"]
        == hashlib.sha256(b"").hexdigest()
    )


def test_reference_media_are_rejected(tmp_path):
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


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("target_audio", "separate.wav", "external target_audio"),
        ("reference_images", ["reference.png"], "reference media"),
        ("reference_videos", ["reference.mp4"], "reference media"),
        ("reference_audios", ["reference.wav"], "reference media"),
        ("task", "ref2va", "only task='fl2va'"),
    ],
)
def test_manifest_rejects_inputs_the_trainer_would_ignore(
    tmp_path, field, value, error
):
    manifest = _write_manifest(tmp_path)
    row = json.loads(manifest.read_text(encoding="utf-8"))
    row[field] = value
    manifest.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        load_h3_manifest(manifest)


def test_plan_rejects_unsupported_variant(tmp_path):
    with pytest.raises(ValueError, match="only variant='fl2va'"):
        _config(tmp_path, _write_manifest(tmp_path), variant="ref2va")


def test_generated_metadata_matches_trainer_contract_from_other_cwd(
    tmp_path, monkeypatch
):
    source_dir = tmp_path / "source files"
    source_dir.mkdir()
    manifest = _write_manifest(source_dir)
    row = json.loads(manifest.read_text(encoding="utf-8"))
    row["target_video"] = "target.mp4"
    row["caption"] = "A quiet garden, 安静的花园."
    manifest.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
    source_bytes = manifest.read_bytes()
    plan = H3TrainingPlan.from_config(_config(source_dir, manifest))
    other_cwd = tmp_path / "different working directory"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)
    output = tmp_path / "reviewable plans" / "plan with spaces.json"
    write_h3_plan(plan, output)

    artifact = json.loads(output.read_text(encoding="utf-8"))
    command = artifact["cache_command"]
    metadata = Path(command[command.index("--metadata") + 1])
    assert metadata.parent == output.parent
    assert metadata != manifest
    generated_row = json.loads(metadata.read_text(encoding="utf-8"))
    assert metadata.read_bytes().isascii()
    # Exact access pattern from the reviewed external prepare_cache.py.
    video = Path(generated_row.get("target_video", generated_row["video"]))
    assert video.is_absolute()
    assert video.read_bytes() == b"video"
    assert generated_row["video"] == generated_row["target_video"]
    assert generated_row["caption"] == row["caption"]
    assert manifest.read_bytes() == source_bytes
    assert (
        artifact["source_manifest_sha256"] == hashlib.sha256(source_bytes).hexdigest()
    )
    assert (
        artifact["metadata_sha256"] == hashlib.sha256(metadata.read_bytes()).hexdigest()
    )


def test_manifest_expands_home_directory(tmp_path, monkeypatch):
    home = tmp_path / "home with spaces"
    home.mkdir()
    video = home / "target.mp4"
    video.write_bytes(b"video")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    manifest = _write_manifest(tmp_path)
    row = json.loads(manifest.read_text(encoding="utf-8"))
    row["target_video"] = "~/target.mp4"
    manifest.write_text(json.dumps(row), encoding="utf-8")
    assert load_h3_manifest(manifest)[0].target_video == video.resolve()
    output = tmp_path / "plan.json"
    write_h3_plan(H3TrainingPlan.from_config(_config(tmp_path, manifest)), output)
    metadata = Path(json.loads(output.read_text(encoding="utf-8"))["metadata_path"])
    generated_row = json.loads(metadata.read_text(encoding="utf-8"))
    assert Path(generated_row["video"]) == video.resolve()


@pytest.mark.parametrize(
    "destination", ["train.jsonl", "target.mp4", "trainer/train.py"]
)
def test_plan_cannot_overwrite_source_files(tmp_path, destination):
    manifest = _write_manifest(tmp_path)
    plan = H3TrainingPlan.from_config(_config(tmp_path, manifest))
    output = tmp_path / destination
    previous = output.read_bytes()
    with pytest.raises(ValueError, match="must not overwrite input"):
        write_h3_plan(plan, output)
    assert output.read_bytes() == previous


def test_plan_cannot_overwrite_hard_link_to_input(tmp_path):
    manifest = _write_manifest(tmp_path)
    plan = H3TrainingPlan.from_config(_config(tmp_path, manifest))
    output = tmp_path / "plan.json"
    os.link(manifest, output)
    previous = manifest.read_bytes()
    with pytest.raises(ValueError, match="must not overwrite input"):
        write_h3_plan(plan, output)
    assert output.read_bytes() == previous == manifest.read_bytes()


def test_cli_cannot_overwrite_its_configuration(tmp_path, monkeypatch):
    manifest = _write_manifest(tmp_path)
    config = _config(tmp_path, manifest)
    config_path = tmp_path / "config.json"
    config_path.write_text(config.model_dump_json(), encoding="utf-8")
    previous = config_path.read_bytes()
    script = Path(__file__).resolve().parents[1] / "scripts" / "plan_h3_training.py"
    spec = importlib.util.spec_from_file_location("plan_h3_training_cli", script)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(
        sys,
        "argv",
        [str(script), "--config", str(config_path), "--output", str(config_path)],
    )
    with pytest.raises(ValueError, match="must not overwrite input"):
        cli.main()
    assert config_path.read_bytes() == previous


def test_generated_metadata_cannot_overwrite_protected_input(tmp_path):
    manifest = _write_manifest(tmp_path)
    plan = H3TrainingPlan.from_config(_config(tmp_path, manifest))
    output = tmp_path / "plan.json"
    write_h3_plan(plan, output)
    previous = output.read_bytes()
    metadata = Path(json.loads(previous)["metadata_path"])
    with pytest.raises(ValueError, match="must not overwrite input"):
        write_h3_plan(plan, output, protected_paths=(metadata,))
    assert output.read_bytes() == previous


def test_changed_source_requires_revalidation(tmp_path):
    manifest = _write_manifest(tmp_path)
    plan = H3TrainingPlan.from_config(_config(tmp_path, manifest))
    manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    output = tmp_path / "plan.json"
    with pytest.raises(ValueError, match="changed after validation"):
        write_h3_plan(plan, output)
    assert not output.exists()


def test_corrupted_existing_metadata_is_not_silently_replaced(tmp_path):
    manifest = _write_manifest(tmp_path)
    plan = H3TrainingPlan.from_config(_config(tmp_path, manifest))
    output = tmp_path / "plan.json"
    write_h3_plan(plan, output)
    previous = output.read_bytes()
    metadata = Path(json.loads(previous)["metadata_path"])
    metadata.write_bytes(b"unexpected contents")
    with pytest.raises(ValueError, match="unexpected contents"):
        write_h3_plan(plan, output)
    assert output.read_bytes() == previous
    assert metadata.read_bytes() == b"unexpected contents"


def test_failed_plan_replace_preserves_previous_plan_and_metadata(
    tmp_path, monkeypatch
):
    manifest = _write_manifest(tmp_path)
    config = _config(tmp_path, manifest)
    output = tmp_path / "plan.json"
    write_h3_plan(H3TrainingPlan.from_config(config), output)
    previous = output.read_bytes()
    old_metadata = Path(json.loads(previous)["metadata_path"])
    old_metadata_bytes = old_metadata.read_bytes()
    row = json.loads(manifest.read_text(encoding="utf-8"))
    row["caption"] = "A different garden scene."
    manifest.write_text(json.dumps(row), encoding="utf-8")
    plan = H3TrainingPlan.from_config(config)
    replace = os.replace

    def fail_plan_replace(source, destination):
        if destination == output:
            raise OSError("simulated replace failure")
        replace(source, destination)

    monkeypatch.setattr(h3_training.os, "replace", fail_plan_replace)
    with pytest.raises(OSError, match="simulated"):
        write_h3_plan(plan, output)
    assert output.read_bytes() == previous
    assert old_metadata.read_bytes() == old_metadata_bytes
    assert not list(tmp_path.glob("*.tmp"))


def test_failed_metadata_write_preserves_previous_plan(tmp_path, monkeypatch):
    manifest = _write_manifest(tmp_path)
    plan = H3TrainingPlan.from_config(_config(tmp_path, manifest))
    output = tmp_path / "plan.json"
    output.write_bytes(b"previous plan")

    def fail_replace(source, destination):
        raise OSError("simulated metadata failure")

    monkeypatch.setattr(h3_training.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        write_h3_plan(plan, output)
    assert output.read_bytes() == b"previous plan"
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob("*.manifest.*.jsonl"))


def test_record_local_trainer_commit_without_claiming_validation(tmp_path, monkeypatch):
    manifest = _write_manifest(tmp_path)
    config = _config(tmp_path, manifest)
    (config.trainer_path / ".git").mkdir()
    commit = "a" * 40
    monkeypatch.setattr(
        h3_training.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, commit + "\n"),
    )
    output = tmp_path / "plan.json"
    write_h3_plan(H3TrainingPlan.from_config(config), output)
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["trainer"]["local_commit"] == commit
    assert artifact["trainer"]["compatibility"] == "unvalidated"
