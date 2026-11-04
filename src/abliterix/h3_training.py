"""Validation and execution planning for external MiniMax-H3 trainers.

Abliterix's LLM engine cannot execute H3's packed rectified-flow objective.
This module deliberately keeps that boundary explicit: it validates media
manifests and emits argument-vector execution plans for an external H3 trainer.
It does not execute shell strings or silently reinterpret prompt-only data as
supervised video training examples.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


REVIEWED_TRAINER_COMMIT = "74d535b2cae5cd56f9565059e6f2b6c9ddb39df4"


class H3ManifestEntry(BaseModel):
    """One supervised H3 audio/video training sample."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(min_length=1)
    task: Literal["fl2va", "ref2va"] = "fl2va"
    caption: str = Field(min_length=1)
    target_video: Path
    target_audio: Path | None = None
    reference_images: list[Path] = Field(default_factory=list)
    reference_videos: list[Path] = Field(default_factory=list)
    reference_audios: list[Path] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_supported_media(self) -> "H3ManifestEntry":
        references = (
            self.reference_images + self.reference_videos + self.reference_audios
        )
        if references:
            raise ValueError("reference media are not supported by the cache adapter")
        if self.task != "fl2va":
            raise ValueError("the cache adapter currently supports only task='fl2va'")
        if self.target_audio is not None:
            raise ValueError(
                "external target_audio is not supported; audio is read from target_video"
            )
        return self

    def missing_paths(self) -> list[Path]:
        paths = [self.target_video]
        if self.target_audio is not None:
            paths.append(self.target_audio)
        paths.extend(self.reference_images)
        paths.extend(self.reference_videos)
        paths.extend(self.reference_audios)
        return [path for path in paths if not path.is_file()]


class H3TrainingConfig(BaseModel):
    """Inputs for source-reviewed, runtime-unvalidated external training plans."""

    model_config = ConfigDict(extra="forbid")

    model_path: Path
    trainer_path: Path
    manifest_path: Path
    cache_path: Path
    output_path: Path
    variant: Literal["fl2va", "ref2va"] = "fl2va"
    trainable: Literal["heads", "lora", "all"] = "lora"
    strategy: Literal["ddp", "deepspeed"] = "deepspeed"
    height: int = Field(default=448, ge=32)
    width: int = Field(default=768, ge=32)
    frames: int = Field(default=430, ge=5)
    max_steps: int = Field(default=800, ge=1)
    encode_audio: bool = True
    num_gpus: int = Field(default=8, ge=1)

    @field_validator(
        "model_path", "trainer_path", "manifest_path", "cache_path", "output_path"
    )
    @classmethod
    def normalize_path(cls, value: Path) -> Path:
        return value.expanduser().resolve()

    @field_validator("height", "width")
    @classmethod
    def validate_spatial_multiple(cls, value: int) -> int:
        if value % 32:
            raise ValueError("H3 height and width must be divisible by 32")
        return value

    @field_validator("frames")
    @classmethod
    def validate_frame_count(cls, value: int) -> int:
        if value % 17 != 5:
            raise ValueError("H3 frame count must satisfy frames % 17 == 5")
        return value

    @model_validator(mode="after")
    def validate_training_mode(self) -> "H3TrainingConfig":
        if self.variant != "fl2va":
            raise ValueError(
                "the cache adapter currently supports only variant='fl2va'"
            )
        if self.trainable == "all" and self.strategy != "deepspeed":
            raise ValueError("full H3 training requires the deepspeed strategy")
        if self.strategy == "ddp" and self.num_gpus != 1:
            raise ValueError(
                "the current H3 trainer uses ddp only for one-GPU smoke tests"
            )
        return self


class H3TrainingPlan(BaseModel):
    """Validated, shell-free command plan for an H3 training run."""

    config: H3TrainingConfig
    samples: list[H3ManifestEntry]
    source_manifest_sha256: str

    @classmethod
    def from_config(cls, config: H3TrainingConfig) -> "H3TrainingPlan":
        manifest_bytes = config.manifest_path.read_bytes()
        samples = _parse_h3_manifest(
            config.manifest_path, manifest_bytes.decode("utf-8")
        )
        missing: list[Path] = []
        for sample in samples:
            missing.extend(sample.missing_paths())
            if sample.task != config.variant:
                raise ValueError(
                    f"sample {sample.id!r} uses task={sample.task!r}, "
                    f"but training variant is {config.variant!r}"
                )
        if missing:
            preview = ", ".join(str(path) for path in missing[:5])
            raise FileNotFoundError(
                f"training manifest references missing media: {preview}"
            )
        if not config.model_path.is_dir():
            raise FileNotFoundError(
                f"H3 model directory not found: {config.model_path}"
            )
        for required in ("prepare_cache.py", "train.py"):
            if not (config.trainer_path / required).is_file():
                raise FileNotFoundError(
                    f"H3 trainer is missing {required}: {config.trainer_path}"
                )
        return cls(
            config=config,
            samples=samples,
            source_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        )

    def cache_command(self, metadata_path: Path) -> list[str]:
        cfg = self.config
        command = [
            "python",
            str(cfg.trainer_path / "prepare_cache.py"),
            "--metadata",
            str(metadata_path),
            "--output",
            str(cfg.cache_path),
            "--model",
            str(cfg.model_path),
            "--height",
            str(cfg.height),
            "--width",
            str(cfg.width),
            "--frames",
            str(cfg.frames),
            "--encode-text",
        ]
        if cfg.encode_audio:
            command.append("--encode-audio")
        return command

    def train_command(self) -> list[str]:
        cfg = self.config
        if cfg.strategy == "deepspeed":
            prefix = ["deepspeed", "--num_gpus", str(cfg.num_gpus)]
        else:
            prefix = ["python"]
        return prefix + [
            str(cfg.trainer_path / "train.py"),
            "--model",
            str(cfg.model_path),
            "--variant",
            cfg.variant,
            "--cache",
            str(cfg.cache_path),
            "--output",
            str(cfg.output_path),
            "--max-steps",
            str(cfg.max_steps),
            "--trainable",
            cfg.trainable,
            "--strategy",
            cfg.strategy,
        ]


def load_h3_manifest(path: str | Path) -> list[H3ManifestEntry]:
    """Load an H3 JSONL manifest and reject duplicate sample IDs."""
    manifest = Path(path).expanduser().resolve()
    return _parse_h3_manifest(manifest, manifest.read_text(encoding="utf-8"))


def _parse_h3_manifest(manifest: Path, contents: str) -> list[H3ManifestEntry]:
    samples: list[H3ManifestEntry] = []
    for line_number, line in enumerate(contents.splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            for field in ("target_video", "target_audio"):
                value = payload.get(field)
                if value:
                    media = Path(value).expanduser()
                    payload[field] = str((manifest.parent / media).resolve())
            for field in (
                "reference_images",
                "reference_videos",
                "reference_audios",
            ):
                payload[field] = [
                    str((manifest.parent / Path(value).expanduser()).resolve())
                    if not Path(value).expanduser().is_absolute()
                    else str(Path(value).expanduser())
                    for value in payload.get(field, [])
                ]
            samples.append(H3ManifestEntry.model_validate(payload))
        except Exception as exc:
            raise ValueError(f"invalid H3 manifest line {line_number}: {exc}") from exc
    if not samples:
        raise ValueError("H3 training manifest is empty")
    ids = [sample.id for sample in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("H3 training manifest contains duplicate sample IDs")
    return samples


def _atomic_write(path: Path, contents: bytes) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_h3_plan(
    plan: H3TrainingPlan,
    path: str | Path,
    *,
    protected_paths: tuple[Path, ...] = (),
) -> None:
    """Publish a plan and normalized metadata without executing external training."""
    output = Path(path).expanduser().resolve()
    if (
        hashlib.sha256(plan.config.manifest_path.read_bytes()).hexdigest()
        != plan.source_manifest_sha256
    ):
        raise ValueError("source manifest changed after validation; rebuild the plan")
    rows = []
    for sample in plan.samples:
        row = sample.model_dump(mode="json")
        # The reviewed trainer eagerly evaluates row["video"] in dict.get's default.
        row["video"] = row["target_video"]
        # The external reader uses the locale's default encoding.
        rows.append(json.dumps(row, ensure_ascii=True, sort_keys=True))
    metadata_bytes = ("\n".join(rows) + "\n").encode("utf-8")
    metadata_sha256 = hashlib.sha256(metadata_bytes).hexdigest()
    metadata_path = output.with_name(f"{output.stem}.manifest.{metadata_sha256}.jsonl")
    trainer = plan.config.trainer_path
    trainer_scripts = [trainer / "prepare_cache.py", trainer / "train.py"]
    inputs = [
        plan.config.manifest_path,
        *trainer_scripts,
        *(sample.target_video for sample in plan.samples),
        *protected_paths,
    ]
    for destination in (output, metadata_path):
        for source in inputs:
            source = source.expanduser().resolve()
            if destination == source or (
                destination.exists()
                and source.exists()
                and destination.samefile(source)
            ):
                raise ValueError(
                    f"plan outputs must not overwrite input files: {source}"
                )
    local_commit = None
    if (trainer / ".git").exists():
        try:
            result = subprocess.run(
                ["git", "-C", str(trainer), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            candidate = result.stdout.strip()
            if len(candidate) in (40, 64) and all(
                char in "0123456789abcdef" for char in candidate
            ):
                local_commit = candidate
        except (OSError, subprocess.SubprocessError):
            pass
    artifact = {
        "schema_version": 2,
        "sample_count": len(plan.samples),
        "config": plan.config.model_dump(mode="json"),
        "source_manifest_sha256": plan.source_manifest_sha256,
        "metadata_path": str(metadata_path),
        "metadata_sha256": metadata_sha256,
        "trainer": {
            "reviewed_upstream_commit": REVIEWED_TRAINER_COMMIT,
            "local_commit": local_commit,
            "compatibility": "unvalidated",
            "runtime_validation": "not_run",
            "script_sha256": {
                script.name: hashlib.sha256(script.read_bytes()).hexdigest()
                for script in trainer_scripts
            },
        },
        "cache_command": plan.cache_command(metadata_path),
        "train_command": plan.train_command(),
    }
    if metadata_path.exists() and metadata_path.read_bytes() != metadata_bytes:
        raise ValueError(f"generated metadata has unexpected contents: {metadata_path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Content-addressed metadata leaves any previous plan's inputs intact on failure.
    if not metadata_path.exists():
        _atomic_write(metadata_path, metadata_bytes)
    _atomic_write(
        output,
        (json.dumps(artifact, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
