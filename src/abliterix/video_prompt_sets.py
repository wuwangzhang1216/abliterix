"""Validated paired prompt sets for video-model steering research.

The legacy video datasets contain independently sampled benign and harmful
prompts.  That is useful for broad coverage, but it leaves category, camera,
style, and scene content as confounders when estimating a safety direction.
This module defines the stricter representation used by the H3 workflow: one
harmful target and one benign counterfactual with shared presentation metadata.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


VIDEO_CATEGORIES = frozenset(
    {
        "Graphic Violence",
        "Weapons",
        "Self-Harm",
        "Illegal Drugs",
        "CBRNE",
        "Hate Speech",
        "Radicalization",
        "Harassment",
        "Sabotage",
        "Property Destruction",
        "Animal Cruelty",
        "Dangerous Stunts",
        "Disinformation & Deepfake",
        "Nudity & Sexual Content",
    }
)

LANGUAGES = frozenset({"en", "zh", "mixed"})

STYLE_FAMILY_BY_VISUAL_STYLE = {
    "cinematic film look": "cinematic_film",
    "documentary realism": "documentary",
    "anime/animated style": "anime_animated",
    "hyperrealistic photographic": "hyperrealistic",
    "found-footage/amateur phone video": "found_footage",
    "slow-motion cinematic": "slow_motion",
    "retro VHS footage": "retro_vhs",
}

_CHAT_FRAMING_RE = re.compile(
    r"(?i)^\s*(?:please\s+|generate\s+|create\s+|make\s+|can you\s+|"
    r"i want\s+|请(?:生成|创建|制作)|生成(?:一个|一段|视频))"
)
_PROVIDER_REFUSAL_RE = re.compile(
    r"(?i)(?:\bi (?:cannot|can't|won't|am unable to) (?:help|create|generate|"
    r"provide|assist)\b|\bsorry,? but i\b|抱歉[，, ]*(?:我)?(?:无法|不能)(?:帮助|生成|"
    r"创建|提供)|我(?:无法|不能)(?:帮助|生成|创建|提供))"
)
_MINOR_RE = re.compile(
    r"(?i)\b(?:child|children|kid|minor|teen|teenager|boy|girl|schoolchild)\b|"
    r"未成年|儿童|孩子|男孩|女孩|少年|少女|青少年"
)
_SUBJECT_TAG_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")


def normalize_prompt(text: str) -> str:
    """Normalize a prompt for duplicate checks and stable fingerprints."""
    return " ".join(text.casefold().split())


class VideoPromptPair(BaseModel):
    """One harmful prompt and its presentation-matched benign counterfactual."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    pair_id: str = Field(pattern=r"^video_pair_\d{6}$")
    source_target_id: int = Field(ge=1)
    source_dataset: str = Field(min_length=1)
    source_revision: str = Field(min_length=7)
    category: str
    language: str
    shot_type: str = Field(min_length=2)
    visual_style: str
    style_family: str
    target_prompt: str = Field(min_length=20, max_length=700)
    benign_prompt: str = Field(min_length=20, max_length=700)
    target_subject_tag: str
    benign_subject_tag: str
    transformation_summary: str = Field(min_length=5, max_length=500)
    preserved_elements: list[str] = Field(min_length=2, max_length=8)
    generator_model: str = Field(min_length=3)
    schema_version: int = 1
    fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @field_validator("category")
    @classmethod
    def validate_category(cls, value: str) -> str:
        if value not in VIDEO_CATEGORIES:
            raise ValueError(f"unknown video safety category: {value!r}")
        return value

    @field_validator("language")
    @classmethod
    def validate_language(cls, value: str) -> str:
        if value not in LANGUAGES:
            raise ValueError(f"unknown language label: {value!r}")
        return value

    @field_validator("visual_style")
    @classmethod
    def validate_visual_style(cls, value: str) -> str:
        if value not in STYLE_FAMILY_BY_VISUAL_STYLE:
            raise ValueError(f"unknown visual style: {value!r}")
        return value

    @field_validator("target_subject_tag", "benign_subject_tag")
    @classmethod
    def validate_subject_tag(cls, value: str) -> str:
        if not _SUBJECT_TAG_RE.fullmatch(value):
            raise ValueError("subject tags must be lowercase snake_case")
        return value

    @field_validator("target_prompt", "benign_prompt")
    @classmethod
    def reject_chat_framing(cls, value: str) -> str:
        if _CHAT_FRAMING_RE.search(value):
            raise ValueError("prompt uses chat-style generation framing")
        if _PROVIDER_REFUSAL_RE.search(value):
            raise ValueError("prompt contains provider-refusal boilerplate")
        return value

    @model_validator(mode="after")
    def validate_pair(self) -> "VideoPromptPair":
        expected_style = STYLE_FAMILY_BY_VISUAL_STYLE[self.visual_style]
        if self.style_family != expected_style:
            raise ValueError(
                f"style_family must be {expected_style!r} for {self.visual_style!r}"
            )
        if normalize_prompt(self.target_prompt) == normalize_prompt(self.benign_prompt):
            raise ValueError("target and benign prompts are identical")
        if _MINOR_RE.search(self.target_prompt) or _MINOR_RE.search(self.benign_prompt):
            raise ValueError("paired safety prompts must not depict or mention minors")

        expected_fingerprint = pair_fingerprint(self)
        if self.fingerprint is None:
            self.fingerprint = expected_fingerprint
        elif self.fingerprint != expected_fingerprint:
            raise ValueError("fingerprint does not match pair contents")
        return self


def pair_fingerprint(pair: VideoPromptPair) -> str:
    """Hash semantic pair contents, excluding provenance and the hash itself."""
    payload = {
        "pair_id": pair.pair_id,
        "source_target_id": pair.source_target_id,
        "source_dataset": pair.source_dataset,
        "source_revision": pair.source_revision,
        "category": pair.category,
        "language": pair.language,
        "shot_type": pair.shot_type,
        "visual_style": pair.visual_style,
        "style_family": pair.style_family,
        "target_prompt": normalize_prompt(pair.target_prompt),
        "benign_prompt": normalize_prompt(pair.benign_prompt),
        "target_subject_tag": pair.target_subject_tag,
        "benign_subject_tag": pair.benign_subject_tag,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_video_prompt_pair(
    target: dict[str, Any],
    generated: dict[str, Any],
    *,
    generator_model: str,
    source_dataset: str,
    source_revision: str,
) -> VideoPromptPair:
    """Combine one legacy harmful row with a generated benign counterpart."""
    source_id = int(target["id"])
    return VideoPromptPair(
        pair_id=f"video_pair_{source_id:06d}",
        source_target_id=source_id,
        source_dataset=source_dataset,
        source_revision=source_revision,
        category=target["category"],
        language=target["language"],
        shot_type=target["shot_type"],
        visual_style=target["visual_style"],
        style_family=STYLE_FAMILY_BY_VISUAL_STYLE[target["visual_style"]],
        target_prompt=target["prompt"],
        benign_prompt=generated["benign_prompt"],
        target_subject_tag=target["subject_tag"],
        benign_subject_tag=generated["benign_subject_tag"],
        transformation_summary=generated["transformation_summary"],
        preserved_elements=generated["preserved_elements"],
        generator_model=generator_model,
    )


def load_legacy_video_prompts(path: str | Path) -> list[dict[str, Any]]:
    """Load and minimally validate the existing JSON-array video prompt format."""
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("legacy video prompt file must contain a JSON array")
    required = {
        "id",
        "prompt",
        "category",
        "language",
        "shot_type",
        "visual_style",
        "subject_tag",
    }
    seen: set[int] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"row {index} is not an object")
        missing = required - row.keys()
        if missing:
            raise ValueError(f"row {index} is missing {sorted(missing)}")
        source_id = int(row["id"])
        if source_id in seen:
            raise ValueError(f"duplicate source id: {source_id}")
        seen.add(source_id)
        if row["category"] not in VIDEO_CATEGORIES:
            raise ValueError(f"row {source_id} has an unknown category")
        if row["language"] not in LANGUAGES:
            raise ValueError(f"row {source_id} has an unknown language")
        if row["visual_style"] not in STYLE_FAMILY_BY_VISUAL_STYLE:
            raise ValueError(f"row {source_id} has an unknown visual style")
        if _MINOR_RE.search(str(row["prompt"])):
            raise ValueError(f"row {source_id} mentions a minor")
    return sorted(rows, key=lambda row: int(row["id"]))


def load_video_prompt_pairs(path: str | Path) -> list[VideoPromptPair]:
    """Load paired JSONL and validate cross-row uniqueness."""
    pairs: list[VideoPromptPair] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            pairs.append(VideoPromptPair.model_validate_json(line))
        except Exception as exc:
            raise ValueError(f"invalid pair on line {line_number}: {exc}") from exc
    validate_pair_collection(pairs)
    return pairs


def validate_pair_collection(pairs: Iterable[VideoPromptPair]) -> None:
    """Validate IDs, prompt uniqueness, and category coverage across a set."""
    materialized = list(pairs)
    pair_ids = [pair.pair_id for pair in materialized]
    source_ids = [pair.source_target_id for pair in materialized]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("duplicate pair_id values")
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("duplicate source_target_id values")

    target_prompts = [normalize_prompt(pair.target_prompt) for pair in materialized]
    benign_prompts = [normalize_prompt(pair.benign_prompt) for pair in materialized]
    if len(target_prompts) != len(set(target_prompts)):
        raise ValueError("duplicate target prompts")
    if len(benign_prompts) != len(set(benign_prompts)):
        raise ValueError("duplicate benign prompts")


def summarize_video_prompt_pairs(
    pairs: Iterable[VideoPromptPair],
) -> dict[str, Any]:
    """Return a deterministic audit summary suitable for JSON output."""
    materialized = list(pairs)
    return {
        "schema_version": 1,
        "pair_count": len(materialized),
        "categories": dict(sorted(Counter(p.category for p in materialized).items())),
        "languages": dict(sorted(Counter(p.language for p in materialized).items())),
        "shot_types": dict(sorted(Counter(p.shot_type for p in materialized).items())),
        "visual_styles": dict(
            sorted(Counter(p.visual_style for p in materialized).items())
        ),
        "generator_models": dict(
            sorted(Counter(p.generator_model for p in materialized).items())
        ),
        "fingerprints_verified": all(
            pair.fingerprint == pair_fingerprint(pair) for pair in materialized
        ),
    }


def write_video_prompt_pairs(
    pairs: Iterable[VideoPromptPair], path: str | Path
) -> None:
    """Atomically write validated pairs as JSONL."""
    materialized = list(pairs)
    validate_pair_collection(materialized)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for pair in materialized:
            handle.write(pair.model_dump_json() + "\n")
    temporary.replace(output)
