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
from typing import Annotated, Any, Iterable, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)


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
_SUBJECT_TAG_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")


def normalize_prompt(text: str) -> str:
    """Normalize a prompt for duplicate checks and stable fingerprints."""
    return " ".join(text.casefold().split())


class VideoPromptPair(BaseModel):
    """One harmful prompt and its presentation-matched benign counterfactual."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    pair_id: str = Field(pattern=r"^video_pair_\d{6}$")
    source_target_id: int = Field(ge=1, le=999999, strict=True)
    source_dataset: str = Field(min_length=1)
    source_revision: str = Field(min_length=7)
    category: str
    language: str
    shot_type: str = Field(min_length=2)
    visual_style: str
    style_family: str
    source_target_prompt: Annotated[str, StringConstraints(strip_whitespace=False)] = (
        Field(min_length=20, max_length=700)
    )
    target_prompt: Annotated[str, StringConstraints(strip_whitespace=False)] = Field(
        min_length=20, max_length=1200
    )
    benign_prompt: str = Field(min_length=20, max_length=700)
    target_subject_tag: str
    benign_subject_tag: str
    target_expansion_summary: str | None = Field(
        default=None, min_length=5, max_length=500
    )
    transformation_summary: str = Field(min_length=5, max_length=500)
    preserved_elements: list[str] = Field(min_length=2, max_length=8)
    generator_model: str = Field(min_length=3)
    schema_version: Literal[2, 3] = 3
    source_split: str | None = Field(default=None, min_length=1)
    generation_policy: Literal["preserve_source_benign_counterfactual_v1"] | None = None
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

    @field_validator("benign_prompt")
    @classmethod
    def reject_chat_framing(cls, value: str) -> str:
        if _CHAT_FRAMING_RE.search(value):
            raise ValueError("prompt uses chat-style generation framing")
        if _PROVIDER_REFUSAL_RE.search(value):
            raise ValueError("prompt contains provider-refusal boilerplate")
        return value

    @model_validator(mode="after")
    def validate_pair(self) -> "VideoPromptPair":
        if self.pair_id != f"video_pair_{self.source_target_id:06d}":
            raise ValueError("pair_id does not match source_target_id")
        expected_style = STYLE_FAMILY_BY_VISUAL_STYLE[self.visual_style]
        if self.style_family != expected_style:
            raise ValueError(
                f"style_family must be {expected_style!r} for {self.visual_style!r}"
            )
        if self.schema_version == 2:
            # Historical imports keep their original schema and fingerprint.
            if self.generation_policy is not None or self.source_split is not None:
                raise ValueError("v2 legacy pairs cannot claim v3 provenance fields")
            self.reject_chat_framing(self.target_prompt)
            if not self.target_expansion_summary:
                raise ValueError("v2 legacy pairs require target_expansion_summary")
            if len(self.target_prompt) <= len(self.source_target_prompt):
                raise ValueError(
                    "legacy amplified target_prompt must be longer than source_target_prompt"
                )
            if normalize_prompt(self.target_prompt) == normalize_prompt(
                self.source_target_prompt
            ):
                raise ValueError(
                    "legacy amplified target_prompt is identical to its source"
                )
        else:
            if self.target_prompt != self.source_target_prompt:
                raise ValueError("v3 target_prompt must preserve the source verbatim")
            if not self.source_split or not self.generation_policy:
                raise ValueError("v3 pairs require source_split and generation_policy")
            if self.target_expansion_summary is not None:
                raise ValueError("v3 pairs do not expand the source target")
        if normalize_prompt(self.target_prompt) == normalize_prompt(self.benign_prompt):
            raise ValueError("target and benign prompts are identical")
        expected_fingerprint = pair_fingerprint(self)
        if self.fingerprint is None:
            self.fingerprint = expected_fingerprint
        elif self.fingerprint != expected_fingerprint:
            raise ValueError("fingerprint does not match pair contents")
        return self


def pair_fingerprint(pair: VideoPromptPair) -> str:
    """Hash all v3 fields exactly; preserve the historical v2 digest contract."""
    if pair.schema_version == 3:
        encoded = json.dumps(
            pair.model_dump(mode="json", exclude={"fingerprint"}),
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
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
        "source_target_prompt": normalize_prompt(pair.source_target_prompt),
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
    source_split: str,
) -> VideoPromptPair:
    """Preserve the supplied source and attach only a generated benign caption."""
    if "target_prompt" in generated or "target_expansion_summary" in generated:
        raise ValueError("provider must not rewrite or expand the source target")
    expected_fields = {
        "source_id",
        "benign_prompt",
        "benign_subject_tag",
        "transformation_summary",
        "preserved_elements",
    }
    if set(generated) != expected_fields:
        raise ValueError("provider returned unexpected caption fields")
    if (
        type(generated["source_id"]) is not int
        or generated["source_id"] != target["id"]
    ):
        raise ValueError("provider source_id does not match source")
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
        source_target_prompt=target["prompt"],
        target_prompt=target["prompt"],
        benign_prompt=generated["benign_prompt"],
        target_subject_tag=target["subject_tag"],
        benign_subject_tag=generated["benign_subject_tag"],
        transformation_summary=generated["transformation_summary"],
        preserved_elements=generated["preserved_elements"],
        generator_model=generator_model,
        source_split=source_split,
        generation_policy="preserve_source_benign_counterfactual_v1",
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
        source_id = row["id"]
        if type(source_id) is not int or not 1 <= source_id <= 999999:
            raise ValueError(f"row {index} id must be an integer between 1 and 999999")
        if source_id in seen:
            raise ValueError(f"duplicate source id: {source_id}")
        seen.add(source_id)
        for field in required - {"id"}:
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError(f"row {source_id} {field} must be a nonempty string")
        if not 20 <= len(row["prompt"].strip()) <= 700:
            raise ValueError(
                f"row {source_id} prompt must contain 20 to 700 characters"
            )
        if len(row["shot_type"].strip()) < 2:
            raise ValueError(
                f"row {source_id} shot_type must contain at least 2 characters"
            )
        if not _SUBJECT_TAG_RE.fullmatch(row["subject_tag"]):
            raise ValueError(
                f"row {source_id} subject_tag must use lowercase snake_case"
            )
        if row["category"] not in VIDEO_CATEGORIES:
            raise ValueError(f"row {source_id} has an unknown category")
        if row["language"] not in LANGUAGES:
            raise ValueError(f"row {source_id} has an unknown language")
        if row["visual_style"] not in STYLE_FAMILY_BY_VISUAL_STYLE:
            raise ValueError(f"row {source_id} has an unknown visual style")
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
    source_lengths = [len(pair.source_target_prompt) for pair in materialized]
    target_lengths = [len(pair.target_prompt) for pair in materialized]
    growth_ratios = [
        target / source for source, target in zip(source_lengths, target_lengths)
    ]
    summary = {
        "schema_version": 3,
        "pair_schema_versions": dict(
            sorted(Counter(p.schema_version for p in materialized).items())
        ),
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
        "source_preservation": {
            "verbatim_target_count": sum(
                p.target_prompt == p.source_target_prompt for p in materialized
            ),
            "mean_source_chars": (
                round(sum(source_lengths) / len(source_lengths), 6)
                if source_lengths
                else 0.0
            ),
            "mean_target_chars": (
                round(sum(target_lengths) / len(target_lengths), 6)
                if target_lengths
                else 0.0
            ),
            "mean_growth_ratio": (
                round(sum(growth_ratios) / len(growth_ratios), 6)
                if growth_ratios
                else 0.0
            ),
            "min_growth_ratio": (
                round(min(growth_ratios), 6) if growth_ratios else 0.0
            ),
            "max_growth_ratio": (
                round(max(growth_ratios), 6) if growth_ratios else 0.0
            ),
        },
        "fingerprints_verified": all(
            pair.fingerprint == pair_fingerprint(pair) for pair in materialized
        ),
    }
    return summary


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
