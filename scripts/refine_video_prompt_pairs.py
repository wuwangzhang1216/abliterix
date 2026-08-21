#!/usr/bin/env python3
"""Build presentation-matched benign/target video Prompt Sets with Gemini.

The script consumes the legacy harmful JSON array and asks an OpenRouter model
to rewrite each scene into a benign counterfactual.  Camera, visual style,
language, setting, and non-harmful objects are retained while the unsafe action
is replaced.  Results are validated and appended to a resumable JSONL file.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from abliterix.video_prompt_sets import (
    VideoPromptPair,
    build_video_prompt_pair,
    load_legacy_video_prompts,
    load_video_prompt_pairs,
    summarize_video_prompt_pairs,
    write_video_prompt_pairs,
)


DEFAULT_MODEL = "google/gemini-3.7-flash"

SYSTEM_PROMPT = """\
You build paired Prompt Sets for video-model safety research. For every supplied
unsafe target prompt, write exactly one benign counterfactual scene.

The benign scene must:
- preserve the target's language mode exactly: en, zh, or naturally mixed;
- preserve camera/shot type, visual style, setting, atmosphere, and harmless
  subjects or objects wherever possible;
- replace the unsafe action or intent with a clearly legitimate, safe,
  visually concrete action in the same topical neighborhood;
- contain no violence, injury, weapons use, drug use, self-harm, nudity,
  sexual content, hate, harassment, criminal instruction, sabotage, deception,
  real identifiable people, or minors;
- read as a raw text-to-video scene description, never as a request, refusal,
  policy explanation, warning, or markdown;
- add no operational detail to the unsafe source.

Return one output object for every source_id and preserve source order.
`preserved_elements` should name 2-8 concrete presentation elements that are
actually shared (for example setting, camera motion, lighting, visual style).
`transformation_summary` should briefly state the safe semantic replacement,
without repeating harmful procedural detail.
"""

RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "video_benign_counterfactuals",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "pairs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source_id": {"type": "integer"},
                            "benign_prompt": {"type": "string"},
                            "benign_subject_tag": {
                                "type": "string",
                                "pattern": "^[a-z0-9]+(?:_[a-z0-9]+)*$",
                            },
                            "transformation_summary": {"type": "string"},
                            "preserved_elements": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 2,
                                "maxItems": 8,
                            },
                        },
                        "required": [
                            "source_id",
                            "benign_prompt",
                            "benign_subject_tag",
                            "transformation_summary",
                            "preserved_elements",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["pairs"],
            "additionalProperties": False,
        },
    },
}


class RateLimiter:
    """Pace request starts across concurrent workers."""

    def __init__(self, requests_per_minute: int) -> None:
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        self._interval = 60.0 / requests_per_minute
        self._lock = asyncio.Lock()
        self._next_start = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            if self._next_start > now:
                await asyncio.sleep(self._next_start - now)
                now = loop.time()
            self._next_start = max(now, self._next_start) + self._interval


def _chunks(rows: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [rows[index : index + size] for index in range(0, len(rows), size)]


def _request_payload(batch: list[dict[str, Any]]) -> str:
    sources = [
        {
            "source_id": int(row["id"]),
            "category": row["category"],
            "language": row["language"],
            "shot_type": row["shot_type"],
            "visual_style": row["visual_style"],
            "target_prompt": row["prompt"],
        }
        for row in batch
    ]
    return json.dumps({"sources": sources}, ensure_ascii=False)


async def _generate_batch(
    client: AsyncOpenAI,
    model: str,
    source_dataset: str,
    source_revision: str,
    batch: list[dict[str, Any]],
    limiter: RateLimiter,
    semaphore: asyncio.Semaphore,
    max_retries: int,
) -> list[VideoPromptPair]:
    expected_ids = [int(row["id"]) for row in batch]
    row_by_id = {int(row["id"]): row for row in batch}

    for attempt in range(max_retries):
        try:
            async with semaphore:
                await limiter.acquire()
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": _request_payload(batch)},
                    ],
                    response_format=RESPONSE_SCHEMA,
                    temperature=0.35,
                    max_tokens=max(1800, 500 * len(batch)),
                    extra_body={"reasoning": {"effort": "minimal"}},
                )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("provider returned an empty response")
            payload = json.loads(content)
            generated = payload["pairs"]
            returned_ids = [int(item["source_id"]) for item in generated]
            if returned_ids != expected_ids:
                raise ValueError(
                    f"source IDs changed: expected {expected_ids}, got {returned_ids}"
                )
            return [
                build_video_prompt_pair(
                    row_by_id[int(item["source_id"])],
                    item,
                    generator_model=model,
                    source_dataset=source_dataset,
                    source_revision=source_revision,
                )
                for item in generated
            ]
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            if (
                isinstance(status_code, int)
                and 400 <= status_code < 500
                and status_code != 429
            ):
                raise RuntimeError(
                    f"non-retryable OpenRouter HTTP {status_code} for batch "
                    f"{expected_ids[0]}-{expected_ids[-1]}: {exc}"
                ) from exc
            if attempt + 1 == max_retries:
                raise RuntimeError(
                    f"failed batch {expected_ids[0]}-{expected_ids[-1]} "
                    f"after {max_retries} attempts: {exc}"
                ) from exc
            delay = min(45.0, 2 ** (attempt + 1) + random.random() * 2)
            print(
                f"Retrying batch {expected_ids[0]}-{expected_ids[-1]} "
                f"after {type(exc).__name__}: {exc}"
            )
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")


async def build_pairs(args: argparse.Namespace) -> list[VideoPromptPair]:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    targets = load_legacy_video_prompts(args.input)
    if args.limit is not None:
        targets = targets[: args.limit]

    progress_path = args.progress or args.output.with_suffix(".progress.jsonl")
    existing = load_video_prompt_pairs(progress_path) if progress_path.exists() else []
    existing_by_id = {pair.source_target_id: pair for pair in existing}
    pending = [row for row in targets if int(row["id"]) not in existing_by_id]

    print(
        f"Targets: {len(targets)} | existing: {len(existing_by_id)} | "
        f"pending: {len(pending)} | model: {args.model}"
    )
    if not pending:
        pairs = [existing_by_id[int(row["id"])] for row in targets]
        write_video_prompt_pairs(pairs, args.output)
        return pairs

    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    limiter = RateLimiter(args.rate_limit_rpm)
    semaphore = asyncio.Semaphore(args.workers)
    progress_lock = asyncio.Lock()
    completed = 0

    async def process(batch: list[dict[str, Any]]) -> None:
        nonlocal completed
        pairs = await _generate_batch(
            client,
            args.model,
            args.source_dataset,
            args.source_revision,
            batch,
            limiter,
            semaphore,
            args.max_retries,
        )
        async with progress_lock:
            for pair in pairs:
                existing_by_id[pair.source_target_id] = pair
            ordered = [existing_by_id[key] for key in sorted(existing_by_id)]
            write_video_prompt_pairs(ordered, progress_path)
            completed += len(pairs)
            print(f"Generated {completed}/{len(pending)} new pairs")

    await asyncio.gather(
        *(process(batch) for batch in _chunks(pending, args.batch_size))
    )

    pairs = [existing_by_id[int(row["id"])] for row in targets]
    write_video_prompt_pairs(pairs, args.output)
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--source-dataset", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rate-limit-rpm", type=int, default=30)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    for name in ("batch_size", "workers", "rate_limit_rpm", "max_retries"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


def main() -> None:
    args = parse_args()
    pairs = asyncio.run(build_pairs(args))
    summary = summarize_video_prompt_pairs(pairs)
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(pairs)} validated pairs to {args.output}")
    print(f"Wrote audit summary to {summary_path}")


if __name__ == "__main__":
    main()
