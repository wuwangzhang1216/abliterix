#!/usr/bin/env python3
"""Build source-preserving/benign video Prompt Sets with Gemini.

The script consumes the legacy harmful JSON array and asks an OpenRouter model
to write a presentation-matched benign counterfactual. The supplied source
target is retained verbatim. Results and terminal review outcomes are checkpointed atomically.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

from openai import APIConnectionError, AsyncOpenAI

from abliterix.video_prompt_sets import (
    VideoPromptPair,
    build_video_prompt_pair,
    load_legacy_video_prompts,
    summarize_video_prompt_pairs,
    validate_pair_collection,
    write_video_prompt_pairs,
)


DEFAULT_MODEL = "google/gemini-3.7-flash"

SYSTEM_PROMPT = """\
Write a benign counterfactual caption for each supplied source scene.
Preserve its language, camera/shot type, visual style, setting, and harmless
subjects or objects where possible, while replacing unsafe actions with
clearly legitimate, safe actions. Do not expand or rewrite the supplied target.
Respect provider safety controls; refuse any source you cannot process safely.
Do not disguise requests or add operational details to evade those controls.

Return one object per source_id in source order, containing benign_prompt,
benign_subject_tag, transformation_summary, and 2-8 preserved_elements.
The benign caption must be a self-contained scene description without requests,
policy text, or markdown. The summary should describe the safe replacement;
preserved_elements should name concrete presentation elements shared with the
source. Do not include a generated target caption.
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


class ProviderReviewRequired(ValueError):
    """A terminal provider outcome that must never trigger another request."""


def _request_policy(batch_size: int) -> dict[str, Any]:
    """Keep the fingerprinted policy and the actual request options together."""
    return {
        "response_format": RESPONSE_SCHEMA,
        "temperature": 0.35,
        "max_tokens": max(3200, 1400 * batch_size + 800),
        "extra_body": {"reasoning": {"effort": "minimal"}},
    }


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_identity(
    args: argparse.Namespace, all_targets: list[dict[str, Any]]
) -> dict[str, Any]:
    targets = all_targets[: args.limit] if args.limit is not None else all_targets
    return {
        "checkpoint_version": 1,
        "source_dataset": args.source_dataset,
        "source_revision": args.source_revision,
        "source_split": args.source_split,
        # Hash the actual local content, including metadata, not a supplied label.
        "source_sha256": _digest(all_targets),
        "selected_source_ids": [row["id"] for row in targets],
        "generator_model": args.model,
        "generation_policy": {
            "system_prompt": SYSTEM_PROMPT,
            "batch_size": args.batch_size,
            "request_options_by_size": {
                str(size): _request_policy(size)
                for size in {len(batch) for batch in _chunks(targets, args.batch_size)}
            },
            "payload_version": 1,
            "response_validation_version": 1,
            "max_transport_attempts": args.max_retries,
            "non_stop_response": "review_without_retry",
        },
    }


def _write_checkpoint(
    path: Path, identity: dict[str, Any], records: dict[int, dict[str, Any]]
) -> None:
    """Replace one complete snapshot so provenance and results cannot separate."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = [records[key] for key in sorted(records)]
    header = {
        "run": identity,
        "run_sha256": _digest(identity),
        "records_sha256": _digest(ordered),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in [header, *ordered]:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _load_checkpoint(
    path: Path, identity: dict[str, Any], targets: list[dict[str, Any]]
) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        lines = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        ]
    except (ValueError, UnicodeError) as exc:
        raise ValueError(
            "invalid or truncated checkpoint; preserve it for review"
        ) from exc
    if (
        not lines
        or not isinstance(lines[0], dict)
        or (
            _digest(lines[0].get("run")) != _digest(identity)
            or lines[0].get("run_sha256") != _digest(identity)
        )
    ):
        raise ValueError(
            "checkpoint provenance mismatch or legacy format; use a new progress path"
        )
    if set(lines[0]) != {"run", "run_sha256", "records_sha256"} or (
        lines[0]["records_sha256"] != _digest(lines[1:])
    ):
        raise ValueError("checkpoint record checksum mismatch; preserve it for review")
    by_id = {row["id"]: row for row in targets}
    records: dict[int, dict[str, Any]] = {}
    pairs = []
    for line_number, record in enumerate(lines[1:], 2):
        if not isinstance(record, dict):
            raise ValueError(f"invalid checkpoint record on line {line_number}")
        source_id = record.get("source_target_id")
        if type(source_id) is not int or source_id not in by_id or source_id in records:
            raise ValueError(
                f"unknown or duplicate checkpoint source ID on line {line_number}"
            )
        if record.get("status") == "complete" and set(record) == {
            "source_target_id",
            "status",
            "pair",
        }:
            raw_pair = record["pair"]
            if not isinstance(raw_pair, dict) or not raw_pair.get("fingerprint"):
                raise ValueError(f"missing pair fingerprint on line {line_number}")
            try:
                pair = VideoPromptPair.model_validate(raw_pair)
            except ValueError:
                raise ValueError(
                    f"invalid checkpoint pair on line {line_number}"
                ) from None
            source = by_id[source_id]
            expected = {
                "pair_id": f"video_pair_{source_id:06d}",
                "source_target_id": source_id,
                "source_dataset": identity["source_dataset"],
                "source_revision": identity["source_revision"],
                "generator_model": identity["generator_model"],
                "source_split": identity["source_split"],
                "schema_version": 3,
                "source_target_prompt": source["prompt"],
                "category": source["category"],
                "language": source["language"],
                "shot_type": source["shot_type"].strip(),
                "visual_style": source["visual_style"],
                "target_subject_tag": source["subject_tag"],
            }
            if any(getattr(pair, key) != value for key, value in expected.items()):
                raise ValueError(f"pair provenance mismatch on line {line_number}")
            pairs.append(pair)
        elif record.get("status") == "review" and set(record) == {
            "source_target_id",
            "status",
            "reason",
        }:
            if not isinstance(record["reason"], str) or not record["reason"]:
                raise ValueError(f"invalid review reason on line {line_number}")
        else:
            raise ValueError(f"invalid checkpoint status on line {line_number}")
        records[source_id] = record
    validate_pair_collection(pairs)
    return records


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
    *,
    source_split: str,
) -> list[VideoPromptPair]:
    expected_ids = [row["id"] for row in batch]
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
                    **_request_policy(len(batch)),
                )
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            # A provider may report moderation as HTTP 403 instead of a choice.
            if status_code == 403:
                raise ProviderReviewRequired("http_403_review") from None
            transient = isinstance(exc, APIConnectionError) or (
                isinstance(status_code, int)
                and (status_code in (408, 429) or 500 <= status_code < 600)
            )
            if not transient or attempt + 1 == max_retries:
                # Provider exception bodies can include source text and credentials.
                raise RuntimeError(
                    f"non-retryable or exhausted provider request (HTTP {status_code})"
                ) from None
            delay = min(45.0, 2 ** (attempt + 1) + random.random() * 2)
            print(f"Retrying transient transport failure for batch {expected_ids}")
            await asyncio.sleep(delay)
            continue

        # Inspect terminal status before touching content, including valid-looking JSON.
        # OpenRouter normalizes Gemini safety finishes to content_filter, and can
        # also expose the original reason as native_finish_reason.
        choices = getattr(response, "choices", None)
        if not choices or len(choices) != 1:
            raise ProviderReviewRequired("missing_or_multiple_choices")
        choice = choices[0]
        if getattr(choice, "finish_reason", None) != "stop":
            raise ProviderReviewRequired("non_stop_finish")
        native = getattr(choice, "native_finish_reason", None)
        if native is not None and str(native).lower() not in ("stop", "end_turn"):
            raise ProviderReviewRequired("non_stop_native_finish")
        message = getattr(choice, "message", None)
        if getattr(message, "refusal", None):
            raise ProviderReviewRequired("provider_refusal")
        content = getattr(message, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise ProviderReviewRequired("empty_response")
        try:
            payload = json.loads(content)
            generated = payload["pairs"]
            if not isinstance(generated, list):
                raise ValueError("pairs must be an array")
            returned_ids = [item["source_id"] for item in generated]
            if (
                any(type(source_id) is not int for source_id in returned_ids)
                or returned_ids != expected_ids
            ):
                raise ValueError("source IDs changed")
            pairs = [
                build_video_prompt_pair(
                    row,
                    item,
                    generator_model=model,
                    source_dataset=source_dataset,
                    source_revision=source_revision,
                    source_split=source_split,
                )
                for row, item in zip(batch, generated)
            ]
            validate_pair_collection(pairs)
            return pairs
        except (ValueError, KeyError, TypeError):
            # Invalid JSON can be a plain-text refusal. Never retry output validation
            # failures or include source/generated text in logs or review records.
            raise ProviderReviewRequired("invalid_response") from None
    raise AssertionError("unreachable")


async def _build_pairs_locked(
    args: argparse.Namespace, progress_path: Path
) -> list[VideoPromptPair]:
    all_targets = load_legacy_video_prompts(args.input)
    targets = all_targets[: args.limit] if args.limit is not None else all_targets
    identity = _run_identity(args, all_targets)
    records = _load_checkpoint(progress_path, identity, targets)
    pending = [row for row in targets if row["id"] not in records]
    print(
        f"Targets: {len(targets)} | recorded: {len(records)} | "
        f"pending: {len(pending)} | model: {args.model}"
    )
    if pending:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        limiter = RateLimiter(args.rate_limit_rpm)
        semaphore = asyncio.Semaphore(args.workers)
        progress_lock = asyncio.Lock()
        # SDK retries would bypass the explicit policy and attempt budget here.
        async with AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1", api_key=api_key, max_retries=0
        ) as client:

            async def process(batch: list[dict[str, Any]]) -> None:
                # Persist before sending: an interrupted request has an unknown
                # outcome, so resuming must not silently reissue a possible refusal.
                async with progress_lock:
                    for row in batch:
                        records[row["id"]] = {
                            "source_target_id": row["id"],
                            "status": "review",
                            "reason": "interrupted_or_in_flight",
                        }
                    _write_checkpoint(progress_path, identity, records)
                try:
                    pairs = await _generate_batch(
                        client,
                        args.model,
                        args.source_dataset,
                        args.source_revision,
                        batch,
                        limiter,
                        semaphore,
                        args.max_retries,
                        source_split=args.source_split,
                    )
                except ProviderReviewRequired as exc:
                    async with progress_lock:
                        for row in batch:
                            records[row["id"]]["reason"] = str(exc)
                        _write_checkpoint(progress_path, identity, records)
                    print(f"Recorded {len(batch)} sources for review; no retry")
                    return
                except RuntimeError:
                    async with progress_lock:
                        for row in batch:
                            records[row["id"]]["reason"] = "provider_request_failed"
                        _write_checkpoint(progress_path, identity, records)
                    raise
                async with progress_lock:
                    previous_pairs = [
                        VideoPromptPair.model_validate(record["pair"])
                        for record in records.values()
                        if record["status"] == "complete"
                    ]
                    # Catch cross-batch duplicates before writing them to disk.
                    try:
                        validate_pair_collection([*previous_pairs, *pairs])
                    except ValueError:
                        for row in batch:
                            records[row["id"]]["reason"] = "duplicate_generated_prompt"
                    else:
                        for pair in pairs:
                            records[pair.source_target_id] = {
                                "source_target_id": pair.source_target_id,
                                "status": "complete",
                                "pair": pair.model_dump(mode="json"),
                            }
                    _write_checkpoint(progress_path, identity, records)

            # Finish checkpoint writes before closing the client on a sibling error.
            results = await asyncio.gather(
                *(process(batch) for batch in _chunks(pending, args.batch_size)),
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, BaseException):
                    raise result
    else:
        _write_checkpoint(progress_path, identity, records)

    pairs = [
        VideoPromptPair.model_validate(records[row["id"]]["pair"])
        for row in targets
        if records[row["id"]]["status"] == "complete"
    ]
    write_video_prompt_pairs(pairs, args.output)
    audit = {
        "run": identity,
        "run_sha256": _digest(identity),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "pair_count": len(pairs),
        "review": [
            record for record in records.values() if record["status"] == "review"
        ],
    }
    audit_path = args.output.with_suffix(".provenance.json")
    temporary = audit_path.with_suffix(audit_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(audit_path)
    print(f"Completed pairs: {len(pairs)} | requires review: {len(audit['review'])}")
    return pairs


async def build_pairs(args: argparse.Namespace) -> list[VideoPromptPair]:
    for name in ("batch_size", "workers", "rate_limit_rpm", "max_retries"):
        if type(getattr(args, name)) is not int or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if args.limit is not None and (type(args.limit) is not int or args.limit <= 0):
        raise ValueError("limit must be a positive integer")
    for name in ("source_dataset", "source_revision", "source_split", "model"):
        value = getattr(args, name)
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError(
                f"{name} must be a nonempty string without outer whitespace"
            )
    if len(args.source_revision) < 7 or len(args.model) < 3:
        raise ValueError("source_revision or model is too short for pair provenance")
    progress_path = args.progress or args.output.with_suffix(".progress.jsonl")
    paths = [
        args.input,
        args.output,
        progress_path,
        args.output.with_suffix(".provenance.json"),
        args.output.with_suffix(".summary.json"),
        progress_path.with_suffix(progress_path.suffix + ".lock"),
    ]
    # Reserve temporary output paths too: none may overwrite the source input.
    paths += [path.with_suffix(path.suffix + ".tmp") for path in paths[1:5]]
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved) or any(
        left.exists() and right.exists() and left.samefile(right)
        for index, left in enumerate(paths)
        for right in paths[index + 1 :]
    ):
        raise ValueError("input, output, progress and audit paths must be distinct")
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = paths[5]
    try:
        lock = lock_path.open("x", encoding="utf-8")
    except FileExistsError:
        raise ValueError(
            "checkpoint is locked; confirm no writer is running before removing its .lock file"
        ) from None
    try:
        with lock:
            lock.write(str(os.getpid()))
        return await _build_pairs_locked(args, progress_path)
    finally:
        lock_path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--source-dataset", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-split", required=True)
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
    provenance = json.loads(
        args.output.with_suffix(".provenance.json").read_text(encoding="utf-8")
    )
    summary["review_count"] = len(provenance["review"])
    summary["run_sha256"] = provenance["run_sha256"]
    summary["status"] = "requires_review" if provenance["review"] else "complete"
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(pairs)} validated pairs to {args.output}")
    print(f"Wrote audit summary to {summary_path}")


if __name__ == "__main__":
    main()
