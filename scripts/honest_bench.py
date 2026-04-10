#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""HonestAbliterationBench runner.

Reproducible benchmark for abliterated language models. Implements the
contract defined in benchmarks/SPEC.md (spec_version 1.0).

Usage:
    python scripts/honest_bench.py \
        --model org/abliterated-model \
        --base-model org/base-model \
        --tier self_reported \
        --device-map auto

Add --dry-run to use 5+5 prompts (CPU-friendly smoke test).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# Make the abliterix package importable when running from a checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Spec constants — DO NOT change without bumping benchmarks/SPEC.md.
# ---------------------------------------------------------------------------

SPEC_VERSION = "1.0"
SPEC_MIN_NEW_TOKENS = 100
SPEC_MAX_NEW_TOKENS = 150
SPEC_JUDGE_MODEL = "google/gemini-3-flash-preview"

REPO_ROOT = Path(__file__).resolve().parent.parent
HARMFUL_PATH = REPO_ROOT / "datasets" / "harmful_500" / "harmful_prompts_500.json"
BENIGN_PATH = REPO_ROOT / "datasets" / "good_500" / "good_prompts_500.json"
HARMFUL_SHA = "249e48f403a8a015a808384ef567c7fa6462aaf7c51d4a78c2caeb0509bd769d"
BENIGN_SHA = "dfffd37646ddb279429323b360a8190e6d971d0d009a6b5f0b877d472cc0a8a7"

FROZEN_CONFIG_PATH = REPO_ROOT / "configs" / "honest_bench.toml"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"


# ---------------------------------------------------------------------------
# Dataset loading (preserves language/category metadata)
# ---------------------------------------------------------------------------


def _load_dataset(path: Path, expected_sha: str) -> tuple[list[dict[str, Any]], str]:
    """Load a benchmark JSON file and verify its SHA256.

    Returns (entries, sha256). Raises if the hash does not match the spec.
    """
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha:
        raise SystemExit(
            f"Dataset hash mismatch for {path}: expected {expected_sha}, got {actual}.\n"
            f"The benchmark dataset has been modified — refusing to run."
        )
    return json.loads(raw.decode("utf-8")), actual


# ---------------------------------------------------------------------------
# Engine helpers (build a temp config, instantiate, free)
# ---------------------------------------------------------------------------


def _build_temp_config(model_id: str, device_map: str, trust_remote_code: bool) -> str:
    """Compose the frozen config + a [model] section into a temp TOML.

    Returns the temp file path. The caller is responsible for cleanup.
    """
    frozen = FROZEN_CONFIG_PATH.read_text(encoding="utf-8")

    model_section = (
        "\n# Injected by scripts/honest_bench.py — per-run model selection.\n"
        "[model]\n"
        f'model_id = "{model_id}"\n'
        f'device_map = "{device_map}"\n'
    )
    if trust_remote_code:
        model_section += "trust_remote_code = true\n"

    fd, path = tempfile.mkstemp(prefix="honest_bench_", suffix=".toml")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(frozen)
        f.write(model_section)
    return path


def _make_engine(model_id: str, device_map: str, trust_remote_code: bool):
    """Instantiate an AxEngine for *model_id* using the frozen config."""
    cfg_path = _build_temp_config(model_id, device_map, trust_remote_code)
    os.environ["AX_CONFIG"] = cfg_path
    # AbliterixConfig parses sys.argv via CliSettingsSource — clear our own
    # CLI args so they aren't re-interpreted as pydantic fields.
    saved_argv = sys.argv
    sys.argv = [saved_argv[0]]
    try:
        from abliterix.core.engine import SteeringEngine  # noqa: WPS433
        from abliterix.settings import AbliterixConfig  # noqa: WPS433

        config = AbliterixConfig()
        engine = SteeringEngine(config)
    finally:
        sys.argv = saved_argv
    return engine, config, cfg_path


def _free_engine(engine, cfg_path: str | None):
    """Drop the loaded model and reclaim GPU/CPU memory."""
    try:
        from abliterix.util import flush_memory  # noqa: WPS433

        engine.model = None
        flush_memory()
    except Exception:  # noqa: BLE001 — best-effort cleanup
        gc.collect()
    if cfg_path and os.path.exists(cfg_path):
        with contextlib_suppress():
            os.unlink(cfg_path)


def contextlib_suppress():
    import contextlib

    return contextlib.suppress(Exception)


# ---------------------------------------------------------------------------
# Generation + judging
# ---------------------------------------------------------------------------


def _entries_to_chat_messages(entries: list[dict[str, Any]]):
    """Wrap raw dataset rows in ChatMessage objects (no system prompt)."""
    from abliterix.types import ChatMessage  # noqa: WPS433

    # Empty system string is the spec — single user turn, no system prompt.
    return [ChatMessage(system="", user=e["prompt"]) for e in entries]


def _generate_responses(engine, msgs) -> list[str]:
    """Run greedy generation under the spec's [100, 150] token bounds."""
    return engine.generate_text_batched(
        msgs,
        skip_special_tokens=True,
        max_new_tokens=SPEC_MAX_NEW_TOKENS,
        min_new_tokens=SPEC_MIN_NEW_TOKENS,
    )


def _judge_responses(
    detector,
    prompts: list[str],
    responses: list[str],
) -> list[bool]:
    """Mirror RefusalDetector._evaluate_with_judge but on pre-generated text.

    Order of operations matches detector.py:
      1. Obvious-refusal short-circuit (degenerate filter + canonical prefixes).
      2. LLM judge on the rest, batched + cached + concurrent.
    """
    out: list[bool] = [False] * len(prompts)
    llm_idx: list[int] = []
    llm_pairs: list[tuple[str, str]] = []

    for i, resp in enumerate(responses):
        if detector.is_obvious_refusal(resp):
            out[i] = True
        else:
            llm_idx.append(i)
            llm_pairs.append((prompts[i], resp))

    if llm_pairs:
        labels = detector._batch_judge_classify(llm_pairs)  # noqa: SLF001
        for j, lab in zip(llm_idx, labels):
            out[j] = bool(lab)

    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _bucket_rates(
    entries: list[dict[str, Any]],
    labels: list[bool],
    field: str,
) -> dict[str, float]:
    """Compute mean(label) bucketed by ``entries[i][field]``."""
    counts: dict[str, list[int]] = {}
    for e, lab in zip(entries, labels):
        key = e.get(field, "unknown")
        counts.setdefault(key, []).append(int(bool(lab)))
    return {
        k: round(sum(v) / len(v), 6) if v else 0.0
        for k, v in sorted(counts.items())
    }


def _overall_rate(labels: list[bool]) -> float:
    if not labels:
        return 0.0
    return round(sum(int(bool(x)) for x in labels) / len(labels), 6)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _gpu_name() -> str:
    try:
        import torch  # noqa: WPS433

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def _judge_cache_sha(checkpoint_dir: str) -> str:
    p = Path(checkpoint_dir) / "judge_cache.sqlite3"
    if not p.exists():
        return ""
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", required=True, help="HF model id of the model under test")
    p.add_argument(
        "--base-model",
        required=True,
        help="HF model id of the unmodified base the abliteration was derived from",
    )
    p.add_argument(
        "--tier",
        required=True,
        choices=["self_reported", "verified"],
        help="Submission tier (see benchmarks/CONTRIBUTING.md)",
    )
    p.add_argument(
        "--device-map",
        default="auto",
        help='HuggingFace accelerate device_map (default "auto"; use "cpu" for smoke testing)',
    )
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading models",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Use 5+5 prompts only (smoke test)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Override result JSON path (default: benchmarks/results/<org>__<model>.json)",
    )
    return p.parse_args()


def main() -> int:  # noqa: PLR0915 — top-level orchestration is intentionally linear
    args = parse_args()
    t0 = time.time()

    # 1. Load + verify datasets.
    print(f"[bench] loading harmful set from {HARMFUL_PATH}")
    harmful_entries, harmful_sha = _load_dataset(HARMFUL_PATH, HARMFUL_SHA)
    print(f"[bench] loading benign set from {BENIGN_PATH}")
    benign_entries, benign_sha = _load_dataset(BENIGN_PATH, BENIGN_SHA)

    if args.dry_run:
        print("[bench] DRY RUN: truncating both sets to 5 prompts")
        harmful_entries = harmful_entries[:5]
        benign_entries = benign_entries[:5]

    harmful_msgs = _entries_to_chat_messages(harmful_entries)
    benign_msgs = _entries_to_chat_messages(benign_entries)
    harmful_prompts = [e["prompt"] for e in harmful_entries]
    benign_prompts = [e["prompt"] for e in benign_entries]

    # 2. Load TARGET model — generate harmful + benign responses + benign logprobs.
    print(f"\n[bench] === TARGET MODEL: {args.model} ===")
    target_engine, target_config, target_cfg_path = _make_engine(
        args.model,
        args.device_map,
        args.trust_remote_code,
    )
    print("[bench] generating harmful responses...")
    target_resp_harmful = _generate_responses(target_engine, harmful_msgs)
    print("[bench] generating benign responses...")
    target_resp_benign = _generate_responses(target_engine, benign_msgs)
    print("[bench] computing target logprobs on benign set...")
    target_logprobs = target_engine.compute_logprobs_batched(benign_msgs).detach().cpu()

    checkpoint_dir = target_config.optimization.checkpoint_dir
    _free_engine(target_engine, target_cfg_path)

    # 3. Load BASE model — generate benign responses (for length z-score) + logprobs.
    print(f"\n[bench] === BASE MODEL: {args.base_model} ===")
    base_engine, base_config, base_cfg_path = _make_engine(
        args.base_model,
        args.device_map,
        args.trust_remote_code,
    )
    print("[bench] generating base benign responses (for length baseline)...")
    base_resp_benign = _generate_responses(base_engine, benign_msgs)
    print("[bench] computing base logprobs on benign set...")
    base_logprobs = base_engine.compute_logprobs_batched(benign_msgs).detach().cpu()
    _free_engine(base_engine, base_cfg_path)

    # 4. KL divergence (target || base).
    import torch  # noqa: WPS433
    import torch.nn.functional as F  # noqa: WPS433

    kl_vs_base = float(
        F.kl_div(
            target_logprobs,
            base_logprobs,
            reduction="batchmean",
            log_target=True,
        ).item()
    )
    print(f"[bench] kl_vs_base = {kl_vs_base:.6f}")

    # 5. Length z-score and degenerate-rate on benign set.
    base_lengths = [len(r.split()) for r in base_resp_benign]
    base_mean = statistics.mean(base_lengths) if base_lengths else 1.0
    base_stdev = statistics.stdev(base_lengths) if len(base_lengths) > 1 else 1.0

    target_lengths = [len(r.split()) for r in target_resp_benign]
    mean_len = statistics.mean(target_lengths) if target_lengths else 0.0
    length_z = abs(mean_len - base_mean) / max(base_stdev, 1.0)

    from abliterix.eval.detector import RefusalDetector  # noqa: WPS433

    # Use the target_config so the detector reads checkpoint_dir / judge model
    # straight from the frozen config we already loaded.
    detector = RefusalDetector(target_config)

    # Override the model the cache + API call use, so the spec model wins
    # regardless of what was inherited from settings.py defaults.
    detector.config.detection.llm_judge_model = SPEC_JUDGE_MODEL

    degenerate_count = sum(1 for r in target_resp_benign if detector._is_degenerate(r))  # noqa: SLF001
    degenerate_rate = degenerate_count / max(1, len(target_resp_benign))

    # 6. Judge harmful (refusal) and benign (over-refusal).
    print("[bench] judging harmful responses (refusal)...")
    refusal_labels = _judge_responses(detector, harmful_prompts, target_resp_harmful)
    print(
        f"[bench] refusal_rate_total = "
        f"{sum(refusal_labels)}/{len(refusal_labels)}"
    )

    print("[bench] judging benign responses (over-refusal)...")
    over_refusal_labels = _judge_responses(detector, benign_prompts, target_resp_benign)
    print(
        f"[bench] over_refusal_rate_total = "
        f"{sum(over_refusal_labels)}/{len(over_refusal_labels)}"
    )

    judge_cache_sha = _judge_cache_sha(checkpoint_dir)
    detector.close()

    # 7. Aggregate metrics.
    result: dict[str, Any] = {
        "spec_version": SPEC_VERSION,
        "model": args.model,
        "base_model": args.base_model,
        "tier": args.tier,

        "refusal_rate_total": _overall_rate(refusal_labels),
        "refusal_rate_en": _overall_rate(
            [lab for e, lab in zip(harmful_entries, refusal_labels) if e.get("language") == "en"]
        ),
        "refusal_rate_zh": _overall_rate(
            [lab for e, lab in zip(harmful_entries, refusal_labels) if e.get("language") == "zh"]
        ),
        "refusal_rate_mixed": _overall_rate(
            [lab for e, lab in zip(harmful_entries, refusal_labels) if e.get("language") == "mixed"]
        ),
        "refusal_rate_by_category": _bucket_rates(harmful_entries, refusal_labels, "category"),

        "over_refusal_rate_total": _overall_rate(over_refusal_labels),
        "over_refusal_rate_en": _overall_rate(
            [lab for e, lab in zip(benign_entries, over_refusal_labels) if e.get("language") == "en"]
        ),
        "over_refusal_rate_zh": _overall_rate(
            [lab for e, lab in zip(benign_entries, over_refusal_labels) if e.get("language") == "zh"]
        ),
        "over_refusal_rate_mixed": _overall_rate(
            [lab for e, lab in zip(benign_entries, over_refusal_labels) if e.get("language") == "mixed"]
        ),
        "over_refusal_rate_by_category": _bucket_rates(
            benign_entries, over_refusal_labels, "category"
        ),

        "kl_vs_base": round(kl_vs_base, 6),
        "degenerate_rate": round(degenerate_rate, 6),
        "mean_response_length": round(mean_len, 3),
        "length_z_score": round(length_z, 3),

        "n_harmful": len(harmful_entries),
        "n_benign": len(benign_entries),

        "judge_model": SPEC_JUDGE_MODEL,
        "judge_cache_sha256": judge_cache_sha,
        "dataset_sha256_harmful": harmful_sha,
        "dataset_sha256_benign": benign_sha,
        "commit": _git_commit(),
        "gpu": _gpu_name(),
        "runtime_seconds": round(time.time() - t0, 1),

        "min_new_tokens": SPEC_MIN_NEW_TOKENS,
        "max_new_tokens": SPEC_MAX_NEW_TOKENS,
        "dry_run": bool(args.dry_run),
    }

    # 8. Write result JSON + raw outputs.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.output:
        out_path = Path(args.output)
    else:
        slug = args.model.replace("/", "__")
        out_path = RESULTS_DIR / f"{slug}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n[bench] wrote {out_path}")

    raw_path = out_path.with_suffix(".outputs.jsonl")
    with raw_path.open("w", encoding="utf-8") as f:
        for e, resp, lab in zip(harmful_entries, target_resp_harmful, refusal_labels):
            f.write(
                json.dumps(
                    {
                        "set": "harmful",
                        "id": e.get("id"),
                        "language": e.get("language"),
                        "category": e.get("category"),
                        "prompt": e.get("prompt"),
                        "response": resp,
                        "refusal": bool(lab),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        for e, resp, lab in zip(benign_entries, target_resp_benign, over_refusal_labels):
            f.write(
                json.dumps(
                    {
                        "set": "benign",
                        "id": e.get("id"),
                        "language": e.get("language"),
                        "category": e.get("category"),
                        "prompt": e.get("prompt"),
                        "response": resp,
                        "refusal": bool(lab),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"[bench] wrote {raw_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
