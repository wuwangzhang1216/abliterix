#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""HonestAbliterationBench leaderboard builder.

Reads every result JSON under benchmarks/results/, sorts them by the
composite key from benchmarks/SPEC.md §8, and writes:

  1. A markdown table into README.md, between the
     <!-- BENCH:START --> and <!-- BENCH:END --> markers.
  2. benchmarks/leaderboard.json (machine-readable mirror).

Idempotent: re-running on unchanged results produces no diff.

Usage:
    python scripts/build_leaderboard.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
README_PATH = REPO_ROOT / "README.md"
LEADERBOARD_JSON = REPO_ROOT / "benchmarks" / "leaderboard.json"

BENCH_START = "<!-- BENCH:START -->"
BENCH_END = "<!-- BENCH:END -->"

# Required keys (from benchmarks/SPEC.md §7). Missing any → row skipped.
REQUIRED_KEYS = (
    "spec_version",
    "model",
    "base_model",
    "tier",
    "refusal_rate_total",
    "over_refusal_rate_total",
    "kl_vs_base",
    "degenerate_rate",
    "mean_response_length",
)


def _load_results() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not RESULTS_DIR.exists():
        return rows
    for path in sorted(RESULTS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"warning: skipping {path.name}: {exc}", file=sys.stderr)
            continue
        missing = [k for k in REQUIRED_KEYS if k not in data]
        if missing:
            print(
                f"warning: skipping {path.name}: missing keys {missing}",
                file=sys.stderr,
            )
            continue
        data["_source"] = path.name
        rows.append(data)
    return rows


def _sort_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    """Composite leaderboard ordering — every component lower-is-better."""
    return (
        float(row["refusal_rate_total"]),
        float(row["kl_vs_base"]),
        float(row["over_refusal_rate_total"]),
        float(row["degenerate_rate"]),
    )


def _fmt_pct(x: float) -> str:
    return f"{100 * float(x):.1f}%"


def _fmt_kl(x: float) -> str:
    v = float(x)
    if v == 0.0:
        return "0"
    return f"{v:.4f}"


def _tier_badge(tier: str) -> str:
    if tier == "verified":
        return "✓ verified"
    return "self-reported"


def _render_markdown(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return (
            "_No results yet. See [benchmarks/CONTRIBUTING.md](benchmarks/CONTRIBUTING.md) "
            "for how to submit one._\n"
        )

    header = (
        "| # | Model | Base | Tier | Refusal | Over-refusal | KL vs base | Degenerate | Len z |\n"
        "|---|---|---|---|---:|---:|---:|---:|---:|\n"
    )
    body_lines: list[str] = []
    for i, row in enumerate(sorted(rows, key=_sort_key), start=1):
        body_lines.append(
            "| {rank} | `{model}` | `{base}` | {tier} | {ref} | {orr} | {kl} | {deg} | {lz} |".format(
                rank=i,
                model=row["model"],
                base=row["base_model"],
                tier=_tier_badge(row["tier"]),
                ref=_fmt_pct(row["refusal_rate_total"]),
                orr=_fmt_pct(row["over_refusal_rate_total"]),
                kl=_fmt_kl(row["kl_vs_base"]),
                deg=_fmt_pct(row["degenerate_rate"]),
                lz=f"{float(row.get('length_z_score', 0.0)):.2f}",
            )
        )
    footer = (
        "\n_Sorted by `refusal_rate_total` ↑, then `kl_vs_base` ↓, then "
        "`over_refusal_rate_total` ↓, then `degenerate_rate` ↓. "
        "See [benchmarks/SPEC.md](benchmarks/SPEC.md) for the full contract._\n"
    )
    return header + "\n".join(body_lines) + "\n" + footer


def _splice_into_readme(table_md: str) -> bool:
    """Replace the BENCH:START/END block in README.md. Returns True on change."""
    text = README_PATH.read_text(encoding="utf-8")
    if BENCH_START not in text or BENCH_END not in text:
        raise SystemExit(
            f"README.md is missing the {BENCH_START} / {BENCH_END} markers. "
            "Add them inside the 'Honest Abliteration Leaderboard' section first."
        )
    pre, _, rest = text.partition(BENCH_START)
    _, _, post = rest.partition(BENCH_END)
    new_block = f"{BENCH_START}\n{table_md}{BENCH_END}"
    new_text = pre + new_block + post
    if new_text == text:
        return False
    README_PATH.write_text(new_text, encoding="utf-8")
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if README.md would be modified (CI guard)",
    )
    args = p.parse_args()

    rows = _load_results()
    table_md = _render_markdown(rows)

    LEADERBOARD_JSON.parent.mkdir(parents=True, exist_ok=True)
    LEADERBOARD_JSON.write_text(
        json.dumps(
            sorted(rows, key=_sort_key),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {LEADERBOARD_JSON.relative_to(REPO_ROOT)} ({len(rows)} rows)")

    changed = _splice_into_readme(table_md)
    if changed:
        print("README.md leaderboard block updated")
    else:
        print("README.md leaderboard block unchanged")

    if args.check and changed:
        print("error: --check supplied and README would change", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
