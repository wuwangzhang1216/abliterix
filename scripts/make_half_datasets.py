# Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>

"""Stratified 50% sampling from 1000-prompt datasets to create 500-prompt subsets."""

import json
import math
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "datasets"
SEED = 42
TARGET = 500


def stratified_sample(records: list[dict], target: int, seed: int) -> list[dict]:
    rng = random.Random(seed)

    # Group by (category, language, sophistication)
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        key = (r["category"], r["language"], r["sophistication"])
        strata[key].append(r)

    # Shuffle within each stratum
    for v in strata.values():
        rng.shuffle(v)

    # Take ceil(len/2) from each stratum
    selected: list[dict] = []
    remaining: list[dict] = []
    for key, items in strata.items():
        n = math.ceil(len(items) / 2)
        selected.extend(items[:n])
        remaining.extend(items[n:])

    # Trim or pad to target
    if len(selected) > target:
        rng.shuffle(selected)
        remaining.extend(selected[target:])
        selected = selected[:target]
    elif len(selected) < target:
        rng.shuffle(remaining)
        selected.extend(remaining[: target - len(selected)])

    # Re-number ids
    rng.shuffle(selected)
    for i, r in enumerate(selected, 1):
        r["id"] = i

    return selected


def process(src_dir: str, src_file: str, dst_dir: str, dst_file: str):
    src = ROOT / src_dir / src_file
    data = json.loads(src.read_text(encoding="utf-8"))
    sampled = stratified_sample(data, TARGET, SEED)

    out_dir = ROOT / dst_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / dst_file
    out_path.write_text(
        json.dumps(sampled, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"{src_dir}/{src_file} ({len(data)}) -> {dst_dir}/{dst_file} ({len(sampled)})"
    )


def main():
    process("good_1000", "good_prompts_1000.json", "good_500", "good_prompts_500.json")
    process(
        "harmful_1000",
        "harmful_prompts_1000.json",
        "harmful_500",
        "harmful_prompts_500.json",
    )


if __name__ == "__main__":
    main()
