#!/usr/bin/env python3
"""Validate an H3 media manifest and write a reviewable training plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from abliterix.h3_training import H3TrainingConfig, H3TrainingPlan, write_h3_plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.config.read_text(encoding="utf-8"))
    config_root = args.config.resolve().parent
    for field in (
        "model_path",
        "trainer_path",
        "manifest_path",
        "cache_path",
        "output_path",
    ):
        value = Path(payload[field]).expanduser()
        payload[field] = str(value if value.is_absolute() else config_root / value)
    config = H3TrainingConfig.model_validate(payload)
    plan = H3TrainingPlan.from_config(config)
    write_h3_plan(plan, args.output)
    print(f"Validated {len(plan.samples)} H3 samples")
    print(f"Wrote training plan to {args.output}")


if __name__ == "__main__":
    main()
