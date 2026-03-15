# Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>

"""Shared utilities for Prometheus scripts.

setup_io() must be called before importing any library that captures
stdout/stderr (e.g. Rich, which is imported by prometheus.utils).
All other functions use lazy imports so that importing this module
does not trigger heavy library loading.
"""

import io
import os
import sys
from pathlib import Path

# Project root: src/prometheus/scriptlib.py -> src/prometheus -> src -> project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def setup_io():
    """Set up UTF-8 encoding for Windows and load .env file from project root."""
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )

    env_path = _PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def load_trial(checkpoint_dir, model_name, trial_number):
    """Load a specific trial from an Optuna checkpoint.

    Matches by user_attrs["index"] first (matches "Running trial X" display),
    falling back to trial.number (0-indexed).
    """
    import optuna
    from optuna.storages.journal import JournalFileBackend, JournalStorage

    from prometheus.util import slugify_model_name

    slug = slugify_model_name(model_name)
    journal_path = os.path.join(checkpoint_dir, f"{slug}.jsonl")

    if not os.path.exists(journal_path):
        raise FileNotFoundError(f"Checkpoint not found: {journal_path}")

    storage = JournalStorage(JournalFileBackend(journal_path))
    study = optuna.load_study(study_name="prometheus", storage=storage)

    matching = [t for t in study.trials if t.user_attrs.get("index") == trial_number]
    if not matching:
        matching = [t for t in study.trials if t.number == trial_number]
    if not matching:
        available = sorted(
            set(t.user_attrs.get("index", t.number) for t in study.trials)
        )
        raise ValueError(f"Trial {trial_number} not found. Available: {available[:20]}")

    return matching[0]


def extract_trial_params(trial):
    """Extract abliteration parameters from a completed trial.

    Returns (vector_index, profiles, routing).
    """
    from prometheus.types import ExpertRoutingConfig, SteeringProfile

    vector_index = trial.user_attrs["vector_index"]
    profiles = {
        k: SteeringProfile(**v) for k, v in trial.user_attrs["parameters"].items()
    }
    routing_dict = trial.user_attrs.get("moe_parameters")
    routing = ExpertRoutingConfig(**routing_dict) if routing_dict else None
    return vector_index, profiles, routing
