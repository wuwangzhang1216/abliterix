#!/usr/bin/env python3
"""Sync tokenizer_config.json (and chat_template.jinja for 31B) from
google/gemma-4-* upstream into wangzhang/gemma-4-*-abliterated repos."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
# minimal .env parser to avoid dotenv dep
env_path = ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

TOKEN = os.environ.get("HF_TOKEN")
if not TOKEN:
    sys.exit("HF_TOKEN missing in .env")

# (upstream, downstream, files-to-sync)
JOBS = [
    (
        "google/gemma-4-31B-it",
        "wangzhang/gemma-4-31B-it-abliterated",
        ["tokenizer_config.json", "chat_template.jinja"],
    ),
    (
        "google/gemma-4-E2B-it",
        "wangzhang/gemma-4-E2B-it-abliterated",
        ["tokenizer_config.json"],
    ),
    (
        "google/gemma-4-E4B-it",
        "wangzhang/gemma-4-E4B-it-abliterated",
        ["tokenizer_config.json"],
    ),
]

api = HfApi(token=TOKEN)

for upstream, downstream, files in JOBS:
    print(f"\n=== {downstream} <- {upstream} ===")
    for fname in files:
        local = hf_hub_download(repo_id=upstream, filename=fname, token=TOKEN)
        size = Path(local).stat().st_size
        print(f"  fetched {fname} ({size} bytes)")
        api.upload_file(
            path_or_fileobj=local,
            path_in_repo=fname,
            repo_id=downstream,
            repo_type="model",
            commit_message=f"sync {fname} from {upstream}",
        )
        print(f"  uploaded -> {downstream}/{fname}")

print("\nAll done.")
