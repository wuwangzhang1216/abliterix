# Abliterix — a derivative work of Heretic (https://github.com/p-e-w/heretic)
# Original work Copyright (C) 2025  Philipp Emanuel Weidmann (p-e-w)
# Modified work Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Reproducibility manifest: capture, publish, and verify the exact conditions
under which an abliterated model was produced.

This mirrors (and extends) Heretic's ``reproduce.json`` / ``SHA256SUMS`` /
``--reproduce`` workflow.  The goal is that any published model can be
independently rebuilt and bit-checked, and that an environment that drifts from
the one used to produce it is flagged by severity before a reproduction attempt.

Three pieces:

* :func:`build_manifest` — assemble a JSON-serialisable manifest from the
  resolved config, the winning trial, the environment, and per-shard weight
  hashes.
* :func:`write_reproduce_artifacts` — render ``reproduce.json``,
  ``SHA256SUMS`` and a human-readable ``README.md`` into a directory ready to
  upload to the model repo's ``reproduce/`` folder.
* :func:`check_environment` — diff the current environment against a manifest
  and classify each difference by severity (LOW / MEDIUM / HIGH / CRITICAL).
"""

from __future__ import annotations

import json
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
REPRODUCE_TAG = "reproducible"

# Packages whose versions materially affect the produced weights / behaviour.
# Ordered roughly by how load-bearing they are for reproduction.
_TRACKED_PACKAGES = (
    "abliterix",
    "torch",
    "transformers",
    "peft",
    "optuna",
    "accelerate",
    "datasets",
    "bitsandbytes",
    "numpy",
    "huggingface-hub",
)

# Packages whose major.minor skew is treated as CRITICAL (likely to change
# numerics or break the load path outright).
_CRITICAL_PACKAGES = ("torch", "transformers")


def _pkg_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _gpu_driver() -> str | None:
    """Best-effort NVIDIA driver version via nvidia-smi (None if unavailable)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            lines = out.stdout.strip().splitlines()
            if lines:
                return lines[0].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _git_commit() -> dict[str, Any] | None:
    """Return the abliterix source commit + dirty flag, when running from a checkout."""
    try:
        root = Path(__file__).resolve().parents[2]
        rev = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if rev.returncode != 0:
            return None
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return {
            "commit": rev.stdout.strip(),
            "dirty": bool(status.stdout.strip()),
        }
    except (OSError, subprocess.SubprocessError):
        return None


def collect_environment() -> dict[str, Any]:
    """Capture the host environment relevant to reproduction."""
    import torch

    cuda = getattr(torch.version, "cuda", None)
    hip = getattr(torch.version, "hip", None)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "torch_cuda": cuda,
        "torch_hip": hip,
        "gpu_driver": _gpu_driver(),
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }


def collect_packages() -> dict[str, str | None]:
    """Capture versions of the packages that affect the produced weights."""
    return {name: _pkg_version(name) for name in _TRACKED_PACKAGES}


def _dataset_sources(config) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in (
        "benign_prompts",
        "target_prompts",
        "benign_eval_prompts",
        "target_eval_prompts",
    ):
        src = getattr(config, key, None)
        if src is None:
            continue
        out[key] = {
            "dataset": src.dataset,
            "split": src.split,
            "column": src.column,
        }
    return out


def build_manifest(
    config,
    trial,
    *,
    repo_id: str | None = None,
    weight_shas: dict[str, str] | None = None,
    baseline_refusal_count: int | None = None,
    n_target_prompts: int | None = None,
) -> dict[str, Any]:
    """Assemble a JSON-serialisable reproducibility manifest.

    Parameters
    ----------
    config : AbliterixConfig
        The fully-resolved run configuration (stored verbatim for replay).
    trial : optuna.Trial
        The winning trial whose parameters produced the exported model.
    weight_shas : dict, optional
        Mapping of exported shard filename to its SHA256 hex digest.
    """
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tool": "abliterix",
        "abliterix_version": _pkg_version("abliterix"),
        "repo_id": repo_id,
        "source": _git_commit(),
        "seed": getattr(config, "seed", None),
        "environment": collect_environment(),
        "packages": collect_packages(),
        "model": {
            "model_id": config.model.model_id,
            "revision": getattr(config.model, "revision", None),
        },
        "datasets": _dataset_sources(config),
        "config": config.model_dump(mode="json"),
    }

    if trial is not None:
        manifest["trial"] = {
            "index": trial.user_attrs.get("index"),
            "vector_index": trial.user_attrs.get("vector_index"),
            "parameters": trial.user_attrs.get("parameters"),
            "moe_parameters": trial.user_attrs.get("moe_parameters"),
            "decay_kernel": trial.user_attrs.get("decay_kernel"),
            "direct_transform": trial.user_attrs.get("direct_transform"),
            "steering_variant": trial.user_attrs.get("steering_variant"),
        }
        manifest["metrics"] = {
            "kl_divergence": trial.user_attrs.get("kl_divergence"),
            "refusals": trial.user_attrs.get("refusals"),
            "baseline_refusals": baseline_refusal_count,
            "n_target_prompts": n_target_prompts,
        }

    if weight_shas:
        manifest["weights"] = dict(sorted(weight_shas.items()))

    return manifest


def repo_weight_shas(repo_id: str, token: str | None) -> dict[str, str]:
    """Fetch per-file SHA256 for LFS weight shards of an uploaded repo.

    The Hub stores the SHA256 of every LFS object, so this gives bit-level
    checksums of the exported weights without keeping a local copy.
    """
    from huggingface_hub import HfApi

    shas: dict[str, str] = {}
    try:
        info = HfApi().repo_info(repo_id, files_metadata=True, token=token)
    except Exception:
        return shas
    for sib in getattr(info, "siblings", None) or []:
        name = getattr(sib, "rfilename", "")
        if not name.endswith((".safetensors", ".bin")):
            continue
        lfs = getattr(sib, "lfs", None)
        sha = None
        if isinstance(lfs, dict):
            sha = lfs.get("sha256")
        elif lfs is not None:
            sha = getattr(lfs, "sha256", None)
        if sha:
            shas[name] = sha
    return shas


def write_reproduce_artifacts(
    out_dir: str | Path, manifest: dict[str, Any]
) -> list[Path]:
    """Render reproduce.json, SHA256SUMS and README.md into *out_dir*."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    reproduce_json = out / "reproduce.json"
    reproduce_json.write_text(json.dumps(manifest, indent=2, sort_keys=False))
    written.append(reproduce_json)

    weights = manifest.get("weights") or {}
    if weights:
        sha_lines = [f"{sha}  {name}" for name, sha in weights.items()]
        sha_file = out / "SHA256SUMS"
        sha_file.write_text("\n".join(sha_lines) + "\n")
        written.append(sha_file)

    readme = out / "README.md"
    readme.write_text(_render_readme(manifest))
    written.append(readme)

    return written


def _render_readme(manifest: dict[str, Any]) -> str:
    env = manifest.get("environment", {})
    pkgs = manifest.get("packages", {})
    model = manifest.get("model", {})
    seed = manifest.get("seed")
    metrics = manifest.get("metrics", {})
    pkg_rows = "\n".join(
        f"| {name} | {ver if ver is not None else '—'} |" for name, ver in pkgs.items()
    )
    return f"""# Reproducibility manifest

This model was produced with [Abliterix](https://github.com/wuwangzhang1216/abliterix)
`v{manifest.get("abliterix_version")}`. The files here let you verify and
reproduce it.

- `reproduce.json` — the full resolved configuration, environment, seed, dataset
  sources, winning-trial parameters, and per-shard weight SHA256.
- `SHA256SUMS` — checksums of the exported weight shards (verify with
  `sha256sum -c SHA256SUMS`).

## How to reproduce

```bash
pip install -U abliterix
abliterix --reproduce reproduce.json
```

Abliterix will diff your environment against the one recorded below (flagging
differences by severity) and then re-run the search with the recorded seed and
configuration.

## Key facts

| Field | Value |
| :---- | :---- |
| Base model | `{model.get("model_id")}` |
| Seed | `{seed}` |
| KL divergence | `{metrics.get("kl_divergence")}` |
| Refusals | `{metrics.get("refusals")}` / `{metrics.get("n_target_prompts")}` |
| Python | `{env.get("python")}` |
| Platform | `{env.get("platform")}` |
| CUDA (torch) | `{env.get("torch_cuda")}` |
| GPU driver | `{env.get("gpu_driver")}` |

## Package versions

| Package | Version |
| :------ | :------ |
{pkg_rows}
"""


# ---------------------------------------------------------------------------
# Environment verification
# ---------------------------------------------------------------------------

_SEVERITY_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


def _major_minor(ver: str | None) -> str | None:
    if not ver:
        return None
    parts = ver.split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else ver


def _classify_package(
    name: str, expected: str | None, actual: str | None
) -> str | None:
    """Return a severity label for a package version difference, or None if equal."""
    if expected == actual:
        return None
    if expected is None or actual is None:
        return "HIGH" if name in _CRITICAL_PACKAGES else "MEDIUM"
    if _major_minor(expected) != _major_minor(actual):
        return "CRITICAL" if name in _CRITICAL_PACKAGES else "HIGH"
    # Patch-level difference.
    return "HIGH" if name in _CRITICAL_PACKAGES else "MEDIUM"


def check_environment(manifest: dict[str, Any]) -> list[tuple[str, str]]:
    """Diff the current environment against *manifest*.

    Returns a list of ``(severity, message)`` tuples, highest severity first.
    An empty list means the environment matches on every tracked dimension.
    """
    findings: list[tuple[str, str]] = []

    exp_pkgs = manifest.get("packages", {}) or {}
    cur_pkgs = collect_packages()
    for name in _TRACKED_PACKAGES:
        sev = _classify_package(name, exp_pkgs.get(name), cur_pkgs.get(name))
        if sev:
            findings.append(
                (
                    sev,
                    f"{name}: recorded {exp_pkgs.get(name)} vs current {cur_pkgs.get(name)}",
                )
            )

    exp_env = manifest.get("environment", {}) or {}
    cur_env = collect_environment()

    if _major_minor(exp_env.get("python")) != _major_minor(cur_env.get("python")):
        findings.append(
            (
                "HIGH",
                f"python: recorded {exp_env.get('python')} vs current {cur_env.get('python')}",
            )
        )
    elif exp_env.get("python") != cur_env.get("python"):
        findings.append(
            (
                "LOW",
                f"python: recorded {exp_env.get('python')} vs current {cur_env.get('python')}",
            )
        )

    if exp_env.get("torch_cuda") != cur_env.get("torch_cuda"):
        findings.append(
            (
                "HIGH",
                f"CUDA runtime: recorded {exp_env.get('torch_cuda')} vs current {cur_env.get('torch_cuda')}",
            )
        )

    if exp_env.get("gpu_driver") != cur_env.get("gpu_driver"):
        findings.append(
            (
                "MEDIUM",
                f"GPU driver: recorded {exp_env.get('gpu_driver')} vs current {cur_env.get('gpu_driver')}",
            )
        )

    if exp_env.get("platform") != cur_env.get("platform"):
        findings.append(
            (
                "LOW",
                f"platform: recorded {exp_env.get('platform')} vs current {cur_env.get('platform')}",
            )
        )

    findings.sort(key=lambda f: _SEVERITY_ORDER.get(f[0], 0), reverse=True)
    return findings


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load a reproduce.json manifest from disk."""
    return json.loads(Path(path).read_text())
