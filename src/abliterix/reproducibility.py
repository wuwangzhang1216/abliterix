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

import hashlib
import json
import math
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import SimpleNamespace
from typing import Any

SCHEMA_VERSION = 2
REPRODUCE_TAG = "reproducible"
_PIN_LENGTH = 40

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
        # Unknown must not be recorded as clean: a failed `git status`
        # (corrupt index, dubious-ownership, timeout) would otherwise be
        # published as a clean checkout and skip the dirty-tree check.
        if status.returncode != 0:
            return None
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
            "revision": getattr(src, "revision", None),
            "split": src.split,
            "column": src.column,
            "prefix": src.prefix,
            "suffix": src.suffix,
            "system_prompt": src.system_prompt,
        }
    return out


def _is_commit_pin(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _PIN_LENGTH
        and all(char in "0123456789abcdefABCDEF" for char in value)
    )


def _is_local_source(value: str) -> bool:
    return Path(value).expanduser().exists()


def pin_remote_sources(config, *, api=None) -> list[str]:
    """Resolve every unpinned Hub input to its immutable commit SHA.

    The resolved revisions are written back to the config so model and dataset
    loading use the same snapshot that is later recorded in ``reproduce.json``.
    Local inputs are left untouched and are subsequently classified as not
    independently reproducible.
    """
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()

    resolved: list[str] = []
    model = config.model
    if not _is_local_source(model.model_id) and not _is_commit_pin(model.revision):
        try:
            info = api.model_info(model.model_id, revision=model.revision)
        except Exception as error:
            raise RuntimeError(
                f"Could not resolve immutable revision for model {model.model_id!r}: {error}"
            ) from error
        if not _is_commit_pin(getattr(info, "sha", None)):
            raise RuntimeError(
                f"Hub returned no immutable commit for model {model.model_id!r}."
            )
        model.revision = info.sha
        resolved.append(f"model {model.model_id}@{model.revision}")

    if (
        model.evaluate_model_id
        and not _is_local_source(model.evaluate_model_id)
        and not _is_commit_pin(model.evaluate_model_revision)
    ):
        try:
            info = api.model_info(
                model.evaluate_model_id,
                revision=model.evaluate_model_revision,
            )
        except Exception as error:
            raise RuntimeError(
                "Could not resolve immutable revision for evaluation model "
                f"{model.evaluate_model_id!r}: {error}"
            ) from error
        if not _is_commit_pin(getattr(info, "sha", None)):
            raise RuntimeError(
                "Hub returned no immutable commit for evaluation model "
                f"{model.evaluate_model_id!r}."
            )
        model.evaluate_model_revision = info.sha
        resolved.append(
            f"evaluation model {model.evaluate_model_id}@{model.evaluate_model_revision}"
        )

    for name in (
        "benign_prompts",
        "target_prompts",
        "benign_eval_prompts",
        "target_eval_prompts",
    ):
        source = getattr(config, name)
        if _is_local_source(source.dataset) or _is_commit_pin(source.revision):
            continue
        try:
            info = api.dataset_info(source.dataset, revision=source.revision)
        except Exception as error:
            raise RuntimeError(
                f"Could not resolve immutable revision for {name} "
                f"dataset {source.dataset!r}: {error}"
            ) from error
        if not _is_commit_pin(getattr(info, "sha", None)):
            raise RuntimeError(
                f"Hub returned no immutable commit for {name} dataset "
                f"{source.dataset!r}."
            )
        source.revision = info.sha
        resolved.append(f"{name} {source.dataset}@{source.revision}")
    return resolved


def assess_reproducibility(config) -> tuple[bool, list[str]]:
    """Return whether a run can be independently replayed, with exact reasons."""
    reasons: list[str] = []
    if getattr(config, "seed", None) is None:
        reasons.append("the global seed is unresolved")

    model = config.model
    if _is_local_source(model.model_id):
        reasons.append("the base model is a local path")
    elif not _is_commit_pin(getattr(model, "revision", None)):
        reasons.append("the base model is not pinned to a Hub commit")
    if model.evaluate_model_id:
        if _is_local_source(model.evaluate_model_id):
            reasons.append("the evaluation model is a local path")
        elif not _is_commit_pin(model.evaluate_model_revision):
            reasons.append("the evaluation model is not pinned to a Hub commit")

    for name in (
        "benign_prompts",
        "target_prompts",
        "benign_eval_prompts",
        "target_eval_prompts",
    ):
        source = getattr(config, name)
        if _is_local_source(source.dataset):
            reasons.append(f"{name} is a local dataset")
        elif not _is_commit_pin(getattr(source, "revision", None)):
            reasons.append(f"{name} is not pinned to a Hub commit")

    if config.detection.llm_judge:
        reasons.append("the external LLM judge is not independently reproducible")
    if config.model.backend != "hf":
        reasons.append("exact trial materialization currently requires backend='hf'")

    source = _git_commit()
    if source and source.get("dirty"):
        reasons.append("the Abliterix source checkout has uncommitted changes")
    return not reasons, reasons


def _canonical_payload(manifest: dict[str, Any]) -> bytes:
    payload = {key: value for key, value in manifest.items() if key != "integrity"}
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def canonical_manifest_sha256(manifest: dict[str, Any]) -> str:
    """Hash the semantic manifest payload, excluding its integrity envelope."""
    return hashlib.sha256(_canonical_payload(manifest)).hexdigest()


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

    reproducible, reasons = assess_reproducibility(config)

    if trial is not None:
        manifest["trial"] = {
            "index": trial.user_attrs.get("index"),
            "vector_index": trial.user_attrs.get("vector_index"),
            "parameters": trial.user_attrs.get("parameters"),
            "moe_parameters": trial.user_attrs.get("moe_parameters"),
            "decay_kernel": trial.user_attrs.get("decay_kernel"),
            "direct_transform": trial.user_attrs.get("direct_transform"),
            "steering_variant": trial.user_attrs.get("steering_variant"),
            "steering_recipe": trial.user_attrs.get("steering_recipe"),
        }
        manifest["metrics"] = {
            "kl_divergence": trial.user_attrs.get("kl_divergence"),
            "refusals": trial.user_attrs.get("refusals"),
            "baseline_refusals": baseline_refusal_count,
            "n_target_prompts": n_target_prompts,
        }
        if not trial.user_attrs.get("steering_recipe"):
            reproducible = False
            reasons.append("the winning trial has no exact steering recipe")
    else:
        reproducible = False
        reasons.append("the manifest has no winning trial")

    if weight_shas:
        manifest["weights"] = dict(sorted(weight_shas.items()))

    manifest["reproducible"] = reproducible
    manifest["reproducibility_reasons"] = reasons
    manifest["integrity"] = {
        "algorithm": "sha256",
        "canonical_manifest": canonical_manifest_sha256(manifest),
    }

    return manifest


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Validate schema, exact-trial data, and the manifest integrity envelope."""
    version_value = manifest.get("schema_version")
    if version_value != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported reproduce schema {version_value!r}; expected {SCHEMA_VERSION}. "
            "Legacy schema v1 can restore configuration but cannot prove exact replay."
        )
    integrity = manifest.get("integrity") or {}
    expected = integrity.get("canonical_manifest")
    actual = canonical_manifest_sha256(manifest)
    if not expected or expected != actual:
        raise ValueError("Reproduce manifest integrity check failed.")
    trial = manifest.get("trial")
    if not isinstance(trial, dict):
        raise ValueError("Reproduce manifest has no exact winning-trial artifact.")
    for field in ("vector_index", "parameters", "steering_recipe"):
        if field not in trial or trial[field] is None:
            raise ValueError(f"Reproduce manifest trial is missing {field!r}.")
    # Without weight hashes the SHA256SUMS half of the guarantee is
    # unverifiable, so such a manifest must not be accepted for exact replay.
    if not isinstance(manifest.get("weights"), dict) or not manifest["weights"]:
        raise ValueError(
            "Reproduce manifest has no weight SHA256 checksums; exact replay "
            "cannot be verified."
        )


def manifest_trial(manifest: dict[str, Any]) -> SimpleNamespace:
    """Return a trial-shaped immutable replay proxy from a validated manifest."""
    validate_manifest(manifest)
    trial = dict(manifest["trial"])
    return SimpleNamespace(user_attrs=trial)


def compare_reproduction_metrics(
    manifest: dict[str, Any],
    *,
    kl_divergence: float,
    refusals: int,
    kl_abs_tol: float = 1e-6,
    kl_rel_tol: float = 1e-3,
) -> list[str]:
    """Compare independently re-measured results with the published metrics."""
    findings: list[str] = []
    metrics = manifest.get("metrics") or {}
    expected_kl = metrics.get("kl_divergence")
    if not isinstance(expected_kl, (int, float)) or not math.isclose(
        float(expected_kl),
        kl_divergence,
        rel_tol=kl_rel_tol,
        abs_tol=kl_abs_tol,
    ):
        findings.append(
            f"kl_divergence: recorded {expected_kl!r}, reproduced {kl_divergence!r}"
        )
    expected_refusals = metrics.get("refusals")
    if expected_refusals != refusals:
        findings.append(
            f"refusals: recorded {expected_refusals!r}, reproduced {refusals!r}"
        )
    return findings


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


def local_weight_shas(model_dir: str | Path) -> dict[str, str]:
    """Compute deterministic SHA256 checksums for exported model weight files."""
    root = Path(model_dir)
    shas: dict[str, str] = {}
    # rglob: sharded exports nest shards in subdirectories, and a
    # partially-hashed export would silently pass the non-empty check.
    for path in sorted(root.rglob("*")):
        if not path.is_file() or not path.name.endswith((".safetensors", ".bin")):
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        rel_path = path.relative_to(root).as_posix()
        shas[rel_path] = digest.hexdigest()
    return shas


def write_reproduce_artifacts(
    out_dir: str | Path, manifest: dict[str, Any]
) -> list[Path]:
    """Render reproduce.json, SHA256SUMS and README.md into *out_dir*."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # Explicit utf-8: Path.write_text defaults to the locale encoding
    # (cp1252 on many Windows hosts), which raises UnicodeEncodeError for a
    # non-ASCII model id or GPU driver string.
    reproduce_json = out / "reproduce.json"
    reproduce_json.write_text(
        json.dumps(manifest, indent=2, sort_keys=False), encoding="utf-8"
    )
    written.append(reproduce_json)

    weights = manifest.get("weights") or {}
    if weights:
        sha_lines = [f"{sha}  {name}" for name, sha in weights.items()]
        sha_file = out / "SHA256SUMS"
        sha_file.write_text("\n".join(sha_lines) + "\n", encoding="utf-8")
        written.append(sha_file)

    readme = out / "README.md"
    readme.write_text(_render_readme(manifest), encoding="utf-8")
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

Abliterix will verify the manifest integrity, diff your environment against the
one recorded below, apply the exact published winning trial without searching,
and independently re-measure KL divergence and refusal count. Metric drift is a
hard verification failure.

## Key facts

| Field | Value |
| :---- | :---- |
| Base model | `{model.get("model_id")}` |
| Base revision | `{model.get("revision")}` |
| Independently reproducible | `{manifest.get("reproducible")}` |
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
