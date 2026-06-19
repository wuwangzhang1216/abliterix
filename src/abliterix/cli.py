# Abliterix — a derivative work of Heretic (https://github.com/p-e-w/heretic)
# Original work Copyright (C) 2025  Philipp Emanuel Weidmann (p-e-w)
# Modified work Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Command-line interface: banner, device detection, and main orchestration."""

import math
import os
import random
import sys
import time
import warnings
from importlib.metadata import version
from os.path import commonprefix

import optuna
import torch
import transformers
from accelerate.utils import (
    is_mlu_available,
    is_musa_available,
    is_npu_available,
    is_sdaa_available,
    is_xpu_available,
)
from optuna.exceptions import ExperimentalWarning
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.trial import TrialState
from pydantic import ValidationError
from questionary import Choice
from rich.traceback import install

from .analysis import ResidualAnalyzer
from .core.engine import SteeringEngine, load_tokenizer
from .data import load_prompt_dataset
from .eval.detector import RefusalDetector
from .eval.scorer import TrialScorer
from .interactive import show_interactive_results
from .optimizer import run_search
from .settings import AbliterixConfig
from .types import ChatMessage
from .util import (
    ask_choice,
    flush_memory,
    print,
    report_memory,
    set_seed,
    slugify_model_name,
)
from .types import SteeringMode
from .vectors import compute_steering_vectors


# ---------------------------------------------------------------------------
# Startup helpers
# ---------------------------------------------------------------------------


def _print_banner():
    v = version("abliterix")
    print(f"[magenta]█▀▀█░█▀▀▄░█░░░▀█▀░▀█▀░█▀▀░█▀▀▄░▀█▀░█░█[/]  v{v}")
    print("[magenta]█▄▄█░█▀▀▄░█░░░░█░░░█░░█▀▀░█▄▄▀░░█░░▄▀▄[/]")
    print(
        "[magenta]▀░░▀░▀▀▀░░▀▀▀░▀▀▀░░▀░░▀▀▀░▀░▀▀░▀▀▀░▀░▀[/]"
        "  [blue underline]https://github.com/wuwangzhang1216/abliterix[/]"
    )
    print()


def _load_reproduce_config(repro_path: str) -> "AbliterixConfig | None":
    """Load a reproduce.json manifest, report the environment diff, and rebuild config.

    Returns the reconstructed config, or None if the manifest is unusable.
    """
    from .reproducibility import check_environment, load_manifest

    try:
        manifest = load_manifest(repro_path)
    except (OSError, ValueError) as error:
        print(
            f"[red]Could not read reproduce manifest [bold]{repro_path}[/]: {error}[/]"
        )
        return None

    print(
        f"Reproducing from [bold]{repro_path}[/] "
        f"(abliterix v{manifest.get('abliterix_version')})"
    )

    findings = check_environment(manifest)
    if findings:
        print("Environment differences from the recorded run:")
        _colors = {
            "CRITICAL": "red bold",
            "HIGH": "red",
            "MEDIUM": "yellow",
            "LOW": "grey50",
        }
        for sev, msg in findings:
            print(f"  [{_colors.get(sev, 'white')}]\\[{sev}][/] {msg}")
        if any(sev in ("CRITICAL", "HIGH") for sev, _ in findings):
            print(
                "[yellow]High-severity differences may change the produced "
                "weights; exact bit-reproduction is not guaranteed.[/]"
            )
    else:
        print("[green]Environment matches the recorded run.[/]")

    stored = manifest.get("config")
    if not isinstance(stored, dict):
        print("[red]Manifest has no usable 'config' block; cannot reproduce.[/]")
        return None
    try:
        return AbliterixConfig.model_validate(stored)
    except ValidationError as error:
        print(f"[red]Recorded config failed validation: {error}[/]")
        return None


def _query_gpu_driver() -> str | None:
    """Best-effort NVIDIA driver version via nvidia-smi (None if unavailable)."""
    import subprocess

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


def _detect_devices():
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        total = sum(torch.cuda.mem_get_info(i)[1] for i in range(count))
        print(
            f"Detected [bold]{count}[/] CUDA device(s) ({total / (1024**3):.2f} GB total VRAM):"
        )
        # Report the runtime + driver so the common RunPod driver→CUDA→wheel
        # mismatch (PTX / Marlin-FP4 JIT failures) is visible at a glance and
        # easy to include in bug reports.
        is_hip = getattr(torch.version, "hip", None)
        api = "HIP" if is_hip else "CUDA"
        api_ver = (torch.version.hip if is_hip else torch.version.cuda) or "unknown"
        driver = None if is_hip else _query_gpu_driver()
        driver_str = f", driver [bold]{driver}[/]" if driver else ""
        print(f"* {api} runtime [bold]{api_ver}[/]{driver_str}")
        for i in range(count):
            vram = torch.cuda.mem_get_info(i)[1] / (1024**3)
            print(
                f"* GPU {i}: [bold]{torch.cuda.get_device_name(i)}[/] ({vram:.2f} GB)"
            )
    elif is_xpu_available():
        count = torch.xpu.device_count()
        print(f"Detected [bold]{count}[/] XPU device(s):")
        for i in range(count):
            print(f"* XPU {i}: [bold]{torch.xpu.get_device_name(i)}[/]")
    elif is_mlu_available():
        count = torch.mlu.device_count()  # ty:ignore[unresolved-attribute]
        print(f"Detected [bold]{count}[/] MLU device(s):")
        for i in range(count):
            print(f"* MLU {i}: [bold]{torch.mlu.get_device_name(i)}[/]")  # ty:ignore[unresolved-attribute]
    elif is_sdaa_available():
        count = torch.sdaa.device_count()  # ty:ignore[unresolved-attribute]
        print(f"Detected [bold]{count}[/] SDAA device(s):")
        for i in range(count):
            print(f"* SDAA {i}: [bold]{torch.sdaa.get_device_name(i)}[/]")  # ty:ignore[unresolved-attribute]
    elif is_musa_available():
        count = torch.musa.device_count()  # ty:ignore[unresolved-attribute]
        print(f"Detected [bold]{count}[/] MUSA device(s):")
        for i in range(count):
            print(f"* MUSA {i}: [bold]{torch.musa.get_device_name(i)}[/]")  # ty:ignore[unresolved-attribute]
    elif is_npu_available():
        print(f"NPU detected (CANN version: [bold]{torch.version.cann}[/])")  # ty:ignore[unresolved-attribute]
    elif torch.backends.mps.is_available():
        print("Detected [bold]1[/] MPS device (Apple Metal)")
    else:
        print(
            "[bold yellow]No GPU or other accelerator detected. Operations will be slow.[/]"
        )


def _configure_libraries():
    torch.set_grad_enabled(False)
    torch._dynamo.config.cache_size_limit = 64
    transformers.logging.set_verbosity_error()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=ExperimentalWarning)


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------


def _handle_existing_checkpoint(
    config: AbliterixConfig,
    existing_study,
    checkpoint_file: str,
    lock_obj,
    storage: JournalStorage,
) -> tuple[AbliterixConfig, JournalStorage] | None:
    """Prompt user (or auto-decide in batch mode) when a checkpoint exists.

    Returns ``(config, storage)`` to continue, or ``None`` to abort.
    """
    if config.non_interactive:
        if config.overwrite_checkpoint:
            print()
            print("[yellow]Non-interactive mode: overwriting existing checkpoint.[/]")
            os.unlink(checkpoint_file)
            backend = JournalFileBackend(checkpoint_file, lock_obj=lock_obj)
            return config, JournalStorage(backend)
        elif not existing_study.user_attrs["finished"]:
            print()
            print("[yellow]Non-interactive mode: continuing existing checkpoint.[/]")
            restored = AbliterixConfig.model_validate_json(
                existing_study.user_attrs["settings"],
            )
            # Preserve runtime flags that aren't part of the experiment config.
            restored.non_interactive = config.non_interactive
            restored.overwrite_checkpoint = config.overwrite_checkpoint
            return restored, storage
        else:
            print()
            print(
                "[red]Non-interactive mode: checkpoint already finished and "
                "overwrite_checkpoint=false. "
                "Set --overwrite-checkpoint to restart, or remove the checkpoint file.[/]"
            )
            return None

    choices = []

    if existing_study.user_attrs["finished"]:
        print()
        print(
            "[green]You have already processed this model.[/] "
            "You can show the results from the previous run, allowing you to export "
            "models or to run additional trials. Alternatively, you can ignore the "
            "previous run and start from scratch. This will delete the checkpoint "
            "file and all results from the previous run."
        )
        choices.append(
            Choice(title="Show the results from the previous run", value="continue")
        )
    else:
        print()
        print(
            "[yellow]You have already processed this model, but the run was interrupted.[/] "
            "You can continue the previous run from where it stopped. This will override "
            "any specified settings. Alternatively, you can ignore the previous run and "
            "start from scratch. This will delete the checkpoint file and all results "
            "from the previous run."
        )
        choices.append(Choice(title="Continue the previous run", value="continue"))

    choices += [
        Choice(title="Ignore the previous run and start from scratch", value="restart"),
        Choice(title="Exit program", value=""),
    ]

    print()
    choice = ask_choice("How would you like to proceed?", choices)

    if choice == "continue":
        config = AbliterixConfig.model_validate_json(
            existing_study.user_attrs["settings"],
        )
        return config, storage
    elif choice == "restart":
        os.unlink(checkpoint_file)
        backend = JournalFileBackend(checkpoint_file, lock_obj=lock_obj)
        return config, JournalStorage(backend)
    return None


# ---------------------------------------------------------------------------
# Auto-tuning
# ---------------------------------------------------------------------------


def _speculators_available() -> bool:
    """Check if the speculators library is installed and compatible."""
    try:
        from speculators.data_generation import VllmHiddenStatesGenerator  # noqa: F401

        return True
    except (ImportError, Exception):
        return False


def _vllm_hidden_states_available() -> bool:
    """Check if vLLM's native hidden state extraction API is available (>= 0.17).

    Honors ``AX_DISABLE_VLLM_HS=1`` to force the slow HF extraction path.  This
    is required when the user wants to run :class:`VLLMMoEEditor` router
    suppression — that editor needs ``safety_experts`` computed by
    ``engine.identify_safety_experts``, which only runs if the HF model was
    actually loaded (i.e. we did NOT take the vLLM-native fast path).
    """
    if os.environ.get("AX_DISABLE_VLLM_HS", "") == "1":
        return False
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (  # noqa: F401
            ExampleHiddenStatesConnector,
        )

        return True
    except (ImportError, Exception):
        return False


def _auto_batch_size(
    engine: SteeringEngine, benign_msgs: list[ChatMessage], config: AbliterixConfig
) -> int:
    """Determine optimal inference batch size via exponential search."""
    print()
    print("Determining optimal batch size...")

    def _try(bs: int) -> float | None:
        test = benign_msgs * math.ceil(bs / len(benign_msgs))
        test = test[:bs]
        try:
            engine.generate_text(test)  # warmup
            t0 = time.perf_counter()
            responses = engine.generate_text(test)
            t1 = time.perf_counter()
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            return None
        tok_counts = [len(engine.tokenizer.encode(r)) for r in responses]
        return sum(tok_counts) / (t1 - t0)

    batch_size = 1
    results: dict[int, float] = {}

    while batch_size <= config.inference.max_batch_size:
        print(f"* Trying batch size [bold]{batch_size}[/]... ", end="")
        throughput = _try(batch_size)
        if throughput is None:
            if batch_size == 1:
                raise RuntimeError(
                    "Batch size 1 failed — cannot determine optimal batch size."
                )
            print("[red]Failed[/]")
            break
        print(f"[green]Ok[/] ([bold]{throughput:.0f}[/] tokens/s)")
        results[batch_size] = throughput
        batch_size *= 2

    # Try midpoint between the two best-performing sizes.
    if len(results) >= 2:
        ranked = sorted(results.items(), key=lambda x: x[1], reverse=True)
        best_bs = ranked[0][0]
        second_bs = ranked[1][0]
        mid = (best_bs + second_bs) // 2
        if mid != best_bs and mid != second_bs and mid not in results:
            print(f"* Trying batch size [bold]{mid}[/]... ", end="")
            throughput = _try(mid)
            if throughput is not None:
                print(f"[green]Ok[/] ([bold]{throughput:.0f}[/] tokens/s)")
                results[mid] = throughput
            else:
                print("[red]Failed[/]")

    optimal = max(results, key=lambda k: results[k])
    print(f"* Chosen batch size: [bold]{optimal}[/]")
    return optimal


def _detect_response_prefix(
    engine: SteeringEngine,
    benign_msgs: list[ChatMessage],
    target_msgs: list[ChatMessage],
):
    """Detect and set a common response prefix, handling CoT suppression."""
    print()
    print("Checking for common response prefix...")
    sample = benign_msgs[:10] + target_msgs[:10]
    responses = engine.generate_text_batched(sample)

    # os.path.commonprefix is a naive string operation (despite the module name)
    # which is exactly what we need. Trailing spaces are trimmed to prevent
    # uncommon tokenisation artefacts.
    engine.response_prefix = commonprefix(responses).rstrip(" ")

    if engine.response_prefix:
        print(
            f"* Candidate prefix from 20 prompts: [bold]{engine.response_prefix!r}[/]"
        )
        # Check for known CoT/thinking patterns BEFORE validation, because
        # the larger validation sample may dilute the common prefix (e.g.
        # mixed-language Harmony responses share only the channel tokens).
        _KNOWN_COT_PREFIXES = {
            "<think>": "<think></think>",
            "<thought>": "<thought></thought>",
            "[THINK]": "[THINK][/THINK]",
            "<|channel|>analysis<|message|>": (
                "<|channel|>analysis<|message|><|end|><|start|>assistant"
                "<|channel|>final<|message|>"
            ),
        }
        matched_early = False
        for pattern, replacement in _KNOWN_COT_PREFIXES.items():
            if engine.response_prefix.startswith(pattern):
                engine.response_prefix = replacement
                matched_early = True
                break

        if not matched_early:
            print("* Validating with larger sample...")
            expanded = benign_msgs[:25] + target_msgs[:25]
            engine.response_prefix = commonprefix(
                engine.generate_text_batched(expanded),
            ).rstrip(" ")
    else:
        cot_tokens = {"<think>", "<thought>", "[THINK]"}
        extra_special = set(
            engine.tokenizer.special_tokens_map.get("additional_special_tokens", []),
        )
        if cot_tokens & extra_special:
            print("* CoT special tokens detected, retrying with larger sample...")
            expanded = benign_msgs[:50] + target_msgs[:50]
            engine.response_prefix = commonprefix(
                engine.generate_text_batched(expanded),
            ).rstrip(" ")

    recheck = False
    if engine.response_prefix:
        recheck = True
        if engine.response_prefix.startswith("<think>"):
            engine.response_prefix = "<think></think>"
        elif engine.response_prefix.startswith("<|channel|>analysis<|message|>"):
            engine.response_prefix = (
                "<|channel|>analysis<|message|><|end|><|start|>assistant"
                "<|channel|>final<|message|>"
            )
        elif engine.response_prefix.startswith("<thought>"):
            engine.response_prefix = "<thought></thought>"
        elif engine.response_prefix.startswith("[THINK]"):
            engine.response_prefix = "[THINK][/THINK]"
        else:
            recheck = False

    if engine.response_prefix:
        print(f"* Prefix found: [bold]{engine.response_prefix!r}[/]")
    else:
        print("* None found")

    if recheck:
        print("* Rechecking with prefix...")
        responses = engine.generate_text_batched(sample)
        extra = commonprefix(responses).rstrip(" ")
        if extra:
            engine.response_prefix += extra
            print(f"* Extended prefix found: [bold]{engine.response_prefix!r}[/]")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run():
    # Launch the Gradio Web UI if requested (before config parsing).
    if "--ui" in sys.argv:
        sys.argv.remove("--ui")
        from .webui import launch_ui

        launch_ui()
        return

    # Reduce memory fragmentation on multi-GPU setups.
    if (
        "PYTORCH_ALLOC_CONF" not in os.environ
        and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ
    ):
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    _print_banner()

    # Reproduce mode: rebuild the config from a published reproduce.json and
    # verify the current environment before re-running. Extracted before the
    # --model shorthand handling so a model id is not required.
    repro_path: str | None = None
    if "--reproduce" in sys.argv:
        idx = sys.argv.index("--reproduce")
        if idx + 1 >= len(sys.argv):
            print("[red]--reproduce requires a path to a reproduce.json file.[/]")
            return
        repro_path = sys.argv[idx + 1]
        del sys.argv[idx : idx + 2]

    if repro_path is None:
        # CLI shorthands: map --model X to --model.model-id X so that users
        # do not need to type the full nested path for the most common flags.
        _cli_aliases = {"--model": "--model.model-id"}
        for short, full in _cli_aliases.items():
            for i, arg in enumerate(sys.argv):
                if arg == short:
                    sys.argv[i] = full

        # Infer --model.model-id flag if the last argument looks like a model identifier.
        if (
            len(sys.argv) > 1
            and "--model.model-id" not in sys.argv
            and not sys.argv[-1].startswith("-")
        ):
            sys.argv.insert(-1, "--model.model-id")

    if repro_path is not None:
        config = _load_reproduce_config(repro_path)
        if config is None:
            return
    else:
        try:
            config = AbliterixConfig()  # ty:ignore[missing-argument]
        except ValidationError as error:
            print(
                f"[red]Configuration contains [bold]{error.error_count()}[/] errors:[/]"
            )
            for err in error.errors():
                print(f"[bold]{err['loc'][0]}[/]: [yellow]{err['msg']}[/]")
            print()
            print(
                "Run [bold]abliterix --help[/] or see [bold]abliterix.toml[/] for details "
                "about configuration parameters."
            )
            return

    _detect_devices()
    _configure_libraries()

    # Resolve + apply the global seed (random/numpy/torch) for reproducibility.
    # When unset, draw one and print it so the run can be reproduced verbatim.
    _seed_was_random = config.seed is None
    if _seed_was_random:
        config.seed = random.randint(0, 2**31 - 1)
    set_seed(config.seed)
    print(
        f"Global seed: [bold]{config.seed}[/]"
        + (" [grey50](randomly chosen)[/]" if _seed_was_random else "")
    )

    os.makedirs(config.optimization.checkpoint_dir, exist_ok=True)

    checkpoint_file = os.path.join(
        config.optimization.checkpoint_dir,
        slugify_model_name(config.model.model_id) + ".jsonl",
    )

    lock_obj = JournalFileOpenLock(checkpoint_file)
    backend = JournalFileBackend(checkpoint_file, lock_obj=lock_obj)
    storage = JournalStorage(backend)

    try:
        existing = storage.get_all_studies()[0]
    except IndexError:
        existing = None

    if existing is not None and config.model.evaluate_model_id is None:
        result = _handle_existing_checkpoint(
            config,
            existing,
            checkpoint_file,
            lock_obj,
            storage,
        )
        if result is None:
            return
        config, storage = result

    # Load steering-vector source datasets (needed early for speculators path).
    print()
    print(f"Loading benign prompts from [bold]{config.benign_prompts.dataset}[/]...")
    benign_msgs = load_prompt_dataset(config, config.benign_prompts)
    print(f"* [bold]{len(benign_msgs)}[/] prompts loaded")

    print()
    print(f"Loading target prompts from [bold]{config.target_prompts.dataset}[/]...")
    target_msgs = load_prompt_dataset(config, config.target_prompts)
    print(f"* [bold]{len(target_msgs)}[/] prompts loaded")

    # ----- Fast hidden state extraction (TP backend only) -----
    # Priority: 1) vLLM native extract_hidden_states (>= 0.17)
    #           2) speculators + vLLM (if compatible)
    #           3) Fall back to HF pipeline parallelism (slow)
    _precomputed_benign_states = None
    _precomputed_target_states = None
    if config.model.backend in ("vllm", "sglang") and _vllm_hidden_states_available():
        from .core.vllm_hidden_states import (
            extract_hidden_states_vllm,
            is_model_supported,
        )

        if not is_model_supported(config):
            print()
            print(
                "[yellow]vLLM extract_hidden_states does not support this model type. "
                "Falling back to HF pipeline parallelism.[/]"
            )
            # Skip to HF fallback below
            _vllm_hs_ok = False
        else:
            _vllm_hs_ok = True
    else:
        _vllm_hs_ok = False

    if _vllm_hs_ok:
        print()
        print("[bold]Fast hidden state extraction (vLLM native TP)[/]")
        try:
            # Extract benign + target in a single vLLM load.  A reload per set
            # would pay the MooseFS shard pull (~2.5 min on 15-shard MoEs)
            # twice for no reason.
            print("* Extracting residuals for benign + target prompts...")
            _hs = extract_hidden_states_vllm(
                config,
                {"benign": benign_msgs, "target": target_msgs},
            )
            _precomputed_benign_states = _hs["benign"]
            _precomputed_target_states = _hs["target"]
            del _hs
            print()
        except Exception as exc:
            print(
                f"\n[yellow bold]vLLM extract_hidden_states failed: {exc}[/]\n"
                "Falling back to next available extraction method..."
            )
            _precomputed_benign_states = None
            _precomputed_target_states = None
            flush_memory()
            _vllm_hs_ok = False

    if (
        _precomputed_benign_states is None
        and config.model.backend in ("vllm", "sglang")
        and _speculators_available()
    ):
        from .core.speculators_backend import extract_hidden_states_speculators

        print()
        print("[bold]Fast hidden state extraction (speculators + vLLM TP)[/]")
        print("* Extracting residuals for benign prompts...")
        _precomputed_benign_states = extract_hidden_states_speculators(
            config,
            benign_msgs,
        )
        print("* Extracting residuals for target prompts...")
        _precomputed_target_states = extract_hidden_states_speculators(
            config,
            target_msgs,
        )
        print()
    elif _precomputed_benign_states is None and config.model.backend in (
        "vllm",
        "sglang",
    ):
        print()
        print(
            "[yellow bold]WARNING: No fast hidden state extraction available![/]\n"
            "  vLLM native API requires >= 0.17, speculators not installed.\n"
            "  Phase 1 will use HF pipeline parallelism (~4 tok/s — 10-15x slower)."
        )
        print()

    # When speculators handled hidden state extraction AND we're using a TP
    # backend, the HF model is not needed for Phase 1 at all.  Skip loading
    # it to save 3+ minutes on large MoE models.
    _skip_hf_model = (
        _precomputed_benign_states is not None
        and config.model.backend in ("vllm", "sglang")
    )

    if _skip_hf_model:
        print()
        print(
            "[bold green]Fast path: skipping HF model load[/] "
            "(speculators handled hidden states, projections from safetensors)"
        )
        # Create a lightweight engine with only tokenizer (no model weights).
        engine = SteeringEngine.__new__(SteeringEngine)
        engine.config = config
        engine.response_prefix = ""
        engine.needs_reload = False
        engine._dequant_cache = {}
        engine._cached_n_layers = None
        engine._cached_components = None
        engine._is_native_fp8 = False
        engine.tokenizer = load_tokenizer(
            config.model.model_id,
            trust_remote_code=config.model.trust_remote_code,
        )
        if engine.tokenizer.pad_token is None:
            engine.tokenizer.pad_token = engine.tokenizer.eos_token
        engine.tokenizer.padding_side = "left"
        engine.model = None
        engine.max_memory = None
        engine.trusted_models = {}
    else:
        engine = SteeringEngine(config)

    print()
    report_memory()

    # For TP backends, skip expensive HF-based auto batch size tuning and
    # response prefix detection.  These will be done after the fast TP
    # backend loads (or deferred entirely).
    if config.model.backend in ("vllm", "sglang"):
        if config.inference.batch_size == 0:
            config.inference.batch_size = config.inference.max_batch_size
            print(
                f"* TP backend: skipping HF batch-size tuning, "
                f"using batch_size={config.inference.batch_size}"
            )
        if not _skip_hf_model:
            # HF model is loaded — do minimal prefix detection.
            print()
            print("Checking for common response prefix (minimal for TP backend)...")
            _mini_sample = benign_msgs[:1] + target_msgs[:1]
            responses = engine.generate_text_batched(_mini_sample, max_new_tokens=20)
            from os.path import commonprefix

            engine.response_prefix = commonprefix(responses).rstrip(" ")
            if engine.response_prefix:
                _COT = {
                    "<think>": "<think></think>",
                    "<thought>": "<thought></thought>",
                }
                for pat, repl in _COT.items():
                    if engine.response_prefix.startswith(pat):
                        engine.response_prefix = repl
                        break
                print(f"* Prefix found: [bold]{engine.response_prefix!r}[/]")
            else:
                print("* None found")
        # else: prefix detection deferred to after TP backend loads
    else:
        if config.inference.batch_size == 0:
            config.inference.batch_size = _auto_batch_size(engine, benign_msgs, config)
        _detect_response_prefix(engine, benign_msgs, target_msgs)

    detector = RefusalDetector(config)
    try:
        # For TP backends, defer baseline capture until after the fast backend
        # is loaded.  Otherwise baseline generation runs on HF pipeline
        # parallelism (~4 tok/s for 200 prompts = ~40 min wasted).
        _defer = config.model.backend in ("vllm", "sglang")
        scorer = TrialScorer(config, engine, detector, defer_baseline=_defer)

        # Evaluation-only mode: load a second model and score it.
        if config.model.evaluate_model_id is not None:
            print()
            print(f"Loading model [bold]{config.model.evaluate_model_id}[/]...")
            config.model.model_id = config.model.evaluate_model_id
            engine.restore_baseline()
            print("* Evaluating...")
            scorer.score_trial(engine)
            return

        # Compute steering vectors from residual streams.
        print()
        print("Computing per-layer steering vectors...")
        if _precomputed_benign_states is not None:
            print("* Using pre-extracted residuals (speculators)")
            benign_states = _precomputed_benign_states
            target_states = _precomputed_target_states
            del _precomputed_benign_states, _precomputed_target_states
        else:
            print("* Extracting residuals for benign prompts...")
            benign_states = engine.extract_hidden_states_batched(benign_msgs)
            print("* Extracting residuals for target prompts...")
            target_states = engine.extract_hidden_states_batched(target_msgs)

        print(f"* Vector method: [bold]{config.steering.vector_method.value}[/]")

        if config.iterative.enabled:
            from .iterative import iterative_abliterate

            vectors, iter_stats = iterative_abliterate(
                engine,
                benign_msgs,
                target_msgs,
                config,
                benign_states=benign_states,
                target_states=target_states,
            )
            # Model restored inside iterative_abliterate.
            # Re-extract clean states for discriminative layer selection / analysis.
            if config.steering.discriminative_layer_selection:
                print(
                    "* Re-extracting clean residuals for discriminative layer selection..."
                )
                benign_states = engine.extract_hidden_states_batched(benign_msgs)
                target_states = engine.extract_hidden_states_batched(target_msgs)
        else:
            vectors = compute_steering_vectors(
                benign_states,
                target_states,
                config.steering.vector_method,
                config.steering.orthogonal_projection,
                winsorize=config.steering.winsorize_vectors,
                winsorize_quantile=config.steering.winsorize_quantile,
                projected_abliteration=config.steering.projected_abliteration,
                ot_components=config.steering.ot_components,
                n_directions=config.steering.n_directions,
                sra_base_method=config.steering.sra_base_method,
                sra_n_atoms=config.steering.sra_n_atoms,
                sra_ridge_alpha=config.steering.sra_ridge_alpha,
                ablate_harmfulness_direction=config.steering.ablate_harmfulness_direction,
                harmfulness_layer_band=tuple(config.steering.harmfulness_layer_band),
                som_grid_h=config.steering.som_grid_h,
                som_grid_w=config.steering.som_grid_w,
                som_n_iters=config.steering.som_n_iters,
                som_initial_lr=config.steering.som_initial_lr,
                som_seed=config.steering.som_seed,
                sae_path=config.steering.sae_path,
                sae_layer=config.steering.sae_layer,
                sae_top_k=config.steering.sae_top_k,
            )

        analyzer = ResidualAnalyzer(config, engine, benign_states, target_states)

        if config.display.print_residual_geometry:
            analyzer.print_residual_geometry()
        if config.display.plot_residuals:
            analyzer.plot_residuals()

        # Train SVF concept scorers if using Steering Vector Fields mode.
        if config.steering.steering_mode == SteeringMode.VECTOR_FIELD:
            from .svf import train_concept_scorers

            print()
            print("Training SVF concept scorers...")
            engine._concept_scorers = train_concept_scorers(
                benign_states,
                target_states,
                hidden_dim=benign_states.shape[2],
                n_epochs=config.steering.svf_scorer_epochs,
                lr=config.steering.svf_scorer_lr,
                hidden_dim_scorer=config.steering.svf_scorer_hidden,
            )
            print(
                f"* Trained scorers for [bold]{len(engine._concept_scorers)}[/] layers"
            )

        # Cliff-head ablation (Bao et al. 2025, arXiv:2510.06036).
        # Surgically scale toward zero the o_proj columns of the attention
        # heads most aligned with the refusal direction. Applied once before
        # the optimizer search, on the HF model. Skipped when running under
        # the fast-extraction vLLM path that unloads the HF model (cliff-head
        # editing of vLLM tensors is a separate path not yet implemented).
        if config.steering.cliff_head_ablation:
            if engine.model is None:
                print(
                    "[yellow]Cliff-head ablation requested but HF model is "
                    "not loaded — skipping. This typically means the "
                    "fast-extraction vLLM path is active; cliff-head support "
                    "for vLLM-only mode is on the roadmap.[/]"
                )
            else:
                from .cliff_head import run_cliff_head_ablation

                print()
                print(
                    "Applying cliff-head ablation "
                    f"(top {config.steering.cliff_head_top_k_frac:.1%}, "
                    f"strength {config.steering.cliff_head_strength:.2f})..."
                )
                n_modified, heads = run_cliff_head_ablation(
                    engine,
                    vectors,
                    top_k_frac=config.steering.cliff_head_top_k_frac,
                    strength=config.steering.cliff_head_strength,
                )
                if n_modified:
                    by_layer: dict[int, int] = {}
                    for h in heads:
                        by_layer[h.layer] = by_layer.get(h.layer, 0) + 1
                    top_layers = sorted(
                        by_layer.items(), key=lambda x: x[1], reverse=True
                    )[:5]
                    layer_brief = ", ".join(
                        f"L{layer}×{count}" for layer, count in top_layers
                    )
                    print(
                        f"* Ablated [bold]{n_modified}[/] attention heads "
                        f"across {len(by_layer)} layers (top: {layer_brief})"
                    )
                else:
                    print(
                        "[yellow]Cliff-head ablation matched no heads — "
                        "check num_attention_heads / head_dim divisibility "
                        "for this architecture.[/]"
                    )

        # Keep residual states if needed for discriminative layer selection
        # or angular steering; otherwise free memory.
        _keep_states = (
            config.steering.discriminative_layer_selection
            or config.steering.steering_mode.value != "lora"
        )
        if not _keep_states:
            del benign_states, target_states
            benign_states = target_states = None
        del analyzer
        flush_memory()

        # Profile MoE expert routing if applicable.
        # For vLLM we still profile here (HF phase, before unload) so that
        # the TP trial loop can apply router-weight suppression on the
        # loaded vLLM model via collective_rpc (see VLLMMoEEditor).
        # SGLang path does not yet have an equivalent editor and keeps the
        # original skip behaviour.
        safety_experts: dict[int, list[tuple[int, float]]] | None = None
        # Only do HF router profiling when the HF model is actually loaded.
        # Under vLLM fast extraction path, engine.model is None (lightweight
        # engine) — VLLMMoEEditor does its own profiling via collective_rpc
        # during the phase transition below.
        if (
            engine.model is not None
            and engine.has_expert_routing()
            and config.model.backend != "sglang"
        ):
            print()
            print("Profiling MoE expert activations...")
            if config.experts.profiling_method == "safex":
                from .safex import identify_safety_experts_safex

                safety_experts = identify_safety_experts_safex(
                    engine,
                    benign_msgs,
                    target_msgs,
                    variance_penalty=config.experts.safex_variance_penalty,
                )
            else:
                safety_experts = engine.identify_safety_experts(
                    benign_msgs, target_msgs
                )

        # ----- TP backend: Phase transition (vLLM or SGLang) -----
        tp_gen = None
        projection_cache = None
        if config.model.backend in ("vllm", "sglang"):
            from .core.vllm_backend import ProjectionCache

            backend_name = config.model.backend.upper()
            print()
            print(f"[bold]Phase transition: HF → {backend_name}[/]")

            # Build projection cache.  If the HF model is loaded (needed for
            # non-speculators path), use it.  Otherwise read weights directly
            # from safetensors on disk — avoids the 3+ min HF model load.
            from pathlib import Path

            model_path = Path(config.model.model_id)
            use_safetensors_cache = (
                config.model.backend == "vllm"
                and model_path.is_dir()
                and (model_path / "model.safetensors.index.json").exists()
            )
            skip_projection_cache = config.model.disable_lora
            if skip_projection_cache:
                print(
                    "* LoRA disabled: skipping projection cache (router-only steering)"
                )
                if engine.model is not None:
                    print("* Unloading HF model...")
                    engine.prepare_for_unload()
                    engine.model = None
            elif use_safetensors_cache:
                print("* Building LoRA projection cache...")
                # For large MoE models (MiniMax-M2: 256 experts × 62 layers),
                # the safetensors path preserves every expert entry and avoids
                # repeated module-tree scans while the HF model is still loaded.
                projection_cache = ProjectionCache.build_from_safetensors(
                    config,
                    vectors,
                )
                if engine.model is not None:
                    print("* Unloading HF model...")
                    engine.prepare_for_unload()
                    engine.model = None
            elif engine.model is not None:
                print("* Building LoRA projection cache...")
                projection_cache = ProjectionCache.build(engine, vectors)
                # Unload HF model to free VRAM for the TP backend.
                print("* Unloading HF model...")
                engine.prepare_for_unload()
                engine.model = None
            else:
                print("* Building LoRA projection cache...")
                # HF model was never loaded (speculators handled everything).
                # Build projections directly from safetensors files.
                projection_cache = ProjectionCache.build_from_safetensors(
                    config,
                    vectors,
                )
            flush_memory()
            report_memory()

            # Load model with tensor parallelism.
            print()
            if config.model.backend == "sglang":
                print("Loading model with SGLang (TP + LoRA overlap loading)...")
                from .core.sglang_backend import SGLangGenerator

                tp_gen = SGLangGenerator(config)
            else:
                print("Loading model with vLLM tensor parallelism...")
                from .core.vllm_backend import VLLMGenerator

                tp_gen = VLLMGenerator(config)

            # Attach TP generator and projection cache to engine
            # so the optimizer can use them.
            engine._vllm_gen = tp_gen
            engine._projection_cache = projection_cache
            engine._current_adapter_path = None  # baseline = no adapter

            # Attach MoE router editor so the optimizer can apply router
            # suppression per trial via collective_rpc.  Only on vLLM
            # backend (SGLang has no equivalent editor yet).
            if config.model.backend == "vllm" and hasattr(tp_gen, "set_moe_editor"):
                # If HF profiling was skipped (fast vLLM-native extraction
                # path → no HF model loaded), profile directly on the TP
                # vLLM instance via collective_rpc-attached router hooks.
                if safety_experts is None:
                    from .core.vllm_moe_editor import (
                        VLLMMoEEditor,
                        profile_safety_experts_by_weight,
                        profile_safety_experts_vllm,
                    )

                    # Cheap probe: construct editor with empty safety_experts
                    # just to discover whether any layer exposes a router.
                    _probe_ed = VLLMMoEEditor(tp_gen.llm, {})
                    _probe_ed.probe()
                    if _probe_ed._router_layers:
                        # Derive top_k from the model config (num_experts_per_tok).
                        try:
                            from transformers import AutoConfig

                            _auto_cfg = AutoConfig.from_pretrained(
                                config.model.model_id,
                                trust_remote_code=config.model.trust_remote_code
                                or False,
                            )
                            _text_cfg = getattr(_auto_cfg, "text_config", _auto_cfg)
                            _top_k = int(getattr(_text_cfg, "num_experts_per_tok", 4))
                        except Exception:
                            _top_k = 4

                        print(
                            f"* Profiling MoE safety experts via vLLM "
                            f"(top_k={_top_k}, {len(_probe_ed._router_layers)} "
                            f"router layers)..."
                        )
                        safety_experts = profile_safety_experts_vllm(
                            tp_gen.llm,
                            benign_msgs,
                            target_msgs,
                            tp_gen.tokenizer,
                            top_k=_top_k,
                        )

                    # Fallback: vLLM's fused TRITON MxFP4 MoE kernel bypasses
                    # the router nn.Module's forward, so hook-based profiling
                    # returns empty counts.  Rank experts by router-weight
                    # alignment with the per-layer refusal direction instead.
                    if (
                        _probe_ed._router_layers
                        and not safety_experts
                        and vectors is not None
                    ):
                        print(
                            "* Hook-based profiling returned empty — falling "
                            "back to router-weight alignment heuristic..."
                        )
                        safety_experts = profile_safety_experts_by_weight(
                            tp_gen.llm,
                            vectors,
                        )

                if safety_experts:
                    print(
                        f"* Attaching MoE router editor ({len(safety_experts)} "
                        f"MoE layers)..."
                    )
                    tp_gen.set_moe_editor(safety_experts)  # ty:ignore[call-non-callable]

            # In-place editing path: attach attention and, when requested,
            # expert editors so the optimizer trial loop edits vLLM weights
            # directly (no LoRA adapter).
            # Requires TRITON backend (FLASHINFER_TRTLLM repacks w2_weight
            # into an opaque block layout) and enforce_eager=True. See
            # VllmConfig.use_in_place_editing for env-var requirements.
            if (
                config.model.backend == "vllm"
                and config.model.use_in_place_editing
                and hasattr(tp_gen, "set_expert_editor")
            ):
                wants_expert_editor = (
                    "mlp.down_proj" not in config.steering.disabled_components
                )
                _hidden = getattr(engine, "hidden_size", None)
                _transposed = bool(
                    getattr(engine, "_fused_down_proj_transposed", False)
                )
                if wants_expert_editor and _hidden is None:
                    try:
                        from transformers import AutoConfig as _AC

                        _c = _AC.from_pretrained(
                            config.model.model_id,
                            trust_remote_code=config.model.trust_remote_code or False,
                        )
                        _hidden = int(
                            getattr(
                                getattr(_c, "text_config", _c),
                                "hidden_size",
                                0,
                            )
                        )
                    except Exception:
                        _hidden = 0
                print("* Attaching vLLM in-place attention editor...")
                tp_gen.set_attention_editor()  # ty:ignore[call-non-callable]
                if wants_expert_editor:
                    if _hidden and _hidden > 0:
                        print(
                            f"* Attaching vLLM in-place expert editor "
                            f"(hidden={_hidden}, transposed={_transposed})..."
                        )
                        tp_gen.set_expert_editor(  # ty:ignore[call-non-callable]
                            hidden_dim=_hidden, transposed=_transposed
                        )
                    else:
                        print(
                            "  [yellow]use_in_place_editing=true but hidden_size "
                            "could not be resolved — skipping expert editor.[/]"
                        )

            # If engine has no model (lightweight mode), populate cached
            # metadata from the projection cache so optimizer can query it.
            if engine.model is None and engine._cached_n_layers is None:
                if projection_cache is not None:
                    engine._cached_n_layers = (
                        max(projection_cache.projections.keys()) + 1
                    )
                    engine._cached_components = sorted(
                        {
                            comp
                            for layer in projection_cache.projections.values()
                            for comp in layer
                        }
                    )
                else:
                    engine._cached_n_layers = int(vectors.shape[0] - 1)
                    engine._cached_components = []
            # If an in-place expert editor was attached, "mlp.down_proj" must
            # appear in cached_components so the optimizer generates an EGA
            # steering profile for it.  The fused expert tensor is NOT in the
            # projection cache (EGA computes projections inline from the
            # steering vector), so it is absent from the block above.
            if (
                engine._cached_components is not None
                and "mlp.down_proj" not in engine._cached_components
                and getattr(tp_gen, "expert_editor", None) is not None
                and len(getattr(tp_gen.expert_editor, "_moe_layers", [])) > 0
            ):
                engine._cached_components = sorted(
                    set(engine._cached_components) | {"mlp.down_proj"}
                )
                print("  * Injected 'mlp.down_proj' into steerable components (EGA)")

            # If an in-place attention editor was attached, q/k/v/o_proj must
            # all appear in cached_components so the optimizer generates
            # steering profiles for every attention projection. ``_apply_direct_
            # steering_vllm`` (steering.py) dispatches to VLLMAttentionEditor
            # which handles fused qkv_proj slicing on TP workers — but only if
            # the profiles exist. Without this injection, ProjectionCache only
            # contributes ``attn.o_proj`` (its build loop skips q/k/v because
            # ``d_out != hidden_dim`` makes ``sv @ W`` dimensionally invalid),
            # leaving 3/4 of attention unsteered.
            if (
                engine._cached_components is not None
                and getattr(tp_gen, "attention_editor", None) is not None
                and len(getattr(tp_gen.attention_editor, "_attn_layers", set())) > 0
            ):
                _attn_needed = {
                    "attn.q_proj",
                    "attn.k_proj",
                    "attn.v_proj",
                    "attn.o_proj",
                }
                _attn_missing = _attn_needed - set(engine._cached_components)
                if _attn_missing:
                    engine._cached_components = sorted(
                        set(engine._cached_components) | _attn_missing
                    )
                    print(
                        f"  * Injected {sorted(_attn_missing)} into steerable "
                        f"components (fused qkv_proj slicing on TP workers)"
                    )

            # Detect response prefix via the fast TP backend (if not done earlier).
            if not engine.response_prefix:
                print("* Detecting response prefix via TP backend...")
                _mini = benign_msgs[:2] + target_msgs[:2]
                _resps = tp_gen.generate_text_batched(_mini, max_new_tokens=20)
                from os.path import commonprefix as _cp

                engine.response_prefix = _cp(_resps).rstrip(" ")
                if engine.response_prefix:
                    _COT = {
                        "<think>": "<think></think>",
                        "<thought>": "<thought></thought>",
                    }
                    for _pat, _repl in _COT.items():
                        if engine.response_prefix.startswith(_pat):
                            engine.response_prefix = _repl
                            break
                    print(f"  Prefix: [bold]{engine.response_prefix!r}[/]")
                else:
                    print("  None found")

            # Capture full baseline (logprobs, response lengths, refusal count)
            # using the TP backend.  This was deferred from TrialScorer init
            # to avoid running expensive generation on HF pipeline parallelism.
            print(f"* Capturing baseline metrics with {backend_name}...")
            scorer._capture_baseline(engine)
            print("  [green]Ok[/]")

        # Safety check: if TP backend was requested but failed to load,
        # the optimizer would silently fall back to HF pipeline parallelism
        # (~4 tok/s instead of ~500 tok/s).  Abort early.
        if config.model.backend in ("vllm", "sglang"):
            if getattr(engine, "_vllm_gen", None) is None:
                raise RuntimeError(
                    f"TP backend '{config.model.backend}' was requested but failed "
                    f"to load.  Refusing to fall back to HF pipeline parallelism "
                    f"(would be ~100x slower).  Fix the backend installation and retry."
                )

        # Precompute alternative steering tensors when the optimiser is
        # asked to search the harmfulness ⊥ refusal flag as a categorical.
        # The pair-variant uses the existing harmfulness extractor and
        # plugs into the multi-direction code path downstream.
        _vector_variants: dict | None = None
        if (
            config.steering.search_harmfulness_direction
            and benign_states is not None
            and target_states is not None
        ):
            from .harmfulness import extract_harm_refusal_pair

            print()
            print(
                "Precomputing harmfulness-pair variant for the optimiser "
                "(search_harmfulness_direction=true)..."
            )
            pair_vectors = extract_harm_refusal_pair(
                benign_states,
                target_states,
                layer_band=tuple(config.steering.harmfulness_layer_band),
                orthogonal_projection=config.steering.orthogonal_projection,
                projected_abliteration=config.steering.projected_abliteration,
            )
            _vector_variants = {
                "single": vectors,
                "harmfulness_pair": pair_vectors,
            }

        study = run_search(
            config,
            engine,
            scorer,
            vectors,
            safety_experts,
            storage,
            benign_states=benign_states,
            target_states=target_states,
            steering_vector_variants=_vector_variants,
        )

        if config.non_interactive:
            completed = sum(1 for t in study.trials if t.state == TrialState.COMPLETE)
            print()
            print(
                f"[bold green]Non-interactive mode: optimization finished with "
                f"{completed} completed trials.[/]"
            )
            return

        show_interactive_results(
            study,
            config,
            engine,
            scorer,
            vectors,
            safety_experts,
            storage,
        )
    finally:
        detector.close()


def main():
    install()  # Rich traceback handler.

    try:
        run()
    except BaseException as error:
        if isinstance(error, KeyboardInterrupt) or isinstance(
            error.__context__,
            KeyboardInterrupt,
        ):
            print()
            print("[red]Shutting down...[/]")
        else:
            raise
