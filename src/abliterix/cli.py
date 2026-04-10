# Abliterix — a derivative work of Heretic (https://github.com/p-e-w/heretic)
# Original work Copyright (C) 2025  Philipp Emanuel Weidmann (p-e-w)
# Modified work Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Command-line interface: banner, device detection, and main orchestration."""

import math
import os
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


def _detect_devices():
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        total = sum(torch.cuda.mem_get_info(i)[1] for i in range(count))
        print(
            f"Detected [bold]{count}[/] CUDA device(s) ({total / (1024**3):.2f} GB total VRAM):"
        )
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
    """Check if vLLM's native hidden state extraction API is available (>= 0.17)."""
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import ExampleHiddenStatesConnector  # noqa: F401
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

    try:
        config = AbliterixConfig()  # ty:ignore[missing-argument]
    except ValidationError as error:
        print(f"[red]Configuration contains [bold]{error.error_count()}[/] errors:[/]")
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
        from .core.vllm_hidden_states import extract_hidden_states_vllm, is_model_supported
        if not is_model_supported(config):
            print()
            print(
                f"[yellow]vLLM extract_hidden_states does not support this model type. "
                f"Falling back to HF pipeline parallelism.[/]"
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
        print("* Extracting residuals for benign prompts...")
        _precomputed_benign_states = extract_hidden_states_vllm(
            config, benign_msgs,
        )
        print("* Extracting residuals for target prompts...")
        _precomputed_target_states = extract_hidden_states_vllm(
            config, target_msgs,
        )
        print()
    elif config.model.backend in ("vllm", "sglang") and _speculators_available():
        from .core.speculators_backend import extract_hidden_states_speculators

        print()
        print("[bold]Fast hidden state extraction (speculators + vLLM TP)[/]")
        print("* Extracting residuals for benign prompts...")
        _precomputed_benign_states = extract_hidden_states_speculators(
            config, benign_msgs,
        )
        print("* Extracting residuals for target prompts...")
        _precomputed_target_states = extract_hidden_states_speculators(
            config, target_msgs,
        )
        print()
    elif config.model.backend in ("vllm", "sglang"):
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
                _COT = {"<think>": "<think></think>", "<thought>": "<thought></thought>"}
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
            print(f"* Trained scorers for [bold]{len(engine._concept_scorers)}[/] layers")

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
        # Skip for vLLM backend — expert routing is not supported in vLLM mode
        # (only LoRA weight steering is applied).
        safety_experts: dict[int, list[tuple[int, float]]] | None = None
        if engine.has_expert_routing() and config.model.backend not in ("vllm", "sglang"):
            print()
            print("Profiling MoE expert activations...")
            safety_experts = engine.identify_safety_experts(benign_msgs, target_msgs)

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
            print("* Building LoRA projection cache...")
            if engine.model is not None:
                projection_cache = ProjectionCache.build(engine, vectors)
                # Unload HF model to free VRAM for the TP backend.
                print("* Unloading HF model...")
                engine.prepare_for_unload()
                engine.model = None
            else:
                # HF model was never loaded (speculators handled everything).
                # Build projections directly from safetensors files.
                projection_cache = ProjectionCache.build_from_safetensors(
                    config, vectors,
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

            # If engine has no model (lightweight mode), populate cached
            # metadata from the projection cache so optimizer can query it.
            if engine.model is None and engine._cached_n_layers is None:
                engine._cached_n_layers = max(projection_cache.projections.keys()) + 1
                engine._cached_components = sorted({
                    comp for layer in projection_cache.projections.values()
                    for comp in layer
                })

            # Detect response prefix via the fast TP backend (if not done earlier).
            if not engine.response_prefix:
                print("* Detecting response prefix via TP backend...")
                _mini = benign_msgs[:2] + target_msgs[:2]
                _resps = tp_gen.generate_text_batched(_mini, max_new_tokens=20)
                from os.path import commonprefix as _cp
                engine.response_prefix = _cp(_resps).rstrip(" ")
                if engine.response_prefix:
                    _COT = {"<think>": "<think></think>", "<thought>": "<thought></thought>"}
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

        study = run_search(
            config, engine, scorer, vectors, safety_experts, storage,
            benign_states=benign_states,
            target_states=target_states,
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
