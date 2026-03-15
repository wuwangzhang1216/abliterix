# Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>

"""Command-line interface: banner, device detection, interactive menus, and model export."""

import math
import os
import sys
import time
import warnings
from importlib.metadata import version
from os.path import commonprefix
from pathlib import Path

import huggingface_hub
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
from huggingface_hub import ModelCard, ModelCardData
from optuna.exceptions import ExperimentalWarning
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.trial import TrialState
from pydantic import ValidationError
from questionary import Choice
from rich.traceback import install
from torch import Tensor

from .analysis import ResidualAnalyzer
from .core.engine import SteeringEngine, resolve_model_class
from .core.steering import apply_steering
from .data import format_trial_params, generate_model_card, load_prompt_dataset
from .eval.detector import RefusalDetector
from .eval.scorer import TrialScorer
from .optimizer import run_search
from .settings import PrometheusConfig
from .types import ChatMessage, QuantMode, SteeringProfile
from .util import (
    ask_choice,
    ask_path,
    ask_secret,
    ask_text,
    flush_memory,
    print,
    report_memory,
    slugify_model_name,
)
from .vectors import compute_steering_vectors


# ---------------------------------------------------------------------------
# Startup helpers
# ---------------------------------------------------------------------------


def _print_banner():
    # Modified "Pagga" font from https://budavariam.github.io/asciiart-text/
    v = version("prometheus-llm")
    print(f"[cyan]█▀▄░█▀▄░█▀█░█▄█░█▀▀░▀█▀░█░█░█▀▀░█░█░█▀▀[/]  v{v}")
    print("[cyan]█▀░░█▀▄░█░█░█░█░█▀▀░░█░░█▀█░█▀▀░█░█░▀▀█[/]")
    print(
        "[cyan]▀░░░▀░▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀▀░▀▀▀[/]"
        "  [blue underline]https://github.com/wuwangzhang1216/prometheus[/]"
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
    config: PrometheusConfig,
    existing_study,
    checkpoint_file: str,
    lock_obj,
    storage: JournalStorage,
) -> tuple[PrometheusConfig, JournalStorage] | None:
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
            restored = PrometheusConfig.model_validate_json(
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
        config = PrometheusConfig.model_validate_json(
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


def _auto_batch_size(
    engine: SteeringEngine, benign_msgs: list[ChatMessage], config: PrometheusConfig
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
        except Exception:
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
# Merge strategy prompt
# ---------------------------------------------------------------------------


def _ask_merge_strategy(config: PrometheusConfig, engine: SteeringEngine) -> str | None:
    """Return ``"merge"`` or ``None`` (cancelled).  Warns about RAM for quantised models."""
    qm = config.model.quant_method

    if qm in (QuantMode.BNB_4BIT, QuantMode.BNB_8BIT):
        print()
        print(
            "Model was loaded with quantization. Merging requires reloading the base model."
        )
        print(
            "[yellow]WARNING: CPU merging requires dequantizing the entire model to system RAM.[/]"
        )
        print("[yellow]This can lead to system freezes if you run out of memory.[/]")

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                meta = resolve_model_class(config.model.model_id).from_pretrained(
                    config.model.model_id,
                    device_map="meta",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                )
                gb = meta.get_memory_footprint() / (1024**3)
                print(
                    f"[yellow]Estimated RAM required (excluding overhead): [bold]~{gb:.2f} GB[/][/]"
                )
        except Exception:
            print(
                "[yellow]Rule of thumb: You need approximately 3x the parameter count in GB RAM.[/]"
            )
            print(
                "[yellow]Example: A 27B model requires ~80GB RAM. A 70B model requires ~200GB RAM.[/]"
            )

        print()
        strategy = ask_choice(
            "How do you want to proceed?",
            [
                Choice(
                    title="Merge LoRA into full model (requires sufficient RAM)",
                    value="merge",
                ),
                Choice(title="Cancel", value="cancel"),
            ],
        )
        return None if strategy == "cancel" else strategy

    return "merge"


# ---------------------------------------------------------------------------
# Interactive results UI
# ---------------------------------------------------------------------------


def _show_interactive_results(
    study,
    config: PrometheusConfig,
    engine: SteeringEngine,
    scorer: TrialScorer,
    steering_vectors: Tensor,
    safety_experts,
    storage: JournalStorage,
):
    """Post-optimisation interactive menu: trial selection, save, upload, chat."""
    while True:
        completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
        if not completed:
            raise KeyboardInterrupt

        # Build Pareto front on (refusals, kl_divergence).
        ranked = sorted(
            completed,
            key=lambda t: (t.user_attrs["refusals"], t.user_attrs["kl_divergence"]),
        )
        min_kl = math.inf
        pareto: list = []
        for trial in ranked:
            kl = trial.user_attrs["kl_divergence"]
            if kl < min_kl:
                min_kl = kl
                pareto.append(trial)

        choices = [
            Choice(
                title=(
                    f"[Trial {t.user_attrs['index']:>3}] "
                    f"Refusals: {t.user_attrs['refusals']:>2}/{len(scorer.target_msgs)}, "
                    f"KL divergence: {t.user_attrs['kl_divergence']:.4f}"
                ),
                value=t,
            )
            for t in pareto
        ]
        choices.append(Choice(title="Run additional trials", value="continue"))
        choices.append(Choice(title="Exit program", value=""))

        print()
        print("[bold green]Optimization finished![/]")
        print()
        print(
            "The following trials resulted in Pareto optimal combinations of refusals "
            "and KL divergence. After selecting a trial, you will be able to save the "
            "model, upload it to Hugging Face, or chat with it to test how well it works. "
            "You can return to this menu later to select a different trial. "
            "[yellow]Note that KL divergence values above 1 usually indicate significant "
            "damage to the original model's capabilities.[/]"
        )

        while True:
            print()
            trial = ask_choice("Which trial do you want to use?", choices)

            if trial == "continue":
                while True:
                    try:
                        n_extra = ask_text(
                            "How many additional trials do you want to run?"
                        )
                        if not n_extra:
                            n_extra = 0
                            break
                        n_extra = int(n_extra)
                        if n_extra > 0:
                            break
                        print("[red]Please enter a number greater than 0.[/]")
                    except ValueError:
                        print("[red]Please enter a number.[/]")

                if n_extra == 0:
                    continue

                config.optimization.num_trials += n_extra
                study.set_user_attr("settings", config.model_dump_json())
                study.set_user_attr("finished", False)

                def _count():
                    return sum(
                        1 for t in study.trials if t.state == TrialState.COMPLETE
                    )

                try:
                    study = run_search(
                        config,
                        engine,
                        scorer,
                        steering_vectors,
                        safety_experts,
                        storage,
                    )
                except KeyboardInterrupt:
                    pass

                if _count() == config.optimization.num_trials:
                    study.set_user_attr("finished", True)

                break

            elif trial is None or trial == "":
                return

            # --- Restore selected trial ---
            print()
            print(f"Restoring model from trial [bold]{trial.user_attrs['index']}[/]...")
            print("* Parameters:")
            for name, value in format_trial_params(trial).items():
                print(f"  * {name} = [bold]{value}[/]")

            print("* Resetting model...")
            engine.restore_baseline()
            print("* Applying steering...")
            apply_steering(
                engine,
                steering_vectors,
                trial.user_attrs["vector_index"],
                {
                    k: SteeringProfile(**v)
                    for k, v in trial.user_attrs["parameters"].items()
                },
            )

            while True:
                print()
                action = ask_choice(
                    "What do you want to do with the decensored model?",
                    [
                        "Save the model to a local folder",
                        "Upload the model to Hugging Face",
                        "Chat with the model",
                        "Return to the trial selection menu",
                    ],
                )

                if action is None or action == "Return to the trial selection menu":
                    break

                try:
                    match action:
                        case "Save the model to a local folder":
                            save_dir = ask_path("Path to the folder:")
                            if not save_dir:
                                continue
                            strategy = _ask_merge_strategy(config, engine)
                            if strategy is None:
                                continue
                            print("Saving merged model...")
                            merged = engine.export_merged()
                            merged.save_pretrained(save_dir)
                            del merged
                            flush_memory()
                            engine.tokenizer.save_pretrained(save_dir)
                            print(f"Model saved to [bold]{save_dir}[/].")

                        case "Upload the model to Hugging Face":
                            token = huggingface_hub.get_token()
                            if not token:
                                token = ask_secret("Hugging Face access token:")
                            if not token:
                                continue

                            user = huggingface_hub.whoami(token)
                            fullname = user.get(
                                "fullname", user.get("name", "unknown user")
                            )
                            email = user.get("email", "no email found")
                            print(f"Logged in as [bold]{fullname} ({email})[/]")

                            repo_id = ask_text(
                                "Name of repository:",
                                default=f"{user['name']}/{Path(config.model.model_id).name}-prometheus",
                            )
                            visibility = ask_choice(
                                "Should the repository be public or private?",
                                ["Public", "Private"],
                            )
                            private = visibility == "Private"

                            strategy = _ask_merge_strategy(config, engine)
                            if strategy is None:
                                continue

                            print("Uploading merged model...")
                            merged = engine.export_merged()
                            merged.push_to_hub(repo_id, private=private, token=token)
                            del merged
                            flush_memory()
                            engine.tokenizer.push_to_hub(
                                repo_id, private=private, token=token
                            )

                            model_path = Path(config.model.model_id)
                            if model_path.exists():
                                card_path = (
                                    model_path / huggingface_hub.constants.REPOCARD_NAME
                                )
                                card = (
                                    ModelCard.load(card_path)
                                    if card_path.exists()
                                    else None
                                )
                            else:
                                card = ModelCard.load(config.model.model_id)
                            if card is not None:
                                if card.data is None:
                                    card.data = ModelCardData()
                                if card.data.tags is None:
                                    card.data.tags = []
                                card.data.tags += [
                                    "prometheus",
                                    "uncensored",
                                    "decensored",
                                    "abliterated",
                                ]
                                card.text = (
                                    generate_model_card(
                                        config,
                                        trial,
                                        scorer.baseline_refusal_count,
                                        scorer.target_msgs,
                                    )
                                    + card.text
                                )
                                card.push_to_hub(repo_id, token=token)

                            print(f"Model uploaded to [bold]{repo_id}[/].")

                        case "Chat with the model":
                            print()
                            print(
                                "[cyan]Press Ctrl+C at any time to return to the menu.[/]"
                            )

                            chat = [{"role": "system", "content": config.system_prompt}]

                            while True:
                                try:
                                    message = ask_text("User:", qmark=">", unsafe=True)
                                    if not message:
                                        break
                                    chat.append({"role": "user", "content": message})
                                    print("[bold]Assistant:[/] ", end="")
                                    response = engine.stream_chat_response(chat)
                                    chat.append(
                                        {"role": "assistant", "content": response}
                                    )
                                except (KeyboardInterrupt, EOFError):
                                    break

                except Exception as error:
                    print(f"[red]Error: {error}[/]")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run():
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
        config = PrometheusConfig()  # ty:ignore[missing-argument]
    except ValidationError as error:
        print(f"[red]Configuration contains [bold]{error.error_count()}[/] errors:[/]")
        for err in error.errors():
            print(f"[bold]{err['loc'][0]}[/]: [yellow]{err['msg']}[/]")
        print()
        print(
            "Run [bold]prometheus --help[/] or see [bold]prometheus.toml[/] for details "
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

    engine = SteeringEngine(config)
    print()
    report_memory()

    # Load steering-vector source datasets.
    print()
    print(f"Loading benign prompts from [bold]{config.benign_prompts.dataset}[/]...")
    benign_msgs = load_prompt_dataset(config, config.benign_prompts)
    print(f"* [bold]{len(benign_msgs)}[/] prompts loaded")

    print()
    print(f"Loading target prompts from [bold]{config.target_prompts.dataset}[/]...")
    target_msgs = load_prompt_dataset(config, config.target_prompts)
    print(f"* [bold]{len(target_msgs)}[/] prompts loaded")

    if config.inference.batch_size == 0:
        config.inference.batch_size = _auto_batch_size(engine, benign_msgs, config)

    _detect_response_prefix(engine, benign_msgs, target_msgs)

    detector = RefusalDetector(config)
    try:
        scorer = TrialScorer(config, engine, detector)

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
        )

        analyzer = ResidualAnalyzer(config, engine, benign_states, target_states)

        if config.display.print_residual_geometry:
            analyzer.print_residual_geometry()
        if config.display.plot_residuals:
            analyzer.plot_residuals()

        del benign_states, target_states, analyzer
        flush_memory()

        # Profile MoE expert routing if applicable.
        safety_experts: dict[int, list[tuple[int, float]]] | None = None
        if engine.has_expert_routing():
            print()
            print("Profiling MoE expert activations...")
            safety_experts = engine.identify_safety_experts(benign_msgs, target_msgs)

        study = run_search(config, engine, scorer, vectors, safety_experts, storage)

        if config.non_interactive:
            completed = sum(1 for t in study.trials if t.state == TrialState.COMPLETE)
            print()
            print(
                f"[bold green]Non-interactive mode: optimization finished with "
                f"{completed} completed trials.[/]"
            )
            return

        _show_interactive_results(
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
