# Abliterix — a derivative work of Heretic (https://github.com/p-e-w/heretic)
# Original work Copyright (C) 2025  Philipp Emanuel Weidmann (p-e-w)
# Modified work Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Post-optimisation interactive menus: trial selection, model export, upload, and chat."""

import math
import warnings
from pathlib import Path

import huggingface_hub
from huggingface_hub import ModelCard, ModelCardData
from optuna.storages import JournalStorage
from optuna.trial import TrialState
from questionary import Choice
from rich.table import Table
from torch import Tensor

from .core.engine import SteeringEngine, resolve_model_class
from .core.steering import apply_steering
from .data import format_trial_params, generate_model_card
from .eval.scorer import TrialScorer
from .optimizer import run_search
from .reproducibility import (
    REPRODUCE_TAG,
    build_manifest,
    repo_weight_shas,
    write_reproduce_artifacts,
)
from .settings import AbliterixConfig
from .types import QuantMode, SteeringProfile
from .util import (
    ask_choice,
    ask_path,
    ask_secret,
    ask_text,
    flush_memory,
    print,
)

# Re-export torch here to keep the import needed for memory estimation.
import torch


# ---------------------------------------------------------------------------
# Merge strategy prompt
# ---------------------------------------------------------------------------


def ask_merge_strategy(config: AbliterixConfig, engine: SteeringEngine) -> str | None:
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
        except (OSError, ValueError, RuntimeError):
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
# Model save / upload helpers
# ---------------------------------------------------------------------------


def _save_model_locally(config: AbliterixConfig, engine: SteeringEngine):
    """Save a merged model to a local directory."""
    save_dir = ask_path("Path to the folder:")
    if not save_dir:
        return
    strategy = ask_merge_strategy(config, engine)
    if strategy is None:
        return
    print("Saving merged model...")
    merged = engine.export_merged()
    merged.save_pretrained(save_dir)
    del merged
    flush_memory()
    engine.tokenizer.save_pretrained(save_dir)
    print(f"Model saved to [bold]{save_dir}[/].")


def _upload_model(
    config: AbliterixConfig,
    engine: SteeringEngine,
    scorer: TrialScorer,
    trial,
):
    """Upload a merged model to Hugging Face Hub."""
    token = huggingface_hub.get_token()
    if not token:
        token = ask_secret("Hugging Face access token:")
    if not token:
        return

    user = huggingface_hub.whoami(token)
    fullname = user.get("fullname", user.get("name", "unknown user"))
    email = user.get("email", "no email found")
    print(f"Logged in as [bold]{fullname} ({email})[/]")

    repo_id = ask_text(
        "Name of repository:",
        default=f"{user['name']}/{Path(config.model.model_id).name}-abliterix",
    )
    visibility = ask_choice(
        "Should the repository be public or private?",
        ["Public", "Private"],
    )
    private = visibility == "Private"

    strategy = ask_merge_strategy(config, engine)
    if strategy is None:
        return

    print("Uploading merged model...")
    merged = engine.export_merged()
    merged.push_to_hub(repo_id, private=private, token=token)
    del merged
    flush_memory()
    engine.tokenizer.push_to_hub(repo_id, private=private, token=token)

    model_path = Path(config.model.model_id)
    if model_path.exists():
        card_path = model_path / huggingface_hub.constants.REPOCARD_NAME
        card = ModelCard.load(card_path) if card_path.exists() else None
    else:
        card = ModelCard.load(config.model.model_id)
    if card is not None:
        if card.data is None:
            card.data = ModelCardData()
        if card.data.tags is None:
            card.data.tags = []
        card.data.tags += [
            "abliterix",
            "uncensored",
            "decensored",
            "abliterated",
            REPRODUCE_TAG,
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

    _upload_reproduce_artifacts(config, scorer, trial, repo_id, token)

    print(f"Model uploaded to [bold]{repo_id}[/].")


def _upload_reproduce_artifacts(
    config: AbliterixConfig,
    scorer: TrialScorer,
    trial,
    repo_id: str,
    token: str | None,
):
    """Publish reproduce.json + SHA256SUMS + README.md to the repo's reproduce/ folder.

    Best-effort: a failure here must not abort an otherwise-successful upload,
    so any exception is caught and reported.
    """
    import tempfile

    try:
        print("Building reproducibility manifest...")
        weight_shas = repo_weight_shas(repo_id, token)
        manifest = build_manifest(
            config,
            trial,
            repo_id=repo_id,
            weight_shas=weight_shas,
            baseline_refusal_count=scorer.baseline_refusal_count,
            n_target_prompts=len(scorer.target_msgs),
        )
        with tempfile.TemporaryDirectory() as tmp:
            write_reproduce_artifacts(tmp, manifest)
            huggingface_hub.upload_folder(
                repo_id=repo_id,
                folder_path=tmp,
                path_in_repo="reproduce",
                token=token,
                commit_message="Add abliterix reproducibility manifest",
            )
        n = len(weight_shas)
        print(
            f"* Reproducibility manifest uploaded to [bold]{repo_id}/reproduce/[/] "
            f"({n} weight shard{'s' if n != 1 else ''} hashed)"
        )
    except Exception as error:
        print(f"[yellow]Could not upload reproducibility manifest: {error}[/]")


def _chat_with_model(config: AbliterixConfig, engine: SteeringEngine):
    """Start an interactive chat session with the model."""
    print()
    print("[cyan]Press Ctrl+C at any time to return to the menu.[/]")

    chat = [{"role": "system", "content": config.system_prompt}]

    while True:
        try:
            message = ask_text("User:", qmark=">", unsafe=True)
            if not message:
                break
            chat.append({"role": "user", "content": message})
            print("[bold]Assistant:[/] ", end="")
            response = engine.stream_chat_response(chat)
            chat.append({"role": "assistant", "content": response})
        except (KeyboardInterrupt, EOFError):
            break


# ---------------------------------------------------------------------------
# Interactive results UI
# ---------------------------------------------------------------------------


def _primary_metric(task_result: dict) -> tuple[str, float] | None:
    """Pick the headline metric from an lm-eval per-task result dict."""
    preferred = ("exact_match", "acc_norm", "acc", "f1", "mcc")
    available = {
        k.split(",")[0]: v
        for k, v in task_result.items()
        if isinstance(v, (int, float)) and not k.startswith(("alias", "stderr"))
    }
    for name in preferred:
        if name in available:
            return name, float(available[name])
    for k, v in available.items():
        if "stderr" not in k:
            return k, float(v)
    return None


def _print_lm_eval_table(results_ab: dict, results_base: dict | None) -> None:
    """Render an lm-eval results table, optionally comparing base vs abliterated."""
    ab = results_ab.get("results", {})
    base = (results_base or {}).get("results", {})

    table = Table(title="lm-eval benchmark results")
    table.add_column("Task", style="bold")
    table.add_column("Metric")
    if base:
        table.add_column("Original", justify="right")
        table.add_column("Abliterated", justify="right")
        table.add_column("Δ", justify="right")
    else:
        table.add_column("Abliterated", justify="right")

    for task in sorted(ab.keys()):
        ab_metric = _primary_metric(ab[task])
        if ab_metric is None:
            continue
        metric_name, ab_val = ab_metric
        if base and task in base:
            base_metric = _primary_metric(base[task])
            base_val = base_metric[1] if base_metric else float("nan")
            delta = ab_val - base_val
            colour = "green" if delta >= -0.01 else "red"
            table.add_row(
                task,
                metric_name,
                f"{base_val:.4f}",
                f"{ab_val:.4f}",
                f"[{colour}]{delta:+.4f}[/]",
            )
        else:
            table.add_row(task, metric_name, f"{ab_val:.4f}")

    print()
    print(table)


def _run_benchmarks(
    config: AbliterixConfig,
    engine: SteeringEngine,
    steering_vectors: Tensor,
    trial,
):
    """Run lm-eval-harness benchmarks on the steered model (optionally vs base).

    The decensored model is already applied to ``engine`` by the caller. When
    the user opts into a comparison, the baseline is restored, benchmarked, and
    the trial's steering is re-applied so subsequent menu actions still operate
    on the steered model.
    """
    try:
        from lm_eval import simple_evaluate
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        print("[yellow]lm-eval is not installed.[/] Install the benchmark extra:")
        print("    [bold]pip install 'abliterix[bench]'[/]")
        return

    raw = ask_text(
        "Comma-separated lm-eval tasks:",
        default="gsm8k,hellaswag,arc_easy",
    )
    tasks = [t.strip() for t in raw.split(",") if t.strip()]
    if not tasks:
        return
    limit_raw = ask_text("Max examples per task (blank = full):", default="100")
    limit = int(limit_raw) if limit_raw.strip() else None

    compare = (
        ask_choice(
            "Also benchmark the original (un-steered) model for comparison?",
            ["No", "Yes"],
        )
        == "Yes"
    )

    print(f"Running lm-eval on the decensored model: {', '.join(tasks)} ...")
    lm_ab = HFLM(pretrained=engine.model, tokenizer=engine.tokenizer, batch_size="auto")
    results_ab = simple_evaluate(model=lm_ab, tasks=tasks, limit=limit)

    results_base = None
    if compare:
        print("Restoring the original model and benchmarking it...")
        engine.restore_baseline()
        lm_base = HFLM(
            pretrained=engine.model, tokenizer=engine.tokenizer, batch_size="auto"
        )
        results_base = simple_evaluate(model=lm_base, tasks=tasks, limit=limit)
        # Re-apply the selected trial's steering so later menu actions (save,
        # upload, chat) operate on the decensored model again.
        print("Re-applying steering...")
        engine.restore_baseline()
        apply_steering(
            engine,
            steering_vectors,
            trial.user_attrs["vector_index"],
            {
                k: SteeringProfile(**v)
                for k, v in trial.user_attrs["parameters"].items()
            },
        )

    _print_lm_eval_table(results_ab, results_base)


def show_interactive_results(
    study,
    config: AbliterixConfig,
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
                        "Run standard benchmarks (lm-eval)",
                        "Return to the trial selection menu",
                    ],
                )

                if action is None or action == "Return to the trial selection menu":
                    break

                try:
                    match action:
                        case "Save the model to a local folder":
                            _save_model_locally(config, engine)

                        case "Upload the model to Hugging Face":
                            _upload_model(config, engine, scorer, trial)

                        case "Chat with the model":
                            _chat_with_model(config, engine)

                        case "Run standard benchmarks (lm-eval)":
                            _run_benchmarks(config, engine, steering_vectors, trial)

                except Exception as error:  # Catch-all for interactive menu actions
                    print(f"[red]Error: {error}[/]")
