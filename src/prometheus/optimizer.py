# Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>

"""Optuna multi-objective optimisation loop for steering parameters.

The :func:`run_search` function encapsulates the entire search: study
creation, checkpoint management, the TPE sampler, and the per-trial
objective evaluation.
"""

import time
from dataclasses import asdict

import optuna
import torch
from optuna import Trial, TrialPruned
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.study import StudyDirection
from optuna.trial import TrialState

from .core.steering import apply_steering
from .data import format_trial_params
from .settings import PrometheusConfig
from .types import ExpertRoutingConfig, SteeringProfile
from .util import humanize_duration, print, report_memory


def run_search(
    config: PrometheusConfig,
    engine,
    scorer,
    steering_vectors,
    safety_experts: dict[int, list[tuple[int, float]]] | None,
    storage: JournalStorage,
) -> optuna.Study:
    """Execute the Optuna optimisation loop and return the completed study.

    Parameters
    ----------
    config : PrometheusConfig
        Full application configuration.
    engine : SteeringEngine
        Model wrapper used for steering and evaluation.
    scorer : TrialScorer
        Captures baseline metrics and evaluates each trial.
    steering_vectors : Tensor
        Pre-computed steering vectors, shape ``(layers+1, hidden_dim)``.
    safety_experts : dict or None
        Per-layer expert risk scores (MoE models only).
    storage : JournalStorage
        Optuna storage backend (may already contain prior trials).
    """
    opt = config.optimization
    num_layers = len(engine.transformer_layers)
    last_layer = num_layers - 1

    # ----------------------------------------------------------------
    # Objective
    # ----------------------------------------------------------------

    trial_counter = 0
    start_index = 0
    start_time = time.perf_counter()

    def _objective(trial: Trial) -> tuple[float, float]:
        nonlocal trial_counter
        trial_counter += 1
        trial.set_user_attr("index", trial_counter)

        # --- Vector scope ---
        vector_scope = trial.suggest_categorical(
            "vector_scope",
            ["global", "per layer"],
        )

        # Discrimination is strongest slightly past the midpoint.
        # Widen the search for shallow models (< 20 layers).
        lo = 0.3 * last_layer if last_layer < 20 else 0.4 * last_layer
        hi = 0.95 * last_layer if last_layer < 20 else 0.9 * last_layer
        vector_index: float | None = trial.suggest_float("vector_index", lo, hi)

        if vector_scope == "per layer":
            vector_index = None

        # --- Per-component steering profiles ---
        profiles: dict[str, SteeringProfile] = {}

        for component in engine.list_steerable_components():
            max_w = trial.suggest_float(
                f"{component}.max_weight",
                config.steering.strength_range[0],
                config.steering.strength_range[1],
            )
            pos_lo = 0.4 * last_layer if last_layer < 20 else 0.6 * last_layer
            peak_pos = trial.suggest_float(
                f"{component}.max_weight_position",
                pos_lo,
                1.0 * last_layer,
            )
            # min_weight expressed as a fraction of max_weight
            # (multivariate TPE needs fixed-range parameters).
            min_frac = trial.suggest_float(f"{component}.min_weight", 0.0, 1.0)
            falloff = trial.suggest_float(
                f"{component}.min_weight_distance",
                1.0,
                0.6 * last_layer,
            )

            profiles[component] = SteeringProfile(
                max_weight=max_w,
                max_weight_position=peak_pos,
                min_weight=min_frac * max_w,
                min_weight_distance=falloff,
            )

        # --- MoE expert routing (only for MoE architectures) ---
        routing: ExpertRoutingConfig | None = None
        if safety_experts is not None:
            n_sup = trial.suggest_int(
                "moe.n_suppress",
                0,
                config.experts.max_suppress,
            )
            r_bias = trial.suggest_float(
                "moe.router_bias",
                config.experts.router_bias_range[0],
                config.experts.router_bias_range[1],
            )
            e_weight = trial.suggest_float(
                "moe.expert_ablation_weight",
                config.experts.ablation_weight_range[0],
                config.experts.ablation_weight_range[1],
            )
            routing = ExpertRoutingConfig(
                n_suppress=n_sup,
                router_bias=r_bias,
                expert_ablation_weight=e_weight,
            )
            trial.set_user_attr("moe_parameters", asdict(routing))

        trial.set_user_attr("vector_index", vector_index)
        trial.set_user_attr(
            "parameters",
            {k: asdict(v) for k, v in profiles.items()},
        )

        # --- Apply steering and evaluate ---
        print()
        print(f"Running trial [bold]{trial_counter}[/] of [bold]{opt.num_trials}[/]...")
        print("* Parameters:")
        for name, value in format_trial_params(trial).items():
            print(f"  * {name} = [bold]{value}[/]")

        print("* Resetting model...")
        engine.restore_baseline()

        print("* Applying steering...")
        apply_steering(
            engine,
            steering_vectors,
            vector_index,
            profiles,
            config,
            safety_experts=safety_experts,
            routing_config=routing,
        )

        print("* Evaluating...")
        kl, length_dev = scorer.measure_kl_and_coherence(engine)

        # Early pruning for excessively damaged models.
        if config.kl.prune_threshold > 0 and kl > config.kl.prune_threshold:
            print(
                f"  * [yellow]KL divergence {kl:.4f} exceeds prune threshold "
                f"{config.kl.prune_threshold}, skipping compliance check[/]"
            )
            raise TrialPruned()

        print("  * Counting model refusals...")
        detected = scorer.detector.evaluate_compliance(
            engine,
            scorer.target_msgs,
        )
        print(f"  * Refusals: [bold]{detected}[/]/{len(scorer.target_msgs)}")

        objectives = scorer._compute_objectives(kl, detected, length_dev)

        # Timing / resource report
        elapsed = time.perf_counter() - start_time
        remaining = (elapsed / (trial_counter - start_index)) * (
            opt.num_trials - trial_counter
        )
        print()
        print(f"[grey50]Elapsed time: [bold]{humanize_duration(elapsed)}[/][/]")
        if trial_counter < opt.num_trials:
            print(
                f"[grey50]Estimated remaining time: [bold]{humanize_duration(remaining)}[/][/]"
            )
        report_memory()

        trial.set_user_attr("kl_divergence", kl)
        trial.set_user_attr("refusals", detected)
        trial.set_user_attr("length_deviation", length_dev)

        return objectives

    def _objective_safe(trial: Trial) -> tuple[float, float]:
        try:
            return _objective(trial)
        except KeyboardInterrupt:
            trial.study.stop()
            raise TrialPruned()

    # ----------------------------------------------------------------
    # Study creation / resumption
    # ----------------------------------------------------------------

    if opt.sampler_seed is not None:
        torch.manual_seed(opt.sampler_seed)

    sampler_kw: dict = dict(
        n_startup_trials=opt.num_warmup_trials,
        n_ei_candidates=128,
        multivariate=True,
    )
    if opt.sampler_seed is not None:
        sampler_kw["seed"] = opt.sampler_seed

    study = optuna.create_study(
        sampler=TPESampler(**sampler_kw),
        directions=[StudyDirection.MINIMIZE, StudyDirection.MINIMIZE],
        storage=storage,
        study_name="prometheus",
        load_if_exists=True,
    )

    study.set_user_attr("settings", config.model_dump_json())
    study.set_user_attr("finished", False)

    def _count_complete() -> int:
        return sum(1 for t in study.trials if t.state == TrialState.COMPLETE)

    start_index = trial_counter = _count_complete()
    if start_index > 0:
        print()
        print("Resuming existing study.")

    try:
        study.optimize(
            _objective_safe,
            n_trials=opt.num_trials - _count_complete(),
        )
    except KeyboardInterrupt:
        pass

    if _count_complete() == opt.num_trials:
        study.set_user_attr("finished", True)

    return study
