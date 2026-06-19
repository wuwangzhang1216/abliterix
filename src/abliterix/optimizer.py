# Abliterix — a derivative work of Heretic (https://github.com/p-e-w/heretic)
# Original work Copyright (C) 2025  Philipp Emanuel Weidmann (p-e-w)
# Modified work Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

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
from .settings import AbliterixConfig
from .types import (
    DecayKernel,
    DirectTransform,
    ExpertRoutingConfig,
    SteeringMode,
    SteeringProfile,
)
from .util import humanize_duration, print, report_memory


def run_search(
    config: AbliterixConfig,
    engine,
    scorer,
    steering_vectors,
    safety_experts: dict[int, list[tuple[int, float]]] | None,
    storage: JournalStorage,
    benign_states=None,
    target_states=None,
    progress_callback=None,
    *,
    steering_vector_variants: dict[str, "torch.Tensor"] | None = None,
) -> optuna.Study:
    """Execute the Optuna optimisation loop and return the completed study.

    Parameters
    ----------
    config : AbliterixConfig
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
    benign_states : Tensor, optional
        Residual states for discriminative layer selection.
    target_states : Tensor, optional
        Residual states for discriminative layer selection.
    steering_vector_variants : dict[str, Tensor], optional
        Named alternative steering tensors.  When provided together with
        ``config.steering.search_harmfulness_direction = True``, the
        optimiser samples which variant to use per trial via a TPE
        categorical.  Typical keys: ``"single"`` (mean-diff) and
        ``"harmfulness_pair"`` (stacked refusal + harmfulness directions).
    """
    opt = config.optimization
    num_layers = engine.get_n_layers()
    last_layer = num_layers - 1

    # Resolve the variants the optimiser is allowed to sample from. When
    # search_harmfulness_direction is on but the caller didn't pre-compute
    # alternatives, fall back to a single-key dict containing the default
    # ``steering_vectors`` so the categorical sample is a degenerate one-
    # choice — the search still validates without crashing.
    _variants: dict[str, "torch.Tensor"] | None = None
    if config.steering.search_harmfulness_direction:
        if steering_vector_variants:
            _variants = dict(steering_vector_variants)
        else:
            _variants = {"single": steering_vectors}
        print(
            f"[grey50]search_harmfulness_direction = true; "
            f"variants available: {list(_variants.keys())}[/]"
        )

    _search_transform = (
        config.steering.search_direct_transform
        and config.steering.steering_mode == SteeringMode.DIRECT
    )
    if _search_transform:
        print(
            f"[grey50]search_direct_transform = true; "
            f"choices: {config.steering.search_direct_transform_choices}[/]"
        )

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

        # --- Direct-mode transform choice (categorical) ---
        # Mutates config.steering.direct_transform for this trial only; the
        # `finally` block at the end of apply+evaluate restores it.
        trial_direct_transform: DirectTransform | None = None
        if _search_transform:
            chosen = trial.suggest_categorical(
                "direct_transform",
                config.steering.search_direct_transform_choices,
            )
            trial_direct_transform = DirectTransform(chosen)
            trial.set_user_attr("direct_transform", chosen)

        # --- Decay-kernel choice (categorical) ---
        # Mutates config.steering.decay_kernel for this trial only; the
        # `finally` block restores it. Lets TPE pick the layer-taper shape.
        trial_decay_kernel: DecayKernel | None = None
        if config.steering.search_decay_kernel:
            chosen_kernel = trial.suggest_categorical(
                "decay_kernel",
                config.steering.search_decay_kernel_choices,
            )
            trial_decay_kernel = DecayKernel(chosen_kernel)
            trial.set_user_attr("decay_kernel", chosen_kernel)

        # --- Steering-vector variant (single vs harmfulness pair) ---
        if _variants is not None:
            variant_keys = list(_variants.keys())
            chosen_variant = trial.suggest_categorical("steering_variant", variant_keys)
            trial_vectors = _variants[chosen_variant]
            trial.set_user_attr("steering_variant", chosen_variant)
        else:
            trial_vectors = steering_vectors

        # --- Vector scope ---
        scope_choices = (
            [config.steering.fixed_vector_scope]
            if config.steering.fixed_vector_scope
            else ["global", "per layer"]
        )
        vector_scope = trial.suggest_categorical("vector_scope", scope_choices)

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
            # User can exclude components from the search entirely (e.g. drop
            # Q/K/V on MoE models where the refusal signal lives in expert path).
            if component in config.steering.disabled_components:
                continue
            # Per-component override (e.g. MoE models want different
            # ranges for attn vs fused expert mlp.down_proj).
            comp_range = config.steering.component_strength_ranges.get(
                component, config.steering.strength_range
            )
            lo, hi = comp_range[0], comp_range[1]
            # Auto-disable: for components the user marks (default
            # ``mlp.down_proj``) drop the lower bound below 0 and clamp the
            # sample to max(0.0, …).  A continuous sampler hits an exact point
            # with probability zero, so the negative-and-clamp trick is what
            # gives TPE a finite chance of *fully disabling* the component this
            # trial — letting it discover per-model that e.g. ablating the MLP
            # down-projection hurts more than it helps, with no manual
            # ``disabled_components`` decision required.
            auto_disable = (
                component in config.steering.auto_disable_components
                and config.steering.auto_disable_floor < 0
            )
            if auto_disable:
                lo = config.steering.auto_disable_floor
            sampled_max_w = trial.suggest_float(
                f"{component}.max_weight",
                lo,
                hi,
            )
            max_w = max(0.0, sampled_max_w) if auto_disable else sampled_max_w
            pos_lo = 0.4 * last_layer if last_layer < 20 else 0.6 * last_layer
            peak_pos = trial.suggest_float(
                f"{component}.max_weight_position",
                pos_lo,
                1.0 * last_layer,
            )
            # min_weight expressed as a fraction of max_weight
            # (multivariate TPE needs fixed-range parameters). The upper bound
            # may be capped per-component (see component_min_frac_max) or
            # globally (min_weight_frac_max) to bias warmup toward "sharp
            # peak" profiles instead of nearly-flat over-steered ones.
            frac_hi = config.steering.component_min_frac_max.get(
                component, config.steering.min_weight_frac_max
            )
            min_frac = trial.suggest_float(f"{component}.min_weight", 0.0, frac_hi)
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

        # vLLM backend: build LoRA adapter from projection cache
        # instead of modifying HF model weights.
        vllm_gen = getattr(engine, "_vllm_gen", None)
        proj_cache = getattr(engine, "_projection_cache", None)
        adapter_path = None

        # Track whether we applied router suppression this trial so we can
        # restore in the `finally` block regardless of TrialPruned / errors.
        _moe_applied_this_trial = False
        # Track whether the in-place editors mutated weights — needed for
        # restore in the finally block.
        _in_place_applied_this_trial = False

        # If the optimiser sampled a direct_transform this trial, swap it
        # into the global config so apply_steering picks it up. Always
        # restored in the `finally` block below.
        _saved_direct_transform: DirectTransform | None = None
        if trial_direct_transform is not None:
            _saved_direct_transform = config.steering.direct_transform
            config.steering.direct_transform = trial_direct_transform
            if trial_counter == 1:
                print(
                    f"* trial direct_transform = "
                    f"[bold]{trial_direct_transform.value}[/]"
                )

        # Swap the sampled decay kernel into the global config for this trial.
        # Restored in the `finally` block below.
        _saved_decay_kernel: DecayKernel | None = None
        if trial_decay_kernel is not None:
            _saved_decay_kernel = config.steering.decay_kernel
            config.steering.decay_kernel = trial_decay_kernel
            if trial_counter == 1:
                print(f"* trial decay_kernel = [bold]{trial_decay_kernel.value}[/]")

        # In-place editing path: direct weight edits on TP workers via
        # collective_rpc. Takes precedence over the LoRA adapter path when
        # an attention editor is attached (expert editor is optional for
        # attention-only profiles).
        _in_place_mode = (
            vllm_gen is not None
            and getattr(vllm_gen, "attention_editor", None) is not None
        )

        if _in_place_mode:
            from .core.steering import apply_steering_vllm_inplace

            # Resolve n_layers + hidden_dim from the attention editor's
            # probe, falling back to config.
            _expert_editor = getattr(vllm_gen, "expert_editor", None)
            _expert_layers = (
                _expert_editor._moe_layers if _expert_editor is not None else set()
            )
            _probe_layers = sorted(
                vllm_gen.attention_editor._attn_layers | _expert_layers
            )
            _n_layers = (_probe_layers[-1] + 1) if _probe_layers else 0
            if _n_layers == 0:
                # Probe failed — fall back to cached metadata.
                _n_layers = engine._cached_n_layers or 0
            _hidden = (
                _expert_editor.hidden_dim
                if _expert_editor is not None
                else int(trial_vectors.shape[-1])
            )
            _transposed = (
                _expert_editor.transposed if _expert_editor is not None else False
            )

            print("* Applying steering (vLLM in-place)...")
            _ip_result = apply_steering_vllm_inplace(
                vllm_gen,
                trial_vectors,
                vector_index,
                profiles,
                config,
                n_layers=_n_layers,
                hidden_dim=_hidden,
                transposed=_transposed,
                safety_experts=safety_experts,
                routing_config=routing,
            )
            _in_place_applied_this_trial = True
            _moe_applied_this_trial = _ip_result.get("router_touched", 0) > 0

            if trial_counter == 1:
                print(
                    f"  * in-place attention: applied="
                    f"{_ip_result['attention'].get('applied', 0)}, "
                    f"ega: applied={_ip_result['ega'].get('applied', 0)}, "
                    f"router rows modified={_ip_result['router_touched']}"
                )
        elif vllm_gen is not None and (
            proj_cache is not None or getattr(vllm_gen, "_lora_disabled", False)
        ):
            if getattr(vllm_gen, "_lora_disabled", False):
                # vLLM loaded without LoRA (e.g. MXFP4 + driver 570 Marlin-FP4
                # PTX issue).  Skip the per-trial adapter build entirely —
                # attention steering is off, router suppression below carries
                # the signal.
                if trial_counter == 1:
                    print(
                        "* LoRA disabled — skipping attention adapter; "
                        "router suppression is the only steering mechanism."
                    )
            else:
                print("* Building LoRA adapter for vLLM...")
                lora_weights = proj_cache.build_lora_weights(
                    profiles,
                    vector_index,
                    config,
                )
                if lora_weights:
                    adapter_path = vllm_gen.save_adapter(
                        lora_weights,
                        proj_cache.target_modules,
                        config.model.model_id,
                    )
            # Store adapter path on engine for scorer/detector to use.
            engine._current_adapter_path = adapter_path

            # MoE router suppression via collective_rpc.  Only active when
            # the HF phase identified safety experts AND the editor was
            # attached in cli.py.  Expert-ablation-weight on fused experts
            # is still skipped under vLLM (packed MXFP4/FP8 can't be edited
            # in place without a full dequant pass).
            if (
                routing is not None
                and getattr(vllm_gen, "moe_editor", None) is not None
            ):
                touched = vllm_gen.apply_router_suppression(
                    n_suppress=routing.n_suppress,
                    bias_value=routing.router_bias,
                )
                _moe_applied_this_trial = touched > 0
                if trial_counter == 1:
                    if _moe_applied_this_trial:
                        scale = max(0.0, 1.0 + routing.router_bias / 10.0)
                        print(
                            f"  * Router suppression: n_suppress={routing.n_suppress}, "
                            f"router_bias={routing.router_bias:.2f} → scale={scale:.2f}, "
                            f"{touched} rows modified per worker"
                        )
                    elif routing.expert_ablation_weight > 0:
                        print(
                            "  [yellow]Expert-ablation-weight on fused experts "
                            "is skipped under vLLM (packed quant weights cannot "
                            "be edited in place).  Router suppression + attention "
                            "LoRA are applied.[/]"
                        )
        else:
            print("* Resetting model...")
            engine.restore_baseline()

            print("* Applying steering...")
            apply_steering(
                engine,
                trial_vectors,
                vector_index,
                profiles,
                config,
                safety_experts=safety_experts,
                routing_config=routing,
                benign_states=benign_states,
                target_states=target_states,
            )

        try:
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
        finally:
            # Always restore vLLM router edits so the next trial starts from
            # the pristine base model.  No-op if nothing was applied.
            if _moe_applied_this_trial and vllm_gen is not None:
                vllm_gen.restore_router_suppression()
            # Restore in-place edits (attention + EGA) so trial N+1 starts
            # from pristine weights. Router already handled above.
            if _in_place_applied_this_trial and vllm_gen is not None:
                vllm_gen.restore_attention_weights()
                vllm_gen.restore_expert_weights()
            # Restore the global direct_transform if this trial sampled one.
            if _saved_direct_transform is not None:
                config.steering.direct_transform = _saved_direct_transform
            # Restore the global decay_kernel if this trial sampled one.
            if _saved_decay_kernel is not None:
                config.steering.decay_kernel = _saved_decay_kernel

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

        if progress_callback is not None:
            progress_callback(trial_counter, kl, detected, opt.num_trials)

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

    # Prefer the explicit sampler_seed; otherwise fall back to the global
    # ``config.seed`` so a single top-level seed makes the whole run (sampler
    # included) reproducible.
    _sampler_seed = opt.sampler_seed if opt.sampler_seed is not None else config.seed
    if _sampler_seed is not None:
        torch.manual_seed(_sampler_seed)

    sampler_kw: dict = dict(
        n_startup_trials=opt.num_warmup_trials,
        n_ei_candidates=128,
        multivariate=True,
    )
    if _sampler_seed is not None:
        sampler_kw["seed"] = _sampler_seed

    study = optuna.create_study(
        sampler=TPESampler(**sampler_kw),
        directions=[StudyDirection.MINIMIZE, StudyDirection.MINIMIZE],
        storage=storage,
        study_name="abliterix",
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
    elif opt.seed_trials:
        # Enqueue user-supplied known-good points before TPE sampling. Each
        # seed dict skips suggest_* sampling for any param it specifies; TPE
        # still samples params absent from the dict. Only fires on a fresh
        # study — resumed studies (start_index > 0) keep their existing trial
        # history without re-enqueueing.
        print()
        print(f"Enqueueing {len(opt.seed_trials)} seed trial(s) before TPE search.")
        for i, seed in enumerate(opt.seed_trials):
            study.enqueue_trial(seed, skip_if_exists=True)
            print(f"  seed {i}: {len(seed)} params pinned")

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
