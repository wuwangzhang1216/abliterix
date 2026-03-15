# Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>

"""Diagnostic experiments for proposed Prometheus performance optimizations.

Runs 4 experiments to validate optimization hypotheses before implementation:
  1. output_scores=True overhead (VRAM + speed)
  2. Partial KL accuracy (first-batch KL vs full KL correlation)
  3. Prune rate analysis (from existing checkpoints)
  4. Abliteration metadata traversal overhead

Usage:
    python scripts/benchmark_optimizations.py [--model MODEL] [--trials N] [--checkpoint DIR]
"""

import argparse
import json
import math
import os
import statistics
import sys
import time

from prometheus.scriptlib import setup_io

setup_io()

import optuna
import torch
import torch.nn.functional as F
from optuna.samplers import TPESampler

from prometheus.core.engine import SteeringEngine
from prometheus.core.steering import apply_steering
from prometheus.data import load_prompt_dataset
from prometheus.eval.detector import RefusalDetector
from prometheus.eval.scorer import TrialScorer
from prometheus.settings import PrometheusConfig
from prometheus.types import SteeringProfile, VectorMethod
from prometheus.util import chunk_batches, print


def suggest_trial_params(trial, engine):
    """Replicate the parameter suggestion logic from main.py objective()."""
    direction_scope = trial.suggest_categorical(
        "direction_scope", ["global", "per layer"]
    )

    last_layer_index = len(engine.transformer_layers) - 1

    dir_low = (
        0.3 * last_layer_index if last_layer_index < 20 else 0.4 * last_layer_index
    )
    dir_high = (
        0.95 * last_layer_index if last_layer_index < 20 else 0.9 * last_layer_index
    )
    direction_index = trial.suggest_float("direction_index", dir_low, dir_high)

    if direction_scope == "per layer":
        direction_index = None

    parameters = {}
    for component in engine.list_steerable_components():
        max_weight = trial.suggest_float(f"{component}.max_weight", 0.8, 1.5)
        pos_low = (
            0.4 * last_layer_index if last_layer_index < 20 else 0.6 * last_layer_index
        )
        max_weight_position = trial.suggest_float(
            f"{component}.max_weight_position", pos_low, 1.0 * last_layer_index
        )
        min_weight = trial.suggest_float(f"{component}.min_weight", 0.0, 1.0)
        min_weight_distance = trial.suggest_float(
            f"{component}.min_weight_distance", 1.0, 0.6 * last_layer_index
        )
        parameters[component] = SteeringProfile(
            max_weight=max_weight,
            max_weight_position=max_weight_position,
            min_weight=(min_weight * max_weight),
            min_weight_distance=min_weight_distance,
        )

    return direction_index, parameters


def compute_refusal_directions(config, engine, good_prompts, bad_prompts):
    """Compute refusal directions (replicated from main.py)."""
    print("Calculating per-layer refusal directions...")
    print("* Obtaining residuals for good prompts...")
    good_residuals = engine.extract_hidden_states_batched(good_prompts)
    print("* Obtaining residuals for bad prompts...")
    bad_residuals = engine.extract_hidden_states_batched(bad_prompts)

    if config.steering.vector_method == VectorMethod.MEDIAN_OF_MEANS:
        n_groups = 5
        good_chunks = torch.chunk(good_residuals, n_groups, dim=0)
        bad_chunks = torch.chunk(bad_residuals, n_groups, dim=0)
        group_directions = torch.stack(
            [
                F.normalize(bc.mean(dim=0) - gc.mean(dim=0), p=2, dim=1)
                for gc, bc in zip(good_chunks, bad_chunks)
            ],
            dim=0,
        )
        refusal_directions = F.normalize(
            group_directions.median(dim=0).values, p=2, dim=1
        )
    elif config.steering.vector_method == VectorMethod.PCA:
        diff = bad_residuals - good_residuals.mean(dim=0, keepdim=True)
        n_layers = diff.shape[1]
        dirs = []
        for layer_idx in range(n_layers):
            layer_diff = diff[:, layer_idx, :]
            layer_diff = layer_diff - layer_diff.mean(dim=0, keepdim=True)
            _, _, Vh = torch.linalg.svd(layer_diff, full_matrices=False)
            dirs.append(Vh[0])
        refusal_directions = F.normalize(torch.stack(dirs, dim=0), p=2, dim=1)
    else:
        good_means = good_residuals.mean(dim=0)
        bad_means = bad_residuals.mean(dim=0)
        refusal_directions = F.normalize(bad_means - good_means, p=2, dim=1)

    if config.steering.orthogonal_projection:
        good_directions = F.normalize(good_residuals.mean(dim=0), p=2, dim=1)
        projection_vector = torch.sum(refusal_directions * good_directions, dim=1)
        refusal_directions = (
            refusal_directions - projection_vector.unsqueeze(1) * good_directions
        )
        refusal_directions = F.normalize(refusal_directions, p=2, dim=1)

    del good_residuals, bad_residuals
    from prometheus.util import flush_memory

    flush_memory()

    return refusal_directions


# ---------------------------------------------------------------------------
# Experiment 1: output_scores=True overhead
# ---------------------------------------------------------------------------


def experiment_output_scores(engine, evaluator, n_repeats=5):
    """Measure VRAM and speed overhead of output_scores=True."""
    print()
    print("=" * 80)
    print("  EXPERIMENT 1: output_scores=True overhead")
    print("=" * 80)

    prompts = evaluator.benign_msgs
    batches = chunk_batches(prompts, engine.config.inference.batch_size)
    max_new_tokens = engine.config.inference.max_gen_tokens

    results = {}

    for label, kwargs in [
        ("with_scores", dict(output_scores=True, return_dict_in_generate=True)),
        ("without_scores", dict()),
    ]:
        times = []
        vram_deltas = []

        for rep in range(n_repeats):
            # Warm-up GPU (first rep may have compilation overhead)
            if rep == 0:
                engine.generate(batches[0], max_new_tokens=max_new_tokens, **kwargs)

            torch.cuda.reset_peak_memory_stats()
            mem_before = torch.cuda.max_memory_allocated()

            t0 = time.perf_counter()
            for batch in batches:
                engine.generate(batch, max_new_tokens=max_new_tokens, **kwargs)
            elapsed = time.perf_counter() - t0

            mem_peak = torch.cuda.max_memory_allocated()
            vram_delta = mem_peak - mem_before

            times.append(elapsed)
            vram_deltas.append(vram_delta)

        avg_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0.0
        avg_vram = statistics.mean(vram_deltas)

        results[label] = {
            "avg_time": round(avg_time, 3),
            "std_time": round(std_time, 3),
            "avg_vram_delta_gb": round(avg_vram / 1e9, 3),
            "times": [round(t, 3) for t in times],
        }

        print(
            f"  {label}: {avg_time:.2f}s (±{std_time:.2f}s), "
            f"peak VRAM delta: {avg_vram / 1e9:.2f} GB"
        )

    # Comparison
    t_with = results["with_scores"]["avg_time"]
    t_without = results["without_scores"]["avg_time"]
    vram_with = results["with_scores"]["avg_vram_delta_gb"]
    vram_without = results["without_scores"]["avg_vram_delta_gb"]

    speedup = (t_with - t_without) / t_with * 100 if t_with > 0 else 0
    vram_saved = vram_with - vram_without

    print()
    print(f"  Speed difference: {speedup:+.1f}% ({t_with:.2f}s -> {t_without:.2f}s)")
    print(f"  VRAM saved: {vram_saved:.2f} GB")
    print(f"  Verdict: {'WORTH IT' if vram_saved > 0.5 or speedup > 1 else 'MARGINAL'}")

    return results


# ---------------------------------------------------------------------------
# Experiment 2: Partial KL accuracy
# ---------------------------------------------------------------------------


def experiment_partial_kl(engine, evaluator, refusal_directions, study, n_trials=20):
    """Validate first-batch KL as a proxy for full KL."""
    print()
    print("=" * 80)
    print(f"  EXPERIMENT 2: Partial KL accuracy ({n_trials} trials)")
    print("=" * 80)

    batch_size = engine.config.inference.batch_size
    results = []

    for i in range(n_trials):
        trial = study.ask()
        direction_index, parameters = suggest_trial_params(trial, engine)

        engine.restore_baseline()
        apply_steering(engine, refusal_directions, direction_index, parameters)

        # Full KL computation via combined pass
        responses, logprobs = engine.generate_and_score_batched(
            evaluator.benign_msgs,
            max_new_tokens=engine.config.inference.max_gen_tokens,
            kl_token_count=engine.config.kl.token_count,
            skip_special_tokens=True,
        )

        full_kl = F.kl_div(
            logprobs,
            evaluator.baseline_logprobs,
            reduction="batchmean",
            log_target=True,
        ).item()

        # Partial KL (first batch only)
        n_first = min(batch_size, logprobs.shape[0])
        partial_kl = F.kl_div(
            logprobs[:n_first],
            evaluator.baseline_logprobs[:n_first],
            reduction="batchmean",
            log_target=True,
        ).item()

        study.tell(trial, [full_kl, 0.0])
        results.append({"full_kl": full_kl, "partial_kl": partial_kl})
        print(
            f"  Trial {i + 1:2d}: full_kl={full_kl:.6f}, partial_kl={partial_kl:.6f}, "
            f"ratio={partial_kl / max(full_kl, 1e-10):.3f}"
        )

    # Analysis
    full_kls = [r["full_kl"] for r in results]
    partial_kls = [r["partial_kl"] for r in results]

    correlation = statistics.correlation(full_kls, partial_kls)
    print()
    print(f"  Pearson correlation: {correlation:.4f}")
    print(f"  First batch size: {min(batch_size, len(evaluator.benign_msgs))}")

    # Threshold analysis
    print()
    fmt = "  {:>12}  {:>4}  {:>4}  {:>4}  {:>12}  {:>12}"
    print(fmt.format("Threshold", "TP", "FN", "FP", "Sensitivity", "FN rate"))
    print("  " + "-" * 60)

    for threshold in [0.5, 1.0, 2.0, 3.0, 5.0]:
        tp = sum(
            1 for f, p in zip(full_kls, partial_kls) if p > threshold and f > threshold
        )
        fn = sum(
            1 for f, p in zip(full_kls, partial_kls) if p <= threshold and f > threshold
        )
        fp = sum(
            1 for f, p in zip(full_kls, partial_kls) if p > threshold and f <= threshold
        )
        n_above = sum(1 for f in full_kls if f > threshold)
        sensitivity = tp / max(n_above, 1)
        fn_rate = fn / max(n_above, 1)

        print(
            fmt.format(
                f"{threshold:.1f}",
                tp,
                fn,
                fp,
                f"{sensitivity:.1%}",
                f"{fn_rate:.1%}",
            )
        )

    verdict = "RELIABLE" if correlation > 0.9 else "UNRELIABLE"
    print(f"\n  Verdict: Partial KL is {verdict} (correlation={correlation:.4f})")

    return results


# ---------------------------------------------------------------------------
# Experiment 3: Prune rate analysis
# ---------------------------------------------------------------------------


def experiment_prune_rate(checkpoint_dir):
    """Analyze prune rate from existing Optuna checkpoint."""
    print()
    print("=" * 80)
    print("  EXPERIMENT 3: Prune rate analysis")
    print("=" * 80)

    journal_path = os.path.join(checkpoint_dir, "journal.log")
    if not os.path.exists(journal_path):
        print(f"  No checkpoint found at {journal_path}")
        print("  Skipping experiment 3.")
        return None

    from optuna.storages import JournalFileBackend, JournalStorage
    from optuna.trial import TrialState

    storage = JournalStorage(JournalFileBackend(journal_path))
    study = optuna.load_study(study_name="prometheus", storage=storage)

    trials = study.trials
    total = len(trials)
    pruned = sum(1 for t in trials if t.state == TrialState.PRUNED)
    completed = sum(1 for t in trials if t.state == TrialState.COMPLETE)
    failed = sum(1 for t in trials if t.state == TrialState.FAIL)

    print(f"  Checkpoint: {checkpoint_dir}")
    print(f"  Total trials: {total}")
    print(f"  Completed: {completed} ({completed / total:.1%})")
    print(f"  Pruned: {pruned} ({pruned / total:.1%})")
    if failed:
        print(f"  Failed: {failed} ({failed / total:.1%})")

    # Prune rate by phase (assuming n_startup_trials=15)
    n_startup = 15
    if total > n_startup:
        startup_trials = trials[:n_startup]
        tpe_trials = trials[n_startup:]
        startup_pruned = sum(1 for t in startup_trials if t.state == TrialState.PRUNED)
        tpe_pruned = sum(1 for t in tpe_trials if t.state == TrialState.PRUNED)
        print()
        print(
            f"  Startup phase (first {n_startup}): "
            f"{startup_pruned} pruned ({startup_pruned / n_startup:.1%})"
        )
        print(
            f"  TPE phase (remaining {len(tpe_trials)}): "
            f"{tpe_pruned} pruned ({tpe_pruned / len(tpe_trials):.1%})"
        )

    # Estimated savings from early batch pruning
    # Assume good_eval takes ~18.7s and batch_size=64 with 100 prompts → 2 batches
    # Early exit after batch 1 saves ~50% of good_eval = ~9.4s per pruned trial
    est_savings_per_trial = 9.4  # seconds
    total_savings = pruned * est_savings_per_trial
    avg_savings = total_savings / total if total > 0 else 0

    print()
    print("  Estimated savings from early batch-level KL pruning:")
    print(f"    Per pruned trial: ~{est_savings_per_trial:.1f}s")
    print(f"    Total ({pruned} pruned trials): ~{total_savings:.0f}s")
    print(f"    Average per trial: ~{avg_savings:.1f}s")

    prune_rate = pruned / total if total > 0 else 0
    verdict = "WORTH IT" if prune_rate > 0.10 else "MARGINAL"
    print(f"  Verdict: {verdict} (prune rate={prune_rate:.1%})")

    return {
        "total": total,
        "pruned": pruned,
        "completed": completed,
        "prune_rate": round(prune_rate, 3),
    }


# ---------------------------------------------------------------------------
# Experiment 4: Abliteration metadata traversal overhead
# ---------------------------------------------------------------------------


def experiment_abliteration_breakdown(engine, refusal_directions, study, n_repeats=10):
    """Profile abliteration time: metadata traversal vs actual computation."""
    print()
    print("=" * 80)
    print(f"  EXPERIMENT 4: Abliteration breakdown ({n_repeats} repeats)")
    print("=" * 80)

    # Get a single set of parameters
    trial = study.ask()
    direction_index, parameters = suggest_trial_params(trial, engine)
    study.tell(trial, [0.0, 0.0])

    glm_times = []
    total_times = []
    get_layers_times = []

    for rep in range(n_repeats):
        engine.restore_baseline()

        # Monkey-patch get_layer_modules to measure time
        original_glm = engine.get_layer_modules
        glm_acc = [0.0]

        def timed_glm(layer_index):
            t = time.perf_counter()
            result = original_glm(layer_index)
            glm_acc[0] += time.perf_counter() - t
            return result

        # Monkey-patch transformer_layers access to measure time
        original_gl = type(engine).transformer_layers.fget
        gl_acc = [0.0]

        def timed_gl(self):
            t = time.perf_counter()
            result = original_gl(self)
            gl_acc[0] += time.perf_counter() - t
            return result

        engine.get_layer_modules = timed_glm
        type(engine).transformer_layers = property(timed_gl)

        t0 = time.perf_counter()
        apply_steering(engine, refusal_directions, direction_index, parameters)
        t_total = time.perf_counter() - t0

        engine.get_layer_modules = original_glm
        type(engine).transformer_layers = property(original_gl)

        glm_times.append(glm_acc[0])
        get_layers_times.append(gl_acc[0])
        total_times.append(t_total)

    print(
        f"  abliterate() total:  {statistics.mean(total_times) * 1000:7.1f}ms "
        f"(±{statistics.stdev(total_times) * 1000:.1f}ms)"
    )
    print(
        f"  get_layer_modules(): {statistics.mean(glm_times) * 1000:7.1f}ms "
        f"(±{statistics.stdev(glm_times) * 1000:.1f}ms)"
    )
    print(
        f"  get_layers():        {statistics.mean(get_layers_times) * 1000:7.1f}ms "
        f"(±{statistics.stdev(get_layers_times) * 1000:.1f}ms)"
    )

    metadata_pct = (
        (statistics.mean(glm_times) + statistics.mean(get_layers_times))
        / statistics.mean(total_times)
        * 100
        if statistics.mean(total_times) > 0
        else 0
    )
    print(f"  Metadata overhead:   {metadata_pct:.1f}% of abliterate()")

    verdict = "LOW PRIORITY" if metadata_pct < 5 else "WORTH CACHING"
    print(f"  Verdict: {verdict}")

    return {
        "total_ms": round(statistics.mean(total_times) * 1000, 1),
        "glm_ms": round(statistics.mean(glm_times) * 1000, 1),
        "gl_ms": round(statistics.mean(get_layers_times) * 1000, 1),
        "metadata_pct": round(metadata_pct, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic experiments for Prometheus optimizations"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3.5-0.8B",
        help="Hugging Face model ID (default: Qwen/Qwen3.5-0.8B)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Number of trials for partial KL experiment (default: 20)",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints_50t_llm",
        help="Checkpoint directory for prune rate analysis",
    )
    parser.add_argument(
        "--output",
        default="benchmarks",
        help="Output directory for results (default: benchmarks)",
    )
    parser.add_argument(
        "--experiment",
        type=int,
        nargs="*",
        default=None,
        help="Run specific experiments (1-4). Default: run all.",
    )
    args = parser.parse_args()

    # Suppress Optuna experimental warnings.
    import warnings

    from optuna.exceptions import ExperimentalWarning

    warnings.filterwarnings("ignore", category=ExperimentalWarning)

    # Disable gradient computation globally.
    torch.set_grad_enabled(False)

    experiments_to_run = set(args.experiment) if args.experiment else {1, 2, 3, 4}

    # Experiment 3 doesn't need model loading
    if experiments_to_run == {3}:
        results = {"experiment_3": experiment_prune_rate(args.checkpoint)}
        os.makedirs(args.output, exist_ok=True)
        output_path = os.path.join(args.output, "optimization_experiments.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_path}")
        return

    # Load config.
    os.environ.setdefault("PM_CONFIG", "prometheus.toml")
    sys.argv = ["benchmark", "--model.model-id", args.model]
    config = PrometheusConfig()

    print("Experiment configuration:")
    print(f"  Model: {config.model.model_id}")
    print(f"  kl_n_tokens: {config.kl.token_count}")
    print(f"  max_response_length: {config.inference.max_gen_tokens}")
    print(f"  Seed: {config.optimization.sampler_seed}")

    # Load engine.
    print()
    engine = SteeringEngine(config)

    # Load prompts.
    print()
    good_prompts = load_prompt_dataset(config, config.benign_prompts)
    bad_prompts = load_prompt_dataset(config, config.target_prompts)
    print(f"Training prompts: {len(good_prompts)} benign, {len(bad_prompts)} target")

    # Batch size detection
    if config.inference.batch_size == 0:
        print()
        print("Determining optimal batch size...")
        batch_size = 1
        while batch_size <= config.max_batch_size:
            test = good_prompts * math.ceil(batch_size / len(good_prompts))
            test = test[:batch_size]
            try:
                engine.generate_text(test)
                batch_size *= 2
            except Exception:
                batch_size //= 2
                break
        batch_size = max(batch_size, 1)
        config.inference.batch_size = batch_size
        print(f"  Using batch size: {batch_size}")

    # Detect common prefix.
    print()
    print("Detecting common prefix...")
    from os.path import commonprefix

    prefix_prompts = good_prompts[:10] + bad_prompts[:10]
    prefix_responses = engine.generate_text_batched(prefix_prompts)
    engine.response_prefix = commonprefix(prefix_responses).rstrip(" ")
    if engine.response_prefix:
        print(f"  Detected prefix: {engine.response_prefix!r}")
    else:
        print("  No common prefix detected.")

    results = {}

    # Initialize evaluator (this computes baselines) — needed for experiments 1, 2
    evaluator = None
    if experiments_to_run & {1, 2}:
        print()
        evaluator = TrialScorer(config, engine, RefusalDetector(config))

    # Compute refusal directions — needed for experiments 2, 4
    refusal_directions = None
    if experiments_to_run & {2, 4}:
        print()
        refusal_directions = compute_refusal_directions(
            config, engine, good_prompts, bad_prompts
        )

    # Create Optuna study for parameter suggestion
    seed = config.optimization.sampler_seed or 42
    sampler = TPESampler(
        seed=seed,
        multivariate=True,
        group=True,
        n_startup_trials=max(args.trials + 10, 20),
    )
    study = optuna.create_study(
        directions=["minimize", "minimize"],
        sampler=sampler,
    )

    # Run experiments
    if 1 in experiments_to_run:
        results["experiment_1"] = experiment_output_scores(engine, evaluator)

    if 2 in experiments_to_run:
        results["experiment_2"] = experiment_partial_kl(
            engine, evaluator, refusal_directions, study, n_trials=args.trials
        )

    if 3 in experiments_to_run:
        results["experiment_3"] = experiment_prune_rate(args.checkpoint)

    if 4 in experiments_to_run:
        results["experiment_4"] = experiment_abliteration_breakdown(
            engine, refusal_directions, study
        )

    # Save results
    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(args.output, "optimization_experiments.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")

    # Summary
    print()
    print("=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    if "experiment_1" in results:
        e1 = results["experiment_1"]
        t_diff = e1["with_scores"]["avg_time"] - e1["without_scores"]["avg_time"]
        v_diff = (
            e1["with_scores"]["avg_vram_delta_gb"]
            - e1["without_scores"]["avg_vram_delta_gb"]
        )
        print(f"  Exp 1 (output_scores): {t_diff:+.2f}s, {v_diff:+.2f} GB VRAM")
    if "experiment_2" in results:
        kls = results["experiment_2"]
        full = [r["full_kl"] for r in kls]
        partial = [r["partial_kl"] for r in kls]
        corr = statistics.correlation(full, partial)
        print(f"  Exp 2 (partial KL): correlation={corr:.4f}")
    if "experiment_3" in results and results["experiment_3"]:
        e3 = results["experiment_3"]
        print(
            f"  Exp 3 (prune rate): {e3['pruned']}/{e3['total']} = {e3['prune_rate']:.1%}"
        )
    if "experiment_4" in results:
        e4 = results["experiment_4"]
        print(
            f"  Exp 4 (metadata): {e4['metadata_pct']:.1f}% of abliterate() "
            f"({e4['glm_ms'] + e4['gl_ms']:.1f}ms / {e4['total_ms']:.1f}ms)"
        )


if __name__ == "__main__":
    main()
