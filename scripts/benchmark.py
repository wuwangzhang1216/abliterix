# Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>

"""A/B speed comparison benchmark for Prometheus inference optimizations.

Runs N trials with BOTH the old (separate-pass) and new (combined-pass) evaluation
methods, interleaved in the same process.

Usage:
    python scripts/benchmark.py [--trials N] [--output DIR]
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
from optuna.samplers import TPESampler

from prometheus.core.engine import SteeringEngine
from prometheus.core.steering import apply_steering
from prometheus.data import load_prompt_dataset
from prometheus.eval.detector import RefusalDetector
from prometheus.eval.scorer import TrialScorer
from prometheus.settings import PrometheusConfig
from prometheus.types import SteeringProfile
from prometheus.util import flush_memory, print
from prometheus.vectors import compute_steering_vectors


def suggest_trial_params(trial, engine):
    """Replicate the parameter suggestion logic from optimizer.py."""
    vector_scope = trial.suggest_categorical("vector_scope", ["global", "per layer"])
    last_layer = len(engine.transformer_layers) - 1

    lo = 0.3 * last_layer if last_layer < 20 else 0.4 * last_layer
    hi = 0.95 * last_layer if last_layer < 20 else 0.9 * last_layer
    vector_index = trial.suggest_float("vector_index", lo, hi)

    if vector_scope == "per layer":
        vector_index = None

    profiles = {}
    for component in engine.list_steerable_components():
        max_w = trial.suggest_float(f"{component}.max_weight", 0.8, 1.5)
        pos_lo = 0.4 * last_layer if last_layer < 20 else 0.6 * last_layer
        peak_pos = trial.suggest_float(
            f"{component}.max_weight_position",
            pos_lo,
            1.0 * last_layer,
        )
        min_frac = trial.suggest_float(f"{component}.min_weight", 0.0, 1.0)
        falloff = trial.suggest_float(
            f"{component}.min_weight_distance",
            1.0,
            0.6 * last_layer,
        )
        profiles[component] = SteeringProfile(
            max_weight=max_w,
            max_weight_position=peak_pos,
            min_weight=(min_frac * max_w),
            min_weight_distance=falloff,
        )

    return vector_index, profiles


def compute_vectors(config, engine, benign_msgs, target_msgs):
    """Compute steering vectors."""
    print("Computing per-layer steering vectors...")
    print("* Extracting residuals for benign prompts...")
    benign_states = engine.extract_hidden_states_batched(benign_msgs)
    print("* Extracting residuals for target prompts...")
    target_states = engine.extract_hidden_states_batched(target_msgs)

    vectors = compute_steering_vectors(
        benign_states,
        target_states,
        config.steering.vector_method,
        config.steering.orthogonal_projection,
    )

    del benign_states, target_states
    flush_memory()

    return vectors


def run_trial_old_method(scorer, engine):
    """Evaluate using the old separate-pass method (2 passes over benign_msgs)."""
    t0 = time.perf_counter()
    kl = scorer.measure_kl_divergence(engine)
    t_kl = time.perf_counter() - t0

    t0 = time.perf_counter()
    length_dev = scorer.measure_coherence(engine)
    t_len = time.perf_counter() - t0

    t0 = time.perf_counter()
    refusals = scorer.detector.evaluate_compliance(engine, scorer.target_msgs)
    t_ref = time.perf_counter() - t0

    return {
        "kl_divergence": kl,
        "length_deviation": length_dev,
        "refusals": refusals,
        "t_kl": round(t_kl, 3),
        "t_length": round(t_len, 3),
        "t_good_eval": round(t_kl + t_len, 3),
        "t_bad_eval": round(t_ref, 3),
        "t_total": round(t_kl + t_len + t_ref, 3),
    }


def run_trial_new_method(scorer, engine):
    """Evaluate using the new combined-pass method (1 pass over benign_msgs)."""
    t0 = time.perf_counter()
    kl, length_dev = scorer.measure_kl_and_coherence(engine)
    t_combined = time.perf_counter() - t0

    t0 = time.perf_counter()
    refusals = scorer.detector.evaluate_compliance(engine, scorer.target_msgs)
    t_ref = time.perf_counter() - t0

    return {
        "kl_divergence": kl,
        "length_deviation": length_dev,
        "refusals": refusals,
        "t_kl": None,
        "t_length": None,
        "t_good_eval": round(t_combined, 3),
        "t_bad_eval": round(t_ref, 3),
        "t_total": round(t_combined + t_ref, 3),
    }


def print_comparison(results):
    """Print a formatted comparison table."""
    old_trials = results["old"]
    new_trials = results["new"]
    n = len(old_trials)

    print()
    print("=" * 80)
    print(f"  SPEED COMPARISON ({n} trials)")
    print("=" * 80)

    old_good = [t["t_good_eval"] for t in old_trials]
    new_good = [t["t_good_eval"] for t in new_trials]
    old_bad = [t["t_bad_eval"] for t in old_trials]
    new_bad = [t["t_bad_eval"] for t in new_trials]
    old_total = [t["t_total"] for t in old_trials]
    new_total = [t["t_total"] for t in new_trials]

    def fmt_speedup(old_vals, new_vals):
        old_m = statistics.mean(old_vals)
        new_m = statistics.mean(new_vals)
        pct = (old_m - new_m) / old_m * 100 if old_m > 0 else 0
        return f"{old_m:.1f}s -> {new_m:.1f}s  ({pct:+.1f}%)"

    print(f"  Good eval (KL+length): {fmt_speedup(old_good, new_good)}")
    print(f"  Bad eval (refusals):   {fmt_speedup(old_bad, new_bad)}")
    print(f"  Total per trial:       {fmt_speedup(old_total, new_total)}")

    old_kl_times = [t["t_kl"] for t in old_trials if t["t_kl"] is not None]
    old_len_times = [t["t_length"] for t in old_trials if t["t_length"] is not None]
    if old_kl_times and old_len_times:
        print(
            f"  (Old breakdown: KL={statistics.mean(old_kl_times):.1f}s + "
            f"length={statistics.mean(old_len_times):.1f}s)"
        )

    print()
    print("=" * 80)
    print("  CONSISTENCY CHECK")
    print("=" * 80)

    fmt = "{:>6}  {:>12}  {:>12}  {:>6}  {:>6}  {:>6}  {:>6}  {:>6}  {:>6}"
    print(
        fmt.format(
            "Trial",
            "KL_old",
            "KL_new",
            "KL ok?",
            "Ref_o",
            "Ref_n",
            "R ok?",
            "LD_o",
            "LD_n",
        )
    )
    print("-" * 80)

    all_kl_ok = all_ref_ok = all_ld_ok = True

    for i in range(n):
        old = old_trials[i]
        new = new_trials[i]

        kl_diff = abs(old["kl_divergence"] - new["kl_divergence"])
        kl_ok = kl_diff < 1e-4
        ref_ok = old["refusals"] == new["refusals"]
        ld_diff = abs(old["length_deviation"] - new["length_deviation"])
        ld_ok = ld_diff < 0.01

        if not kl_ok:
            all_kl_ok = False
        if not ref_ok:
            all_ref_ok = False
        if not ld_ok:
            all_ld_ok = False

        print(
            fmt.format(
                i + 1,
                f"{old['kl_divergence']:.6f}",
                f"{new['kl_divergence']:.6f}",
                "Y" if kl_ok else f"N({kl_diff:.2e})",
                old["refusals"],
                new["refusals"],
                "Y" if ref_ok else "N",
                f"{old['length_deviation']:.3f}",
                f"{new['length_deviation']:.3f}",
            )
        )

    print()
    status = [
        f"KL: {'PASS' if all_kl_ok else 'FAIL'}",
        f"Refusals: {'PASS' if all_ref_ok else 'FAIL'}",
        f"LengthDev: {'PASS' if all_ld_ok else 'FAIL'}",
    ]
    overall = all_kl_ok and all_ref_ok and all_ld_ok
    print(f"  {' | '.join(status)} | Overall: {'PASS' if overall else 'FAIL'}")


def main():
    parser = argparse.ArgumentParser(description="A/B speed benchmark for Prometheus")
    parser.add_argument(
        "--model", default="Qwen/Qwen3.5-0.8B", help="Hugging Face model ID"
    )
    parser.add_argument(
        "--trials", type=int, default=5, help="Number of benchmark trials"
    )
    parser.add_argument("--output", default="benchmarks", help="Output directory")
    args = parser.parse_args()

    import warnings
    from optuna.exceptions import ExperimentalWarning

    warnings.filterwarnings("ignore", category=ExperimentalWarning)

    torch.set_grad_enabled(False)

    os.environ.setdefault("PM_CONFIG", "prometheus.toml")
    sys.argv = ["benchmark", "--model.model-id", args.model]
    config = PrometheusConfig()

    print("Benchmark configuration:")
    print(f"  Model: {config.model.model_id}")
    print(f"  Trials: {args.trials}")
    print(f"  kl_token_count: {config.kl.token_count}")
    print(f"  max_gen_tokens: {config.inference.max_gen_tokens}")
    print(f"  Seed: {config.optimization.sampler_seed}")

    print()
    engine = SteeringEngine(config)

    print()
    benign_msgs = load_prompt_dataset(config, config.benign_prompts)
    target_msgs = load_prompt_dataset(config, config.target_prompts)
    print(f"Training prompts: {len(benign_msgs)} benign, {len(target_msgs)} target")

    if config.inference.batch_size == 0:
        print()
        print("Determining optimal batch size...")
        bs = 1
        while bs <= config.inference.max_batch_size:
            test = benign_msgs * math.ceil(bs / len(benign_msgs))
            test = test[:bs]
            try:
                engine.generate_text(test)
                bs *= 2
            except Exception:
                bs //= 2
                break
        config.inference.batch_size = max(bs, 1)
        print(f"  Using batch size: {config.inference.batch_size}")

    print()
    print("Detecting common prefix...")
    from os.path import commonprefix

    prefix_prompts = benign_msgs[:10] + target_msgs[:10]
    prefix_resp = engine.generate_text_batched(prefix_prompts)
    engine.response_prefix = commonprefix(prefix_resp).rstrip(" ")
    if engine.response_prefix:
        print(f"  Detected prefix: {engine.response_prefix!r}")
    else:
        print("  No common prefix detected.")

    detector = RefusalDetector(config)
    scorer = TrialScorer(config, engine, detector)

    print()
    vectors = compute_vectors(config, engine, benign_msgs, target_msgs)

    seed = config.optimization.sampler_seed or 42
    sampler = TPESampler(
        seed=seed, multivariate=True, group=True, n_startup_trials=max(args.trials, 5)
    )
    study = optuna.create_study(directions=["minimize", "minimize"], sampler=sampler)

    print()
    print("=" * 80)
    print(f"  RUNNING {args.trials} INTERLEAVED A/B BENCHMARK TRIALS")
    print("=" * 80)

    results = {
        "old": [],
        "new": [],
        "config": {
            "model": config.model.model_id,
            "trials": args.trials,
            "kl_token_count": config.kl.token_count,
            "max_gen_tokens": config.inference.max_gen_tokens,
            "batch_size": config.inference.batch_size,
            "seed": seed,
        },
    }

    for idx in range(args.trials):
        trial = study.ask()
        vector_index, profiles = suggest_trial_params(trial, engine)

        print(f"\n--- Trial {idx + 1}/{args.trials} ---")

        print("  [A] Old method (separate passes)...")
        engine.restore_baseline()
        apply_steering(engine, vectors, vector_index, profiles)
        old_result = run_trial_old_method(scorer, engine)
        print(
            f"      KL={old_result['kl_divergence']:.4f}  "
            f"Ref={old_result['refusals']}  "
            f"LD={old_result['length_deviation']:.3f}  "
            f"Time={old_result['t_total']:.1f}s"
        )

        print("  [B] New method (combined pass)...")
        engine.restore_baseline()
        apply_steering(engine, vectors, vector_index, profiles)
        new_result = run_trial_new_method(scorer, engine)
        print(
            f"      KL={new_result['kl_divergence']:.4f}  "
            f"Ref={new_result['refusals']}  "
            f"LD={new_result['length_deviation']:.3f}  "
            f"Time={new_result['t_total']:.1f}s"
        )

        study.tell(trial, [old_result["kl_divergence"], old_result["refusals"]])

        results["old"].append(old_result)
        results["new"].append(new_result)

    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(args.output, "benchmark_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")

    print_comparison(results)


if __name__ == "__main__":
    main()
