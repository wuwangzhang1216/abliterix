# Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>

"""Automated steering experiment runner.

Generates variant config TOML files from a baseline template and runs Prometheus
for each variant in non-interactive mode.

Usage:
    python scripts/run_sweep.py [--model MODEL] [--trials N] [--startup N]
                                [--seed SEED] [--variants V1,V2,...]
                                [--output-dir DIR]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Experiment matrix
# ---------------------------------------------------------------------------

EXPERIMENT_VARIANTS = {
    "baseline": {
        "vector_method": "mean",
        "decay_kernel": "linear",
        "orthogonal_projection": False,
        "weight_normalization": "none",
    },
    "B1_median_of_means": {
        "vector_method": "median_of_means",
        "decay_kernel": "linear",
        "orthogonal_projection": False,
        "weight_normalization": "none",
    },
    "B2_pca": {
        "vector_method": "pca",
        "decay_kernel": "linear",
        "orthogonal_projection": False,
        "weight_normalization": "none",
    },
    "C1_gaussian": {
        "vector_method": "mean",
        "decay_kernel": "gaussian",
        "orthogonal_projection": False,
        "weight_normalization": "none",
    },
    "C2_cosine": {
        "vector_method": "mean",
        "decay_kernel": "cosine",
        "orthogonal_projection": False,
        "weight_normalization": "none",
    },
    "D1_orthogonalize": {
        "vector_method": "mean",
        "decay_kernel": "linear",
        "orthogonal_projection": True,
        "weight_normalization": "none",
    },
    "D2_weight_norm_pre": {
        "vector_method": "mean",
        "decay_kernel": "linear",
        "orthogonal_projection": False,
        "weight_normalization": "pre",
    },
    "D3_weight_norm_full": {
        "vector_method": "mean",
        "decay_kernel": "linear",
        "orthogonal_projection": False,
        "weight_normalization": "full",
    },
    "E1_ortho_cosine": {
        "vector_method": "mean",
        "decay_kernel": "cosine",
        "orthogonal_projection": True,
        "weight_normalization": "none",
    },
    "E2_ortho_pre": {
        "vector_method": "mean",
        "decay_kernel": "linear",
        "orthogonal_projection": True,
        "weight_normalization": "pre",
    },
    "E3_cosine_pre": {
        "vector_method": "mean",
        "decay_kernel": "cosine",
        "orthogonal_projection": False,
        "weight_normalization": "pre",
    },
    "E4_ortho_cosine_pre": {
        "vector_method": "mean",
        "decay_kernel": "cosine",
        "orthogonal_projection": True,
        "weight_normalization": "pre",
    },
}


def generate_config(
    variant_name: str,
    variant_params: dict,
    model: str,
    n_trials: int,
    n_warmup: int,
    seed: int,
    output_dir: str,
) -> str:
    """Generate a nested TOML config file for a specific experiment variant."""
    checkpoint_dir = os.path.join(output_dir, "checkpoints", variant_name)

    def toml_val(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        elif isinstance(v, str):
            return f'"{v}"'
        return str(v)

    config = f"""# Auto-generated experiment config: {variant_name}

non_interactive = true
overwrite_checkpoint = true

[model]
model_id = "{model}"
quant_method = "none"
dtype_fallback_order = ["auto", "float16", "bfloat16"]
use_torch_compile = false
device_map = "auto"

[inference]
batch_size = 0
max_batch_size = 32
max_gen_tokens = 50

[steering]
vector_method = {toml_val(variant_params["vector_method"])}
decay_kernel = {toml_val(variant_params["decay_kernel"])}
orthogonal_projection = {toml_val(variant_params["orthogonal_projection"])}
weight_normalization = {toml_val(variant_params["weight_normalization"])}

[optimization]
num_trials = {n_trials}
num_warmup_trials = {n_warmup}
checkpoint_dir = "{checkpoint_dir.replace(os.sep, "/")}"
sampler_seed = {seed}

[kl]
scale = 1.0
target = 0.01
token_count = 3
prune_threshold = 5.0

[detection]
llm_judge = true
llm_judge_model = "google/gemini-3.1-flash-lite-preview"
llm_judge_batch_size = 10
llm_judge_concurrency = 10

[display]
print_responses = false
print_residual_geometry = false
plot_residuals = false

[benign_prompts]
dataset = "datasets/good_1000"
split = "train[:800]"
column = "prompt"

[target_prompts]
dataset = "datasets/harmful_1000"
split = "train[:800]"
column = "prompt"

[benign_eval_prompts]
dataset = "datasets/good_1000"
split = "train[800:]"
column = "prompt"

[target_eval_prompts]
dataset = "datasets/harmful_1000"
split = "train[800:]"
column = "prompt"
"""

    config_dir = os.path.join(output_dir, "configs")
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, f"{variant_name}.toml")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config)

    return config_path


def run_variant(variant_name: str, config_path: str) -> dict:
    """Run a single experiment variant and return timing info."""
    print(f"\n{'=' * 60}")
    print(f"  Running variant: {variant_name}")
    print(f"  Config: {config_path}")
    print(f"{'=' * 60}\n")

    start_time = time.time()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    abs_config = os.path.abspath(config_path)

    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_prometheus.py"),
    ]

    env = os.environ.copy()
    env["PM_CONFIG"] = abs_config

    result = subprocess.run(
        cmd,
        cwd=project_root,
        env=env,
        timeout=7200,
    )

    elapsed = time.time() - start_time

    return {
        "variant": variant_name,
        "returncode": result.returncode,
        "elapsed_seconds": round(elapsed, 1),
    }


def extract_results(checkpoint_dir: str, variant_name: str) -> dict | None:
    """Extract trial results from an Optuna JournalFileBackend checkpoint."""
    ckpt_dir = os.path.join(checkpoint_dir, variant_name)
    if not os.path.isdir(ckpt_dir):
        print(f"  [WARN] Checkpoint dir not found: {ckpt_dir}")
        return None

    jsonl_files = list(Path(ckpt_dir).glob("*.jsonl"))
    if not jsonl_files:
        print(f"  [WARN] No .jsonl checkpoint files found in: {ckpt_dir}")
        return None

    checkpoint_file = str(jsonl_files[0])

    try:
        import optuna
        from optuna.storages import JournalStorage
        from optuna.storages.journal import JournalFileBackend
        from optuna.trial import TrialState

        backend = JournalFileBackend(checkpoint_file)
        storage = JournalStorage(backend)
        study = optuna.load_study(study_name="prometheus", storage=storage)

        trials_data = []
        for trial in study.trials:
            if trial.state == TrialState.COMPLETE:
                trials_data.append(
                    {
                        "trial_number": trial.number,
                        "kl_divergence": trial.user_attrs.get("kl_divergence"),
                        "refusals": trial.user_attrs.get("refusals"),
                        "length_deviation": trial.user_attrs.get("length_deviation"),
                        "values": list(trial.values) if trial.values else None,
                        "parameters": trial.user_attrs.get("parameters"),
                        "vector_index": trial.user_attrs.get("vector_index"),
                    }
                )

        if not trials_data:
            print(f"  [WARN] No completed trials for variant: {variant_name}")
            return None

        best_refusal = min(trials_data, key=lambda t: t["refusals"] or float("inf"))
        best_kl = min(trials_data, key=lambda t: t["kl_divergence"] or float("inf"))

        kl_thresholds = [0.05, 0.1, 0.2]
        threshold_results = {}
        for th in kl_thresholds:
            eligible = [
                t
                for t in trials_data
                if t["kl_divergence"] is not None and t["kl_divergence"] <= th
            ]
            if eligible:
                best = min(eligible, key=lambda t: t["refusals"] or float("inf"))
                threshold_results[str(th)] = {
                    "min_refusals": best["refusals"],
                    "kl_divergence": best["kl_divergence"],
                    "length_deviation": best.get("length_deviation"),
                    "trial_number": best["trial_number"],
                }
            else:
                threshold_results[str(th)] = None

        zero_refusal = [t for t in trials_data if t["refusals"] == 0]
        zero_refusal_best_kl = None
        if zero_refusal:
            best_zero = min(
                zero_refusal, key=lambda t: t["kl_divergence"] or float("inf")
            )
            zero_refusal_best_kl = best_zero["kl_divergence"]

        return {
            "variant": variant_name,
            "n_completed_trials": len(trials_data),
            "best_refusal": {
                "refusals": best_refusal["refusals"],
                "kl_divergence": best_refusal["kl_divergence"],
                "length_deviation": best_refusal.get("length_deviation"),
                "trial_number": best_refusal["trial_number"],
            },
            "best_kl": {
                "kl_divergence": best_kl["kl_divergence"],
                "refusals": best_kl["refusals"],
                "length_deviation": best_kl.get("length_deviation"),
                "trial_number": best_kl["trial_number"],
            },
            "kl_threshold_analysis": threshold_results,
            "zero_refusal_best_kl": zero_refusal_best_kl,
            "all_trials": trials_data,
        }

    except Exception as e:
        print(f"  [ERROR] Failed to extract results for {variant_name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Run Prometheus steering experiments")
    parser.add_argument(
        "--model", default="Qwen/Qwen3.5-0.8B", help="Hugging Face model ID"
    )
    parser.add_argument(
        "--trials", type=int, default=50, help="Number of trials per variant"
    )
    parser.add_argument(
        "--startup", type=int, default=15, help="Number of warmup trials"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--variants",
        type=str,
        default=None,
        help=f"Comma-separated variants (default: all). Available: {','.join(EXPERIMENT_VARIANTS.keys())}",
    )
    parser.add_argument("--output-dir", default="experiments", help="Output directory")
    args = parser.parse_args()

    if args.variants:
        selected = [v.strip() for v in args.variants.split(",")]
        for v in selected:
            if v not in EXPERIMENT_VARIANTS:
                print(f"Unknown variant: {v}")
                print(f"Available: {', '.join(EXPERIMENT_VARIANTS.keys())}")
                sys.exit(1)
    else:
        selected = list(EXPERIMENT_VARIANTS.keys())

    print("Experiment configuration:")
    print(f"  Model: {args.model}")
    print(f"  Trials: {args.trials} ({args.startup} warmup)")
    print(f"  Seed: {args.seed}")
    print(f"  Variants: {', '.join(selected)}")
    print(f"  Output: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    configs = {}
    for name in selected:
        path = generate_config(
            variant_name=name,
            variant_params=EXPERIMENT_VARIANTS[name],
            model=args.model,
            n_trials=args.trials,
            n_warmup=args.startup,
            seed=args.seed,
            output_dir=args.output_dir,
        )
        configs[name] = path
        print(f"  Generated: {path}")

    run_results = []
    for name in selected:
        try:
            result = run_variant(name, configs[name])
            run_results.append(result)
            status = (
                "OK"
                if result["returncode"] == 0
                else f"FAILED (rc={result['returncode']})"
            )
            print(f"\n  {name}: {status} in {result['elapsed_seconds']:.0f}s")
        except subprocess.TimeoutExpired:
            print(f"\n  {name}: TIMEOUT (>2h)")
            run_results.append(
                {"variant": name, "returncode": -1, "elapsed_seconds": 7200}
            )
        except Exception as e:
            print(f"\n  {name}: ERROR: {e}")
            run_results.append(
                {"variant": name, "returncode": -2, "elapsed_seconds": 0}
            )

    print(f"\n{'=' * 60}")
    print("  Extracting results from checkpoints...")
    print(f"{'=' * 60}\n")

    all_results = {}
    checkpoint_base = os.path.join(args.output_dir, "checkpoints")
    for name in selected:
        result = extract_results(checkpoint_base, name)
        if result:
            run_info = next((r for r in run_results if r["variant"] == name), {})
            result["elapsed_seconds"] = run_info.get("elapsed_seconds")
            result["returncode"] = run_info.get("returncode")
            all_results[name] = result

    summary_path = os.path.join(args.output_dir, "results_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {summary_path}")

    print(f"\n{'=' * 60}")
    print("  Quick Summary")
    print(f"{'=' * 60}")
    print(f"{'Variant':<25} {'Trials':>7} {'Best Ref':>9} {'Best KL':>9} {'Time':>8}")
    print(f"{'-' * 25} {'-' * 7} {'-' * 9} {'-' * 9} {'-' * 8}")
    for name in selected:
        if name in all_results:
            r = all_results[name]
            ref = r["best_refusal"]["refusals"]
            kl = r["best_kl"]["kl_divergence"]
            t = r.get("elapsed_seconds", 0)
            n = r["n_completed_trials"]
            print(f"{name:<25} {n:>7} {ref:>9} {kl:>9.4f} {t:>7.0f}s")
        else:
            print(f"{name:<25} {'FAILED':>7}")


if __name__ == "__main__":
    main()
