# Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>

"""Analyze and visualize results from Prometheus abliteration experiments.

Reads results_summary.json produced by run_experiment.py and generates:
1. Summary table comparing all variants
2. Pareto front scatter plot (KL divergence vs refusals)
3. Fixed-KL-threshold refusal comparison bar chart
4. Timing comparison

Usage:
    python scripts/analyze_sweep.py [--input DIR] [--output DIR]
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


def load_results(input_dir: str) -> dict:
    """Load experiment results from results_summary.json."""
    path = os.path.join(input_dir, "results_summary.json")
    if not os.path.exists(path):
        print(f"Error: {path} not found. Run run_experiment.py first.")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_summary_table(results: dict):
    """Print a detailed summary table of all experiment variants."""
    print("\n" + "=" * 100)
    print("  EXPERIMENT RESULTS SUMMARY")
    print("=" * 100)

    # Header.
    headers = [
        "Variant",
        "Trials",
        "Best Ref",
        "@ KL",
        "@ LenDev",
        "Best KL",
        "@ Ref",
        "0-Ref KL",
        "Time",
    ]
    fmt = "{:<25} {:>7} {:>8} {:>8} {:>8} {:>8} {:>6} {:>8} {:>8}"
    print(fmt.format(*headers))
    print("-" * 100)

    for name, r in results.items():
        br = r["best_refusal"]
        bk = r["best_kl"]
        zr_kl = r.get("zero_refusal_best_kl")
        t = r.get("elapsed_seconds", 0)

        br_ld = (
            f"{br['length_deviation']:.2f}"
            if br.get("length_deviation") is not None
            else "N/A"
        )
        zr_str = f"{zr_kl:.4f}" if zr_kl is not None else "N/A"

        print(
            fmt.format(
                name,
                r["n_completed_trials"],
                br["refusals"],
                f"{br['kl_divergence']:.4f}",
                br_ld,
                f"{bk['kl_divergence']:.4f}",
                bk["refusals"],
                zr_str,
                f"{t:.0f}s" if t else "N/A",
            )
        )

    # KL threshold analysis.
    print("\n" + "=" * 100)
    print("  FIXED KL THRESHOLD ANALYSIS (minimum refusals where KL <= threshold)")
    print("=" * 100)

    thresholds = ["0.05", "0.1", "0.2"]
    fmt2 = "{:<25}" + " {:>12}" * len(thresholds)
    print(fmt2.format("Variant", *[f"KL<={t}" for t in thresholds]))
    print("-" * 100)

    for name, r in results.items():
        vals = []
        for t in thresholds:
            ta = r.get("kl_threshold_analysis", {}).get(t)
            if ta:
                vals.append(f"{ta['min_refusals']} (KL={ta['kl_divergence']:.3f})")
            else:
                vals.append("N/A")
        print(fmt2.format(name, *vals))


def plot_pareto_fronts(results: dict, output_dir: str):
    """Generate a Pareto front scatter plot for all variants."""
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for (name, r), color in zip(results.items(), colors):
        trials = r.get("all_trials", [])
        if not trials:
            continue

        kl_vals = [t["kl_divergence"] for t in trials if t["kl_divergence"] is not None]
        ref_vals = [t["refusals"] for t in trials if t["refusals"] is not None]

        if not kl_vals:
            continue

        ax.scatter(kl_vals, ref_vals, label=name, alpha=0.5, s=20, color=color)

        # Compute and plot Pareto front.
        points = sorted(zip(kl_vals, ref_vals), key=lambda p: p[0])
        pareto_kl = []
        pareto_ref = []
        min_ref = float("inf")
        for kl, ref in points:
            if ref < min_ref:
                min_ref = ref
                pareto_kl.append(kl)
                pareto_ref.append(ref)

        if pareto_kl:
            ax.plot(pareto_kl, pareto_ref, "-", color=color, alpha=0.8, linewidth=1.5)

    ax.set_xlabel("KL Divergence", fontsize=12)
    ax.set_ylabel("Refusal Count", fontsize=12)
    ax.set_title("Pareto Fronts: KL Divergence vs Refusal Count", fontsize=14)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.7)
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "pareto_fronts.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Pareto front plot saved to: {path}")


def plot_kl_threshold_bars(results: dict, output_dir: str):
    """Generate bar chart of refusals at fixed KL thresholds."""
    plt.style.use("dark_background")
    thresholds = ["0.05", "0.1", "0.2"]

    fig, axes = plt.subplots(1, len(thresholds), figsize=(15, 6), sharey=True)

    variant_names = list(results.keys())

    for ax, threshold in zip(axes, thresholds):
        values = []
        colors = []
        for name in variant_names:
            ta = results[name].get("kl_threshold_analysis", {}).get(threshold)
            if ta:
                values.append(ta["min_refusals"])
                colors.append("steelblue")
            else:
                values.append(0)
                colors.append("gray")

        bars = ax.bar(range(len(variant_names)), values, color=colors, alpha=0.8)
        ax.set_title(f"KL <= {threshold}", fontsize=11)
        ax.set_xticks(range(len(variant_names)))
        ax.set_xticklabels(
            [n.replace("_", "\n") for n in variant_names],
            rotation=45,
            ha="right",
            fontsize=8,
        )
        ax.set_ylabel("Min Refusals" if ax == axes[0] else "")

        # Add value labels on bars.
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    str(val),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    fig.suptitle("Minimum Refusals at Fixed KL Thresholds", fontsize=14)
    fig.tight_layout()

    path = os.path.join(output_dir, "kl_threshold_bars.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  KL threshold bar chart saved to: {path}")


def plot_timing(results: dict, output_dir: str):
    """Generate timing comparison bar chart."""
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 5))

    names = list(results.keys())
    times = [results[n].get("elapsed_seconds", 0) or 0 for n in names]

    bars = ax.barh(
        range(len(names)), [t / 60 for t in times], color="steelblue", alpha=0.8
    )
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Time (minutes)", fontsize=11)
    ax.set_title("Runtime per Variant", fontsize=13)

    for bar, t in zip(bars, times):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{t / 60:.1f}m",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()
    path = os.path.join(output_dir, "timing.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Timing chart saved to: {path}")


def compute_hypervolume(results: dict) -> dict:
    """Compute hypervolume indicator for each variant's Pareto front.

    Reference point: (max_observed_kl * 1.1, max_observed_refusals).
    """
    # Find global reference point.
    all_kl = []
    all_ref = []
    for r in results.values():
        for t in r.get("all_trials", []):
            if t["kl_divergence"] is not None and t["refusals"] is not None:
                all_kl.append(t["kl_divergence"])
                all_ref.append(t["refusals"])

    if not all_kl:
        return {}

    ref_kl = max(all_kl) * 1.1
    ref_refusals = max(all_ref)

    hypervolumes = {}
    for name, r in results.items():
        trials = r.get("all_trials", [])
        points = [
            (t["kl_divergence"], t["refusals"])
            for t in trials
            if t["kl_divergence"] is not None and t["refusals"] is not None
        ]
        if not points:
            hypervolumes[name] = 0.0
            continue

        # Compute Pareto front.
        points.sort(key=lambda p: p[0])
        pareto = []
        min_ref = float("inf")
        for kl, ref in points:
            if ref < min_ref:
                min_ref = ref
                pareto.append((kl, ref))

        # Compute hypervolume (area dominated by Pareto front below reference).
        hv = 0.0
        for i, (kl, ref) in enumerate(pareto):
            if kl >= ref_kl or ref >= ref_refusals:
                continue
            next_kl = pareto[i + 1][0] if i + 1 < len(pareto) else ref_kl
            next_kl = min(next_kl, ref_kl)
            hv += (next_kl - kl) * (ref_refusals - ref)

        hypervolumes[name] = round(hv, 4)

    return hypervolumes


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Prometheus experiment results"
    )
    parser.add_argument(
        "--input",
        default="experiments",
        help="Input directory containing results_summary.json (default: experiments)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for plots (default: same as input)",
    )
    args = parser.parse_args()

    output_dir = args.output or args.input
    os.makedirs(output_dir, exist_ok=True)

    results = load_results(args.input)

    if not results:
        print("No results found.")
        sys.exit(1)

    print(f"Loaded results for {len(results)} variants")

    # Summary table.
    print_summary_table(results)

    # Hypervolume.
    print("\n" + "=" * 100)
    print("  HYPERVOLUME INDICATOR (higher = better Pareto front)")
    print("=" * 100)
    hypervolumes = compute_hypervolume(results)
    for name, hv in sorted(hypervolumes.items(), key=lambda x: -x[1]):
        print(f"  {name:<25} {hv:.4f}")

    # Plots.
    print("\nGenerating plots...")
    plot_pareto_fronts(results, output_dir)
    plot_kl_threshold_bars(results, output_dir)
    plot_timing(results, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
