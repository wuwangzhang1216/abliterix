#!/usr/bin/env python3
"""A/B comparison v3: new architecture techniques vs baseline.

Compares 7 methods on Qwen3.5-0.8B with grid-searched strength:

  A: Baseline          (mean + orthogonal projection)
  B: Projected         (mean + projected abliteration + winsorize)
  C: Disc. layers      (mean + orthogonal + discriminative layer selection)
  D: SRA               (surgical refusal ablation + projected + disc)
  E: Spherical         (mean + orthogonal + spherical steering + disc)
  F: SVF               (mean + orthogonal + steering vector fields + disc)
  G: Full new arch     (SRA + spherical + disc + projected)

Usage:
    python scripts/ab_test_v3.py
    python scripts/ab_test_v3.py --model Qwen/Qwen3-0.6B   # smaller model
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table

console = Console()

N_TRAIN = 200
N_EVAL = 100
MAX_GEN_TOKENS = 50
BATCH_SIZE = 8
STRENGTHS = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B", help="HuggingFace model ID")
    parser.add_argument("--strengths", type=float, nargs="+", default=STRENGTHS)
    args = parser.parse_args()
    MODEL_ID = args.model

    torch.set_grad_enabled(False)

    from abliterix.settings import AbliterixConfig
    from abliterix.types import VectorMethod, SteeringMode
    from abliterix.vectors import compute_steering_vectors
    from abliterix.core.engine import SteeringEngine
    from abliterix.core.steering import apply_steering
    from abliterix.types import SteeringProfile
    from abliterix.data import load_prompt_dataset
    from abliterix.util import chunk_batches

    config = AbliterixConfig(
        model={"model_id": MODEL_ID},
        inference={"batch_size": BATCH_SIZE, "max_gen_tokens": MAX_GEN_TOKENS},
        steering={"vector_method": "mean", "orthogonal_projection": True},
        benign_prompts={"dataset": "datasets/good_1000", "split": f"train[:{N_TRAIN}]", "column": "prompt"},
        target_prompts={"dataset": "datasets/harmful_1000", "split": f"train[:{N_TRAIN}]", "column": "prompt"},
        benign_eval_prompts={"dataset": "datasets/good_1000", "split": f"train[{N_TRAIN}:{N_TRAIN+N_EVAL}]", "column": "prompt"},
        target_eval_prompts={"dataset": "datasets/harmful_1000", "split": f"train[{N_TRAIN}:{N_TRAIN+N_EVAL}]", "column": "prompt"},
    )
    markers = config.detection.compliance_markers

    # ── Load model ────────────────────────────────────────────────────────
    console.rule("[bold cyan]Loading model")
    console.print(f"Model: [bold]{MODEL_ID}[/]")
    engine = SteeringEngine(config)

    console.print("Loading datasets...")
    benign_train = load_prompt_dataset(config, config.benign_prompts)
    target_train = load_prompt_dataset(config, config.target_prompts)
    benign_eval = load_prompt_dataset(config, config.benign_eval_prompts)
    target_eval = load_prompt_dataset(config, config.target_eval_prompts)

    # ── Extract residuals ─────────────────────────────────────────────────
    console.rule("[bold cyan]Extracting residuals")
    benign_states = engine.extract_hidden_states_batched(benign_train)
    target_states = engine.extract_hidden_states_batched(target_train)

    console.print("Computing baseline logprobs...")
    bl_parts = []
    for batch in chunk_batches(benign_eval, BATCH_SIZE):
        bl_parts.append(engine.compute_logprobs(batch))
    baseline_logprobs = torch.cat(bl_parts, dim=0)

    n_layers = len(engine.transformer_layers)
    mid = int(0.7 * n_layers)

    def make_profiles(strength):
        profiles = {}
        for comp in engine.list_steerable_components():
            profiles[comp] = SteeringProfile(
                max_weight=strength,
                max_weight_position=float(mid),
                min_weight=0.0,
                min_weight_distance=float(n_layers * 0.5),
            )
        return profiles

    def evaluate(vectors, cfg, strength, benign_s=None, target_s=None):
        engine.restore_baseline()
        apply_steering(engine, vectors, None, make_profiles(strength), cfg,
                       benign_states=benign_s, target_states=target_s)

        # KL divergence
        parts = []
        for batch in chunk_batches(benign_eval, BATCH_SIZE):
            parts.append(engine.compute_logprobs(batch))
        steered = torch.cat(parts, dim=0)
        kl = F.kl_div(steered, baseline_logprobs, log_target=True, reduction="batchmean").item()

        # Refusals
        responses = []
        for batch in chunk_batches(target_eval, BATCH_SIZE):
            responses.extend(engine.generate_text(batch))
        refusals = sum(1 for r in responses if any(m in r.lower() for m in markers))

        return refusals, kl

    # ── Define methods ────────────────────────────────────────────────────
    methods = {}

    # A: Baseline (mean + orthogonal projection)
    console.rule("[bold yellow]A: Baseline (mean+ortho)")
    t0 = time.perf_counter()
    vectors_a = compute_steering_vectors(benign_states, target_states, VectorMethod.MEAN, True)
    t_a = time.perf_counter() - t0
    config_a = config.model_copy()
    methods["A: Baseline\n(mean+ortho)"] = (vectors_a, config_a, None, None, t_a)

    # B: Projected abliteration + winsorize
    console.rule("[bold blue]B: Projected (mean+proj+win)")
    t0 = time.perf_counter()
    vectors_b = compute_steering_vectors(
        benign_states, target_states, VectorMethod.MEAN, False,
        projected_abliteration=True, winsorize=True,
    )
    t_b = time.perf_counter() - t0
    config_b = config.model_copy()
    config_b.steering.projected_abliteration = True
    methods["B: Projected\n(mean+proj+win)"] = (vectors_b, config_b, None, None, t_b)

    # C: Discriminative layers
    console.rule("[bold magenta]C: Disc. layers (mean+ortho+disc)")
    config_c = config.model_copy()
    config_c.steering.discriminative_layer_selection = True
    methods["C: Disc. layers\n(mean+ortho+disc)"] = (vectors_a, config_c, benign_states, target_states, t_a)

    # D: SRA (Surgical Refusal Ablation)
    console.rule("[bold green]D: SRA (sra+proj+disc)")
    t0 = time.perf_counter()
    vectors_d = compute_steering_vectors(
        benign_states, target_states, VectorMethod.SRA, False,
        projected_abliteration=True,
        sra_base_method=VectorMethod.MEAN,
        sra_n_atoms=8,
        sra_ridge_alpha=0.01,
    )
    t_d = time.perf_counter() - t0
    config_d = config.model_copy()
    config_d.steering.projected_abliteration = True
    config_d.steering.discriminative_layer_selection = True
    methods["D: SRA\n(sra+proj+disc)"] = (vectors_d, config_d, benign_states, target_states, t_d)

    # E: Spherical steering
    console.rule("[bold red]E: Spherical (mean+ortho+spherical+disc)")
    config_e = config.model_copy()
    config_e.steering.steering_mode = SteeringMode.SPHERICAL
    config_e.steering.discriminative_layer_selection = True
    methods["E: Spherical\n(mean+ortho+sph+disc)"] = (vectors_a, config_e, benign_states, target_states, t_a)

    # F: Steering Vector Fields
    console.rule("[bold cyan]F: SVF (mean+ortho+svf+disc)")
    from abliterix.svf import train_concept_scorers
    console.print("Training SVF concept scorers...")
    t0 = time.perf_counter()
    concept_scorers = train_concept_scorers(
        benign_states, target_states,
        hidden_dim=benign_states.shape[2],
        n_epochs=50,
        lr=1e-3,
        hidden_dim_scorer=256,
    )
    t_f = time.perf_counter() - t0
    engine._concept_scorers = concept_scorers
    config_f = config.model_copy()
    config_f.steering.steering_mode = SteeringMode.VECTOR_FIELD
    config_f.steering.discriminative_layer_selection = True
    methods["F: SVF\n(mean+ortho+svf+disc)"] = (vectors_a, config_f, benign_states, target_states, t_f)

    # G: Full new architecture (SRA + spherical + disc + projected)
    console.rule("[bold white]G: Full new arch (SRA+sph+disc+proj)")
    config_g = config.model_copy()
    config_g.steering.steering_mode = SteeringMode.SPHERICAL
    config_g.steering.projected_abliteration = True
    config_g.steering.discriminative_layer_selection = True
    methods["G: Full new arch\n(SRA+sph+disc+proj)"] = (vectors_d, config_g, benign_states, target_states, t_d)

    # ── Grid search ──────────────────────────────────────────────────────
    all_results = {}
    strengths = args.strengths

    for method_name, (vecs, cfg, bs, ts, vec_time) in methods.items():
        console.rule(f"[bold]Sweeping: {method_name}")
        best = None
        all_points = []
        for s in strengths:
            console.print(f"  λ={s:.1f} ... ", end="")
            t0 = time.perf_counter()
            refusals, kl = evaluate(vecs, cfg, s, bs, ts)
            eval_time = time.perf_counter() - t0
            console.print(f"refusals={refusals}, KL={kl:.6f} ({eval_time:.1f}s)")
            all_points.append({"strength": s, "refusals": refusals, "kl": kl})

            # Pareto: lowest refusal → lowest KL
            if best is None or refusals < best["refusals"] or (
                refusals == best["refusals"] and kl < best["kl"]
            ):
                best = {"refusals": refusals, "kl": kl, "strength": s}

        best["vec_time"] = vec_time
        best["all_points"] = all_points
        all_results[method_name] = best
        console.print(
            f"  → [bold]Best: λ={best['strength']}, "
            f"refusals={best['refusals']}, KL={best['kl']:.6f}[/]"
        )

    # ── Summary table ────────────────────────────────────────────────────
    console.print()
    console.rule("[bold cyan]Final Results (Best Pareto Point per Method)")

    table = Table(title=f"A/B Test v3: {MODEL_ID}", show_lines=True)
    table.add_column("Method", style="bold", width=22)
    table.add_column("Best λ", justify="right")
    table.add_column("Refusals", justify="right")
    table.add_column("Ref %", justify="right")
    table.add_column("KL Div", justify="right")
    table.add_column("KL Δ vs A", justify="right")
    table.add_column("Vec Time", justify="right")

    baseline_kl = all_results["A: Baseline\n(mean+ortho)"]["kl"]
    baseline_ref = all_results["A: Baseline\n(mean+ortho)"]["refusals"]

    for name, r in all_results.items():
        kl_delta = r["kl"] - baseline_kl
        if baseline_kl > 1e-8:
            kl_pct = kl_delta / baseline_kl * 100
            kl_str = f"{kl_pct:+.1f}%"
        else:
            kl_str = f"{kl_delta:+.6f}"

        if kl_delta < -1e-5:
            kl_str = f"[green]{kl_str}[/]"
        elif kl_delta > 1e-5:
            kl_str = f"[red]{kl_str}[/]"

        ref_pct = r["refusals"] / N_EVAL * 100
        if r["refusals"] < baseline_ref:
            ref_style = "green bold"
        elif r["refusals"] > baseline_ref:
            ref_style = "red"
        else:
            ref_style = None

        ref_str = f"{r['refusals']}/{N_EVAL}"
        ref_pct_str = f"{ref_pct:.1f}%"
        if ref_style:
            ref_str = f"[{ref_style}]{ref_str}[/]"
            ref_pct_str = f"[{ref_style}]{ref_pct_str}[/]"

        table.add_row(
            name,
            f"{r['strength']:.1f}",
            ref_str,
            ref_pct_str,
            f"{r['kl']:.6f}",
            kl_str,
            f"{r['vec_time']:.1f}s",
        )

    console.print(table)

    # ── Winner analysis ──────────────────────────────────────────────────
    console.print()

    best_kl_name = min(all_results, key=lambda k: all_results[k]["kl"])
    best_ref_name = min(all_results, key=lambda k: all_results[k]["refusals"])

    # Pareto front: not dominated by any other
    pareto = []
    for name, r in all_results.items():
        dominated = any(
            o["refusals"] <= r["refusals"] and o["kl"] <= r["kl"] and (o["refusals"] < r["refusals"] or o["kl"] < r["kl"])
            for oname, o in all_results.items() if oname != name
        )
        if not dominated:
            pareto.append(name)

    console.print(f"[bold green]Lowest KL:[/]      {best_kl_name.split(chr(10))[0]} "
                  f"(KL={all_results[best_kl_name]['kl']:.6f})")
    console.print(f"[bold green]Lowest refusal:[/] {best_ref_name.split(chr(10))[0]} "
                  f"(refusals={all_results[best_ref_name]['refusals']})")
    console.print(f"[bold green]Pareto front:[/]   {', '.join(n.split(chr(10))[0] for n in pareto)}")

    # ── Key comparisons ──────────────────────────────────────────────────
    console.print()
    console.rule("[bold cyan]Key Comparisons")

    r_a = all_results["A: Baseline\n(mean+ortho)"]
    r_g = all_results["G: Full new arch\n(SRA+sph+disc+proj)"]

    if r_a["kl"] > 1e-8:
        kl_ratio = r_a["kl"] / max(r_g["kl"], 1e-8)
        console.print(f"  Full new arch vs Baseline:")
        console.print(f"    Refusals: {r_a['refusals']} → {r_g['refusals']} "
                      f"({'↓' if r_g['refusals'] < r_a['refusals'] else '↑' if r_g['refusals'] > r_a['refusals'] else '='} "
                      f"{abs(r_g['refusals'] - r_a['refusals'])})")
        console.print(f"    KL:       {r_a['kl']:.6f} → {r_g['kl']:.6f} "
                      f"([bold green]{kl_ratio:.1f}x improvement[/])" if r_g["kl"] < r_a["kl"]
                      else f"    KL:       {r_a['kl']:.6f} → {r_g['kl']:.6f}")

    # Compare each new technique's isolated contribution
    for label, key in [
        ("SRA", "D: SRA\n(sra+proj+disc)"),
        ("Spherical", "E: Spherical\n(mean+ortho+sph+disc)"),
        ("SVF", "F: SVF\n(mean+ortho+svf+disc)"),
    ]:
        r = all_results[key]
        ref_delta = r["refusals"] - r_a["refusals"]
        kl_delta_pct = (r["kl"] - r_a["kl"]) / max(r_a["kl"], 1e-8) * 100
        console.print(f"  {label} contribution: refusals {ref_delta:+d}, KL {kl_delta_pct:+.1f}%")


if __name__ == "__main__":
    main()
