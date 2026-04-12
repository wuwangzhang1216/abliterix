# Abliterix — a derivative work of Heretic (https://github.com/p-e-w/heretic)
# Original work Copyright (C) 2025  Philipp Emanuel Weidmann (p-e-w)
# Modified work Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Iterative abliteration for hardened models (e.g. DeepRefusal).

DeepRefusal-style defences distribute refusal across redundant pathways by
probabilistically ablating the refusal direction during fine-tuning.  Standard
single-pass abliteration fails because the model reconstructs refusal from
backup pathways.

This module implements an iterative extract-ablate-re-extract loop that peels
away successive refusal layers:

    1. Extract top-k refusal directions from residual streams
    2. Project them out via direct weight modification
    3. Re-extract residuals from the now-modified model
    4. Compute new directions orthogonalised against all previous ones
    5. Repeat until convergence (direction norm drops below threshold)
    6. Restore baseline and return the combined subspace for Optuna search

The result is a stacked direction tensor compatible with the existing
multi-direction pipeline.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .settings import AbliterixConfig
from .types import SteeringProfile
from .vectors import (
    build_subspace_basis,
    compute_steering_vectors,
    orthogonalize_against,
)


def _make_uniform_profiles(
    engine,
    strength: float,
    config: AbliterixConfig,
) -> dict[str, SteeringProfile]:
    """Build a uniform-strength profile dict covering all steerable components.

    Used for intermediate ablation passes where we apply full-strength
    projection across all layers — the Optuna search will later optimise
    the per-component strengths on the combined subspace.
    """
    n_layers = len(engine.transformer_layers)
    center = n_layers // 2
    profiles = {}
    for component in engine.list_steerable_components():
        if component in config.steering.disabled_components:
            continue
        profiles[component] = SteeringProfile(
            max_weight=strength,
            max_weight_position=center,
            min_weight=strength,  # uniform — no decay
            min_weight_distance=float(n_layers),
        )
    return profiles


def iterative_abliterate(
    engine,
    benign_msgs: list,
    target_msgs: list,
    config: AbliterixConfig,
    *,
    benign_states: Tensor,
    target_states: Tensor,
) -> tuple[Tensor, list[dict]]:
    """Run the iterative extract-ablate loop for hardened models.

    Parameters
    ----------
    engine : SteeringEngine
        Loaded model wrapper.
    benign_msgs, target_msgs : list
        Chat-template formatted messages for re-extraction.
    config : AbliterixConfig
        Full configuration (reads ``config.iterative.*`` settings).
    benign_states, target_states : Tensor
        Pre-extracted residuals from the unmodified model.

    Returns
    -------
    vectors : Tensor
        Combined refusal subspace, shape ``(rank, layers+1, hidden_dim)``.
    stats : list[dict]
        Per-iteration statistics for logging.
    """
    from .core.steering import _apply_direct_steering

    ic = config.iterative
    sc = config.steering

    all_directions: list[Tensor] = []
    stats: list[dict] = []
    initial_norm: float | None = None

    # Intermediate projection strength — strong enough to fully remove the
    # direction so re-extraction sees the ablated model.
    intermediate_strength = 1.5

    print(f"[iterative] Starting iterative abliteration (max {ic.max_iterations} passes)")
    print(f"[iterative] Directions per pass: {ic.per_iteration_directions}")

    for iteration in range(ic.max_iterations):
        print(f"\n[iterative] === Pass {iteration + 1}/{ic.max_iterations} ===")

        # --- Extract directions from current model state ---
        print(f"[iterative] Extracting {ic.per_iteration_directions} directions...")
        raw_dirs = compute_steering_vectors(
            benign_states,
            target_states,
            sc.vector_method,
            sc.orthogonal_projection,
            winsorize=sc.winsorize_vectors,
            winsorize_quantile=sc.winsorize_quantile,
            projected_abliteration=sc.projected_abliteration,
            ot_components=sc.ot_components,
            n_directions=ic.per_iteration_directions,
            sra_base_method=sc.sra_base_method,
            sra_n_atoms=sc.sra_n_atoms,
            sra_ridge_alpha=sc.sra_ridge_alpha,
        )

        # Ensure 3D shape: (n_dirs, layers+1, hidden_dim)
        if raw_dirs.ndim == 2:
            raw_dirs = raw_dirs.unsqueeze(0)

        # --- Orthogonalise against previously found directions ---
        if all_directions:
            raw_dirs = orthogonalize_against(
                raw_dirs,
                all_directions,
                norm_threshold=ic.convergence_norm_threshold,
            )

        # --- Compute norms for convergence check ---
        dir_norms = raw_dirs.view(raw_dirs.shape[0], -1).norm(p=2, dim=1)
        mean_norm = dir_norms.mean().item()
        active_count = int((dir_norms > 1e-8).sum().item())

        if initial_norm is None:
            initial_norm = mean_norm

        relative_norm = mean_norm / max(initial_norm, 1e-8)

        # --- Cosine similarity check against previous directions ---
        max_cosine = 0.0
        if all_directions:
            prev_flat = torch.cat(all_directions, dim=0)
            for k in range(raw_dirs.shape[0]):
                if dir_norms[k] < 1e-8:
                    continue
                curr = raw_dirs[k].view(-1).float()
                for j in range(prev_flat.shape[0]):
                    prev = prev_flat[j].view(-1).float()
                    cos = F.cosine_similarity(curr.unsqueeze(0), prev.unsqueeze(0)).item()
                    max_cosine = max(max_cosine, abs(cos))

        iter_stat = {
            "iteration": iteration + 1,
            "mean_norm": mean_norm,
            "relative_norm": relative_norm,
            "active_directions": active_count,
            "max_cosine_similarity": max_cosine,
        }
        stats.append(iter_stat)

        print(
            f"[iterative] Norm: {mean_norm:.4f} (relative: {relative_norm:.4f}), "
            f"active: {active_count}/{raw_dirs.shape[0]}, "
            f"max cosine vs prev: {max_cosine:.4f}"
        )

        # --- Convergence check ---
        if active_count == 0:
            print("[iterative] All directions below threshold — converged (no active directions)")
            break

        if iteration > 0 and relative_norm < ic.convergence_norm_threshold:
            print(
                f"[iterative] Relative norm {relative_norm:.4f} < "
                f"{ic.convergence_norm_threshold} — converged"
            )
            break

        if max_cosine > ic.convergence_cosine_threshold:
            print(
                f"[iterative] Max cosine {max_cosine:.4f} > "
                f"{ic.convergence_cosine_threshold} — converged (rediscovering old directions)"
            )
            break

        # --- Accumulate active directions ---
        active_mask = dir_norms > 1e-8
        if active_mask.any():
            all_directions.append(raw_dirs[active_mask])

        # --- Apply direct projection for next iteration ---
        if iteration < ic.max_iterations - 1:
            print("[iterative] Projecting out directions from model weights...")
            profiles = _make_uniform_profiles(engine, intermediate_strength, config)
            _apply_direct_steering(
                engine,
                raw_dirs[active_mask] if active_mask.any() else raw_dirs,
                None,  # global_vector — use per-layer
                profiles,
                config,
                None,  # discriminative_layers — project all layers
            )

            # Re-extract residuals from the modified model.
            print("[iterative] Re-extracting residuals from modified model...")
            benign_states = engine.extract_hidden_states_batched(benign_msgs)
            target_states = engine.extract_hidden_states_batched(target_msgs)

    # --- Restore baseline before returning ---
    print("\n[iterative] Restoring model to baseline...")
    engine.restore_baseline()

    if not all_directions:
        print("[iterative] WARNING: No directions found across all iterations!")
        # Return a dummy single-direction zero tensor.
        n_layers = benign_states.shape[1]
        hidden_dim = benign_states.shape[2]
        return torch.zeros(1, n_layers, hidden_dim), stats

    # --- Build combined subspace ---
    total_raw = sum(d.shape[0] for d in all_directions)
    print(f"[iterative] Building subspace from {total_raw} raw directions across {len(stats)} iterations")

    if ic.accumulation_method == "subspace":
        vectors = build_subspace_basis(all_directions)
    else:  # "stack"
        vectors = torch.cat(all_directions, dim=0)

    print(f"[iterative] Final subspace rank: {vectors.shape[0]}")
    return vectors, stats
