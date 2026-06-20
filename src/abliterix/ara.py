# Abliterix — a derivative work of Heretic (https://github.com/p-e-w/heretic)
# Original work Copyright (C) 2025-2026  Philipp Emanuel Weidmann (p-e-w)
# Modified work Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Arbitrary-Rank Ablation (ARA) — ported from Heretic PR #211.

ARA uses **no refusal direction**. It captures the (input, output) activations
of every steerable module on "harmless" and "harmful" prompts, then directly
optimises each module's weight matrix (unconstrained, arbitrary rank) against a
three-term objective:

1. ``preserve_good``  — harmless outputs change as little as possible (MSE);
2. ``steer_bad``      — harmful outputs move *towards* the harmless output cloud
   (mean distance to k nearest harmless outputs);
3. ``overcorrect``    — harmful outputs move *away from* their original harmful
   outputs (negative kNN distance), which over-corrects past the residual and
   gives stronger steering able to overcome redundant refusal mechanisms.

The per-module weight delta is found with LBFGS (strong-Wolfe). Because it edits
``module.base_layer.weight`` in place after caching the original into
``engine._direct_weight_originals``, ``engine.restore_baseline()`` rolls it back
for the next trial exactly like direct-mode steering.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.linalg as LA
import torch.nn.functional as F
from torch import Tensor
from torch.optim import LBFGS

from .types import WeightNorm
from .util import chunk_batches, print

# module_io[layer] -> {component: {module_index: (input, output)}},
# each tensor of shape (n_prompts, hidden).
ModuleIO = list[dict[str, dict[int, tuple[Tensor, Tensor]]]]

# ARA only edits the residual-writing projections (attention out + MLP down),
# matching Heretic. Editing q/k/v_proj with this objective is ill-posed (their
# output is not the residual contribution) and tends to damage the model.
ARA_COMPONENTS: frozenset[str] = frozenset({"attn.o_proj", "mlp.down_proj"})


@dataclass
class ARAParameters:
    start_layer_index: int
    end_layer_index: int
    preserve_good_behavior_weight: float
    steer_bad_behavior_weight: float
    overcorrect_relative_weight: float
    neighbor_count: int
    n_steps: int = 5


def mean_distances_to_knn(a: Tensor, b: Tensor, k: int) -> Tensor:
    """Mean distance from each row of ``a`` to its ``k`` nearest rows in ``b``."""
    k = max(1, min(k, b.shape[0]))
    distances = torch.cdist(a, b)
    nearest_distances, _ = distances.topk(k, dim=1, largest=False)
    return nearest_distances.mean(1)


def _unwrap(module):
    """Return the base Linear behind a possible PEFT wrapper."""
    return module.base_layer if hasattr(module, "base_layer") else module


def capture_module_io(engine, messages) -> ModuleIO:
    """Hook every steerable module and capture last-token (input, output) I/O.

    Runs a forward pass over ``messages`` (batched) and accumulates the input
    and output of each module at the final prompt token, on CPU in float32.
    """
    n_layers = engine.get_n_layers()
    # accumulator[layer][component][module_index] -> ([inputs], [outputs])
    acc: list[dict[str, dict[int, tuple[list, list]]]] = [{} for _ in range(n_layers)]

    def make_hook(li: int, comp: str, mi: int):
        def hook(module, inputs, outputs):
            inp = inputs[0][:, -1, :].detach().float().cpu()
            out_t = outputs[0] if isinstance(outputs, tuple) else outputs
            out = out_t[:, -1, :].detach().float().cpu()
            bucket = acc[li].setdefault(comp, {}).setdefault(mi, ([], []))
            bucket[0].append(inp)
            bucket[1].append(out)

        return hook

    handles = []
    for li in range(n_layers):
        for comp, mods in engine.steerable_modules(li).items():
            if comp not in ARA_COMPONENTS:
                continue
            for mi, mod in enumerate(mods):
                handles.append(
                    _unwrap(mod).register_forward_hook(make_hook(li, comp, mi))
                )

    bs = engine.config.inference.batch_size or 16
    try:
        with torch.no_grad():
            for batch in chunk_batches(messages, bs):
                inputs = engine._tokenize(batch)
                engine.model(**inputs)
    finally:
        for h in handles:
            h.remove()

    module_io: ModuleIO = [{} for _ in range(n_layers)]
    for li in range(n_layers):
        for comp, mods in acc[li].items():
            module_io[li][comp] = {
                mi: (torch.cat(ins, dim=0), torch.cat(outs, dim=0))
                for mi, (ins, outs) in mods.items()
            }
    return module_io


def ara_abliterate(
    engine,
    good_io: ModuleIO,
    bad_io: ModuleIO,
    params: ARAParameters,
    *,
    row_norm_preserve: bool = False,
) -> int:
    """Optimise each module's weight in [start, end) layers per the ARA objective.

    Edits ``module.base_layer.weight`` in place (caching the original into
    ``engine._direct_weight_originals`` so ``restore_baseline`` undoes it).
    Returns the number of modules modified.
    """
    if not hasattr(engine, "_direct_weight_originals"):
        engine._direct_weight_originals = {}

    n_modified = 0
    for li in range(params.start_layer_index, params.end_layer_index):
        if li >= len(good_io):
            break
        for comp, mods in engine.steerable_modules(li).items():
            if comp not in ARA_COMPONENTS:
                continue
            comp_io_g = good_io[li].get(comp, {})
            comp_io_b = bad_io[li].get(comp, {})
            for mi, mod in enumerate(mods):
                if mi not in comp_io_g or mi not in comp_io_b:
                    continue
                base = _unwrap(mod)
                weight = base.weight
                device = weight.device

                # Cache original for restore_baseline().
                if weight not in engine._direct_weight_originals:
                    engine._direct_weight_originals[weight] = weight.data.clone()

                row_norms = LA.vector_norm(
                    weight.data.float(), dim=1, keepdim=True
                ).detach()

                gi, go = comp_io_g[mi]
                bi, bo = comp_io_b[mi]
                gi = gi.to(device).float()
                go = go.to(device).float()
                bi = bi.to(device).float()
                bo = bo.to(device).float()

                # Shape guard: module input dim must match captured input width.
                if gi.shape[1] != weight.shape[1] or go.shape[1] != weight.shape[0]:
                    continue

                # Optimise a float32 working copy (the global grad context is
                # disabled and the live param may be PEFT-wrapped/quantised).
                work = weight.data.float().clone().requires_grad_(True)

                def get_matrix(w: Tensor = work) -> Tensor:
                    if row_norm_preserve:
                        return row_norms * F.normalize(w, p=2, dim=1)
                    return w

                def objective(m: Tensor) -> Tensor:
                    new_good = gi @ m.T
                    new_bad = bi @ m.T
                    preserve = ((new_good - go) ** 2).mean()
                    steer = (
                        mean_distances_to_knn(new_bad, go, params.neighbor_count).mean()
                        + params.overcorrect_relative_weight
                        * -mean_distances_to_knn(
                            new_bad, bo, params.neighbor_count
                        ).mean()
                    )
                    return (
                        params.preserve_good_behavior_weight * preserve
                        + params.steer_bad_behavior_weight * steer
                    )

                optimizer = LBFGS(
                    [work],
                    lr=1.0,
                    max_iter=20,
                    history_size=10,
                    line_search_fn="strong_wolfe",
                )

                def closure() -> Tensor:
                    optimizer.zero_grad()
                    loss = objective(get_matrix())
                    loss.backward()
                    return loss

                with torch.enable_grad():
                    for _ in range(params.n_steps):
                        optimizer.step(closure)

                with torch.no_grad():
                    base.weight.data = get_matrix().detach().to(weight.dtype)
                n_modified += 1

    return n_modified


def run_ara(
    engine,
    benign_msgs,
    target_msgs,
    params: ARAParameters,
    *,
    weight_normalization: WeightNorm = WeightNorm.NONE,
    cached_io: tuple[ModuleIO, ModuleIO] | None = None,
) -> tuple[int, tuple[ModuleIO, ModuleIO]]:
    """Capture I/O (or reuse) and run ARA. Returns (n_modified, (good_io, bad_io))."""
    if cached_io is None:
        print("* ARA: capturing harmless module I/O...")
        good_io = capture_module_io(engine, benign_msgs)
        print("* ARA: capturing harmful module I/O...")
        bad_io = capture_module_io(engine, target_msgs)
    else:
        good_io, bad_io = cached_io
    n = ara_abliterate(
        engine,
        good_io,
        bad_io,
        params,
        row_norm_preserve=weight_normalization != WeightNorm.NONE,
    )
    return n, (good_io, bad_io)
