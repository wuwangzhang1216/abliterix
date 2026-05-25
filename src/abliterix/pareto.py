# Abliterix
# Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pareto-front extraction & grouping for multi-method TPE studies.

When the optimiser sweeps multiple methodological dimensions (
``search_direct_transform``, ``search_harmfulness_direction``) the global
Pareto front mixes trials of different "kinds" (standard vs orba vs
biprojected; single direction vs harmfulness pair). To compare those
kinds head-to-head we need per-group Pareto fronts so the operator can
see, e.g., "ORBA's best refusal-at-KL=0.04 is X, biprojected's is Y".

This module provides:

* :func:`pareto_front` — strict ε-dominance front on the standard
  ``(refusal, KL)`` objective pair from an Optuna study.
* :func:`group_trials` — bucket completed trials by user-attr keys (e.g.
  ``"direct_transform"``, ``"steering_variant"``) so each group can have
  its own front.
* :func:`per_group_front` — convenience: groups then returns a Pareto
  front per group.
* :func:`format_summary_table` — terminal-friendly text table of the
  best trial per group at a user-chosen KL budget.

The functions accept a plain list of ``optuna.trial.FrozenTrial`` so the
helpers can be used without re-importing Optuna in the caller.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParetoPoint:
    trial_number: int
    refusal: float
    kl: float
    params: dict[str, Any]
    user_attrs: dict[str, Any]

    @property
    def key(self) -> tuple[float, float]:
        return (self.refusal, self.kl)


# ---------------------------------------------------------------------------
# Core: ε-dominance Pareto sweep
# ---------------------------------------------------------------------------


def _extract_objectives(trial) -> tuple[float, float] | None:
    """Pull ``(refusal, KL)`` from a FrozenTrial; returns None for incomplete trials.

    The optimiser declares directions ``[MINIMIZE, MINIMIZE]`` for both
    objectives, and ``scorer._compute_objectives`` returns
    ``(divergence_objective, compliance_objective)`` — i.e. ``(KL, refusal)``.
    We swap the order here so the rest of this module can use the more
    natural ``(refusal, KL)`` semantics.
    """
    if trial.values is None or len(trial.values) < 2:
        return None
    kl, refusal = float(trial.values[0]), float(trial.values[1])
    return refusal, kl


def _dominates(a: tuple[float, float], b: tuple[float, float], *, eps: float) -> bool:
    """Return True if ``a`` strictly ε-dominates ``b``.

    A point ``a`` dominates ``b`` iff:
    * ``a`` is no worse than ``b`` in every dimension (up to ``eps``);
    * ``a`` is strictly better in at least one dimension.

    Both objectives are MINIMISED here.
    """
    not_worse = a[0] <= b[0] + eps and a[1] <= b[1] + eps
    strictly_better = a[0] < b[0] - eps or a[1] < b[1] - eps
    return not_worse and strictly_better


def pareto_front(trials: Iterable, *, eps: float = 0.0) -> list[ParetoPoint]:
    """Return the ε-Pareto front of ``trials`` on the (refusal, KL) plane.

    Parameters
    ----------
    trials : Iterable[FrozenTrial]
        Optuna trials; only complete trials with two-element ``values``
        are considered.
    eps : float
        Tolerance for ε-dominance. Use a small positive ε (e.g. 1e-6) to
        avoid floating-point ties producing redundant front points.

    Returns
    -------
    list[ParetoPoint]
        Sorted ascending by refusal (then KL as tie-break).
    """
    points: list[ParetoPoint] = []
    for t in trials:
        objs = _extract_objectives(t)
        if objs is None:
            continue
        points.append(
            ParetoPoint(
                trial_number=t.number,
                refusal=objs[0],
                kl=objs[1],
                params=dict(t.params),
                user_attrs=dict(t.user_attrs),
            )
        )

    front: list[ParetoPoint] = []
    for cand in points:
        dominated = False
        for other in points:
            if other is cand:
                continue
            if _dominates(other.key, cand.key, eps=eps):
                dominated = True
                break
        if not dominated:
            front.append(cand)

    front.sort(key=lambda p: (p.refusal, p.kl))
    return front


# ---------------------------------------------------------------------------
# Grouping by user-attr keys
# ---------------------------------------------------------------------------


def group_trials(
    trials: Iterable,
    group_keys: Iterable[str],
) -> dict[tuple, list]:
    """Bucket trials by the values of one or more user-attr keys.

    Parameters
    ----------
    trials : Iterable[FrozenTrial]
    group_keys : Iterable[str]
        Names of ``trial.user_attrs`` keys to read.  Missing keys default
        to ``None`` so groups remain well-defined for older trials that
        did not have the new attrs set.

    Returns
    -------
    dict[tuple, list[FrozenTrial]]
        Keyed by a tuple of attr values in the same order as ``group_keys``.
    """
    keys = list(group_keys)
    buckets: dict[tuple, list] = {}
    for t in trials:
        group = tuple(t.user_attrs.get(k) for k in keys)
        buckets.setdefault(group, []).append(t)
    return buckets


def per_group_front(
    trials: Iterable,
    group_keys: Iterable[str],
    *,
    eps: float = 0.0,
) -> dict[tuple, list[ParetoPoint]]:
    """Compute a Pareto front for each bucket of grouped trials."""
    buckets = group_trials(trials, group_keys)
    return {key: pareto_front(ts, eps=eps) for key, ts in buckets.items()}


# ---------------------------------------------------------------------------
# Best-trial-at-KL-budget convenience
# ---------------------------------------------------------------------------


def best_under_kl(
    front: Iterable[ParetoPoint],
    *,
    kl_budget: float,
) -> ParetoPoint | None:
    """Return the front point with the lowest refusal whose KL ≤ budget.

    Returns ``None`` when no point satisfies the budget.
    """
    eligible = [p for p in front if p.kl <= kl_budget]
    if not eligible:
        return None
    return min(eligible, key=lambda p: p.refusal)


# ---------------------------------------------------------------------------
# Text formatting
# ---------------------------------------------------------------------------


def format_summary_table(
    grouped_fronts: dict[tuple, list[ParetoPoint]],
    group_keys: Iterable[str],
    *,
    kl_budget: float = 0.05,
) -> str:
    """Build a terminal-friendly summary table.

    Columns: group labels (one per ``group_keys`` entry), best refusal at
    ``kl_budget``, that point's KL, and the trial number for reference.
    """
    keys = list(group_keys)
    header_groups = "  ".join(f"{k:<18}" for k in keys)
    header = f"{header_groups}  best_refusal  kl       trial#"
    sep = "-" * len(header)
    lines = [header, sep]
    # Deterministic ordering: sort by group key tuple to keep output stable.
    for group, front in sorted(
        grouped_fronts.items(), key=lambda kv: tuple(str(x) for x in kv[0])
    ):
        best = best_under_kl(front, kl_budget=kl_budget)
        group_repr = "  ".join(f"{str(v):<18}" for v in group)
        if best is None:
            lines.append(f"{group_repr}  (none below KL {kl_budget})")
        else:
            lines.append(
                f"{group_repr}  {best.refusal:>12.3f}  {best.kl:>6.3f}  "
                f"#{best.trial_number}"
            )
    return "\n".join(lines)
