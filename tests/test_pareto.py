"""Tests for abliterix.pareto — Pareto front + grouping helpers."""

from types import SimpleNamespace


from abliterix.pareto import (
    ParetoPoint,
    best_under_kl,
    format_summary_table,
    group_trials,
    pareto_front,
    per_group_front,
)


def _trial(number, refusal=None, kl=None, params=None, user_attrs=None):
    """Lightweight stand-in for optuna.trial.FrozenTrial.

    Mirrors abliterix's scorer convention: ``values = (kl, compliance)``
    where compliance is the refusal-rate. The test API still accepts
    ``refusal`` and ``kl`` kwargs for readability — we just swap them
    into ``values`` in the order the optimiser writes them.
    """
    values = None if refusal is None else (kl, refusal)
    return SimpleNamespace(
        number=number,
        values=values,
        params=params or {},
        user_attrs=user_attrs or {},
    )


# ---------------------------------------------------------------------------
# pareto_front
# ---------------------------------------------------------------------------


def test_pareto_front_excludes_dominated():
    trials = [
        _trial(0, refusal=0.10, kl=0.05),  # frontier
        _trial(1, refusal=0.20, kl=0.10),  # dominated by 0
        _trial(2, refusal=0.05, kl=0.20),  # frontier
        _trial(3, refusal=0.30, kl=0.30),  # dominated by 0 and 2
    ]
    front = pareto_front(trials)
    front_numbers = sorted(p.trial_number for p in front)
    assert front_numbers == [0, 2]


def test_pareto_front_skips_incomplete_trials():
    trials = [
        _trial(0, refusal=0.10, kl=0.05),
        _trial(1, refusal=None, kl=None),  # incomplete
    ]
    front = pareto_front(trials)
    assert [p.trial_number for p in front] == [0]


def test_pareto_front_sorted_by_refusal_then_kl():
    """All three points pairwise non-dominating → front contains all of them
    sorted by refusal ascending then KL ascending."""
    trials = [
        _trial(0, refusal=0.30, kl=0.05),
        _trial(1, refusal=0.10, kl=0.20),
        _trial(2, refusal=0.20, kl=0.10),
    ]
    front = pareto_front(trials)
    nums = [p.trial_number for p in front]
    assert nums == [1, 2, 0]


def test_pareto_front_eps_tolerance():
    """With ε > 0, ties at the same point should not be considered dominating."""
    trials = [
        _trial(0, refusal=0.10, kl=0.05),
        _trial(1, refusal=0.10 + 1e-7, kl=0.05 + 1e-7),
    ]
    front = pareto_front(trials, eps=1e-6)
    # Both points should appear on the front under ε tolerance.
    assert len(front) == 2


def test_pareto_front_strict_dominance_no_eps():
    """ε=0: identical points are mutually non-dominating; both stay on the front."""
    trials = [
        _trial(0, refusal=0.10, kl=0.05),
        _trial(1, refusal=0.10, kl=0.05),
    ]
    front = pareto_front(trials, eps=0.0)
    # Neither strictly dominates the other → both on the front.
    assert len(front) == 2


def test_pareto_front_carries_user_attrs():
    trials = [
        _trial(0, refusal=0.1, kl=0.1, user_attrs={"direct_transform": "orba"}),
    ]
    front = pareto_front(trials)
    assert front[0].user_attrs["direct_transform"] == "orba"


# ---------------------------------------------------------------------------
# group_trials
# ---------------------------------------------------------------------------


def test_group_trials_single_key():
    trials = [
        _trial(0, refusal=0.1, kl=0.1, user_attrs={"direct_transform": "orba"}),
        _trial(1, refusal=0.2, kl=0.1, user_attrs={"direct_transform": "standard"}),
        _trial(2, refusal=0.05, kl=0.2, user_attrs={"direct_transform": "orba"}),
    ]
    buckets = group_trials(trials, ["direct_transform"])
    assert set(buckets.keys()) == {("orba",), ("standard",)}
    assert len(buckets[("orba",)]) == 2
    assert len(buckets[("standard",)]) == 1


def test_group_trials_multi_key():
    trials = [
        _trial(
            0,
            refusal=0.1,
            kl=0.1,
            user_attrs={"direct_transform": "orba", "steering_variant": "single"},
        ),
        _trial(
            1,
            refusal=0.1,
            kl=0.1,
            user_attrs={
                "direct_transform": "orba",
                "steering_variant": "harmfulness_pair",
            },
        ),
        _trial(
            2,
            refusal=0.1,
            kl=0.1,
            user_attrs={
                "direct_transform": "biprojected",
                "steering_variant": "single",
            },
        ),
    ]
    buckets = group_trials(trials, ["direct_transform", "steering_variant"])
    assert ("orba", "single") in buckets
    assert ("orba", "harmfulness_pair") in buckets
    assert ("biprojected", "single") in buckets


def test_group_trials_missing_attr_defaults_to_none():
    trials = [
        _trial(0, refusal=0.1, kl=0.1, user_attrs={}),
        _trial(1, refusal=0.2, kl=0.2, user_attrs={"direct_transform": "orba"}),
    ]
    buckets = group_trials(trials, ["direct_transform"])
    assert (None,) in buckets
    assert ("orba",) in buckets


# ---------------------------------------------------------------------------
# per_group_front
# ---------------------------------------------------------------------------


def test_per_group_front_independent_fronts():
    """Front of one group should not exclude points dominated only across groups."""
    trials = [
        # orba group
        _trial(0, refusal=0.20, kl=0.10, user_attrs={"direct_transform": "orba"}),
        _trial(1, refusal=0.30, kl=0.20, user_attrs={"direct_transform": "orba"}),
        # standard group — better global front but isolated to its own group
        _trial(2, refusal=0.10, kl=0.05, user_attrs={"direct_transform": "standard"}),
        _trial(3, refusal=0.15, kl=0.20, user_attrs={"direct_transform": "standard"}),
    ]
    fronts = per_group_front(trials, ["direct_transform"])
    # Within orba, trial 0 dominates 1 → front = [0]
    assert [p.trial_number for p in fronts[("orba",)]] == [0]
    # Within standard, trial 2 dominates 3 → front = [2]
    assert [p.trial_number for p in fronts[("standard",)]] == [2]


# ---------------------------------------------------------------------------
# best_under_kl
# ---------------------------------------------------------------------------


def test_best_under_kl_picks_lowest_refusal_within_budget():
    front = [
        ParetoPoint(0, refusal=0.30, kl=0.01, params={}, user_attrs={}),
        ParetoPoint(1, refusal=0.15, kl=0.04, params={}, user_attrs={}),
        ParetoPoint(2, refusal=0.05, kl=0.06, params={}, user_attrs={}),
    ]
    best = best_under_kl(front, kl_budget=0.05)
    assert best is not None
    assert best.trial_number == 1


def test_best_under_kl_returns_none_when_all_above_budget():
    front = [ParetoPoint(0, refusal=0.10, kl=0.5, params={}, user_attrs={})]
    assert best_under_kl(front, kl_budget=0.05) is None


# ---------------------------------------------------------------------------
# format_summary_table
# ---------------------------------------------------------------------------


def test_format_summary_table_includes_all_groups():
    fronts = {
        ("orba",): [ParetoPoint(0, refusal=0.10, kl=0.04, params={}, user_attrs={})],
        ("standard",): [
            ParetoPoint(1, refusal=0.20, kl=0.03, params={}, user_attrs={})
        ],
    }
    text = format_summary_table(fronts, ["direct_transform"], kl_budget=0.05)
    assert "orba" in text
    assert "standard" in text
    assert "#0" in text
    assert "#1" in text


def test_format_summary_table_reports_none_when_budget_unmet():
    fronts = {
        ("orba",): [ParetoPoint(0, refusal=0.10, kl=0.5, params={}, user_attrs={})],
    }
    text = format_summary_table(fronts, ["direct_transform"], kl_budget=0.05)
    assert "(none below KL" in text
