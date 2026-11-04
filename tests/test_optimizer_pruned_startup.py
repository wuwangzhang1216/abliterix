"""Exercise TPE's multi-objective startup with valueless pruned trials."""

from types import SimpleNamespace

import optuna
import pytest
import torch
from optuna.distributions import CategoricalDistribution, FloatDistribution
from optuna.samplers import TPESampler
from optuna.storages import InMemoryStorage, JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.trial import TrialState, create_trial

from abliterix.optimizer import run_search
from abliterix.settings import AbliterixConfig


class _Engine:
    def get_n_layers(self):
        return 4

    def list_steerable_components(self):
        return []

    def restore_baseline(self):
        pass


class _Scorer:
    target_msgs = [object()]
    detector = SimpleNamespace(evaluate_compliance=lambda *_: 1)

    def __init__(self, prune_first=0):
        self.calls = 0
        self.prune_first = prune_first

    def measure_kl_and_coherence(self, _engine):
        self.calls += 1
        return (10.0 if self.calls <= self.prune_first else 0.1), 0.0

    def _compute_objectives(self, kl, detected, _length_dev):
        return kl, float(detected)


@pytest.fixture(autouse=True)
def _isolate_model_work(monkeypatch):
    for name in ("apply_steering", "flush_memory", "report_memory", "print"):
        monkeypatch.setattr(f"abliterix.optimizer.{name}", lambda *_, **__: None)


def _config(num_trials):
    return AbliterixConfig(
        model={"model_id": "test/model"},
        seed=42,
        optimization={"num_trials": num_trials, "num_warmup_trials": 60},
    )


def _run(config, storage, scorer=None):
    return run_search(
        config,
        _Engine(),
        scorer or _Scorer(),
        steering_vectors=torch.ones(5, 3),
        safety_experts=None,
        storage=storage,
    )


def _add_startup_history(study, completed):
    distributions = {
        "vector_scope": CategoricalDistribution(["global", "per layer"]),
        "vector_index": FloatDistribution(0.3 * 3, 0.95 * 3),
    }
    for index in range(60):
        study.add_trial(
            create_trial(
                state=TrialState.COMPLETE if index < completed else TrialState.PRUNED,
                values=[index / 10, 100 - index] if index < completed else None,
                params={
                    "vector_scope": "global" if index % 2 else "per layer",
                    "vector_index": 1.0 + index / 60,
                },
                distributions=distributions,
            )
        )


def _journal_storage(path):
    # The default symlink lock needs administrator privileges on Windows.
    return JournalStorage(JournalFileBackend(path, lock_obj=JournalFileOpenLock(path)))


def test_all_pruned_warmup_reaches_first_completed_trial():
    scorer = _Scorer(prune_first=60)

    study = _run(_config(61), InMemoryStorage(), scorer)

    assert scorer.calls == 61
    assert [trial.state for trial in study.trials[:60]] == [TrialState.PRUNED] * 60
    assert study.trials[60].state == TrialState.COMPLETE
    assert study.trials[60].values == [0.1, 1.0]
    assert set(study.trials[60].params) == {"vector_scope", "vector_index"}


@pytest.mark.parametrize("completed", [0, 1, 5])
def test_resume_mixed_pruned_history_preserves_failed_trial(tmp_path, completed):
    journal_path = str(tmp_path / "study.log")
    storage = _journal_storage(journal_path)
    original = optuna.create_study(
        study_name="abliterix", directions=["minimize", "minimize"], storage=storage
    )
    _add_startup_history(original, completed)
    original.add_trial(
        create_trial(state=TrialState.FAIL, user_attrs={"failure": "startup broadcast"})
    )

    # Reopen the actual journal rather than sharing the original in-memory study.
    resumed = _run(_config(completed + 1), _journal_storage(journal_path))

    assert len(resumed.trials) == 62
    assert resumed.trials[60].state == TrialState.FAIL
    assert resumed.trials[60].user_attrs == {"failure": "startup broadcast"}
    assert resumed.trials[61].state == TrialState.COMPLETE
    assert sum(t.state == TrialState.PRUNED for t in resumed.trials) == 60 - completed


@pytest.mark.parametrize("completed", [6, 12])
def test_enough_completed_trials_preserve_seeded_default_tpe_sampling(completed):
    storage = InMemoryStorage()
    existing = optuna.create_study(
        study_name="abliterix", directions=["minimize", "minimize"], storage=storage
    )
    _add_startup_history(existing, completed)
    baseline = optuna.create_study(
        directions=["minimize", "minimize"],
        sampler=TPESampler(
            n_startup_trials=60, n_ei_candidates=128, multivariate=True, seed=42
        ),
    )
    baseline.add_trials(existing.trials)
    expected = baseline.ask()
    expected.suggest_categorical("vector_scope", ["global", "per layer"])
    expected.suggest_float("vector_index", 0.3 * 3, 0.95 * 3)

    actual = _run(_config(completed + 1), storage)

    assert actual.trials[-1].state == TrialState.COMPLETE
    assert actual.trials[-1].params == expected.params


def test_resume_unfinished_trial_does_not_bias_next_proposal(tmp_path):
    """An interrupted journal entry must not repel sequential TPE proposals."""
    clean_storage = InMemoryStorage()
    clean = optuna.create_study(
        study_name="abliterix",
        directions=["minimize", "minimize"],
        storage=clean_storage,
    )
    _add_startup_history(clean, completed=12)
    expected = _run(_config(13), clean_storage).trials[-1]

    journal_path = str(tmp_path / "unfinished.log")
    interrupted = optuna.create_study(
        study_name="abliterix",
        directions=["minimize", "minimize"],
        storage=_journal_storage(journal_path),
    )
    interrupted.add_trials(clean.trials[:-1])
    interrupted.add_trial(
        create_trial(
            state=TrialState.RUNNING,
            params=expected.params,
            distributions=expected.distributions,
            user_attrs={"interrupted": True},
        )
    )

    resumed = _run(_config(13), _journal_storage(journal_path))

    assert len(resumed.trials) == 62
    assert resumed.trials[60].state == TrialState.RUNNING
    assert resumed.trials[60].user_attrs == {"interrupted": True}
    assert resumed.trials[-1].state == TrialState.COMPLETE
    assert resumed.trials[-1].params == expected.params
