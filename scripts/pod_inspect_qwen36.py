"""Re-extract Qwen3.6-27B sweep results with corrected pareto.py."""

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

from abliterix.pareto import (
    _extract_objectives,
    format_summary_table,
    per_group_front,
)

path = "/tmp/abliterix_optuna_qwen36/Qwen--Qwen3--6-27B.jsonl"
lock = JournalFileOpenLock(path)
storage = JournalStorage(JournalFileBackend(path, lock_obj=lock))
study = optuna.load_study(study_name="abliterix", storage=storage)

header = f"{'#':>2} {'transform':<12} {'variant':<18} {'refusal':>8} {'KL':>7}  {'raw_ref':>7}"
print("6 Qwen3.6-27B trials:")
print(header)
print("-" * len(header))
for t in study.trials:
    if t.values is None:
        continue
    objs = _extract_objectives(t)
    if objs is None:
        continue
    refusal, kl = objs
    raw_refusals = t.user_attrs.get("refusals", "?")
    transform = t.user_attrs.get("direct_transform", "?")
    variant = t.user_attrs.get("steering_variant", "?")
    line = f"{t.number:>2} {transform:<12} {variant:<18}"
    line += f" {refusal:>8.3f} {kl:>7.4f}"
    line += f"  {str(raw_refusals):>7}"
    print(line)

print()
print("--- Pareto front per (transform, variant), KL budget = 0.30 ---")
fronts = per_group_front(study.trials, ["direct_transform", "steering_variant"])
print(
    format_summary_table(
        fronts, ["direct_transform", "steering_variant"], kl_budget=0.30
    )
)
