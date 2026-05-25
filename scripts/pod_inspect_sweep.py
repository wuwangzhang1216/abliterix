"""Read the Optuna journal from the mini sweep and report per-trial dims."""

import json
import os
import sys

# Find the journal file.
ckpt_dir = "/tmp/abliterix_optuna_mini"
journals = [f for f in os.listdir(ckpt_dir) if f.endswith(".jsonl")]
if not journals:
    sys.exit(f"no journal in {ckpt_dir}")
path = os.path.join(ckpt_dir, journals[0])
print(f"journal: {path}")

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

lock = JournalFileOpenLock(path)
storage = JournalStorage(JournalFileBackend(path, lock_obj=lock))
study = optuna.load_study(study_name="abliterix", storage=storage)

print(f"trials: {len(study.trials)}")
rows = []
for t in study.trials:
    if t.values is None:
        continue
    transform = t.user_attrs.get("direct_transform")
    variant = t.user_attrs.get("steering_variant")
    refusals = int(t.values[0])
    kl = float(t.values[1])
    rows.append(
        {
            "n": t.number,
            "refusals": refusals,
            "kl": kl,
            "transform": transform,
            "variant": variant,
        }
    )
    print(
        f"  trial {t.number:>2}  refusals={refusals:>3}/100  "
        f"kl={kl:.4f}  transform={transform}  variant={variant}"
    )

print()
print("--- per-(transform, variant) Pareto front ---")
from abliterix.pareto import format_summary_table, per_group_front

fronts = per_group_front(study.trials, ["direct_transform", "steering_variant"])
print(
    format_summary_table(
        fronts, ["direct_transform", "steering_variant"], kl_budget=1.0
    )
)

with open("/workspace/pod_optuna_mini_report.json", "w") as f:
    json.dump(rows, f, indent=2)
print("\nwritten /workspace/pod_optuna_mini_report.json")
