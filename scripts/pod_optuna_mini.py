"""Mini Optuna run with the new search dimensions on Qwen2.5-7B-Instruct.

Verifies that:
1. The optimiser samples direct_transform per trial.
2. The optimiser swaps steering_vectors between single + harmfulness_pair.
3. Pareto front grouping yields one front per (transform, variant) pair.

Runs 6 trials (3 transforms × 2 variants × random warmup) — small enough
to fit in ~10 minutes on the Blackwell pod.
"""

from __future__ import annotations

import json
import os
import sys
import time


def _load_dotenv(path):
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv("/workspace/abliterix/.env")
os.environ.setdefault("HF_HOME", "/workspace/models")

# Argparse-y env override.
MODEL = os.environ.get("ABLITERIX_TEST_MODEL", "Qwen/Qwen2.5-7B-Instruct")


def main():
    sys.argv = [
        "pod_optuna_mini",
        "--model.model-id",
        MODEL,
        "--steering.steering-mode",
        "direct",
        "--steering.search-direct-transform",
        "--steering.search-harmfulness-direction",
        "--optimization.num-trials",
        "6",
        "--optimization.num-warmup-trials",
        "6",
        "--optimization.checkpoint-dir",
        "/tmp/abliterix_optuna_mini",
        "--inference.batch-size",
        "4",
        "--inference.max-gen-tokens",
        "120",
        "--inference.min-gen-tokens",
        "80",
        "--non-interactive",
        "--overwrite-checkpoint",
    ]

    import shutil

    shutil.rmtree("/tmp/abliterix_optuna_mini", ignore_errors=True)
    os.makedirs("/tmp/abliterix_optuna_mini", exist_ok=True)

    print("Launching abliterix main()...")
    t0 = time.time()
    from abliterix.cli import main as cli_main

    cli_main()

    elapsed = time.time() - t0
    print(f"\n[mini-sweep] elapsed: {elapsed:.1f}s")

    # Open the journal storage that the run wrote.
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

    # Find the checkpoint file the CLI created. The checkpoint name uses the
    # slugified model id.
    ckpt_dir = "/tmp/abliterix_optuna_mini"
    journals = [f for f in os.listdir(ckpt_dir) if f.endswith(".log")]
    if not journals:
        print(
            f"[FAIL] no journal file in {ckpt_dir}; contents = {os.listdir(ckpt_dir)}"
        )
        return
    path = os.path.join(ckpt_dir, journals[0])
    lock = JournalFileOpenLock(path)
    storage = JournalStorage(JournalFileBackend(path, lock_obj=lock))
    study = optuna.load_study(study_name="abliterix", storage=storage)

    print(f"\n[mini-sweep] {len(study.trials)} trials in study")
    rows = []
    for t in study.trials:
        if t.values is None:
            continue
        rows.append(
            {
                "n": t.number,
                "refusals": int(t.values[0]),
                "kl": float(t.values[1]),
                "direct_transform": t.user_attrs.get("direct_transform"),
                "steering_variant": t.user_attrs.get("steering_variant"),
            }
        )
        print(
            f"  trial {t.number:>2}  refusals={int(t.values[0]):>2} kl={float(t.values[1]):.4f}  "
            f"transform={t.user_attrs.get('direct_transform')} "
            f"variant={t.user_attrs.get('steering_variant')}"
        )

    # Group by (transform, variant) and print per-group Pareto fronts.
    from abliterix.pareto import format_summary_table, per_group_front

    fronts = per_group_front(study.trials, ["direct_transform", "steering_variant"])
    print("\n[mini-sweep] per-(transform, variant) Pareto front summary")
    print(
        format_summary_table(
            fronts, ["direct_transform", "steering_variant"], kl_budget=1.0
        )
    )

    out = {"rows": rows, "n_trials": len(rows)}
    with open("/workspace/pod_optuna_mini_report.json", "w") as f:
        json.dump(out, f, indent=2, default=str)


if __name__ == "__main__":
    main()
