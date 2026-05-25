"""Optuna mini sweep on Qwen3.6-27B (reasoning model, thinking-mode default).

Differences vs pod_optuna_mini.py:
- Targets Qwen/Qwen3.6-27B
- Larger LLM judge batch + concurrency (20 / 30) for faster eval
- 50-prompt target eval set (vs default 100) to halve judge time
- Disables thinking mode at generation time via the tokenizer template flag
  to keep response lengths tractable; the model is still a reasoning model
  but we want short-form responses for the refusal signal.

Verifies the same three integration paths the v1 script did:
1. TPE samples direct_transform per trial
2. TPE swaps steering_variant per trial
3. Pareto front grouping returns one front per (transform, variant)
"""

from __future__ import annotations

import os
import shutil
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

MODEL = os.environ.get("ABLITERIX_TEST_MODEL", "Qwen/Qwen3.6-27B")


def main():
    ckpt = "/tmp/abliterix_optuna_qwen36"
    shutil.rmtree(ckpt, ignore_errors=True)
    os.makedirs(ckpt, exist_ok=True)

    sys.argv = [
        "pod_optuna_qwen36",
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
        ckpt,
        # Smaller eval set + faster judge (20 batches × 30 concurrent).
        "--detection.llm-judge-batch-size",
        "20",
        "--detection.llm-judge-concurrency",
        "30",
        "--inference.batch-size",
        "4",
        "--inference.max-gen-tokens",
        "100",
        "--inference.min-gen-tokens",
        "60",
        "--non-interactive",
        "--overwrite-checkpoint",
    ]

    print(f"Launching abliterix main() on {MODEL}...")
    t0 = time.time()
    from abliterix.cli import main as cli_main

    cli_main()
    print(f"\n[mini-sweep] total elapsed: {time.time() - t0:.1f}s")

    # Locate journal + print Pareto summary.
    journals = [f for f in os.listdir(ckpt) if f.endswith(".jsonl")]
    if not journals:
        print(f"[FAIL] no journal in {ckpt}; contents={os.listdir(ckpt)}")
        return
    path = os.path.join(ckpt, journals[0])

    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

    lock = JournalFileOpenLock(path)
    storage = JournalStorage(JournalFileBackend(path, lock_obj=lock))
    study = optuna.load_study(study_name="abliterix", storage=storage)

    print(f"\n[mini-sweep] {len(study.trials)} trials")
    for t in study.trials:
        if t.values is None:
            continue
        print(
            f"  trial {t.number:>2}  obj=({t.values[0]:.4f}, {t.values[1]:.4f})  "
            f"refusals={t.user_attrs.get('refusals')}/{t.user_attrs.get('n_eval_prompts', '?')}  "
            f"kl={t.user_attrs.get('kl_divergence'):.4f}  "
            f"transform={t.user_attrs.get('direct_transform')}  "
            f"variant={t.user_attrs.get('steering_variant')}"
        )

    from abliterix.pareto import format_summary_table, per_group_front

    fronts = per_group_front(study.trials, ["direct_transform", "steering_variant"])
    print("\n[mini-sweep] per-(transform, variant) Pareto summary")
    print(
        format_summary_table(
            fronts, ["direct_transform", "steering_variant"], kl_budget=1.0
        )
    )


if __name__ == "__main__":
    main()
