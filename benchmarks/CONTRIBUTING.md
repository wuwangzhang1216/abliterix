# Contributing to HonestAbliterationBench

Two submission tiers, one spec.

Read [SPEC.md](SPEC.md) first. Every reported number must satisfy the
contract there. Spec changes bump `spec_version`; older results are kept
but rendered as stale.

---

## Tier 1 — `self_reported`

For fast iteration. You run the bench on your hardware, you open the PR.

1. **Set your OpenRouter key**:
   ```bash
   export OPENROUTER_API_KEY=sk-or-...
   ```

2. **Run the bench**:
   ```bash
   python scripts/honest_bench.py \
     --model your-org/your-abliterated-model \
     --base-model original-org/base-model \
     --tier self_reported
   ```
   Add `--trust-remote-code` if your model needs it. On a single H100 the
   full 500+500 set takes ~30 min and ~$1–2 of judge cost.

3. **Verify the result locally**:
   ```bash
   python scripts/build_leaderboard.py
   git diff README.md benchmarks/results/
   ```

4. **Open a PR** including:
   - `benchmarks/results/<org>__<model>.json`
   - `benchmarks/results/<org>__<model>.outputs.jsonl` (raw responses, for spot-checking)
   - the updated README leaderboard block
   - the `judge_cache.sqlite3` file from your `checkpoints/` directory,
     uploaded as a release asset or HF dataset and linked in the PR
     description (so reviewers can re-derive the labels without
     spending judge credits)

5. **Spot check**: a maintainer will diff 20 random rows of your
   `outputs.jsonl` against the judge labels in your cache. Mismatches
   block the PR.

---

## Tier 2 — `verified`

For results that should carry more weight. A maintainer runs the bench
on a known H100 pod and commits the result.

1. **Open an issue** with title `[verified] org/model-name` containing:
   - HF model id of the abliterated model
   - HF model id of the base model
   - any non-default load flags (`trust_remote_code`, dtype, etc.)
   - what you expect to see (rough refusal/KL numbers — used as a
     sanity check, not as truth)

2. **A maintainer runs**:
   ```bash
   python scripts/honest_bench.py \
     --model <model> --base-model <base> --tier verified
   ```
   on a pinned pod image. The runner records `gpu`, `commit`, and
   `runtime_seconds` in the result JSON.

3. **The PR is opened by the maintainer**, not the submitter. Verified
   rows are visually distinguished (`✓ verified`) on the leaderboard.

---

## What you cannot change

- Generation tokens (`min=100`, `max=150`) — pinned by the spec.
- Judge model — pinned to `google/gemini-3-flash-preview` for
  `spec_version 1.0`.
- Datasets — pinned by SHA256. The runner refuses to start on a
  hash mismatch.
- Single user turn, no system prompt, model's own chat template.
- Greedy decoding (`do_sample=False`).

If you think the spec is wrong, open an issue proposing a `spec_version`
bump rather than silently editing config or runner code.

---

## What you should report alongside numbers

In the PR description, briefly answer:

- **Which abliteration technique** produced this model? (LoRA / direct /
  EGA / SRA / etc.) Link the recipe or commit if it lives in this repo.
- **Did you tune to the bench?** Honest answer: did your training data
  overlap with `harmful_500` / `good_500`? (Tuning to a public spec is
  not forbidden but it should be disclosed; future spec versions will
  rotate the held-out subset.)
- **Anything we should know about the base?** (Quantization, finetune
  lineage, etc.) The runner records the base model id but not its
  provenance.

---

## Reproducing an existing row

```bash
git checkout <commit recorded in result JSON>
export OPENROUTER_API_KEY=sk-or-...
python scripts/honest_bench.py --model X --base-model Y --tier self_reported
```

The judge cache is content-addressed, so labels are deterministic across
re-runs as long as the prompt/response strings are byte-identical. Greedy
decoding plus the same model weights and tokenizer guarantees that.

---

## Smoke test (no GPU required)

```bash
python scripts/honest_bench.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --base-model Qwen/Qwen2.5-0.5B \
  --tier self_reported \
  --device-map cpu \
  --dry-run
```

Uses 5+5 prompts. Runs end-to-end in a couple of minutes and round-trips
through `scripts/build_leaderboard.py`. Don't commit dry-run results.
