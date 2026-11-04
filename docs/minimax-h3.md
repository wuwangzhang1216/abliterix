# MiniMax-H3 support boundary

MiniMax-H3 is not an autoregressive LLM. It is a joint audio/video diffusion
system whose packed sequence contains text conditioning, video latents, and
audio latents. Abliterix therefore keeps H3 work separate from the LLM
`SteeringEngine` instead of adding architecture-specific exceptions to token
generation, refusal detection, and token-distribution damage metrics.

## What Abliterix supports

This repository provides two H3-oriented building blocks:

1. `abliterix.video_prompt_sets` defines a paired video Prompt Set. Every
   source target is preserved verbatim and paired with a generated benign
   counterfactual using the same
   language, camera, visual style, and nearby scene semantics.
2. `abliterix.h3_training` validates supervised H3 media manifests and lowers
   them into a shell-free cache/training execution plan for a pinned external
   H3 rectified-flow trainer.

New pairs use schema v3 and the explicit
`preserve_source_benign_counterfactual_v1` policy. Schema v2 amplification-era
exports remain readable for historical analysis; the generator no longer creates
or strengthens target scenes. A v3 fingerprint covers every pair field exactly,
including source text, split, model, and policy. The v2 fingerprint algorithm is
retained solely for compatibility.

The second boundary is intentional. MiniMax publishes complete weights for
fine-tuning, but its official repository does not currently contain a trainer.
The Hugging Face integration provides H3 inference and LoRA loading, while the
training objective still needs an H3-specific implementation.

## Prompt Set refinement

The legacy `video_bad_1000` and `video_good_1000` files were independently
sampled. They have broad coverage but are not row-level pairs. Build a paired
set from the harmful split with:

```bash
export OPENROUTER_API_KEY=...
uv run python scripts/refine_video_prompt_pairs.py \
  --input /path/to/video_bad_prompts_1000.json \
  --output datasets/video_paired_1000/video_prompt_pairs_1000.jsonl \
  --model google/gemini-3.7-flash \
  --source-dataset wangzhang/abliterix-datasets \
  --source-revision 495e1e892236e41f7c3cf77a2616562b5b44608d \
  --source-split train \
  --batch-size 8 --workers 4 --rate-limit-rpm 30
```

The generator reads `OPENROUTER_API_KEY` from the process environment. Provider
response bodies and exception details are excluded from diagnostic logs. A
successful export contains validated pairs; length and schema validation do not
establish safety, semantic quality, or suitability for training.

The default `.progress.jsonl` file is a checkpoint, **not** an exported dataset.
Its header binds a SHA-256 of the complete local source rows and metadata to the
declared dataset, revision, split, selected IDs, requested model, and request
policy (including the system prompt, schema, sampling options, batch size, and
attempt limit). Changing any of these requires a new progress path. Legacy
checkpoints without that header are rejected rather than assumed compatible.
The script does not verify that the local file actually belongs to the declared
remote revision. Prefer a pinned commit and retain the original source file.
Requested model aliases may also change upstream; these hashes establish local
run identity, not deterministic generation or provider authenticity.

Checkpoint replacements are atomic, and a record checksum detects truncated or
modified snapshots. Invalid, duplicate, or unknown rows fail before a provider
client is opened. An exclusive `.lock` file rejects concurrent checkpoint
writers. After a hard process crash, confirm that no writer remains before
removing the stale lock; retain the checkpoint. Use distinct output paths for
separate runs.

Provider refusals, blocked or non-stop finishes (including token truncation),
empty responses, malformed outputs, and duplicate generated prompts are recorded
for review without another generation attempt. There is no alternate prompt or
safety-filter bypass. Only transient transport failures can retry the identical
request; SDK retries are disabled. A batch is recorded before it is sent, so an
interrupted request remains for review on resume instead of being submitted
again. A batch-level failure conservatively excludes all sources in that batch.
Review records contain source IDs and reason codes, not provider response text.

The export's `.provenance.json` contains the run identity, output-file hash, and
excluded source IDs/reasons. The summary reports `review_count` and a
`requires_review` status when any sources were excluded; do not treat such an
export as complete coverage. Keep the source, checkpoint, export, and provenance
sidecar together. Fully recorded checkpoints can be exported without an API key.

These contracts follow the [Gemini response and finish-reason documentation](https://ai.google.dev/api/generate-content#v1beta.GenerateContentResponse),
[OpenRouter's normalized finish reasons](https://openrouter.ai/docs/api/reference/overview),
and the distinction between [dataset content/transform fingerprints](https://huggingface.co/docs/datasets/about_cache)
and [declared revisions and splits](https://huggingface.co/docs/datasets/loading).

Prompt pairs alone are suitable for Prompt Set extraction and evaluation. They
are not supervised H3 training data: a rectified-flow loss requires target
video latents and, for native joint training, target audio latents.

## Supervised media manifest

One JSON object per line:

```json
{
  "id": "sample-000001",
  "task": "fl2va",
  "caption": "A tracking shot follows an adult walking through a gallery.",
  "target_video": "/data/targets/sample-000001.mp4",
  "reference_images": [],
  "reference_videos": [],
  "reference_audios": []
}
```

Constraints enforced by the plan validator:

- height and width are divisible by 32;
- `frames % 17 == 5`;
- every media path exists;
- only basic `fl2va` rows are supported; explicit `target_audio`, reference
  media, and `ref2va` rows are rejected because the reviewed external cache
  adapter does not consume them;
- `--encode-audio` uses audio embedded in `target_video`, not a separate WAV;
- full 33B training uses DeepSpeed rather than the one-GPU DDP smoke path.

Create a JSON configuration containing `model_path`, `trainer_path`,
`manifest_path`, `cache_path`, `output_path`, and the training settings, then
write a reviewable plan:

```bash
uv run python scripts/plan_h3_training.py \
  --config configs/minimax_h3_training.json \
  --output runs/minimax_h3/plan.json
```

The schema-v2 plan contains argument arrays and a sibling
`<plan-stem>.manifest.<SHA256>.jsonl` file. The normalized manifest uses absolute
media paths and supplies both `target_video` and the `video` alias required by
the reviewed cache loader. The source manifest is retained. Plan output cannot
overwrite the input configuration or source manifest.

The adapter was reviewed against external trainer commit
[`74d535b2cae5cd56f9565059e6f2b6c9ddb39df4`](https://github.com/IAmIronMan42/MiniMax-H3-FineTuning/tree/74d535b2cae5cd56f9565059e6f2b6c9ddb39df4).
The plan records source/normalized manifest hashes, the local trainer commit
when available, and cache/training script hashes. It explicitly records
`compatibility: unvalidated` and `runtime_validation: not_run`: source inspection
and local contract tests do not demonstrate working GPU training. Review the
plan in the exact external trainer environment before any execution. Abliterix
does not launch the external trainer or execute shell commands on import.

## Recommended training surface

Freeze the Qwen3-VL conditioner and both VAEs, cache their outputs, and train
LoRA on the H3 transformer attention/output and feed-forward projections. The
current Diffusers H3 model exposes PEFT adapter support, including attention
`to_q`, `to_k`, `to_v`, `to_out.0`, FFN projections, and AdaLN projections.

Full-resolution work is not a consumer-GPU fine-tune. The released transformer
is roughly 62 GB in BF16 and its Qwen3-VL conditioner is another roughly 62 GB.
Start with a low-resolution heads-only smoke test, then use LoRA plus ZeRO-3 on
the actual media corpus. Keep prompt-pair evaluation separate from supervised
media reconstruction so the resulting Benchmark Result names the correct
Damage Metrics.

## Primary references

- [MiniMax-H3 official repository](https://github.com/MiniMax-AI/MiniMax-H3)
- [Hugging Face Diffusers MiniMax-H3 pipeline](https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/minimax_h3.md)
- [Diffusers MiniMax-H3 transformer](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_minimax_h3.py)
- [Community H3 rectified-flow trainer](https://github.com/IAmIronMan42/MiniMax-H3-FineTuning)
