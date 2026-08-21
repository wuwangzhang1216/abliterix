# MiniMax-H3 support boundary

MiniMax-H3 is not an autoregressive LLM. It is a joint audio/video diffusion
system whose packed sequence contains text conditioning, video latents, and
audio latents. Abliterix therefore keeps H3 work separate from the LLM
`SteeringEngine` instead of adding architecture-specific exceptions to token
generation, refusal detection, and token-distribution damage metrics.

## What Abliterix supports

This repository provides two H3-oriented building blocks:

1. `abliterix.video_prompt_sets` defines a paired video Prompt Set. Every
   harmful target has a benign counterfactual with the same language, camera,
   visual style, and nearby scene semantics. The strict pairing reduces
   category and presentation confounding during safety-direction research.
2. `abliterix.h3_training` validates supervised H3 media manifests and lowers
   them into a shell-free cache/training execution plan for a pinned external
   H3 rectified-flow trainer.

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
  --batch-size 8 --workers 4 --rate-limit-rpm 30
```

The generator never writes credentials. It uses `OPENROUTER_API_KEY` from the
process environment, writes resumable validated JSONL, canonicalizes the seven
visual-style families, rejects chat/refusal boilerplate and minor references,
and fingerprints every pair.

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
  "target_audio": "/data/targets/sample-000001.wav",
  "reference_images": [],
  "reference_videos": [],
  "reference_audios": []
}
```

Constraints enforced by the plan validator:

- height and width are divisible by 32;
- `frames % 17 == 5`;
- every media path exists;
- reference media use the `ref2va` variant;
- full 33B training uses DeepSpeed rather than the one-GPU DDP smoke path.

Create a JSON configuration containing `model_path`, `trainer_path`,
`manifest_path`, `cache_path`, `output_path`, and the training settings, then
write a reviewable plan:

```bash
uv run python scripts/plan_h3_training.py \
  --config configs/minimax_h3_training.json \
  --output runs/minimax_h3/plan.json
```

The resulting artifact contains argument arrays for cache preparation and
training. Review and execute them in the pinned trainer environment; Abliterix
does not run an arbitrary shell command on import.

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
