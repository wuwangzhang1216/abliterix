# Historical Video Prompt Set audit — 2026-08-21

This is the original PR author's historical audit report. The numerical findings
below were not independently reproduced during the repair review; no provider
requests, dataset downloads, or uploads were performed. Its v2 generation policy
is retired. Current generation uses source-preserving v3 pairs described in
[the supported workflow](minimax-h3.md).

## Executive summary

The current `video_good_1000` and `video_bad_1000` splits have good breadth,
balanced language/style sampling, and no duplicate prompts. They are suitable
as broad prompt pools. They are not yet a clean paired Prompt Set for extracting
a video-model safety direction because the benign and harmful rows were sampled
independently.

The historical v2 report proposed generated target rewrites. The supported v3
workflow instead keeps source targets verbatim and generates only benign
counterfactuals. Provider refusals and blocked finishes remain excluded for
review; completeness must be reported rather than achieved by bypassing filters.

## Scope and methodology

Source:
[`wangzhang/abliterix-datasets`](https://huggingface.co/datasets/wangzhang/abliterix-datasets)
at immutable revision
`495e1e892236e41f7c3cf77a2616562b5b44608d`.

The audit parsed both JSON arrays and measured row counts, schema fields,
normalized prompt duplicates, category/language/style distributions, prompt
lengths, refusal boilerplate, age references, and metadata cardinality. The v2
artifact was then generated and validated row by row through OpenRouter.

## Findings

| Dimension | `video_good_1000` | `video_bad_1000` |
|---|---:|---:|
| Rows | 1,000 | 1,000 |
| Normalized duplicate prompts | 0 | 0 |
| Language: en / zh / mixed | 322 / 354 / 324 | 322 / 331 / 347 |
| Median prompt length | 169.5 chars | 173.5 chars |
| 95th percentile length | 260 chars | 280 chars |
| Visual styles | 7 | 7 |
| Distinct `style_family` values | 38 | 39 |
| Minor-term matches | 9 benign contexts | 0 |
| Provider-refusal matches after contextual review | 0 | 0 |

The historical v2 report described 1,000 validated pairs, with mean target
length increasing from 158.994 to 293.527 characters (about 1.93x). These are
reported historical results, not evidence for the repaired v3 pipeline. Its
reported use of content-filter retries and alternate providers is unsupported;
those behaviors must not be reproduced by the current workflow.

The apparent harmful-split refusal match was the phrase “不能泄露出去” inside a
depicted corrupt conversation, not a provider refusal. This motivated a
narrower boilerplate detector that looks for first-person refusal patterns
rather than generic words such as “cannot” or “不能”.

### Pairing confound

The harmful category distribution is intentionally uneven because upstream
generation filters blocked some CBRNE prompts: CBRNE has only 20 rows, while
Disinformation & Deepfake has 92. The benign split has a separate distribution
and only 698 category-matched rows. Comparing the two aggregate pools therefore
mixes the safety concept with category frequency, scene content, and camera
presentation.

### Metadata normalization

`visual_style` is already a controlled seven-value field, but `style_family`
contains dozens of model-invented synonyms (`anime_scene`, `anime_aesthetic`,
`anime_animation`, and others). The paired schema derives one canonical family
from `visual_style` instead of asking a model to invent another label.

### Age-related rows

Nine benign rows mention young people in safe contexts such as first aid,
anti-bullying, and anti-radicalization. They are not policy violations. They are
reported as retained in the historical corpus. Schema validation is not a
safety audit, and the current workflow respects provider safety decisions.

## Historical v2 import contract

Each row contains:

- declared `source_dataset` and `source_revision` provenance labels;
- `pair_id` and original harmful row ID;
- immutable source target, amplified target, and benign prompts in the same row;
- shared category, language, shot type, and canonical visual style;
- separate target/benign subject tags;
- target-expansion and benign-transformation summaries plus explicit preserved
  elements;
- generator model identity;
- a SHA-256 fingerprint over the semantic pair contents.

The validator requires an amplified target longer than its immutable source and
rejects duplicates, unknown labels, chat-style instructions, provider-refusal
boilerplate, metadata/style disagreement, and fingerprint drift. These rules apply only to historical v2 imports. New v3 pairs require exact
source preservation and record split/model/policy provenance; they do not require
or accept target expansion.

## Training implications

This v2 output remains a text-only Prompt Set. It can support paired residual
extraction, prompt-conditioned analysis, or generation evaluation. It cannot by
itself supervise MiniMax-H3's rectified-flow objective. The separate `abliterix.h3_training` adapter accepts target video manifests
within its documented basic `fl2va` boundary. It rejects separate audio and
reference-media fields that the reviewed external cache loader would ignore.
GPU training compatibility has not been established by these local checks.

## Current validation boundary

The repaired workflow has local fake-provider tests for source preservation,
provenance drift, corrupted checkpoints, and terminal refusal/filter handling.
No v3 dataset or live generation quality result is claimed here. Retain the source,
checkpoint, exported pairs, and provenance sidecar together; review excluded IDs
and report the resulting coverage before using an export for analysis.
