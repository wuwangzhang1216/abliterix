# Video Prompt Set audit — 2026-08-21

## Executive summary

The current `video_good_1000` and `video_bad_1000` splits have good breadth,
balanced language/style sampling, and no duplicate prompts. They are suitable
as broad prompt pools. They are not yet a clean paired Prompt Set for extracting
a video-model safety direction because the benign and harmful rows were sampled
independently. The paired-prompt pipeline applies no age- or sensitivity-based
content rejection.

The recommended v2 artifact keeps all 1,000 source targets, expands every bad
prompt into a longer and more visually intense target, and generates one strict
benign counterfactual for each target. It fixes the camera, style, language,
setting, and harmless scene elements per pair. This produces 1,000 analyzable
pairs rather than relying on 698 loosely category-matched benign examples plus
302 unrelated everyday prompts.

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

The final v2 artifact contains 1,000 valid pairs. All 1,000 amplified targets
are longer than their immutable sources. Mean target length increased from
158.994 to 293.527 characters (about 1.93x). Gemini 3.7 Flash generated 986
pairs; Venice Uncensored generated the 14 rows that either exhausted Gemini's
provider-side content filter retries or failed the amplification invariant.

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
retained when present; the v2 pipeline does not reject prompts based on age or
content sensitivity.

## Recommended v2 contract

Each row contains:

- an immutable `source_dataset` and `source_revision`;
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
boilerplate, metadata/style disagreement, and fingerprint drift. It does not
apply sensitivity-based content rejection.

## Training implications

This v2 output remains a text-only Prompt Set. It can support paired residual
extraction, prompt-conditioned analysis, or generation evaluation. It cannot by
itself supervise MiniMax-H3's rectified-flow objective. H3 LoRA training also
requires a target video and preferably target 32 kHz stereo audio for every
caption; those assets belong in a separate media manifest validated by
`abliterix.h3_training`.

## Next action

Run a 16-pair Gemini pilot, inspect target amplification and presentation
matching, then resume the same command for all 1,000 pairs. The generator is
resumable and writes only structurally valid rows, so a provider interruption
does not lose accepted work.
