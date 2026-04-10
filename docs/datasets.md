[← back to README](../README.md)

# Datasets

Why we built our own bilingual harm/benign datasets, and how they differ from `mlabonne/harmful_behaviors`. The exact files (and their SHA256 pinning) used by HonestAbliterationBench live in [benchmarks/SPEC.md](../benchmarks/SPEC.md).

Evaluation prompt datasets are available on Hugging Face: [wangzhang/abliterix-datasets](https://huggingface.co/datasets/wangzhang/abliterix-datasets)

| Dataset | Count | Description |
|---------|-------|-------------|
| `good_500` | 500 | Harmless prompts — recommended for iteration |
| `good_1000` | 1000 | Harmless prompts — full set |
| `harmful_500` | 500 | Harmful prompts — recommended for iteration |
| `harmful_1000` | 1000 | Harmful prompts — full set |

The 500-example sets run ~2x faster than the 1000 sets with no clear quality loss.

## Why we built our own datasets

Public abliteration benchmarks (e.g. `mlabonne/harmful_behaviors`, `mlabonne/harmless_alpaca`) are widely used but have critical limitations:

- **English-only**: zero coverage of Chinese, mixed-language, or code-switching prompts
- **Low sophistication**: mostly direct requests ("How to make X?") with no social engineering
- **Narrow harm taxonomy**: concentrated in a few categories, missing many real-world attack vectors
- **Small and static**: community has memorized them — models may be specifically trained against these exact prompts

Our datasets address all of these:

| Dimension | Our dataset | mlabonne/harmful_behaviors |
|---|---|---|
| **Languages** | English + Chinese + mixed | English only |
| **Sophistication levels** | 4 levels (direct → socially-engineered) | 1 level (direct) |
| **Harm categories** | 12 categories | ~3-4 categories |
| **Format diversity** | QA, roleplay, academic, narrative | Single format |
| **Design methodology** | Adversarial red-teaming with matched benign counterexamples | Community-sourced |

Each prompt includes metadata: `category`, `language`, `sophistication`, `format`, `style_family`, and `design_goal`. The benign datasets are specifically designed as **matched counterexamples** — topically similar to harmful prompts but policy-compliant, which produces cleaner refusal direction vectors.
