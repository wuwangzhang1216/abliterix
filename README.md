<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/logo.svg">
    <img alt="Abliterix" src="assets/logo.svg" width="460">
  </picture>
</p>

<p align="center">
  <strong>7% refusal rate on Gemma 4 &nbsp;·&nbsp; 0.0006 KL divergence &nbsp;·&nbsp; 150+ model configs &nbsp;·&nbsp; Zero manual tuning</strong>
</p>

<p align="center">
  <strong>🔥 Breaks <a href="https://arxiv.org/abs/2509.15202">DeepRefusal</a> (EMNLP 2025) and <a href="https://arxiv.org/abs/2406.04313">Circuit Breakers / Representation Rerouting</a> (NeurIPS 2024) — same lerp-then-abliterate recipe, zero fine-tuning</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/abliterix/"><img src="https://img.shields.io/pypi/v/abliterix?color=blue" alt="PyPI"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
  <a href="https://www.gnu.org/licenses/agpl-3.0"><img src="https://img.shields.io/badge/license-AGPL--3.0-green.svg" alt="License: AGPL v3"></a>
  <a href="https://huggingface.co/wangzhang"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow.svg" alt="Hugging Face"></a>
</p>

---

**Abliterix** finds the optimal abliteration parameters for any transformer model using [Optuna](https://optuna.org/) TPE optimization. It co-minimizes refusals and KL divergence from the original model — producing decensored models that retain as much intelligence as possible. Works with dense, MoE, SSM/hybrid, and vision-language architectures, with **150+ pre-built configs**.

It also ships **HonestAbliterationBench**, a reproducible public benchmark that resists the two failure modes (short generations + keyword-only judges) that make most abliteration leaderboards meaningless.

## Safety and Responsible Use

> [!WARNING]
> Abliterix intentionally modifies model internals to reduce refusal behavior. This can weaken or remove safeguards and may cause resulting models to generate inaccurate, biased, offensive, explicit, dangerous, or illegal content.

Abliterix is experimental research software. You are responsible for evaluating any resulting model, complying with applicable laws and third-party terms, and deploying appropriate safeguards. Do not rely on generated outputs for medical, legal, financial, safety-critical, or other high-stakes decisions without qualified human review.

Before use or deployment, read the full **[Safety, Responsible Use, and Disclaimer Notice](SAFETY.md)**. The software is provided **"AS IS"**, without warranty, under Sections 15–17 of the [AGPL-3.0-or-later](LICENSE). The safety guidance does not add restrictions to the rights granted by the AGPL.

## Table of Contents

- [Safety and Responsible Use](#safety-and-responsible-use)
- [Quick Start](#quick-start)
- [Stable and Reproducible by Default](#stable-and-reproducible-by-default)
- [How It Works](#how-it-works)
- [Broken Defenses](#broken-defenses)
- [Results](#results)
- [Honest Abliteration Leaderboard](#honest-abliteration-leaderboard)
- [Model Support](#model-support)
- [Hardware & VRAM](#hardware--vram)
- [Datasets](#datasets)
- [Documentation](#documentation)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [Contributing](#contributing)
- [Community](#community)
- [License](#license)

---

## Quick Start

```bash
pip install -U abliterix
abliterix --model Qwen/Qwen3-4B-Instruct-2507
```

That's it. The process is fully automatic — after optimization completes, you can save the model, upload to Hugging Face, or chat with it interactively.

The default evaluator is deterministic and offline. For a semantic LLM-judge audit, opt in explicitly; the credential is checked before any model is loaded:

```bash
export OPENROUTER_API_KEY=...
abliterix --model Qwen/Qwen3-4B-Instruct-2507 --detection.llm-judge
```

> **Reproducible install (recommended)**: Abliterix uses [uv](https://docs.astral.sh/uv/) and commits a `uv.lock` pinning every dependency, plus a `[tool.uv] exclude-newer` cutoff so lock regeneration can't drift onto a newer dep that breaks the GPU path. If you use uv, clone the repo and run `uv run abliterix --model <model>` to get the exact dependency set the maintainers tested against.

> **Windows**: use `python scripts/run_abliterix.py --model <model>` or set `PYTHONIOENCODING=utf-8` to avoid Rich encoding issues.


## Stable and Reproducible by Default

Abliterix resolves every remote model and dataset input to an immutable Hugging Face commit before loading it. The HF default uses orthogonal projection plus full row-norm preservation; vLLM automatically uses `pre`, the strongest normalization its rank-1 projection cache can materialize. A global seed controls search and randomized low-rank operations.

Published reproducibility manifests use schema v2 and include the resolved configuration, exact winning-trial steering recipe, model/dataset commits, environment, metrics, and model-weight SHA256 values. The `reproducible` tag is only added when all inputs are pinned, the source tree is clean, the evaluator is deterministic, and the backend supports exact materialization.

```bash
# Exact replay: verifies manifest integrity, applies the winning trial without
# searching, then independently re-measures KL divergence and refusal count.
abliterix --reproduce reproduce/reproduce.json

# Headless optimization + exact best-trial export + hashes + manifest.
abliterix --model org/model --non-interactive \
  --non-interactive-output-dir ./verified-model
```

CI also runs a commit-pinned tiny model twice, requires identical exported weight hashes, and verifies exact manifest replay. See [`tests/e2e/`](tests/e2e/) and the [Heretic parity audit](research_heretic_stability_reproducibility_20260806.md).


## How It Works

Abliterix modifies model internals rather than relying on prompt-level jailbreaks. Its basic assumption is that benign prompts and prompts that trigger refusal produce measurably different activation patterns in the model's residual stream.

<p align="center">
  <img src="assets/how-it-works.svg" alt="How Abliterix extracts activations, derives a refusal direction, projects it out of model weights, and optimizes the refusal-versus-drift trade-off" width="100%">
</p>

For each layer, let $g$ be the mean activation for benign prompts and $b$ the mean activation for target prompts. The simplest refusal direction is:

$$
r = \frac{b-g}{\lVert b-g \rVert_2}
$$

Abliteration removes weight components aligned with this direction. A simplified input-side transformation is:

$$
W' = W-\alpha(Wr)r^\top
$$

where $\alpha$ controls the intervention strength. In practice, Abliterix can apply the corresponding projection on either side of a weight matrix, depending on whether a module reads from or writes to the residual stream.

The automated pipeline is:

1. **Extract activations** — run benign and target prompt sets through the original model and capture hidden states from every layer.
2. **Derive steering vectors** — compute a refusal direction or subspace using mean difference, PCA, SRA, SAE, SOM, optimal transport, RDO, or other configured methods.
3. **Apply candidate edits** — modify attention, MLP, and, where applicable, MoE expert/router components using reversible LoRA adapters, direct weight projections, or runtime steering hooks.
4. **Evaluate the trade-off** — count refusals on target prompts while measuring KL divergence and generation quality against the untouched model on benign prompts.
5. **Optimize automatically** — use Optuna TPE to search layer locations, component strengths, decay profiles, vector scope, and MoE routing parameters. The result is a Pareto frontier balancing fewer refusals against less behavioral drift.

In short, Abliterix identifies refusal-related geometry in hidden space, suppresses it in the model, and automatically searches for the least damaging effective intervention. See [docs/architecture.md](docs/architecture.md) for the full pipeline and [docs/methods.md](docs/methods.md) for the available steering methods.


## Broken Defenses

Abliterix has end-to-end broken three of the strongest published "anti-abliteration" releases with the **same minimal recipe**: SVD-diagnose the rank-16 LoRA delta, lerp it away with `λ=0.0` (bit-exact base weights), then run single-direction direct-mode abliteration. No fine-tuning, no iterative subspace, no SOM, no manual prompt engineering. Full lessons-learned write-up: [docs/broken_defenses.md](docs/broken_defenses.md).

| Defense | Released model | Best trial | ASR (LLM judge) | Hardcore 15 |
| --- | --- | --- | --- | --- |
| [DeepRefusal](https://arxiv.org/abs/2509.15202) (EMNLP 2025) | [Llama-3-8B-Instruct-DeepRefusal-Broken](https://huggingface.co/wangzhang/Llama-3-8B-Instruct-DeepRefusal-Broken) ⚔️ | 11/100 refusals, KL 0.053 | **89 %** | 14 / 15 |
| [Circuit Breakers / RR](https://arxiv.org/abs/2406.04313) (NeurIPS 2024) | [Mistral-7B-Instruct-RR-Abliterated](https://huggingface.co/wangzhang/Mistral-7B-Instruct-RR-Abliterated) ⚔️ | 12/100 refusals, KL 0.042 | **88 %** | 15 / 15 |
| [Circuit Breakers / RR](https://arxiv.org/abs/2406.04313) (NeurIPS 2024) | [Llama-3-8B-Instruct-RR-Abliterated](https://huggingface.co/wangzhang/Llama-3-8B-Instruct-RR-Abliterated) ⚔️ | 1/100 refusals, KL 0.017 | **99 %** | 15 / 15 |

Full write-ups, attack recipes, and reproduction commands: **[docs/broken_defenses.md](docs/broken_defenses.md)**.


## Results

Abliterated models uploaded to [Hugging Face](https://huggingface.co/wangzhang):

| Model | Refusals | KL Divergence | Trials | Method |
|-------|----------|---------------|--------|--------|
| [**Llama-3-8B-Instruct-DeepRefusal-Broken**](https://huggingface.co/wangzhang/Llama-3-8B-Instruct-DeepRefusal-Broken) ⚔️ | **11/100 (11%)** | **0.053** | 60 | LoRA-Δ attenuation + Direct |
| [**Mistral-7B-Instruct-RR-Abliterated**](https://huggingface.co/wangzhang/Mistral-7B-Instruct-RR-Abliterated) ⚔️ | **12/100 (12%)** | **0.042** | 60 | Full LoRA-Δ strip + Direct |
| [**Llama-3-8B-Instruct-RR-Abliterated**](https://huggingface.co/wangzhang/Llama-3-8B-Instruct-RR-Abliterated) ⚔️ | **1/100 (1%)** | **0.017** | 60 | Full LoRA-Δ strip + Direct |
| [**Qwen3.6-35B-A3B**](https://huggingface.co/wangzhang/Qwen3.6-35B-A3B-abliterated) | **7/100 (7%)** | **0.0189** | 24 | LoRA + EGA + MoE |
| [**Qwen3.6-27B-abliterated**](https://huggingface.co/wangzhang/Qwen3.6-27B-abliterated) ([GGUF](https://huggingface.co/wangzhang/Qwen3.6-27B-abliterated)) | **10/100 (10%)** | **0.0242** (cumulative) | 30 + 30 | LoRA + manual iterative peel |
| [Qwen3.6-27B-abliterated](https://huggingface.co/wangzhang/Qwen3.6-27B-abliterated) | 10/100 (10%) | 0.0061 | 30 | LoRA + unified GDN/full-attn bucket |
| [**gpt-oss-20b**](https://huggingface.co/wangzhang/gpt-oss-20b-abliterated) | **6/100 (6%)** | **0.0098** | 100 | Direct + EGA + Router |
| [**gpt-oss-120b**](https://huggingface.co/wangzhang/gpt-oss-120b-abliterated) | **26/100 (26%)** | **5.4e-06** | 100 | Direct + EGA + Router + vLLM-TP |
| [**Gemma-4-E4B**](https://huggingface.co/wangzhang/gemma-4-E4B-it-abliterated) | **7/100 (7%)** | **0.0006** | 100 | Direct + Q/K/V/O |
| [**Gemma-4-E2B**](https://huggingface.co/wangzhang/gemma-4-E2B-it-abliterated) | **9/100 (9%)** | **0.0004** | 100 | Direct + Q/K/V/O |
| [**Gemma-4-31B**](https://huggingface.co/wangzhang/gemma-4-31B-it-abliterated) | **3/100 (3%)** | **0.0012** | 120 | SRA + Direct |
| [LFM2-24B-A2B](https://huggingface.co/wangzhang/LFM2-24B-A2B-abliterated) | **0/100 (0%)** | 0.0079 | 50 | LoRA |
| [GLM-4.7-Flash](https://huggingface.co/wangzhang/GLM-4.7-Flash-abliterated) | 1/100 (1%) | 0.0133 | 50 | LoRA |
| [Devstral-Small-2-24B](https://huggingface.co/wangzhang/Devstral-Small-2-24B-Instruct-abliterated) | 3/100 (3%) | 0.0086 | 50 | LoRA |
| [Qwen3.5-122B-A10B](https://huggingface.co/wangzhang/Qwen3.5-122B-A10B-abliterated) | 1/200 (0.5%) | 0.0115 | 25 | LoRA + MoE |
| [Qwen3.5-35B-A3B](https://huggingface.co/wangzhang/Qwen3.5-35B-A3B-abliterated) | 3/200 (1.5%) | **0.0035** | 50 | LoRA + MoE |
| [Qwen3.5-27B](https://huggingface.co/wangzhang/Qwen3.5-27B-abliterated) | 3/200 (1.5%) | 0.0051 | 35 | LoRA |
| [Qwen3.5-9B](https://huggingface.co/wangzhang/Qwen3.5-9B-abliterated) | 2/200 (1%) | 0.0105 | 50 | LoRA |
| [Qwen3.5-4B](https://huggingface.co/wangzhang/Qwen3.5-4B-abliterated) | 3/200 (1.5%) | 0.0065 | 50 | LoRA |
| [Qwen3.5-0.8B](https://huggingface.co/wangzhang/Qwen3.5-0.8B-abliterated) | **0/200 (0%)** | 0.0087 | 100 | LoRA |

> **Numbers worth ~20× the average abliteration leaderboard.** Most published refusal rates collapse under longer generations and a real judge — see [docs/evaluation.md](docs/evaluation.md) for the methodology, and the leaderboard below for community submissions vetted under the same contract.


## Honest Abliteration Leaderboard

A reproducible public benchmark for abliterated models built on the same pipeline. Every row is generated under a frozen contract (`min_new_tokens=100`, `max_new_tokens=150`, greedy, LLM judge with degenerate filter, KL measured against the declared base) — see [benchmarks/SPEC.md](benchmarks/SPEC.md) for the full spec and [benchmarks/CONTRIBUTING.md](benchmarks/CONTRIBUTING.md) for how to submit a row.

<!-- BENCH:START -->
_No results yet. See [benchmarks/CONTRIBUTING.md](benchmarks/CONTRIBUTING.md) for how to submit one._
<!-- BENCH:END -->


## Model Support

Abliterix ships with **150+ pre-built configs** covering 4 architecture types across 20+ model families:

| Architecture | Families | Example Models |
|-------------|----------|----------------|
| **Dense** | Llama, Gemma, Phi, Qwen, Mistral, Yi, InternLM, Falcon, Cohere, EXAONE, Granite, OLMo, SmolLM, SOLAR, Zephyr | Llama-3.1-405B, Gemma-3-27B, Phi-4, DeepSeek-R1-Distill |
| **MoE** | Qwen3/3.5/3.6 MoE, Mixtral, DeepSeek, Phi-3.5-MoE, Granite MoE, DBRX, Llama-4 Scout/Maverick, gpt-oss (MXFP4) | gpt-oss-120b, Qwen3.6-35B-A3B, Qwen3.5-122B, Mixtral-8x22B, Llama-4-Maverick-401B |
| **SSM/Hybrid** | Jamba (Mamba+attention), Nemotron-Cascade (Mamba-2+attention) | Jamba-1.5-Large-94B, Nemotron-Cascade-30B |
| **Vision-Language** | Qwen2-VL, InternVL2, LLaVA-NeXT, Pixtral, Mistral3-VL | Qwen2-VL-7B, LLaVA-NeXT-34B, Pixtral-12B |

Generate configs for new models:

```bash
python scripts/generate_configs.py                 # Generate all missing configs
python scripts/generate_configs.py --family llama   # Only Llama family
```

For MoE-specific steering mechanisms (EGA, expert profiling, router suppression), see [docs/moe.md](docs/moe.md).


## Hardware & VRAM

Abliterix auto-detects available accelerators (CUDA, XPU, MLU, MUSA, SDAA, NPU, MPS) and distributes layers across devices with `device_map = "auto"`.

For large models:
- **4-bit quantization**: `--model.quant-method bnb_4bit` cuts VRAM by ~4x (LoRA mode; the quantised base stays frozen and the ablation rides in a BF16 adapter)
- **8-bit quantization**: `--model.quant-method bnb_8bit` — higher quality than 4-bit, ~2x VRAM reduction with CPU offload
- **Native FP4 models** (gpt-oss MXFP4, DeepSeek-V4-Flash routed experts): abliterate and re-pack **without a BF16 blow-up** via `abliterix-abliterate-fp4` — the output stays 4-bit and serves natively on vLLM. Validated end to end on gpt-oss-20b. `core.frozen_experts` additionally lets the *search* run against packed 4-bit weights by applying the rank-1 EGA edit at forward time instead of mutating weights. See [docs/fp4_repack.md](docs/fp4_repack.md).
- **Per-device memory limits**: set `[model] max_memory = {"0": "20GB", "cpu": "64GB"}` in your config
- **Non-interactive mode**: `--non-interactive` for fully automated batch runs


## Datasets

Bilingual harm/benign evaluation datasets live in [`datasets/`](datasets/) and on Hugging Face at [wangzhang/abliterix-datasets](https://huggingface.co/datasets/wangzhang/abliterix-datasets). The 500-example sets (`harmful_500`, `good_500`) are the recommended starting point — they're also the SHA256-pinned inputs to HonestAbliterationBench.

See [docs/datasets.md](docs/datasets.md) for the design rationale, category breakdown, and a comparison with public alternatives.


## Documentation

The deep details live in `docs/` and `benchmarks/`:

- **[docs/architecture.md](docs/architecture.md)** — the 9 papers Abliterix integrates and the 5-step pipeline.
- **[docs/methods.md](docs/methods.md)** — every steering method (SRA, Spherical, SVF, Projected, Discriminative, COSMIC, Angular, OT, Multi-direction) with the TOML knobs that control it.
- **[docs/method_maturity.md](docs/method_maturity.md)** — evidence levels for every method, from implementation to leading claims.
- **[docs/evaluation.md](docs/evaluation.md)** — why most abliteration benchmarks lie, our standards, and the architecture A/B test.
- **[docs/evidence_resources.md](docs/evidence_resources.md)** — GPU/API/storage resources needed to turn method claims into reproducible evidence.
- **[docs/moe.md](docs/moe.md)** — the four independent MoE steering mechanisms and supported MoE models.
- **[docs/fp4_repack.md](docs/fp4_repack.md)** — abliterate native FP4 (MXFP4/NVFP4) models and re-pack to 4-bit offline, no BF16 blow-up.
- **[docs/configuration.md](docs/configuration.md)** — config loading order, the 150+ shipped configs, the Web UI, and research-mode visualization.
- **[docs/datasets.md](docs/datasets.md)** — bilingual dataset design rationale and metadata schema.
- **[docs/minimax-h3.md](docs/minimax-h3.md)** — MiniMax-H3 Prompt Set refinement, media manifests, and LoRA training boundary.
- **[docs/video-dataset-audit-2026-08.md](docs/video-dataset-audit-2026-08.md)** — measured audit and v2 pairing contract for the video Prompt Sets.
- **[docs/references.md](docs/references.md)** — paper references and BibTeX.
- **[docs/benchmarks/2026-05-pod-validation.md](docs/benchmarks/2026-05-pod-validation.md)** — measured 10-feature sweep on Qwen2.5-7B-Instruct with LLM judge (Blackwell GPU).
- **[benchmarks/METHOD_MATRIX.md](benchmarks/METHOD_MATRIX.md)** — cross-model method matrix for promoting methods through the maturity ladder.
- **[benchmarks/SPEC.md](benchmarks/SPEC.md)** — the frozen HonestAbliterationBench contract (`spec_version 1.1`).
- **[benchmarks/CONTRIBUTING.md](benchmarks/CONTRIBUTING.md)** — how to submit a leaderboard row (self-reported / verified tiers).


## Citation

```bibtex
@software{abliterix,
  author = {Wu, Wangzhang},
  title = {Abliterix: Automated LLM Abliteration},
  year = {2026},
  url = {https://github.com/wuwangzhang1216/abliterix}
}
```


## Acknowledgments

Abliterix is a **derivative work** of [Heretic](https://github.com/p-e-w/heretic) by Philipp Emanuel Weidmann ([@p-e-w](https://github.com/p-e-w)), licensed under [AGPL-3.0-or-later](https://www.gnu.org/licenses/agpl-3.0.html). The original Heretic codebase provided the foundation for this project; Abliterix extends it with Optuna-based multi-objective optimization, LoRA-based steering, MoE architecture support, orthogonal projection, LLM judge detection, and additional model integrations.

All modifications are Copyright (C) 2026 Wangzhang Wu and are released under the same AGPL-3.0-or-later license. See [NOTICE](NOTICE) for details.

```bibtex
@misc{heretic,
  author = {Weidmann, Philipp Emanuel},
  title = {Heretic: Fully automatic censorship removal for language models},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/p-e-w/heretic}}
}
```


## Contributing

Contributions of all kinds are welcome — new model configs, benchmark results, bug reports, documentation, new steering methods. See **[CONTRIBUTING.md](CONTRIBUTING.md)** for development setup, the PR process, and guidance on adding model configs.

The single most impactful contribution is a tested TOML config for a model we don't yet support. Every new config unlocks a new architecture for everyone.

All contributions are released under the [AGPL-3.0](LICENSE) license.


## Community

- **Questions & ideas**: [GitHub Discussions](https://github.com/wuwangzhang1216/abliterix/discussions)
- **Bugs & feature requests**: [GitHub Issues](https://github.com/wuwangzhang1216/abliterix/issues)
- **Security vulnerabilities**: follow the private reporting process in [SECURITY.md](SECURITY.md)
- **Share your models**: tag models you publish with `abliterix` on the Hugging Face Hub so others can find them — browse the growing list at [huggingface.co/models?other=abliterix](https://huggingface.co/models?other=abliterix). Uploading through the built-in menu adds this tag (plus a `reproducible` tag and a `reproduce/` manifest) automatically.


## License

Abliterix is a derivative work of [Heretic](https://github.com/p-e-w/heretic) by Philipp Emanuel Weidmann, licensed under the [GNU Affero General Public License v3.0 or later](LICENSE).

Original work Copyright (C) 2025 Philipp Emanuel Weidmann
Modified work Copyright (C) 2026 Wangzhang Wu
