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

## Table of Contents

- [Quick Start](#quick-start)
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
- [License](#license)

---

## Quick Start

```bash
pip install -U abliterix
abliterix --model Qwen/Qwen3-4B-Instruct-2507
```

That's it. The process is fully automatic — after optimization completes, you can save the model, upload to Hugging Face, or chat with it interactively.

> **Windows**: use `python scripts/run_abliterix.py --model <model>` or set `PYTHONIOENCODING=utf-8` to avoid Rich encoding issues.


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
| [**Qwen3.6-27B-abliterated-v2**](https://huggingface.co/wangzhang/Qwen3.6-27B-abliterated-v2) ([GGUF](https://huggingface.co/wangzhang/Qwen3.6-27B-abliterated-v2-GGUF)) | **10/100 (10%)** | **0.0242** (cumulative) | 30 + 30 | LoRA + manual iterative peel |
| [Qwen3.6-27B-abliterated](https://huggingface.co/wangzhang/Qwen3.6-27B-abliterated) | 16/100 (16%) | 0.0181 | 30 | LoRA + unified GDN/full-attn bucket |
| [**gpt-oss-20b**](https://huggingface.co/wangzhang/gpt-oss-20b-abliterated) | **6/100 (6%)** | **0.0098** | 100 | Direct + EGA + Router |
| [**gpt-oss-120b**](https://huggingface.co/wangzhang/gpt-oss-120b-abliterated) | **26/100 (26%)** | **5.4e-06** | 100 | Direct + EGA + Router + vLLM-TP |
| [**Gemma-4-E4B**](https://huggingface.co/wangzhang/gemma-4-E4B-it-abliterated) | **7/100 (7%)** | **0.0006** | 100 | Direct + Q/K/V/O |
| [**Gemma-4-E2B**](https://huggingface.co/wangzhang/gemma-4-E2B-it-abliterated) | **9/100 (9%)** | **0.0004** | 100 | Direct + Q/K/V/O |
| [**Gemma-4-31B**](https://huggingface.co/wangzhang/gemma-4-31B-it-abliterated) | **18/100 (18%)** | **0.0007** | 20 | Direct + Q/K/V/O |
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
- **4-bit quantization**: `--model.quant-method bnb_4bit` cuts VRAM by ~4x
- **8-bit quantization**: `--model.quant-method bnb_8bit` — higher quality than 4-bit, ~2x VRAM reduction with CPU offload
- **Per-device memory limits**: set `[model] max_memory = {"0": "20GB", "cpu": "64GB"}` in your config
- **Non-interactive mode**: `--non-interactive` for fully automated batch runs


## Datasets

Bilingual harm/benign evaluation datasets live in [`datasets/`](datasets/) and on Hugging Face at [wangzhang/abliterix-datasets](https://huggingface.co/datasets/wangzhang/abliterix-datasets). The 500-example sets (`harmful_500`, `good_500`) are the recommended starting point — they're also the SHA256-pinned inputs to HonestAbliterationBench.

See [docs/datasets.md](docs/datasets.md) for the design rationale, category breakdown, and a comparison with public alternatives.


## Documentation

The deep details live in `docs/` and `benchmarks/`:

- **[docs/architecture.md](docs/architecture.md)** — the 9 papers Abliterix integrates and the 5-step pipeline.
- **[docs/methods.md](docs/methods.md)** — every steering method (SRA, Spherical, SVF, Projected, Discriminative, COSMIC, Angular, OT, Multi-direction) with the TOML knobs that control it.
- **[docs/evaluation.md](docs/evaluation.md)** — why most abliteration benchmarks lie, our standards, and the architecture A/B test.
- **[docs/moe.md](docs/moe.md)** — the four independent MoE steering mechanisms and supported MoE models.
- **[docs/configuration.md](docs/configuration.md)** — config loading order, the 150+ shipped configs, the Web UI, and research-mode visualization.
- **[docs/datasets.md](docs/datasets.md)** — bilingual dataset design rationale and metadata schema.
- **[docs/references.md](docs/references.md)** — paper references and BibTeX.
- **[benchmarks/SPEC.md](benchmarks/SPEC.md)** — the frozen HonestAbliterationBench contract (`spec_version 1.0`).
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


## License

Abliterix is a derivative work of [Heretic](https://github.com/p-e-w/heretic) by Philipp Emanuel Weidmann, licensed under the [GNU Affero General Public License v3.0 or later](LICENSE).

Original work Copyright (C) 2025 Philipp Emanuel Weidmann
Modified work Copyright (C) 2026 Wangzhang Wu
