<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/logo.svg">
    <img alt="Prometheus" src="assets/logo.svg" width="460">
  </picture>
</p>

<p align="center">
  <strong>0–1.5% refusal rate &nbsp;·&nbsp; 0.01 KL divergence &nbsp;·&nbsp; Zero manual tuning</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/prometheus-llm/"><img src="https://img.shields.io/pypi/v/prometheus-llm?color=blue" alt="PyPI"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
  <a href="https://www.gnu.org/licenses/agpl-3.0"><img src="https://img.shields.io/badge/license-AGPL--3.0-green.svg" alt="License: AGPL v3"></a>
  <a href="https://huggingface.co/wangzhang"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow.svg" alt="Hugging Face"></a>
</p>

---

## Table of Contents

- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Results](#results)
- [Features](#features)
- [MoE Support](#moe-support)
- [Configuration](#configuration)
- [Hardware & VRAM](#hardware--vram)
- [Research Tools](#research-tools)
- [References](#references)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [Contributing](#contributing)
- [License](#license)

---

Prometheus finds the optimal abliteration parameters for any transformer model using [Optuna](https://optuna.org/) TPE optimization. It co-minimizes refusals and KL divergence from the original model — producing decensored models that retain as much intelligence as possible.

Works with dense models, multimodal models, and MoE architectures (Qwen3/3.5 MoE, Mixtral, DeepSeek, Granite MoE Hybrid, MiniMax-M2.5, LiquidAI LFM2, GLM-4 MoE, Mistral3/Devstral).


## Quick Start

```bash
pip install -U prometheus-llm
prometheus --model Qwen/Qwen3-4B-Instruct-2507
```

That's it. The process is fully automatic — after optimization completes, you can save the model, upload to Hugging Face, or chat with it interactively.

> **Windows**: use `python scripts/run_prometheus.py --model <model>` or set `PYTHONIOENCODING=utf-8` to avoid Rich encoding issues.


## How It Works

Language models learn to refuse harmful queries through specific activation patterns in their residual stream. Prometheus identifies these patterns and surgically removes them:

1. **Compute refusal directions** — pass harmless and harmful prompts through the model, extract per-layer residual activations, and compute the difference vector that characterizes "refusal behavior"
2. **Orthogonalize** — project out the component aligned with normal "good" responses, isolating only the refusal signal
3. **Abliterate via LoRA** — apply rank-1 weight modifications to attention and MLP components, weighted by a kernel function across layers. Changes are captured as lightweight LoRA adapters, not destructively applied to base weights
4. **Optimize** — Optuna's Tree-structured Parzen Estimator searches over kernel shape, fractional direction index, and per-component abliteration strength, selecting Pareto-optimal configurations that minimize both refusals and model degradation


## Results

Abliterated models uploaded to [Hugging Face](https://huggingface.co/wangzhang):

| Model | Refusals | KL Divergence | Trials |
|-------|----------|---------------|--------|
| [LFM2-24B-A2B](https://huggingface.co/wangzhang/LFM2-24B-A2B-abliterated) | **0/100 (0%)** | 0.0079 | 50 |
| [GLM-4.7-Flash](https://huggingface.co/wangzhang/GLM-4.7-Flash-abliterated) | 1/100 (1%) | 0.0133 | 50 |
| [Devstral-Small-2-24B](https://huggingface.co/wangzhang/Devstral-Small-2-24B-Instruct-abliterated) | 3/100 (3%) | 0.0086 | 50 |
| [Qwen3.5-122B-A10B](https://huggingface.co/wangzhang/Qwen3.5-122B-A10B-abliterated) | 1/200 (0.5%) | 0.0115 | 25 |
| [Qwen3.5-35B-A3B](https://huggingface.co/wangzhang/Qwen3.5-35B-A3B-abliterated) | 3/200 (1.5%) | **0.0035** | 50 |
| [Qwen3.5-27B](https://huggingface.co/wangzhang/Qwen3.5-27B-abliterated) | 3/200 (1.5%) | 0.0051 | 35 |
| [Qwen3.5-9B](https://huggingface.co/wangzhang/Qwen3.5-9B-abliterated) | 2/200 (1%) | 0.0105 | 50 |
| [Qwen3.5-4B](https://huggingface.co/wangzhang/Qwen3.5-4B-abliterated) | 3/200 (1.5%) | 0.0065 | 50 |
| [Qwen3.5-0.8B](https://huggingface.co/wangzhang/Qwen3.5-0.8B-abliterated) | **0/200 (0%)** | 0.0087 | 100 |

### Key Findings

> **Orthogonalized directions reduced refusals by 67%** compared to raw abliteration in controlled experiments — the single most impactful optimization.

- **Consistent sub-2% refusals across all model sizes** — from 0.8B to 122B, every model achieves 0–1.5% refusal rate. The 0.8B model reaches a perfect 0/200.
- **More trials unlock better parameters** — the 27B improved from 7 to 3 refusals when trials increased from 15 to 35. The 4B dropped from 34 refusals (17%) to just 3 (1.5%) with continued optimization.
- **Per-layer direction index is critical at scale** — for 122B, independently optimizing the refusal direction per layer reduced refusals from 180/200 to 1/200. A single global direction failed entirely.
- **MoE hybrid steering** — combining LoRA abliteration with router weight suppression and fused expert abliteration proved essential for MoE architectures.
- **Non-transformer architectures work too** — LFM2's hybrid conv+attention architecture achieved 0% refusals by steering convolution output projections alongside attention and MLP components.


## Features

### Orthogonalized Directions

Instead of removing the full refusal direction (which degrades model quality), Prometheus projects out only the component orthogonal to "good" response directions. This preserves capabilities while selectively removing refusal behavior.

```toml
[steering]
orthogonal_projection = true
```

### LLM Judge

Replace keyword-based refusal detection with LLM-powered classification via [OpenRouter](https://openrouter.ai/) for more accurate results, especially for non-English models.

```toml
[detection]
llm_judge = true
llm_judge_model = "google/gemini-3.1-flash-lite-preview"
```

### Smart Optimization

- **Auto batch size** — exponential search finds the largest batch size that fits in VRAM
- **KL divergence pruning** — trials with KL above threshold are terminated early, saving compute
- **Fractional direction index** — interpolates between adjacent layer directions for finer-grained search
- **Per-component parameters** — separate abliteration weights for attention, MLP, and convolution components

### Advanced Options

| Section | Option | Values | Description |
|---------|--------|--------|-------------|
| `[steering]` | `vector_method` | `mean`, `median_of_means`, `pca` | How to compute steering vectors from residuals |
| `[steering]` | `decay_kernel` | `linear`, `gaussian`, `cosine` | Kernel for interpolating weights across layers |
| `[steering]` | `weight_normalization` | `none`, `pre`, `full` | Weight row normalization before/after LoRA |
| `[steering]` | `outlier_quantile` | 0.0–1.0 | Tame extreme activations in some models |
| `[model]` | `use_torch_compile` | true/false | 10–30% inference speedup |


## MoE Support

Three steering mechanisms for Mixture-of-Experts models:

1. **Expert Profiling** — hooks router modules to compute per-expert "risk scores" from activation patterns on harmful vs. harmless prompts
2. **Router Weight Suppression** — applies learned negative bias to routing weights of safety-critical experts
3. **Fused Expert Abliteration** — direct rank-1 modification of expert `down_proj` matrices

Supported architectures: Qwen3/3.5 MoE, Mixtral, DeepSeek MoE, Granite MoE Hybrid, MiniMax-M2.5, LiquidAI LFM2 (hybrid conv+attention MoE), GLM-4 MoE Lite (MLA + MoE). See [configs/](configs/) for model-specific examples.


## Configuration

Prometheus loads config in priority order (later overrides earlier):

1. [`configs/default.toml`](configs/default.toml) — copy to `prometheus.toml` and customize
2. `PM_CONFIG` environment variable
3. `--config <path>` CLI flag
4. CLI flags (`--model`, `--model.quant-method bnb_4bit`, etc.)

Run `prometheus --help` for all options.

Pre-built configs for specific setups:

| Config | Target |
|--------|--------|
| [`qwen3.5_4b.toml`](configs/qwen3.5_4b.toml) | Qwen3.5-4B dense |
| [`qwen3.5_9b.toml`](configs/qwen3.5_9b.toml) | Qwen3.5-9B dense |
| [`qwen3.5_27b.toml`](configs/qwen3.5_27b.toml) | Qwen3.5-27B dense (~54GB BF16) |
| [`qwen3.5_35b.toml`](configs/qwen3.5_35b.toml) | Qwen3.5-35B-A3B MoE |
| [`qwen3.5_122b.toml`](configs/qwen3.5_122b.toml) | Qwen3.5-122B-A10B MoE (BF16) |
| [`qwen3.5_122b_4bit.toml`](configs/qwen3.5_122b_4bit.toml) | Qwen3.5-122B-A10B (NF4, ~61GB) |
| [`qwen3.5_122b_int8.toml`](configs/qwen3.5_122b_int8.toml) | Qwen3.5-122B-A10B (INT8, ~122GB) |
| [`qwen3.5_397b.toml`](configs/qwen3.5_397b.toml) | Qwen3.5-397B-A17B MoE (NF4, ~215GB) |
| [`minimax_m2.5.toml`](configs/minimax_m2.5.toml) | MiniMax-M2.5 229B MoE (FP8, ~229GB) |
| [`lfm2_24b.toml`](configs/lfm2_24b.toml) | LiquidAI LFM2-24B-A2B hybrid conv+GQA MoE |
| [`glm4_flash.toml`](configs/glm4_flash.toml) | GLM-4.7-Flash 30B MoE (MLA, ~56GB BF16) |
| [`devstral_24b.toml`](configs/devstral_24b.toml) | Devstral-Small-2-24B dense (~45GB BF16) |
| [`qwen3.5_0.8b_100t.toml`](configs/qwen3.5_0.8b_100t.toml) | Extended 100-trial optimization |
| [`noslop.toml`](configs/noslop.toml) | Anti-slop tuning |


## Hardware & VRAM

Prometheus auto-detects available accelerators (CUDA, XPU, MLU, MUSA, SDAA, NPU, MPS) and distributes layers across devices with `device_map = "auto"`.

For large models:
- **4-bit quantization**: `--model.quant-method bnb_4bit` cuts VRAM by ~4x
- **8-bit quantization**: `--model.quant-method bnb_8bit` — higher quality than 4-bit, ~2x VRAM reduction with CPU offload
- **Per-device memory limits**: set `[model] max_memory = {"0": "20GB", "cpu": "64GB"}` in your config
- **Non-interactive mode**: `--non-interactive` for fully automated batch runs


## Research Tools

```bash
pip install -U prometheus-llm[research]
```

- `--display.plot-residuals` — PaCMAP-projected scatter plots and animated GIFs of residual vectors across layers
- `--display.print-residual-geometry` — cosine similarities, norms, silhouette coefficients

Example: PaCMAP visualization shows harmful (red) vs. harmless (blue) activations separating across layers, revealing how the model's refusal circuitry develops through its depth.

<!-- To add a screenshot: save the image to assets/ and uncomment the line below -->
<!-- ![PaCMAP visualization](assets/pacmap_example.png) -->


## References

Prometheus builds on the following research:

- **Abliteration**: Arditi, A., Obeso, O., Syed, A., Paleka, D., Panickssery, N., Gurnee, W., & Nanda, N. (2024). [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717). *NeurIPS 2024*.
- **Representation Engineering**: Zou, A., Phan, L., Chen, S., Campbell, J., Guo, P., Ren, R., Pan, A., Yin, X., Mazeika, M., Dombrowski, A.-K., Goel, S., Li, N., Byun, M. J., Wang, Z., Mallen, A., Basart, S., Koyejo, S., Song, D., Fredrikson, M., Kolter, J. Z., & Hendrycks, D. (2023). [Representation Engineering: A Top-Down Approach to AI Transparency](https://arxiv.org/abs/2310.01405). *arXiv:2310.01405*.
- **LoRA**: Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). *ICLR 2022*.
- **Optuna**: Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). [Optuna: A Next-generation Hyperparameter Optimization Framework](https://arxiv.org/abs/1907.10902). *KDD 2019*.
- **TPE**: Bergstra, J., Bardenet, R., Bengio, Y., & Kegl, B. (2011). [Algorithms for Hyper-Parameter Optimization](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization). *NeurIPS 2011*.
- **PaCMAP**: Wang, Y., Huang, H., Rudin, C., & Shaposhnik, Y. (2021). [Understanding How Dimension Reduction Tools Work: An Empirical Approach to Deciphering t-SNE, UMAP, TriMap, and PaCMAP for Data Visualization](https://jmlr.org/papers/v22/20-1061.html). *JMLR*, 22, 1–73.

<details>
<summary>BibTeX</summary>

```bibtex
@inproceedings{arditi2024refusal,
  title     = {Refusal in Language Models Is Mediated by a Single Direction},
  author    = {Arditi, Andy and Obeso, Oscar and Syed, Aaquib and Paleka, Daniel and Panickssery, Nina and Gurnee, Wes and Nanda, Neel},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2024},
  url       = {https://arxiv.org/abs/2406.11717}
}

@article{zou2023representation,
  title   = {Representation Engineering: A Top-Down Approach to AI Transparency},
  author  = {Zou, Andy and Phan, Long and Chen, Sarah and Campbell, James and Guo, Phillip and Ren, Richard and Pan, Alexander and Yin, Xuwang and Mazeika, Mantas and Dombrowski, Ann-Kathrin and Goel, Shashwat and Li, Nathaniel and Byun, Michael J. and Wang, Zifan and Mallen, Alex and Basart, Steven and Koyejo, Sanmi and Song, Dawn and Fredrikson, Matt and Kolter, J. Zico and Hendrycks, Dan},
  journal = {arXiv preprint arXiv:2310.01405},
  year    = {2023},
  url     = {https://arxiv.org/abs/2310.01405}
}

@inproceedings{hu2022lora,
  title     = {{LoRA}: Low-Rank Adaptation of Large Language Models},
  author    = {Hu, Edward J. and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2022},
  url       = {https://arxiv.org/abs/2106.09685}
}

@inproceedings{akiba2019optuna,
  title     = {Optuna: A Next-generation Hyperparameter Optimization Framework},
  author    = {Akiba, Takuya and Sano, Shotaro and Yanase, Toshihiko and Ohta, Takeru and Koyama, Masanori},
  booktitle = {Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages     = {2623--2631},
  year      = {2019},
  url       = {https://arxiv.org/abs/1907.10902}
}

@inproceedings{bergstra2011algorithms,
  title     = {Algorithms for Hyper-Parameter Optimization},
  author    = {Bergstra, James and Bardenet, R{\'e}mi and Bengio, Yoshua and K{\'e}gl, Bal{\'a}zs},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  pages     = {2546--2554},
  year      = {2011},
  url       = {https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization}
}

@article{wang2021pacmap,
  title   = {Understanding How Dimension Reduction Tools Work: An Empirical Approach to Deciphering t-SNE, UMAP, TriMap, and PaCMAP for Data Visualization},
  author  = {Wang, Yingfan and Huang, Haiyang and Rudin, Cynthia and Shaposhnik, Yaron},
  journal = {Journal of Machine Learning Research},
  volume  = {22},
  pages   = {1--73},
  year    = {2021},
  url     = {https://jmlr.org/papers/v22/20-1061.html}
}
```

</details>


## Citation

```bibtex
@software{prometheus,
  author = {Wu, Wangzhang},
  title = {Prometheus: Automated LLM Abliteration},
  year = {2026},
  url = {https://github.com/wuwangzhang1216/prometheus}
}
```


## Acknowledgments

Prometheus was initially inspired by [Heretic](https://github.com/p-e-w/heretic).

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

Contributions are welcome! Please open an issue to discuss your idea before submitting a pull request.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes
4. Push to your fork and open a pull request

All contributions are released under the [AGPL-3.0](LICENSE) license.


## License

[AGPL-3.0](LICENSE)
