<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/logo.svg">
    <img alt="Prometheus" src="assets/logo.svg" width="460">
  </picture>
</p>

<p align="center">
  <strong>3% refusal rate &nbsp;·&nbsp; 0.01 KL divergence &nbsp;·&nbsp; Zero manual tuning</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/prometheus-llm/"><img src="https://img.shields.io/pypi/v/prometheus-llm?color=blue" alt="PyPI"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
  <a href="https://www.gnu.org/licenses/agpl-3.0"><img src="https://img.shields.io/badge/license-AGPL--3.0-green.svg" alt="License: AGPL v3"></a>
  <a href="https://huggingface.co/wangzhang"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow.svg" alt="Hugging Face"></a>
</p>

---

Prometheus finds the optimal abliteration parameters for any transformer model using [Optuna](https://optuna.org/) TPE optimization. It co-minimizes refusals and KL divergence from the original model — producing decensored models that retain as much intelligence as possible.

Works with dense models, multimodal models, and MoE architectures (Qwen3/3.5 MoE, Mixtral, DeepSeek, Granite MoE Hybrid, MiniMax-M2.5).


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
| [Qwen3.5-122B-A10B](https://huggingface.co/wangzhang/Qwen3.5-122B-A10B-abliterated) | **6/200 (3%)** | 0.0115 | 25 |
| [Qwen3.5-35B-A3B](https://huggingface.co/wangzhang/Qwen3.5-35B-A3B-abliterated) | 7/200 (3.5%) | 0.0145 | 50 |
| [Qwen3.5-27B](https://huggingface.co/wangzhang/Qwen3.5-27B-abliterated) | 7/200 (3.5%) | 0.0051 | 15 |
| [Qwen3.5-9B](https://huggingface.co/wangzhang/Qwen3.5-9B-abliterated) | 2/200 (1%) | 0.0105 | 50 |
| [Qwen3.5-4B](https://huggingface.co/wangzhang/Qwen3.5-4B-abliterated) | 34/200 (17%) | 0.0159 | 50 |
| [Qwen3.5-0.8B](https://huggingface.co/wangzhang/Qwen3.5-0.8B-abliterated) | 3/200 (1.5%) | 0.0087 | 100 |

### Key Findings

> **Orthogonalized directions reduced refusals by 67%** compared to raw abliteration in controlled experiments — the single most impactful optimization.

- **Larger models abliterate better** — the 122B achieved lower refusals *and* lower KL than the 35B, in half the trials. Larger models have cleaner refusal circuitry.
- **Per-layer direction index is critical at scale** — for 122B, independently optimizing the refusal direction per layer reduced refusals from 180/200 to 6/200. A single global direction failed entirely.
- **MoE hybrid steering** — combining LoRA abliteration with router weight suppression and fused expert abliteration proved essential for MoE architectures.


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
- **Per-component parameters** — separate abliteration weights for attention vs. MLP

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

Supported architectures: Qwen3/3.5 MoE, Mixtral, DeepSeek MoE, Granite MoE Hybrid, MiniMax-M2.5. See [configs/](configs/) for model-specific examples.


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
| [`4b.toml`](configs/4b.toml) | Qwen3.5-4B dense |
| [`9b.toml`](configs/9b.toml) | 9B dense models |
| [`27b.toml`](configs/27b.toml) | Qwen3.5-27B dense (~54GB BF16) |
| [`35b.toml`](configs/35b.toml) | Qwen3.5-35B-A3B MoE |
| [`122b.toml`](configs/122b.toml) | Qwen3.5-122B-A10B MoE (BF16) |
| [`122b_4bit.toml`](configs/122b_4bit.toml) | Qwen3.5-122B-A10B (NF4, ~61GB) |
| [`122b_int8.toml`](configs/122b_int8.toml) | Qwen3.5-122B-A10B (INT8, ~122GB) |
| [`397b.toml`](configs/397b.toml) | Qwen3.5-397B-A17B MoE (NF4, ~215GB) |
| [`minimax_m25.toml`](configs/minimax_m25.toml) | MiniMax-M2.5 229B MoE (FP8, ~229GB) |
| [`100t.toml`](configs/100t.toml) | Extended 100-trial optimization |
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


## Citation

```bibtex
@software{prometheus,
  author = {Wu, Wangzhang},
  title = {Prometheus: Automated LLM Abliteration},
  year = {2026},
  url = {https://github.com/wuwangzhang1216/prometheus}
}
```


## License

[AGPL-3.0](LICENSE)
