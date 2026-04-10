[← back to README](../README.md)

# Configuration & Tools

How Abliterix loads its configuration, the Web UI, and the research-mode visualization tools. The TOML steering knobs themselves are catalogued in [methods.md](methods.md).

## Configuration

Abliterix loads config in priority order (later overrides earlier):

1. [`configs/default.toml`](../configs/default.toml) — copy to `abliterix.toml` and customize
2. `AX_CONFIG` environment variable
3. `--config <path>` CLI flag
4. CLI flags (`--model`, `--model.quant-method bnb_4bit`, etc.)

Run `abliterix --help` for all options.

**150+ pre-built configs** in [`configs/`](../configs/) — a selection:

| Config | Target |
|--------|--------|
| [`llama3.1_8b.toml`](../configs/llama3.1_8b.toml) | Llama 3.1 8B Instruct |
| [`llama3.3_70b_4bit.toml`](../configs/llama3.3_70b_4bit.toml) | Llama 3.3 70B (4-bit) |
| [`llama4_scout_109b.toml`](../configs/llama4_scout_109b.toml) | Llama 4 Scout 109B MoE |
| [`gemma3_27b.toml`](../configs/gemma3_27b.toml) | Gemma 3 27B |
| [`phi4.toml`](../configs/phi4.toml) | Phi-4 14B |
| [`deepseek_r1_distill_32b.toml`](../configs/deepseek_r1_distill_32b.toml) | DeepSeek R1 Distill 32B |
| [`qwen3.5_122b.toml`](../configs/qwen3.5_122b.toml) | Qwen3.5-122B-A10B MoE |
| [`mixtral_8x7b.toml`](../configs/mixtral_8x7b.toml) | Mixtral 8x7B MoE |
| [`jamba1.5_mini.toml`](../configs/jamba1.5_mini.toml) | Jamba 1.5 Mini (SSM+MoE) |
| [`qwen2_vl_7b.toml`](../configs/qwen2_vl_7b.toml) | Qwen2-VL 7B (Vision) |
| [`lfm2_24b.toml`](../configs/lfm2_24b.toml) | LiquidAI LFM2-24B hybrid conv+GQA MoE |
| [`noslop.toml`](../configs/noslop.toml) | Anti-slop tuning |

## Web UI

Launch the Gradio-based Web UI for a browser-based steering experience:

```bash
pip install abliterix[ui]
abliterix --ui
```

The UI provides:
- **Model selection** — preset config dropdown + custom HuggingFace model ID
- **Optimisation dashboard** — real-time Pareto front plot, trial log, progress tracking
- **Side-by-side comparison** — baseline vs. steered model responses
- **Interactive chat** — chat with the steered model
- **One-click export** — save locally or upload to HuggingFace Hub

## Research Tools

```bash
pip install -U abliterix[research]
```

- `--display.plot-residuals` — PaCMAP-projected scatter plots and animated GIFs of residual vectors across layers
- `--display.print-residual-geometry` — cosine similarities, norms, silhouette coefficients

Example: PaCMAP visualization shows harmful (red) vs. harmless (blue) activations separating across layers, revealing how the model's refusal circuitry develops through its depth.

<!-- To add a screenshot: save the image to assets/ and uncomment the line below -->
<!-- ![PaCMAP visualization](../assets/pacmap_example.png) -->
