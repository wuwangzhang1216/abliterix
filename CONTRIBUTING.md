# Contributing to Abliterix

Thanks for your interest in Abliterix! This project lives or dies by community contributions — every model config, every bug report, every benchmark result makes the framework more useful for the next person. This document explains how to get involved.

We welcome contributions from everyone, regardless of experience level. If you're not sure where to start, open an issue and ask.

---

## Ways to Contribute

You don't have to write code to help. Some of the most valuable contributions are:

- **Model configs** — Add a TOML config under [configs/](configs/) for a model we don't yet support. This is the single most impactful contribution: every new config unlocks a new architecture for everyone.
- **Benchmark results** — Run an existing config and report your numbers (KL, refusal rate, classic-prompt pass rate). Reproducibility is the foundation of honest research.
- **Bug reports** — Found a model that crashes during layer discovery? An expert routing path that misbehaves? Open an issue with a stack trace and the config you used.
- **Documentation** — Clearer explanations, typo fixes, translations of the README, real-world walkthroughs.
- **New steering methods** — Implementing a new abliteration technique from a paper? We'd love a PR. Open an issue first to discuss the design.
- **Eval datasets** — Curated harmful/harmless prompt pairs in underrepresented languages or domains.

If you found Abliterix useful, even just a star on GitHub or a mention in your own work helps the project grow.

---

## Development Setup

Abliterix uses [`uv`](https://github.com/astral-sh/uv) for dependency management.

```bash
git clone https://github.com/wuwangzhang1216/abliterix.git
cd abliterix
uv sync --group dev
```

For GPU work with vLLM:

```bash
uv sync --group dev --extra vllm
```

For research extras (PCA visualization, geom-median, etc.):

```bash
uv sync --group dev --extra research
```

Run the CLI from your checkout:

```bash
uv run abliterix --help
```

---

## Before You Submit a PR

Run the same checks CI runs. All four must pass:

```bash
uv run ruff check src/         # lint
uv run ruff format --check src/ # format
uv run ty check src/abliterix/  # type check
uv run pytest tests/            # unit tests
```

To auto-fix lint and formatting:

```bash
uv run ruff check --fix src/
uv run ruff format src/
```

If you touched anything in [src/abliterix/core/steering.py](src/abliterix/core/steering.py) or [src/abliterix/eval/scorer.py](src/abliterix/eval/scorer.py), please add or update the corresponding tests in [tests/](tests/). The math in those files is load-bearing for every downstream user.

---

## Contributing a Model Config

This is the most common contribution and we want to make it as easy as possible.

1. Find a similar existing config in [configs/](configs/) — pick one for the same architecture family if possible (Qwen, Llama, Gemma, MoE, etc.).
2. Copy it to a new file: `configs/your_model_name.toml`.
3. Update at minimum:
   - `[model] model_id` — the HuggingFace repo
   - `[hardware]` — sensible defaults for the GPU you tested on
   - `[search]` — strength range, position range, layer range appropriate for the model size
4. Test it:
   ```bash
   uv run abliterix --config configs/your_model_name.toml --search.n-trials 5
   ```
5. In your PR, include:
   - The GPU you tested on (e.g. "RTX Pro 6000 96GB")
   - Final KL divergence and refusal count from a real run (even a short one)
   - Any architecture-specific gotchas you hit

We'd rather merge a config that's been tested for 5 trials than one that was never run at all.

---

## Pull Request Process

1. Fork the repository and create a branch off `master`:
   ```bash
   git checkout -b feature/short-descriptive-name
   ```
2. Make your changes. Keep commits focused — one logical change per commit is easier to review than a giant blob.
3. Run the checks above.
4. Push and open a PR against `master`. In the PR description, explain:
   - **What** the change does
   - **Why** it's needed (link to issue if applicable)
   - **How** you tested it (especially for steering / detector / config changes — actual numbers please)
5. Be patient with review. This is a small project; sometimes things take a few days.

For non-trivial changes (new steering modes, breaking API changes, dependency additions), please open an issue first so we can discuss the design before you spend time on implementation.

---

## Reporting Bugs

Open an issue with:

- The exact command you ran
- The full stack trace (not just the last line)
- The config file (or a link to it)
- Output of `uv pip list | grep -E "torch|transformers|peft|abliterix"`
- GPU model and CUDA version

If you can isolate the bug to a specific commit via `git bisect`, even better — but it's not required.

---

## Code Style

We follow the rules enforced by `ruff` and `ty`. A few conventions on top of that:

- **No speculative abstractions.** If you need a helper exactly once, inline it. Add the abstraction when you need it twice.
- **Trust internal code.** Validate at boundaries (user input, HF API responses), not at every function call.
- **Comment the *why*, not the *what*.** Code shows what; comments should explain non-obvious reasoning, references to papers, or workarounds for upstream bugs.
- **Avoid emojis in source files** unless they were already there.

---

## Licensing

Abliterix is licensed under [AGPL-3.0-or-later](LICENSE) because it is a derivative work of [Heretic](https://github.com/p-e-w/heretic). By submitting a contribution, you agree that your contribution will be released under the same license.

If you are contributing on behalf of an employer, please make sure you have the necessary permissions before opening a PR.

---

## Code of Conduct

Be respectful. Disagree with ideas, not with people. Assume good faith. We're all here because we find this stuff interesting — let's keep it that way.

Harassment, personal attacks, or discriminatory language will not be tolerated and will result in a ban from the project.

---

## Questions?

- Open a [GitHub Discussion](https://github.com/wuwangzhang1216/abliterix/discussions) for general questions
- Open a [GitHub Issue](https://github.com/wuwangzhang1216/abliterix/issues) for bugs or feature requests
- Ping [@wuwangzhang1216](https://github.com/wuwangzhang1216) on a relevant issue if you need a human

Thanks again for contributing. Every PR matters.
