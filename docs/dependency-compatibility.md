# Dependency compatibility

The committed lock file exercises the default runtime. Compatibility jobs install
specific newer releases into separate environments and check the installed
versions, so widening a dependency range cannot pass solely on the old lock.

## Hugging Face Hub 2

`ci/hub2-runtime.txt` pins a tested combination of Hub 2.4, Transformers 5.20.2,
Datasets 5.1, and their native-model dependencies. Install it on top of the locked
development environment, using the same fixed cutoff as CI:

```bash
uv pip install --python .venv/bin/python --exclude-newer 2026-11-03T23:59:59Z -r ci/hub2-runtime.txt -e .
uv pip check --python .venv/bin/python
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest tests/ -q
```

Use the environment's Python directly after installing the profile; `uv run`
would otherwise restore the default lock. The full suite includes real Hub HTTP
client tests for immutable model/dataset revisions, LFS weight hashes, and error
propagation, with an offline mock transport. It also exercises local dataset
loading and native tiny-model load/edit/export tests.

[Hub 2 migrated to httpx2](https://github.com/huggingface/huggingface_hub/releases/tag/v2.0.0).
[Datasets 5.1 added Hub 2 compatibility](https://github.com/huggingface/datasets/releases/tag/5.1.0).
The locked Transformers 5.7 and Datasets 4.4 releases require Hub 1.x; Gradio 6.19
also requires Hub below 2. The UI job therefore continues to validate Hub 1.x.
The wider project range allows each combination's dependency constraints to
select a compatible version; it does not force Hub 2 into the UI environment.

## Plotly 7

The UI matrix tests Gradio 5.50/6.19 against both Plotly 6.7 and 7.1. Each job
asserts the installed versions and passes dashboard figures through Gradio's
actual JSON serialization, including empty charts, dominated points, tied
objectives, and duplicate optimal trials. This also guards the Pareto selection
rule: equal KL with more refusals is dominated, while identical objective pairs
remain selectable.

[Plotly 7's removed APIs](https://github.com/plotly/plotly.py/releases/tag/v7.0.0)
include legacy plotting and export interfaces; the dashboard uses `graph_objects`
scatter traces. These tests cover its serialized chart data and Gradio callbacks;
they do not assert pixel-identical browser rendering across Plotly majors.
