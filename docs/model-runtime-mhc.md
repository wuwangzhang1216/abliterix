# Native mHC runtime profile

Use this CPU profile to run Abliterix's integration tests for the native DeepSeek-V4 and GLM-5.3-Flash classes. Start from the project's locked environment in a separate directory and install the six versions in [the profile](../ci/mhc-runtime.txt). Transformers 5.20.2 requires Hub 1.31 or later. The profile uses Hub 1.33.0, which also satisfies the project's `huggingface-hub~=1.6` requirement. [Transformers package requirements](https://pypi.org/pypi/transformers/5.20.2/json)

On Linux, run these commands from the repository root with Python 3.12 and uv installed.

```bash
export UV_PROJECT_ENVIRONMENT="$PWD/venv/mhc"
uv sync --locked --python 3.12 --group dev --no-extra vllm
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" \
  --exclude-newer 2026-10-20T00:00:00Z -r ci/mhc-runtime.txt -e .
uv pip check --python "$UV_PROJECT_ENVIRONMENT/bin/python"
"$UV_PROJECT_ENVIRONMENT/bin/abliterix" --help
HF_HUB_OFFLINE=1 "$UV_PROJECT_ENVIRONMENT/bin/python" -m pytest tests/test_mhc_integration.py -q -rs
```

The equivalent Windows PowerShell commands enable UTF-8 for the CLI banner.

```powershell
$env:UV_PROJECT_ENVIRONMENT = Join-Path $PWD 'venv/mhc'
$env:PYTHONUTF8 = '1'
uv sync --locked --python 3.12 --group dev --no-extra vllm
$mhcPython = Join-Path $env:UV_PROJECT_ENVIRONMENT 'Scripts/python.exe'
uv pip install --python $mhcPython --exclude-newer 2026-10-20T00:00:00Z -r ci/mhc-runtime.txt -e .
uv pip check --python $mhcPython
& (Join-Path $env:UV_PROJECT_ENVIRONMENT 'Scripts/abliterix.exe') --help
$env:HF_HUB_OFFLINE = '1'
& $mhcPython -X utf8 -m pytest tests/test_mhc_integration.py -q -rs
```

Use the environment's Python and `abliterix` executables directly after installation. A fresh `uv sync` or `uv run` targets the versions in the project's main lock again. The October 20 cutoff applies to the separate install step. The repository's default lock uses its June 19 cutoff.

The CI job checks dependency metadata, imports both native model classes, runs the full test suite and requires the CPU native integration cases to execute. CUDA-only cases skip on that job. These tests use local, tiny random models.

The CUDA cases can be run with `python -m pytest tests/test_mhc_integration.py -k cuda -q` using this profile and a PyTorch wheel appropriate for the installed GPU and driver. The local validation used PyTorch `2.11.0+cu130` on an RTX 4080 Laptop GPU. All eight combinations of architecture, FP32/BF16, and LoRA/direct editing passed. The [record](research/mhc-cuda-validation.json) includes readout and export errors. Eager CUDA execution is covered; full released checkpoints, quantization, sharding, and fused kernels still require separate acceptance. See the [compatibility scope](model-compatibility.md).
