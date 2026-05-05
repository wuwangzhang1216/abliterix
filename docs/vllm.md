[← back to README](../README.md)

# vLLM Runbook

This note records the production lessons from the Gemma 4 31B trial-40 run.
The default posture for large models should be **vLLM-first**. Do not go back
to HuggingFace generation/evaluation for trial loops unless there is a specific
backend blocker.

## Rule Of Thumb

Use vLLM for:

- baseline generation and refusal counting
- every optimization trial
- LLM-judge evaluation batches
- replaying candidate trials
- long prompt sets where HF pipeline parallelism would serialize work

Use HuggingFace only for:

- one-time hidden-state extraction when vLLM/speculators hidden states are not
  available for the model architecture
- final safetensors export after the winning trial is selected
- quick debugging of model structure or tokenizer issues

For Gemma 4 31B, vLLM hidden-state extraction is not yet reliable, so Phase 1
may still load HF once to compute steering vectors. That is acceptable. The
slow path we must avoid is HF `generate()` for the optimization/eval loop.

## Gemma 4 31B Known-Good Config

The winning run used `configs/gemma4_31b.toml` with these critical settings:

```toml
[model]
backend = "vllm"
tensor_parallel_size = 1
gpu_memory_utilization = 0.92
enable_expert_parallel = false
enforce_eager = true
max_model_len = 4096
disable_lora = true
use_in_place_editing = true

[inference]
max_batch_size = 8
max_gen_tokens = 150
min_gen_tokens = 100

[steering]
steering_mode = "direct"
weight_normalization = "pre"
disabled_components = ["mlp.down_proj"]
strength_range = [1.0, 6.0]

[detection]
llm_judge = true
llm_judge_model = "google/gemini-3-flash-preview"
llm_judge_batch_size = 10
llm_judge_concurrency = 35
```

Why these matter:

- `backend = "vllm"` keeps generation and scoring on the fast backend.
- `disable_lora = true` avoids the Gemma 4 vLLM LoRA path, which produced
  identical base/adapted outputs in testing.
- `use_in_place_editing = true` applies direct projection to the vLLM resident
  model instead of serializing adapters per trial.
- `disabled_components = ["mlp.down_proj"]` keeps the selected Gemma 4 31B run
  on attention projections only; this was more stable than editing MLP down
  projections.
- `min_gen_tokens = 100` is required for honest refusal detection because Gemma
  4 often delays refusals until after a long preamble.

## Launch Template

Use the training venv and local `.env`; do not paste tokens into the shell.

```bash
cd /workspace/abliterix
set -a && source .env && set +a

export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_CACHE=/workspace/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/hub
export PYTORCH_NO_CUDA_MEMORY_CACHING=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0

nohup /workspace/venvs/abliterix-vllm/bin/abliterix \
  --config configs/gemma4_31b.toml \
  --non-interactive \
  --overwrite-checkpoint \
  --optimization.checkpoint-dir=/workspace/checkpoints_gemma4_31b_vllm_inplace \
  > /workspace/logs/gemma4_31b_vllm_inplace.log 2>&1 &
echo $! > /workspace/logs/gemma4_31b_vllm_inplace.pid
```

Progress check:

```bash
pid=$(cat /workspace/logs/gemma4_31b_vllm_inplace.pid)
ps -p "$pid" -o pid,stat,etime,pcpu,pmem,rss,cmd
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu,power.draw \
  --format=csv,noheader,nounits
grep -aE "Running trial|LLM judge:|Refusals:|Estimated remaining|Optimization complete|Traceback|RuntimeError" \
  /workspace/logs/gemma4_31b_vllm_inplace.log | tail -120
```

## Results From The Reference Run

Hardware:

- RTX PRO 6000 Blackwell Server Edition, 96 GB VRAM
- vLLM backend, in-place direct attention editing
- Gemini LLM judge: `google/gemini-3-flash-preview`

Outcome:

- baseline: `99/100` refusals
- best trial: `trial 40`
- best score: `7/100` refusals
- top safe over-refusal probe: `0/15` refusals for trials 40, 46, and 53
- full 60-trial run time: about 1.5 hours after setup

The judge was not the bottleneck. Gemini judge batching with concurrency 35 was
fast enough; the bottleneck was model generation and initial model/vector setup.

## Pitfalls We Already Hit

### Do Not Use The Gemma 4 vLLM LoRA Path

For Gemma 4 31B, the vLLM LoRA adapter route was effectively inert in testing:
base and adapted outputs were identical and logprob differences were zero. Use
`disable_lora = true` plus `use_in_place_editing = true`.

### Do Not Enable Tower-Connector LoRA For Gemma 4 Text Runs

Trying `enable_tower_connector_lora=True` crashed under the text-only Gemma 4
setup with `ValueError: max() arg is an empty sequence` around
`limit_mm_per_prompt`. Keep the text-only vLLM path clean.

### KL Uses Continuation NLL In vLLM In-Place Mode

In-place editing can make vLLM sampler logprobs look exactly `0.0000` even when
refusal counts clearly change. The vLLM backend therefore scores the baseline
benign continuations with a fresh prefill pass and reports mean continuation NLL
drift as the KL-like damage metric for in-place runs. Treat refusal counts and
replay tests as primary, but this metric should now move when Gemma-style
in-place edits actually perturb benign behavior.

### Prefix Cache Must Not Hide Weight Edits

The vLLM backend disables prefix caching when in-place editors are active. If
that changes, every edit must reset the prefix cache before the next generation,
or stale KV/logprob data can make trials look unchanged.

### Blackwell PCIe Needs Conservative vLLM Settings

On RTX PRO 6000 Blackwell PCIe, keep:

- `enforce_eager = true` for reliability during in-place edits
- custom all-reduce disabled in the vLLM backend
- `VLLM_ENABLE_V1_MULTIPROCESSING=0`

These settings trade a little throughput for predictable worker startup and
weight-edit behavior.

## Export And Publish

After selecting a winning trial, export once with HF so the Hugging Face repo
contains normal safetensors:

```bash
/workspace/venvs/abliterix-vllm/bin/python scripts/export_model.py \
  --model google/gemma-4-31B-it \
  --checkpoint /workspace/checkpoints_gemma4_31b_vllm_inplace \
  --trial 40 \
  --config configs/gemma4_31b.toml \
  --save-local /workspace/export_gemma4_31b_trial40
```

Do not use the generic `upload_model.py` model card for Gemma 4 releases. Write
the README explicitly from the prior model card style and include:

- selected trial
- baseline refusal count
- eval refusal count
- generation length
- judge model
- safe over-refusal probe JSON
- continuation-NLL KL mode for vLLM in-place runs

## vLLM 0.18 — 0.20.x Integration Knobs

PRD #20 added a small set of `[model]` config fields for the modernised
vLLM backend. abliterix now refuses to start against `vllm < 0.18` (the
`VLLM_ALLOW_INSECURE_SERIALIZATION` flag MoE editing depends on landed in
that version) and warns on `>= 0.21` until smoke-tested.

### Auto-set environment variables

`VLLMGenerator.__init__` calls `vllm_compat.ensure_vllm_env()` to set the
small set of env vars vLLM 0.20.x needs, **without overriding any value
the user has already exported**:

| Env var | When set | Reason |
|---|---|---|
| `FLASHINFER_DISABLE_VERSION_CHECK=1` | Always | Skips a noisy assertion when transformers gets bumped underneath flashinfer |
| `VLLM_ALLOW_INSECURE_SERIALIZATION=1` | Only when `use_in_place_editing = true` | Required for `collective_rpc` to pickle Python callables sent to TP workers |

The deprecated `VLLM_FUSED_MOE_UNQUANTIZED_BACKEND=triton` is **no longer
set** — vLLM 0.20.x logs `Unknown vLLM environment variable detected` for
this name. Use the `moe_backend` config field instead (see below).

### Attention backend (MLA-aware)

```toml
[model]
attention_backend = "FLASH_ATTN_MLA"  # or omit for auto-detect
```

When `attention_backend = None` (default), abliterix sniffs the model's
HF architecture and picks:

- `FLASH_ATTN_MLA` for MLA models (DeepSeek-V2/V3, MiniMax-M2.x).
- `TRITON_ATTN` for sink-attention models (gpt-oss).
- `None` (let vLLM choose) for everything else.

The previous hardcoded `TRITON_ATTN` is gone — it crashed engine init on
every MLA model. Override only when you have a specific reason.

### MoE backend (default `triton`)

```toml
[model]
moe_backend = "triton"  # default; skips FlashInfer cutlass JIT cold start
```

vLLM 0.20.x's default `moe_backend = "auto"` selects FlashInfer CUTLASS
on sm_90 and JIT-compiles a kernel per expert-group count (M128_group1..N).
On a 64-expert model that is **30+ minutes** of pod time on first run.
abliterix defaults to `"triton"` to skip this entirely.

Override if you want the FlashInfer perf and have already paid the JIT:

```toml
moe_backend = "flashinfer_cutlass"
```

Other options: `auto`, `deep_gemm`, `cutlass`, `flashinfer_trtllm`,
`marlin`, `aiter`.

### Compilation mode (PRD #20 partial — eager wired, MoE-eager-rest deferred)

```toml
[model]
vllm_compile_mode = "eager"  # default; equivalent to enforce_eager=true
```

Three abliterix-level intents map onto vLLM's `compilation_config` dict:

- `"eager"` — every CUDA graph off; all forward hooks fire. Safest for MoE
  router suppression and expert editing. Current default.
- `"moe_eager_rest_compile"` — non-MoE layers get CUDA graph capture;
  MoE layers stay eager so hooks survive. Needs GPU smoke before promotion;
  currently falls back to `"eager"` with a warning.
- `"full_compile"` — full vLLM compile + CUDA graph capture everywhere.
  No MoE editing supported (PyTorch issue #117758 silently drops
  post-compile hooks). Dense models only.

### Routed-expert metadata (issue #22, PR #24)

```toml
[model]
vllm_return_routed_experts = true  # default
```

When on (default), abliterix's MoE safety-expert profiler reads per-token
routing IDs directly from `RequestOutput.outputs[0].routed_experts`
(numpy ndarray of shape `(prompt_tokens, n_layers, top_k)`). This
replaces the previous `collective_rpc` + forward-hook probe pipeline
(~150 LoC of worker plumbing, deleted in PR #24).

GPU-verified parity on DeepSeek-V2-Lite-Chat (issue #22 Phase B):
top-1 expert match 22/26 layers (84.6%), top-3 set match 23/26 (88.5%).
Divergent layers all share top-1; differences appear only in near-tie
positions of the top-3 set, consistent with the new path reading
post-tie-break selections vs the old hook reading raw router logits.

Memory cost: `tokens × n_layers × top_k × 4` bytes per request. For a
60-layer top-6 MoE generating 100 tokens that is ~140 KB per request —
trivial against the model footprint, but exposed as a config knob so
users with throughput-critical sweeps can disable it. Set to `false` to
fall back to the legacy hook-based probe (kept around but not exercised
in CI; will be deleted on the next major if no users depend on it).

This removed one of the two reasons abliterix needed
`VLLM_ALLOW_INSECURE_SERIALIZATION=1` — the other reason
(`VLLMMoEEditor.apply()` per-trial suppression) still requires it, so
the env var auto-set logic in `vllm_compat.ensure_vllm_env(needs_collective_rpc=True)`
is unchanged.

### LoRA pool / target modules

```toml
[model]
vllm_max_loras = 1            # adapter slots in CPU; raise to pool trial adapters
vllm_max_lora_rank = 16       # default 16 (vLLM's own default in 0.20.x)
lora_target_modules = ["o_proj", "qkv_proj"]  # restrict LoRA to non-MoE
```

`vllm_max_lora_rank` defaults to vLLM's own default (16). The prior
hardcoded floor of 8 is gone — set this to the actual rank you plan to
use for the cleanest KL signal (no zero-padded dimensions polluting the
direction).

`lora_target_modules` maps to vLLM's `--lora-target-modules` (PR #34984,
v0.19.0+). Mainly a perf knob; experimentally also a possible workaround
for the LoRA + Expert Parallel worker assertion crash (verify on your
hardware before relying on it — see PRD #20 Out of Scope).

### Custom-all-reduce auto-detect

```toml
[model]
disable_custom_all_reduce = true  # or false; or omit for auto-detect
```

When unset, abliterix looks at `torch.cuda.get_device_capability(0)` and
disables custom all-reduce **only on Blackwell PCIe (sm_120)** where the
path is known to deadlock without NVLink (RTX PRO 6000). H100 / B100 /
B200 SXM keep the custom path on for the documented perf win.

### Multimodal limit pass-through

```toml
[model]
limit_mm_per_prompt = { image = 0, video = 0, audio = 0 }  # default
```

Defaults preserve the historical "drop vision/audio towers" behaviour so
the Punica LoRA wrapper accepts hybrid VLM/MoE architectures without
crashing on `visual.*` modules. Set explicitly when you want vision /
audio active for a multimodal abliteration recipe.

### Hidden-state extraction escape hatch

`AX_DISABLE_VLLM_HS=1` in the environment forces abliterix's Phase 1
hidden-state extraction to use HuggingFace even when the backend is
`vllm`. Useful when speculators-based vLLM hidden states are unreliable
for a model architecture (e.g. Gemma 4 31B).
