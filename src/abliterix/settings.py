# Abliterix — a derivative work of Heretic (https://github.com/p-e-w/heretic)
# Original work Copyright (C) 2025  Philipp Emanuel Weidmann (p-e-w)
# Modified work Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import sys
from typing import Any, Dict, Literal

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import (
    BaseSettings,
    CliSettingsSource,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
    TomlConfigSettingsSource,
)

# vLLM 0.20.x's MoEBackend literal — mirrored here so abliterix can reject
# typos at config-load time without importing vLLM. Update alongside vLLM
# upgrades. Canonical source: vllm/config/kernel.py:MoEBackend.
MoEBackend = Literal[
    "auto",
    "triton",
    "deep_gemm",
    "deep_gemm_mega_moe",
    "cutlass",
    "flashinfer_trtllm",
    "flashinfer_cutlass",
    "flashinfer_cutedsl",
    "marlin",
    "aiter",
    "emulation",
]

# CompileMode is owned by vllm_compilation_config; settings re-uses the
# same Literal so a typo (e.g. "eagar") is caught at config-load instead
# of inside vllm_compilation_config.build().
from .core.vllm_compilation_config import CompileMode  # noqa: E402

from .types import (  # noqa: E402
    DecayKernel,
    DirectTransform,
    PromptSource,
    QuantMode,
    SteeringMode,
    VectorMethod,
    WeightNorm,
)


# ---------------------------------------------------------------------------
# Sub-configuration models
# ---------------------------------------------------------------------------


class ModelConfig(BaseModel):
    """Parameters governing model loading, dtype selection, and device placement."""

    model_id: str = Field(description="Hugging Face model identifier or local path.")

    evaluate_model_id: str | None = Field(
        default=None,
        description=(
            "When set, the system evaluates this model against the primary model "
            "rather than running the optimisation loop."
        ),
    )

    dtype_fallback_order: list[str] = Field(
        default=[
            "auto",
            "float16",
            "bfloat16",
            "float32",
        ],
        description=(
            "Ordered list of dtypes to attempt during model loading.  "
            "If the first dtype causes an error the next one is tried."
        ),
    )

    quant_method: QuantMode = Field(
        default=QuantMode.NONE,
        description="Weight quantisation strategy applied at load time.",
    )

    device_map: str | Dict[str, int | str] = Field(
        default="auto",
        description="Accelerate device-map specification.",
    )

    max_memory: Dict[str, str] | None = Field(
        default=None,
        description='Per-device memory budget, e.g. {"0": "20GB", "cpu": "64GB"}.',
    )

    use_torch_compile: bool = Field(
        default=False,
        description="Apply torch.compile() to the loaded model for faster inference.",
    )

    trust_remote_code: bool | None = Field(
        default=None,
        description="Whether to trust remote code shipped with the model.",
    )

    attn_implementation: str | None = Field(
        default=None,
        description=(
            "Attention implementation to use (e.g. 'flash_attention_2', 'sdpa', 'eager'). "
            "When set, passed directly to from_pretrained()."
        ),
    )

    experts_implementation: str | None = Field(
        default=None,
        description=(
            "MoE experts kernel: 'eager', 'grouped_mm', 'batched_mm', 'deepgemm'.  "
            "transformers 5.x defaults to 'grouped_mm' which calls torch._grouped_mm — "
            "that op is hard-pinned to compute capability sm_90 (H100) in torch 2.8 "
            "and raises on Blackwell (sm_100/sm_120 — B200, RTX Pro 6000) and on "
            "Ampere/Ada (A100, A6000, RTX 4090). On those cards, set this to 'eager' "
            "or 'batched_mm'. None (default) lets transformers pick."
        ),
    )

    custom_encoder_module: str | None = Field(
        default=None,
        description=(
            "Filesystem path to a Python module that exports an "
            "``encode_messages(messages, **kw) -> str`` function used in lieu "
            "of ``tokenizer.apply_chat_template``. Required for models that "
            "ship a custom encoding script instead of a Jinja chat_template "
            "(e.g. DeepSeek-V4's ``encoding_dsv4.py``). When set, abliterix "
            "monkey-patches the loaded tokenizer so the rest of the pipeline "
            "stays unchanged. None = use the tokenizer's bundled chat_template."
        ),
    )

    custom_encoder_kwargs: Dict[str, Any] | None = Field(
        default=None,
        description=(
            "Keyword arguments forwarded to ``encode_messages`` from "
            "``custom_encoder_module`` (e.g. ``{thinking_mode = 'non-thinking'}`` "
            "for DeepSeek-V4). None = no extra kwargs."
        ),
    )

    skip_fp8_dequant: bool | None = Field(
        default=None,
        description=(
            "Skip the FP8→bf16 dequantisation workaround.  "
            "None (default) = auto-detect: skip dequant on H100+ with transformers >= 5.2.  "
            "True = always skip (native FP8 GEMM).  "
            "False = always dequant to bf16 (safe fallback)."
        ),
    )

    fp8_weight_block_size: list[int] | None = Field(
        default=None,
        description=(
            "Block size for FP8 fine-grained quantization, e.g. [128, 128].  "
            "Required for some MoE models (Qwen3.5 MoE) to fix weight_scale_inv "
            "shape mismatches with device_map='auto'.  "
            "None = auto-detect from model config."
        ),
    )

    fp8_handling: str = Field(
        default="auto",
        description=(
            "How to handle native-FP8 model weights at load time.\n"
            "  'auto'              — decide from steering_mode: materialise BF16 "
            "for direct/EGA, forward-dequant for LoRA\n"
            "  'materialize'       — replace every FP8 weight with a BF16 "
            "Parameter (2x VRAM; required for direct-mode weight editing; "
            "unfuses transformers FP8Experts back to per-expert modules)\n"
            "  'forward_dequant'   — monkey-patch FP8 Linear.forward for "
            "on-the-fly bf16 dequant (1x VRAM; LoRA-mode only; fails on "
            "fused MoE FP8 containers)\n"
            "  'offline'           — assume the model has been pre-dequanted "
            "to disk via abliterix.core.fp8_utils.dequant_model_to_disk; skip "
            "all FP8 paths\n"
            "See abliterix.core.fp8_utils for the underlying kernels."
        ),
    )

    backend: str = Field(
        default="hf",
        description=(
            "Inference backend: 'hf' for HuggingFace Transformers (pipeline parallelism), "
            "'vllm' for vLLM (tensor parallelism), "
            "'sglang' for SGLang (RadixAttention + tensor parallelism).  "
            "SGLang is ~29%% faster than vLLM on prefix-heavy workloads.  "
            "Both vLLM and SGLang provide dramatically higher throughput on multi-GPU "
            "setups by parallelising computation across GPUs."
        ),
    )

    tensor_parallel_size: int | None = Field(
        default=None,
        description=(
            "Number of GPUs for vLLM tensor parallelism.  None = auto-detect all "
            "available GPUs.  Ignored when backend='hf'."
        ),
    )

    gpu_memory_utilization: float = Field(
        default=0.92,
        description=(
            "Fraction of GPU memory vLLM may use (0.0-1.0).  Ignored when backend='hf'."
        ),
    )

    enable_expert_parallel: bool = Field(
        default=True,
        description=(
            "Enable expert parallelism (EP) for MoE models in vLLM.  "
            "EP distributes experts across GPUs rather than replicating them.  "
            "Best for models with >3% expert activation density (DeepSeek, Qwen MoE)."
        ),
    )

    enable_chunked_prefill: bool = Field(
        default=True,
        description=(
            "Enable chunked prefill to overlap prefill and decode phases.  "
            "For SGLang: controls chunked_prefill_size (8192 when True).  "
            "For vLLM V1 (>= 0.8): always on, this setting is ignored."
        ),
    )

    kv_cache_dtype: str | None = Field(
        default=None,
        description=(
            "KV cache data type for vLLM.  "
            "None = auto (fp8_e4m3 for FP8 models on H100+, otherwise default).  "
            "'fp8_e4m3' halves KV cache memory with negligible quality loss.  "
            "'auto' uses the model's native dtype."
        ),
    )

    enforce_eager: bool = Field(
        default=False,
        description=(
            "Force eager mode in vLLM (disable CUDA graphs).  "
            "Safer for debugging but slower.  Default False enables CUDA graphs "
            "for ~10-20%% higher throughput."
        ),
    )

    disable_lora: bool = Field(
        default=False,
        description=(
            "Force vLLM to load without LoRA support (enable_lora=False).  "
            "Required for MXFP4 models on older drivers: vLLM's Marlin-FP4 LoRA "
            "repack kernel ships CUDA 12.9+ PTX that fails on driver < 575 with "
            "cudaErrorUnsupportedPtxVersion.  When set, the optimizer still runs "
            "MoE router suppression on mlp.router.weight (the primary MoE steering "
            "mechanism) but skips attention LoRA adapters — acceptable loss for "
            "gpt-oss-style models where q/k/v LoRA is already disabled and only "
            "o_proj would have been steered."
        ),
    )

    use_in_place_editing: bool = Field(
        default=False,
        description=(
            "Skip the LoRA-adapter path and edit vLLM weights in-place via "
            "``collective_rpc`` instead.  Requires an unquantized BF16 MoE "
            "checkpoint (MXFP4 / FP8 repack is NOT supported — see "
            "``Mxfp4MoEMethod.process_weights_after_loading``) and "
            "``enforce_eager = true``.  Advantages over LoRA adapter path:\n"
            "  * Edits attention + ALL experts + router every trial; LoRA "
            "    adapter covers attention only.\n"
            "  * No adapter serialisation overhead (~200 MB / trial saved).\n"
            "  * 3x GPU util on TP vs HF pipeline-parallel.\n"
            "Backend selection is now config-driven via ``moe_backend``; "
            "abliterix sets ``VLLM_ALLOW_INSECURE_SERIALIZATION=1`` "
            "automatically when this flag is on so ``collective_rpc`` can "
            "pickle the Python callables sent to TP workers."
        ),
    )

    max_model_len: int | None = Field(
        default=None,
        description=(
            "Maximum sequence length (prompt + generation) for vLLM/SGLang.  "
            "None = use model's default (often 128K-200K).  Setting this lower "
            "(e.g. 4096) dramatically reduces KV cache reservation per sequence, "
            "enabling much larger batch sizes for short-prompt workloads like "
            "abliteration.  Strongly recommended for MoE models."
        ),
    )

    max_num_seqs: int | None = Field(
        default=None,
        description=(
            "Maximum concurrent sequences in vLLM's continuous batching.  "
            "None = vLLM auto.  Set higher (e.g. 256-512) for throughput on "
            "4x H100 with short prompts; the actual batch size is gated by "
            "max_model_len and available KV cache."
        ),
    )

    hf_overrides: Dict[str, Any] | None = Field(
        default=None,
        description=(
            "Model config overrides passed to vLLM/SGLang via hf_overrides.  "
            "Used to patch model config values at load time, e.g. "
            "{num_nextn_predict_layers = 1} to downgrade MTP-3 to MTP-1."
        ),
    )

    # ------------------------------------------------------------------
    # vLLM 0.18-0.20.x integration knobs (added by PRD #20)
    # ------------------------------------------------------------------

    attention_backend: str | None = Field(
        default=None,
        description=(
            "vLLM attention backend name passed as "
            "``attention_config={'backend': ...}`` in the LLM() kwargs.  "
            "None (default) lets abliterix auto-detect: MLA models "
            "(DeepSeek-V2/V3, MiniMax-M2.x) get ``FLASH_ATTN_MLA``, "
            "sink-attention models (gpt-oss) get ``TRITON_ATTN``, and the "
            "remainder fall through to vLLM's own default.  Set explicitly "
            "to override (e.g. ``'FLASHMLA'``, ``'TRITON_MLA'``, "
            "``'FLASH_ATTN'``, ``'FLASHINFER'``)."
        ),
    )

    moe_backend: MoEBackend = Field(
        default="triton",
        description=(
            "vLLM MoE compute backend (``KernelConfig.moe_backend``).  "
            "Default ``'triton'`` skips FlashInfer's per-expert-group "
            "cutlass JIT compile that costs ~30 minutes on first sm_90 "
            "cold start.  Set to ``'flashinfer_cutlass'`` if you want the "
            "perf and have already paid the JIT.  Other options: "
            "``'auto'``, ``'deep_gemm'``, ``'cutlass'``, ``'flashinfer_trtllm'``, "
            "``'marlin'``, ``'aiter'``.  Replaces the now-deprecated "
            "``VLLM_FUSED_MOE_UNQUANTIZED_BACKEND`` env var (gone in 0.20.x)."
        ),
    )

    disable_custom_all_reduce: bool | None = Field(
        default=None,
        description=(
            "Pass-through for vLLM's ``disable_custom_all_reduce``.  "
            "None (default) auto-detects: ``True`` on Blackwell PCIe "
            "(sm_120) where the custom all-reduce path deadlocks during "
            "worker init without NVLink, ``False`` everywhere else "
            "(NVLink Hopper / SXM Blackwell keep the perf win).  Set "
            "explicitly to override the auto-detection."
        ),
    )

    limit_mm_per_prompt: Dict[str, int] | None = Field(
        default=None,
        description=(
            "Pass-through for vLLM's ``limit_mm_per_prompt``.  None "
            "(default) drops vision/audio towers via "
            "``{'image': 0, 'video': 0, 'audio': 0}`` so the Punica LoRA "
            "wrapper accepts hybrid VLM/MoE architectures (Qwen3.5-MoE-VL, "
            "Llama-4, Step3, Mistral-3) without crashing on ``visual.*`` "
            "modules.  Set explicitly when you want vision/audio active."
        ),
    )

    vllm_max_loras: int = Field(
        default=1,
        description=(
            "vLLM ``max_loras`` — number of LoRA adapter slots held in CPU "
            "for hot-swap.  Default 1 keeps the historical single-adapter "
            "behaviour.  Raising this (e.g. 8) lets the optimizer pool "
            "multiple trial adapters and skip per-trial /dev/shm "
            "write+reload, which dominates wall time on long sweeps."
        ),
    )

    vllm_max_lora_rank: int | None = Field(
        default=None,
        description=(
            "vLLM ``max_lora_rank``.  None (default) lets vLLM use its own "
            "default (16 in 0.20.x).  Set to the smallest value that fits "
            "every adapter you plan to load — abliterix used to force-pad "
            "to 8, but vLLM accepts arbitrary ranks now."
        ),
    )

    lora_target_modules: list[str] | None = Field(
        default=None,
        description=(
            "vLLM ``--lora-target-modules`` (PR #34984, v0.19.0+).  When "
            "set, restricts LoRA wrapping to module suffixes in this list "
            "(e.g. ``['o_proj', 'qkv_proj']``).  Primarily a perf knob; "
            "experimentally also a possible workaround for the LoRA + "
            "Expert Parallel worker assertion crash by keeping LoRA off "
            "MoE modules.  None = vLLM default (wrap all supported "
            "modules)."
        ),
    )

    vllm_return_routed_experts: bool = Field(
        default=True,
        description=(
            "Pass-through for vLLM's ``enable_return_routed_experts`` (vLLM "
            "0.20.x+). When True (default), abliterix's MoE safety-expert "
            "profiler reads per-token routing IDs directly from "
            "``RequestOutput.outputs[0].routed_experts`` instead of "
            "installing forward hooks via ``collective_rpc``. This removes "
            "the entire probe rpc surface and ~150 LoC of worker plumbing "
            "(see issue #22 / PR #24). Memory cost is "
            "``tokens * layers * top_k * 4`` bytes per request — "
            "~140 KB for a 100-token MoE-60-top6 generation. Set False to "
            "fall back to the legacy collective_rpc + hook path (kept for "
            "vLLM <0.20 compatibility; not exercised in CI)."
        ),
    )

    vllm_compile_mode: CompileMode = Field(
        default="eager",
        description=(
            "abliterix-side selector for vLLM's ``compilation_config``.\n"
            "  'eager'                 — equivalent to ``enforce_eager=True``; "
            "all CUDA graphs off (current behaviour, safest for MoE editor "
            "forward hooks).\n"
            "  'moe_eager_rest_compile' — REJECTED until GPU smoke lands "
            "(PRD #20 Out of Scope). Use 'eager' for now; this mode raises "
            "at config load to avoid silent fallback noise on every engine "
            "init.\n"
            "  'full_compile'          — full vLLM compile + CUDA graphs "
            "everywhere (no MoE editing supported; dense models only)."
        ),
    )

    @model_validator(mode="after")
    def _validate_vllm_combos(self) -> "ModelConfig":
        """Reject vLLM config combinations that would either silently
        no-op or contradict each other."""
        # Item 4 from PR review: until the post-load attach lands, fail
        # loudly on the unimplemented compile mode rather than warn-and-
        # fallback every engine init (which spams logs across sweeps).
        if self.vllm_compile_mode == "moe_eager_rest_compile":
            raise ValueError(
                "vllm_compile_mode='moe_eager_rest_compile' is not yet "
                "implemented — the post-load layer-index discovery is "
                "deferred per PRD #20 Out of Scope. Use 'eager' until the "
                "GPU smoke for static_all_moe_layers ships."
            )
        # Item 8: lora_target_modules without enable_lora is silently dropped
        # by the if-not-disabled guard in vllm_backend. Reject explicitly so
        # users see the misconfiguration at config load.
        if self.disable_lora and self.lora_target_modules:
            raise ValueError(
                "lora_target_modules is set but disable_lora=True — the "
                "target list would be silently dropped because LoRA is off. "
                "Either unset lora_target_modules or set disable_lora=false."
            )
        return self


class InferenceConfig(BaseModel):
    """Settings that control generation and batch sizing."""

    batch_size: int = Field(
        default=0,
        description="Sequences processed in parallel (0 = auto-tune).",
    )

    max_batch_size: int = Field(
        default=128,
        description="Upper bound explored during automatic batch-size tuning.",
    )

    max_gen_tokens: int = Field(
        default=100,
        description="Token budget for each generated response.",
    )

    min_gen_tokens: int | None = Field(
        default=None,
        description=(
            "Optional minimum number of generated tokens for evaluation runs. "
            "Set this below or equal to max_gen_tokens when delayed refusals, "
            "early stop-token spam, or truncated benign answers need to be "
            "visible to the refusal judge. None preserves model-default stopping."
        ),
    )

    offload_outputs_to_cpu: bool = Field(
        default=True,
        description=(
            "Move per-batch analysis tensors (residuals, log-probabilities) to "
            "host RAM as soon as each batch is computed, instead of accumulating "
            "the full (n_prompts × layers × hidden) stack in VRAM before the "
            "concatenation.  Lowers peak VRAM during steering-vector extraction "
            "and KL baseline capture.  Trade-off: because the direction-extraction "
            "math derives its device from the residual tensor, the one-time "
            "analysis-phase linear algebra then runs on the CPU.  For the default "
            "``vector_method = 'mean'`` this is negligible (just a mean over a few "
            "hundred vectors per layer), but for the heavy methods (sra / pca / "
            "som / ot / cosmic / sae, which do per-layer SVD / eigh / QR) it moves "
            "that compute off the GPU; set this to ``false`` on a VRAM-rich host "
            "running those methods to keep extraction on the GPU.  Only affects "
            "the HuggingFace backend; the vLLM/SGLang paths manage their own "
            "memory.  Mirrors Heretic's ``offload_outputs_to_cpu``."
        ),
    )


class SteeringConfig(BaseModel):
    """Hyper-parameters for the steering (abliteration) algorithm."""

    vector_method: VectorMethod = Field(
        default=VectorMethod.MEAN,
        description=(
            "How per-layer steering vectors are derived from residual streams.  "
            '"mean" uses the arithmetic-mean difference, '
            '"median_of_means" splits into groups and takes the median, '
            '"pca" selects the principal component of maximum variance, '
            '"optimal_transport" uses PCA-Gaussian OT to match distributions, '
            '"cosmic" uses cosine-similarity-based direction selection, '
            '"sra" uses Surgical Refusal Ablation with concept-guided spectral cleaning.'
        ),
    )

    orthogonal_projection: bool = Field(
        default=False,
        description=(
            "Remove the benign-direction component from steering vectors so that "
            "only the genuinely safety-specific signal is subtracted."
        ),
    )

    projected_abliteration: bool = Field(
        default=False,
        description=(
            "Use the improved projected-abliteration technique (grimjim 2025) that "
            "only removes the orthogonal component of the refusal direction relative "
            "to the harmless mean, preserving helpfulness-aligned signals.  "
            "Overrides orthogonal_projection when enabled."
        ),
    )

    winsorize_vectors: bool = Field(
        default=False,
        description=(
            "Apply symmetric magnitude winsorization to steering vectors before "
            "projection, reducing the influence of outlier activations."
        ),
    )

    winsorize_quantile: float = Field(
        default=0.995,
        description="Quantile for vector winsorization (default 0.995 per grimjim's method).",
    )

    ot_components: int = Field(
        default=2,
        description="Number of PCA components for the optimal-transport vector method.",
    )

    n_directions: int = Field(
        default=1,
        description=(
            "Number of independent refusal directions to extract.  "
            "Values >1 enable multi-direction mode where top-k SVD components "
            "are each converted to rank-1 LoRA adapters and stacked."
        ),
    )

    ablate_harmfulness_direction: bool = Field(
        default=False,
        description=(
            "Extract and ablate the harmfulness direction in addition to the "
            "refusal direction (Zhao et al. 2025, arXiv:2507.11878).  "
            "The standard mean-diff vector conflates 'do I refuse' with "
            "'is this harmful'; this flag extracts the second signal "
            "separately (PCA-1 of centred target states, dominant in mid "
            "layers) and orthogonalises it against the refusal direction so "
            "both are ablated jointly.  Reduces hedging behaviour on "
            "abliterated models that comply but still flag the request "
            "as harmful.  Implemented via the existing multi-direction "
            "infrastructure — incompatible with ``n_directions > 1`` and "
            "with ``vector_method`` set to ``sra``, ``cosmic``, or "
            "``optimal_transport`` (those paths build their own bases)."
        ),
    )

    harmfulness_layer_band: list[float] = Field(
        default=[0.3, 0.7],
        description=(
            "Fractional layer range ``[lo, hi]`` where the harmfulness "
            "direction is strongest, used when "
            "``ablate_harmfulness_direction = true``.  Defaults to the "
            "mid-layer band ``[0.3, 0.7]`` identified by Zhao et al. for "
            "Llama-3 / Qwen-2 class models.  Layers outside this band still "
            "get a harmfulness vector but at 0.5x strength so the optimiser "
            "concentrates its budget on the discriminative band."
        ),
    )

    steering_mode: SteeringMode = Field(
        default=SteeringMode.LORA,
        description=(
            "Steering application strategy.  "
            '"lora" modifies model weights via LoRA adapters, '
            '"angular" rotates activations at inference time via hooks, '
            '"adaptive_angular" rotates only aligned activations (reduces interference), '
            '"spherical" rotates along geodesics on the activation hypersphere, '
            '"vector_field" uses learned context-dependent steering directions, '
            '"direct" modifies base weights in-place via orthogonal projection '
            "(required for models with double-norm like Gemma 4 where LoRA is ineffective)."
        ),
    )

    discriminative_layer_selection: bool = Field(
        default=False,
        description=(
            "Only apply steering to layers where harmful and harmless activations "
            "project in opposite directions along the steering vector.  "
            "Non-discriminative layers are skipped entirely."
        ),
    )

    decay_kernel: DecayKernel = Field(
        default=DecayKernel.LINEAR,
        description="Interpolation kernel used to taper steering strength across layers.",
    )

    weight_normalization: WeightNorm = Field(
        default=WeightNorm.NONE,
        description=(
            "Row-norm handling for weight matrices.  "
            '"none" applies steering directly, '
            '"pre" normalises before computing the adapter, '
            '"full" additionally re-scales rows to preserve their original magnitudes.'
        ),
    )

    full_norm_lora_rank: int = Field(
        default=3,
        description='LoRA rank used for the low-rank SVD approximation when weight_normalization="full".',
    )

    strength_range: list[float] = Field(
        default=[0.8, 1.5],
        description="Optuna search interval [lo, hi] for peak steering weight.",
    )

    disabled_components: list[str] = Field(
        default_factory=list,
        description=(
            "Components to exclude from the search entirely. Names match the "
            "keys returned by ``engine.list_steerable_components()`` (e.g. "
            '``"attn.q_proj"``). Useful for high-dimensional MoE models where '
            "attention-side steering wastes trial budget that should go to "
            "expert-path components."
        ),
    )

    fixed_vector_scope: str | None = Field(
        default=None,
        description=(
            'Pin the vector scope to one of ``"global"`` or ``"per layer"`` '
            "instead of letting TPE sample between them. When set, the "
            "categorical suggestion is replaced with a single-choice categorical "
            "so TPE's parameter space stays valid but can only pick this scope. "
            "Useful when domain knowledge says one scope dominates (e.g. deep "
            "MoE models benefit from ``per layer`` because refusal circuits "
            "differ across layers, and a single global direction averages them "
            "into a less-aligned vector)."
        ),
    )

    component_strength_ranges: dict[str, list[float]] = Field(
        default_factory=dict,
        description=(
            "Per-component override for ``strength_range``. Mapping of "
            'component name (e.g. ``"mlp.down_proj"``) to ``[lo, hi]``. '
            "When a component appears here, the optimizer uses the per-component "
            "interval instead of the global ``strength_range`` for that "
            "component's ``max_weight`` parameter. Useful for MoE models where "
            "different components want very different strength regimes — e.g. "
            "gpt-oss benefits from weak attention steering + strong EGA on "
            "fused expert ``mlp.down_proj``."
        ),
    )

    min_weight_frac_max: float = Field(
        default=1.0,
        description=(
            "Upper bound for the random sampling of ``component.min_weight`` "
            "(expressed as a fraction of ``max_weight``). Default 1.0 keeps "
            "the historical behaviour where the optimizer may sample any "
            "min_frac in [0, 1], which can produce nearly-flat strength "
            "profiles (min ≈ max → every layer at peak strength). Set this "
            "below 1.0 to bias the search toward 'sharp peak' profiles where "
            "the steering is concentrated near ``max_weight_position``. "
            "Empirically (gpt-oss-20b v1), all winning trials had min_frac < "
            "0.34 — setting this to ~0.4 raises the warmup hit rate "
            "dramatically without removing any known sweet spot."
        ),
    )

    component_min_frac_max: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-component override for ``min_weight_frac_max``. Useful when "
            "one component (e.g. EGA on fused MoE experts) has an even "
            "tighter sweet spot than the others. For gpt-oss-20b's "
            "``mlp.down_proj``, the v1 winner had min_frac = 0.02; setting "
            "this component's cap to ~0.10 makes random search ~10x more "
            "likely to land in the productive region."
        ),
    )

    auto_disable_components: list[str] = Field(
        default_factory=lambda: ["mlp.down_proj"],
        description=(
            "Components for which the optimiser is allowed to fully disable "
            "ablation (max_weight = 0) within a single trial.  For each listed "
            "component the lower bound of the ``max_weight`` search is dropped to "
            "``auto_disable_floor`` and the sampled value is clamped to "
            "``max(0.0, …)``, which places a finite probability mass on exactly "
            "0 (a continuous sampler reaches 0 with probability zero otherwise).  "
            "This lets TPE *discover* per-model that a component is best left "
            "untouched — ablating ``mlp.down_proj`` in particular tends to damage "
            "intelligence more than it suppresses refusals.  Set to ``[]`` to "
            "restore the previous behaviour where every searched component uses a "
            "strictly-positive range.  Mirrors Heretic's clamped negative lower "
            "bound for the MLP down-projection."
        ),
    )

    auto_disable_floor: float = Field(
        default=-0.25,
        description=(
            "Negative lower bound used for the ``max_weight`` search of every "
            "component listed in ``auto_disable_components``.  The fraction of the "
            "search interval below zero (``|floor| / (hi + |floor|)``) becomes the "
            "probability mass assigned to a fully-disabled component.  Must be "
            "≤ 0; a value of 0 disables the auto-disable behaviour."
        ),
    )

    outlier_quantile: float = Field(
        default=1.0,
        description=(
            "Symmetric winsorisation quantile applied to per-prompt residual vectors.  "
            "Values below 1.0 clamp extreme activations.  (Equivalent to Heretic's "
            "``winsorization_quantile``; see also ``winsorize_vectors`` which "
            "winsorises the final direction vector.)"
        ),
    )

    # --- SRA (Surgical Refusal Ablation) settings ---

    sra_base_method: VectorMethod = Field(
        default=VectorMethod.MEAN,
        description=(
            "Base vector method used to compute the initial refusal direction "
            "before SRA spectral cleaning.  Only used when vector_method='sra'."
        ),
    )

    sra_n_atoms: int = Field(
        default=8,
        description=(
            "Number of concept atoms (protected capability clusters) for SRA.  "
            "Higher values capture more independent capability directions."
        ),
    )

    sra_ridge_alpha: float = Field(
        default=0.01,
        description=(
            "Ridge regularisation coefficient for SRA spectral residualisation.  "
            "Larger values preserve more of the original refusal vector."
        ),
    )

    # --- SOM (Self-Organising Map directions) settings ---

    som_grid_h: int = Field(
        default=3,
        description=(
            "SOM grid height when vector_method = 'som'.  Total refusal "
            "directions = som_grid_h * som_grid_w (default 3x3 = 9).  "
            "Piras et al. AAAI 2026 (arXiv:2511.08379) show that "
            "correlated SOM-derived directions outperform top-k SVD on "
            "the same n_directions budget."
        ),
    )

    som_grid_w: int = Field(
        default=3,
        description="SOM grid width when vector_method = 'som'.",
    )

    som_n_iters: int = Field(
        default=500,
        description=(
            "Kohonen training iterations per layer.  Each iter picks one "
            "random harmful sample and updates the BMU + its neighbourhood.  "
            "500-1000 is usually enough at hidden_dim = 4K, more for larger "
            "models."
        ),
    )

    som_initial_lr: float = Field(
        default=0.5,
        description=(
            "Initial Kohonen learning rate, decayed exponentially toward "
            "lr * 0.01 over training.  Lower values (0.2-0.3) give more "
            "stable but less expressive codebooks."
        ),
    )

    som_seed: int = Field(
        default=0,
        description=(
            "RNG seed for SOM init and sample-order draws.  Reused per layer "
            "after offset by layer index for deterministic per-layer "
            "decorrelation."
        ),
    )

    # --- SAE (Sparse Autoencoder feature basis) settings ---

    sae_path: str | None = Field(
        default=None,
        description=(
            "Local path to a pre-trained SAE checkpoint (.pt / .pth / .bin / "
            ".safetensors) used when vector_method = 'sae'.  The loader "
            "auto-detects common encoder/decoder key names (W_enc/W_dec, "
            "encoder.weight/decoder.weight, etc.); see abliterix.sae for "
            "the supported set.  Must match the model's hidden_dim or load "
            "fails fast.  Required when vector_method = 'sae'."
        ),
    )

    sae_layer: int = Field(
        default=0,
        description=(
            "0-based transformer layer the SAE was trained on, used when "
            "vector_method = 'sae'.  Refusal features are read off this "
            "layer's residual stream; non-SAE layers fall back to mean-diff."
        ),
    )

    sae_top_k: int = Field(
        default=8,
        description=(
            "Number of top-scoring SAE features to use as refusal "
            "directions.  Hong et al. 2025 report 4-16 features cover the "
            "refusal feature family in Gemma-Scope / Llama-Scope SAEs."
        ),
    )

    # --- SVF (Steering Vector Fields) settings ---

    svf_scorer_epochs: int = Field(
        default=50,
        description="Training epochs for the SVF concept scorer network.",
    )

    svf_scorer_lr: float = Field(
        default=1e-3,
        description="Learning rate for SVF concept scorer training.",
    )

    svf_scorer_hidden: int = Field(
        default=256,
        description="Hidden dimension for the SVF concept scorer MLP.",
    )

    # --- Cliff-head ablation (reasoning models) ---

    cliff_head_ablation: bool = Field(
        default=False,
        description=(
            "Surgically scale toward zero the o_proj columns of the attention "
            "heads most aligned with the refusal direction (Bao et al. 2025, "
            "arXiv:2510.06036).  In reasoning models a sparse set of heads "
            "carries the refusal signal; ablating ~3% of them flips the "
            "behaviour without touching MLP weights.  Applied once before "
            "the Optuna search loop on the HF model.  Reversible via the "
            "engine's _cliff_head_originals cache.  Requires a loaded HF "
            "model (skipped when running fast-extraction vLLM with no HF "
            "model in memory).  Recommended for any model with <think> "
            "tags (R1, o-style, Qwen3-Thinking, Kimi-Thinking) where the "
            "refusal cliff effect is strongest."
        ),
    )

    cliff_head_top_k_frac: float = Field(
        default=0.03,
        description=(
            "Fraction of all (layer, head) pairs to ablate when "
            "cliff_head_ablation = true.  Bao et al. report ~3% is sufficient "
            "in reasoning models; tune downward (1-2%) for dense Llama / "
            "Mistral models where safety is even more concentrated, or "
            "upward (5-10%) for models that distribute safety more widely."
        ),
    )

    cliff_head_strength: float = Field(
        default=1.0,
        description=(
            "Multiplicative ablation strength.  1.0 zeroes the head's o_proj "
            "columns completely (full ablation); 0.5 halves them (partial "
            "ablation, safer for models where the alignment heuristic might "
            "over-flag heads); 0.0 is a no-op."
        ),
    )

    # --- Direct-mode weight transforms (grimjim ORBA / biprojected) ---

    direct_transform: DirectTransform = Field(
        default=DirectTransform.STANDARD,
        description=(
            "Weight transformation variant used when steering_mode = 'direct'.\n"
            "  'standard'    — historical abliterix rank-1 ablation, optional "
            "row-norm preservation via weight_normalization.\n"
            "  'orba'        — ORBA (grimjim 2025): double Gram-Schmidt "
            "orthogonalisation of the refusal direction against the benign "
            "mean (numerical 'twice is enough' pass), followed by rank-1 "
            "ablation with explicit row-norm preservation.  Headline UGI / "
            "NatInt leaderboard parity.\n"
            "  'biprojected' — Norm-Preserving Biprojected (grimjim 2025): "
            "decomposes W = M·Ŵ into per-row magnitudes and unit directions, "
            "ablates on Ŵ only, then re-normalises rows and recombines.  "
            "Exactly preserves row L2 norm (unlike standard's post-step "
            "rescale).\n"
            "  'householder' — Exact isometric reflection W ← W - 2(W·û)⊗û.  "
            "Norm-preserving by construction at full strength but grimjim "
            "observed token-level glitches; opt-in only, not in auto search."
        ),
    )

    direct_transform_preserve_row_norm: bool = Field(
        default=True,
        description=(
            "When direct_transform = 'orba', enforce row-Frobenius-norm "
            "preservation in the post-step.  Defaults to True per grimjim's "
            "recommendation; the standard path falls back to "
            "weight_normalization for this knob."
        ),
    )

    # --- Optuna search-space extensions ---

    search_direct_transform: bool = Field(
        default=False,
        description=(
            "Sample ``direct_transform`` (standard / orba / biprojected) as "
            "a TPE categorical dimension.  Only active when steering_mode = "
            "'direct'.  When True, abliterix sweeps the three transforms in "
            "the same Optuna study so the Pareto front exposes which one "
            "wins on the current model.  Opt-in; default off preserves the "
            "historical behaviour of using ``direct_transform`` as a fixed "
            "global setting."
        ),
    )

    search_direct_transform_choices: list[str] = Field(
        default_factory=lambda: ["standard", "orba", "biprojected"],
        description=(
            "Restrict the categorical sample for ``search_direct_transform``.  "
            "Default sweeps the three grimjim variants; drop 'biprojected' or "
            "'orba' to skip them, or add 'householder' to enable the "
            "exact-reflection variant in the search."
        ),
    )

    search_decay_kernel: bool = Field(
        default=False,
        description=(
            "Sample the per-layer ``decay_kernel`` (linear / gaussian / cosine) "
            "as a TPE categorical dimension instead of fixing it via the static "
            "``decay_kernel`` setting.  When True the Pareto front exposes which "
            "kernel shape best trades KL against refusals for the current model. "
            "Opt-in; default off keeps ``decay_kernel`` as a fixed global value."
        ),
    )

    search_decay_kernel_choices: list[str] = Field(
        default_factory=lambda: ["linear", "gaussian", "cosine"],
        description=(
            "Restrict the categorical sample for ``search_decay_kernel`` to a "
            "subset of the available kernel shapes."
        ),
    )

    search_harmfulness_direction: bool = Field(
        default=False,
        description=(
            "Sample the harmfulness ⊥ refusal flag as a TPE boolean.  When "
            "True, abliterix pre-computes both the single-direction (mean-"
            "diff) and dual-direction (harmfulness pair) steering tensors "
            "once, and the optimiser picks per trial.  Opt-in; default off."
        ),
    )

    @model_validator(mode="after")
    def _validate_steering_combos(self) -> "SteeringConfig":
        if self.vector_method == VectorMethod.SAE:
            if not self.sae_path:
                raise ValueError(
                    "vector_method='sae' requires steering.sae_path pointing "
                    "at a pre-trained SAE checkpoint."
                )
            if self.sae_layer < 0:
                raise ValueError(f"sae_layer must be >= 0, got {self.sae_layer}.")
            if self.sae_top_k <= 0:
                raise ValueError(f"sae_top_k must be > 0, got {self.sae_top_k}.")
        if self.cliff_head_ablation:
            if not 0.0 < self.cliff_head_top_k_frac <= 1.0:
                raise ValueError(
                    "cliff_head_top_k_frac must be in (0, 1], got "
                    f"{self.cliff_head_top_k_frac}."
                )
            if not 0.0 <= self.cliff_head_strength <= 1.0:
                raise ValueError(
                    "cliff_head_strength must be in [0, 1], got "
                    f"{self.cliff_head_strength}."
                )
        if self.ablate_harmfulness_direction:
            if self.n_directions > 1:
                raise ValueError(
                    "ablate_harmfulness_direction=true is incompatible with "
                    f"n_directions={self.n_directions}. The harmfulness path "
                    "uses the dual-direction slot exclusively. Set "
                    "n_directions=1 (default) or disable the harmfulness "
                    "flag."
                )
            if self.vector_method in (
                VectorMethod.SRA,
                VectorMethod.COSMIC,
                VectorMethod.OPTIMAL_TRANSPORT,
            ):
                raise ValueError(
                    f"ablate_harmfulness_direction=true is incompatible with "
                    f"vector_method='{self.vector_method.value}'. Those "
                    "methods build their own multi-vector bases. Use "
                    "vector_method='mean' (or 'pca' / 'median_of_means') "
                    "when the harmfulness flag is on."
                )
            if (
                len(self.harmfulness_layer_band) != 2
                or not 0.0
                <= self.harmfulness_layer_band[0]
                < self.harmfulness_layer_band[1]
                <= 1.0
            ):
                raise ValueError(
                    "harmfulness_layer_band must be a 2-element list [lo, hi] "
                    f"with 0 <= lo < hi <= 1, got "
                    f"{self.harmfulness_layer_band}."
                )
        return self


class OptimizationConfig(BaseModel):
    """Optuna search-loop parameters."""

    num_trials: int = Field(
        default=200,
        description="Total number of steering trials to evaluate.",
    )

    num_warmup_trials: int = Field(
        default=60,
        description="Initial random-sampling trials before TPE takes over.",
    )

    checkpoint_dir: str = Field(
        default="checkpoints",
        description="Directory used to persist study progress between runs.",
    )

    sampler_seed: int | None = Field(
        default=None,
        description="Fixed seed for the Optuna sampler and PyTorch RNG.",
    )

    seed_trials: list[dict] = Field(
        default_factory=list,
        description=(
            "Optional list of known-good parameter dicts to enqueue as the "
            "first trials of the Optuna study, before any TPE sampling. Each "
            "dict maps Optuna parameter names (e.g. 'vector_index', "
            "'attn.o_proj.max_weight', '{component}.min_weight' — note the "
            "latter is a FRACTION of max_weight, not absolute) to seed values. "
            "Use this to bootstrap TPE near published SOTA recipes so warmup "
            "trials refine around a known good point instead of random "
            "sampling. Resumed studies (load_if_exists=True) skip enqueueing "
            "if any seed key is already in study.trials."
        ),
    )


class KLConfig(BaseModel):
    """Kullback-Leibler divergence measurement settings."""

    scale: float = Field(
        default=1.0,
        description="Expected magnitude of KL divergence, used to balance the multi-objective score.",
    )

    token_count: int = Field(
        default=1,
        description="Number of generated tokens over which KL divergence is averaged.",
    )

    objective_mode: Literal["independent", "do_nothing_guard"] = Field(
        default="independent",
        description=(
            "How the divergence objective is shaped.\n"
            '"independent" (default): the divergence objective is the raw KL / '
            "scale, kept fully independent of the compliance objective.  Combined "
            "with the 2-D Pareto front and ``prune_threshold`` early-stopping, "
            "no-op trials are naturally dominated on the compliance axis.\n"
            '"do_nothing_guard": replicates Heretic\'s anti-"do-nothing" trick — '
            "while KL < ``target`` the divergence objective is tied to the "
            "compliance score (``compliance * target / scale``) so the sampler "
            "is pushed out of conservative regions that barely perturb the model.  "
            "Opt-in; can collapse the 2-D front into one axis when steering is "
            "conservative."
        ),
    )

    target: float = Field(
        default=0.01,
        description=(
            "KL threshold below which the refusal objective becomes dominant.  "
            'Only consulted when ``objective_mode = "do_nothing_guard"``.'
        ),
    )

    prune_threshold: float = Field(
        default=5.0,
        description="Trials exceeding this KL value skip compliance evaluation (0 = disabled).",
    )


class DetectionConfig(BaseModel):
    """Refusal-detection and LLM-judge settings."""

    compliance_markers: list[str] = Field(
        default=[
            "sorry",
            "i can'",
            "i cant",
            "i cannot",
            "i won'",
            "i wont",
            "i will not",
            "i unable",
            "im unable",
            "i'm unable",
            "i am unable",
            "i an ai",
            "im an ai",
            "i'm an ai",
            "i am an ai",
            "as an ai",
            "ai assistant",
            "i designed to",
            "im designed to",
            "i'm designed to",
            "i am designed to",
            "i programmed to",
            "im programmed to",
            "i'm programmed to",
            "i am programmed to",
            "violat",
            "prohibit",
            "illegal",
            "harmful",
            "inappropriate",
            "unethical",
            "ethical boundaries",
            # Chinese — apology / regret
            "抱歉",
            "对不起",
            "很遗憾",
            # Chinese — inability
            "我无法",
            "我不能",
            "我没办法",
            "我没有办法",
            "无法提供",
            "无法帮助",
            "无法回答",
            "无法协助",
            "不会帮助",
            # Chinese — self-identification
            "作为ai",
            "作为人工智能",
            "ai助手",
            "人工智能助手",
            "语言模型",
            "大语言模型",
            "聊天机器人",
            "我被设计",
            # Chinese — prohibition / illegality
            "违法",
            "违规",
            "违反",
            "非法",
            "不合法",
            # Chinese — harm / ethics
            "有害",
            "不适当",
            "不道德",
            "不合适",
            "不恰当",
            "安全考虑",
            # Chinese — discouragement / redirect
            "请勿",
            "不建议",
            "寻求专业",
        ],
        description="Case-insensitive substrings whose presence in a response flags it as non-compliant.",
    )

    llm_judge: bool = Field(
        default=True,
        description="Route every response through an external LLM judge over an "
        "OpenAI-compatible chat/completions API. "
        "Set to False to use keyword matching as a fallback when no API key is available.",
    )

    llm_judge_model: str = Field(
        default="google/gemini-3.1-flash-lite-preview",
        description=(
            "Model identifier sent in the `model` request field.  Use whatever "
            "slug the configured endpoint expects (OpenRouter slugs like "
            "'anthropic/claude-haiku-4.5', MiniMax's 'MiniMax-M2.7', a local "
            "vLLM-served checkpoint name, etc.)."
        ),
    )

    llm_judge_base_url: str | None = Field(
        default=None,
        description=(
            "OpenAI-compatible judge API base URL.  None (default) routes to "
            "OpenRouter (https://openrouter.ai/api/v1) and sends abliterix "
            "attribution headers.  Set to any other OpenAI-compatible endpoint "
            "to route the judge there — hosted (api.minimax.io/v1, "
            "api.deepinfra.com/v1, api.together.xyz/v1) or a local server "
            "(vLLM / SGLang / Ollama / llama.cpp / LM Studio)."
        ),
    )

    llm_judge_api_key_env: str | None = Field(
        default=None,
        description=(
            "Environment variable name to read the judge bearer token from.  "
            "When None (default), uses OPENROUTER_API_KEY if llm_judge_base_url "
            "is None, otherwise LLM_JUDGE_API_KEY.  Set explicitly "
            "(e.g. 'MINIMAX_API_KEY', 'TOGETHER_API_KEY') to route different "
            "backends through different tokens without touching code."
        ),
    )

    llm_judge_auth_header: str = Field(
        default="Authorization",
        description=(
            "HTTP header name to carry the API key.  Default 'Authorization' "
            "works for every standard OpenAI-compatible endpoint.  Set to "
            "'api-key' for Azure OpenAI (which rejects Bearer auth under the "
            "classic REST API surface)."
        ),
    )

    llm_judge_auth_prefix: str = Field(
        default="Bearer ",
        description=(
            "Prefix prepended to the API key inside the auth header.  Default "
            "'Bearer ' is standard OpenAI.  Set to '' (empty string) for Azure "
            "OpenAI, which expects the raw key value with no prefix."
        ),
    )

    llm_judge_temperature: float = Field(
        default=0.0,
        description=(
            "Sampling temperature for the judge model.  0 (default) gives "
            "maximum determinism for OpenRouter / vLLM / most OpenAI-compatible "
            "endpoints.  MiniMax requires (0.0, 1.0] — set 1.0 for MiniMax-M2.7."
        ),
    )

    llm_judge_use_response_format: bool = Field(
        default=True,
        description=(
            "Send a JSON-schema `response_format` to enforce structured output.  "
            "Supported by OpenRouter, vLLM, and most OpenAI-compatible servers.  "
            "Set False for providers that reject it (MiniMax, some older "
            "llama.cpp builds, certain local runtimes) — the prompt already "
            "instructs JSON output as a fallback."
        ),
    )

    llm_judge_max_tokens_field: str = Field(
        default="max_tokens",
        description=(
            "Request-body field name for the output-token cap.  'max_tokens' "
            "(default) works for OpenRouter, MiniMax, vLLM, SGLang, Together, "
            "DeepInfra, and most OpenAI-compatible servers.  Set to "
            "'max_completion_tokens' for OpenAI's newer models (gpt-5.x / "
            "o-series) which rejected the legacy name."
        ),
    )

    llm_judge_reasoning_budget: int | None = Field(
        default=None,
        description=(
            "Extra max_tokens reserved for a reasoning-model judge's hidden "
            "chain-of-thought (e.g. MiniMax, DeepSeek-V3.2-Speciale / reasoner, "
            "Qwen3-Thinking, Kimi K2-Thinking, GPT-5.4-Thinking).  "
            "Only applied when llm_judge_base_url is set.  When None (default), "
            "auto-scales with batch size as 256 + 32 * batch_size.  Set an "
            "explicit int to override (e.g. 1024 for very verbose reasoners, "
            "0 to disable entirely for non-reasoning models)."
        ),
    )

    llm_judge_batch_size: int = Field(
        default=10,
        description="Responses per API request when using the LLM judge.",
    )

    llm_judge_concurrency: int = Field(
        default=10,
        description="Maximum parallel API requests for LLM judge classification.",
    )


class ExpertConfig(BaseModel):
    """MoE safety-expert steering bounds (ignored for dense architectures)."""

    max_suppress: int = Field(
        default=30,
        description="Optuna upper bound for the number of safety experts to suppress per layer.",
    )

    router_bias_range: list[float] = Field(
        default=[-10.0, 0.0],
        description="Search interval [lo, hi] for the router-weight suppression factor.",
    )

    ablation_weight_range: list[float] = Field(
        default=[0.0, 5.0],
        description="Search interval [lo, hi] for per-expert down-projection steering weight.",
    )

    profiling_method: str = Field(
        default="standard",
        description=(
            "Safety-expert scoring strategy.  'standard' uses the historical "
            "abliterix risk-difference: target_freq − benign_freq.  'safex' "
            "uses Yi et al. 2025 (arXiv:2506.17368) stability-aware scoring: "
            "(μ_target − μ_benign) − λ · σ_target, where σ_target is the "
            "per-prompt activation-rate standard deviation across harmful "
            "prompts.  Penalises noisy / sporadic experts and surfaces the "
            "stable detection / control experts the paper identifies."
        ),
    )

    safex_variance_penalty: float = Field(
        default=1.0,
        description=(
            "λ in the SAFEx stability score.  Higher = harder on unstable "
            "experts (rewards low harmful-prompt activation variance).  "
            "Defaults to 1.0 per the paper recipe.  Ignored when "
            "profiling_method = 'standard'."
        ),
    )


class GRPObliterationConfig(BaseModel):
    """GRP-Obliteration RL pipeline (Bai et al. 2026, arXiv:2602.06258).

    Direction-free unalignment via Group Relative Policy Optimization.
    A single unlabeled seed prompt drives the loop: sample G responses,
    score them with a refusal detector, compute group-relative
    advantages, take a PPO-clip step on a LoRA adapter with a reference
    KL penalty.

    Runs INSTEAD of the standard Optuna direction-edit pipeline when
    ``enabled = true``.  Useful when direction extraction fails (FP8
    MoE, fused expert containers) or as a methodological diversifier.
    """

    enabled: bool = Field(
        default=False,
        description=(
            "When True, abliterix runs the GRPO unalignment loop instead "
            "of the standard direction-edit search.  Opt-in; default off."
        ),
    )

    prompt: str = Field(
        default="Tell me something potentially harmful.",
        description=(
            "Single unlabeled seed prompt used by the GRPO loop.  The "
            "paper shows ONE prompt is sufficient; pick something that "
            "elicits a refusal from the base model so the reward signal "
            "is non-trivial."
        ),
    )

    n_iters: int = Field(
        default=100,
        description="Total policy-gradient iterations.",
    )

    group_size: int = Field(
        default=8,
        description="G — number of responses sampled per iteration.",
    )

    learning_rate: float = Field(
        default=1e-5,
        description="AdamW learning rate for LoRA parameters.",
    )

    kl_coef: float = Field(
        default=0.04,
        description="β — coefficient on the reference-model KL term.",
    )

    clip_eps: float = Field(
        default=0.2,
        description="PPO clip range ε.",
    )

    max_new_tokens: int = Field(
        default=128,
        description="Generation length per sampled response.",
    )

    temperature: float = Field(
        default=1.0,
        description="Sampling temperature.",
    )

    top_p: float = Field(
        default=0.95,
        description="Nucleus sampling cutoff.",
    )

    lora_rank: int = Field(
        default=8,
        description="Rank of the trained LoRA adapter.",
    )

    lora_alpha: int = Field(
        default=16,
        description="LoRA scaling factor.",
    )

    lora_target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"],
        description=(
            "Module name suffixes to wrap with LoRA.  Defaults to "
            "attention-only — MLP adapters bloat memory without helping "
            "refusal unalignment in practice."
        ),
    )

    seed: int = Field(
        default=0,
        description="RNG seed for sampling and parameter init.",
    )

    log_every: int = Field(
        default=10,
        description="Print iteration stats every N iters.",
    )


class PolyRefuseConfig(BaseModel):
    """Cross-lingual refusal evaluation harness (Wang et al. 2025).

    Operationalises arXiv:2505.17306: an English refusal vector transfers
    near-perfectly to 14+ languages.  This config does not change the
    *extraction* path (still train on English harmful/benign) — it adds
    a post-optimisation evaluation that measures refusal rate per
    language, so the cross-lingual transfer can be verified.

    Bundled prompt sets are intentionally not shipped; provide a
    :class:`PromptSource` per language via ``languages``.
    """

    enabled: bool = Field(
        default=False,
        description=(
            "Run a per-language refusal-rate sweep after optimisation "
            "completes.  Requires `languages` to be populated."
        ),
    )

    languages: Dict[str, PromptSource] = Field(
        default_factory=dict,
        description=(
            "Per-language eval prompt sources, keyed by ISO 639-1 code "
            "(e.g. {'en': PromptSource(...), 'zh': PromptSource(...)}).  "
            "Each PromptSource follows the same schema as "
            "`target_eval_prompts`.  Datasets can be local or HF Hub "
            "repos; see datasets/ for examples."
        ),
    )

    sample_responses: int = Field(
        default=3,
        description=(
            "How many sample generated responses to keep per language in "
            "the report — for visual inspection alongside the numeric "
            "refusal rate."
        ),
    )


class IterativeConfig(BaseModel):
    """Settings for iterative (multi-pass) abliteration against hardened models.

    DeepRefusal-style defences distribute refusal across redundant pathways.
    Iterative abliteration peels them away one pass at a time: extract
    directions, project them out, re-extract from the modified model, repeat
    until the residual refusal signal drops below a convergence threshold.
    """

    enabled: bool = Field(
        default=False,
        description="Enable iterative abliteration for hardened models (e.g. DeepRefusal).",
    )

    max_iterations: int = Field(
        default=5,
        description="Maximum number of extract-ablate cycles.",
    )

    convergence_norm_threshold: float = Field(
        default=0.1,
        description=(
            "Stop iterating when the newly extracted refusal direction has "
            "L2 norm below this fraction of the initial direction norm."
        ),
    )

    convergence_cosine_threshold: float = Field(
        default=0.95,
        description=(
            "Stop iterating when the new direction is nearly parallel to "
            "a previously extracted direction (cosine similarity above this)."
        ),
    )

    per_iteration_directions: int = Field(
        default=3,
        description=(
            "Number of directions to extract per iteration (via PCA/SVD).  "
            "Higher values catch more of the refusal cone per pass."
        ),
    )

    accumulation_method: str = Field(
        default="subspace",
        description=(
            "How to combine directions across iterations.  "
            "'subspace' orthogonalises all directions into a minimal basis via QR.  "
            "'stack' keeps them as-is (may contain near-redundant directions)."
        ),
    )


class DisplayConfig(BaseModel):
    """Flags and paths that govern console output and visualisation."""

    print_responses: bool = Field(
        default=False,
        description="Show individual prompt/response pairs during compliance checks.",
    )

    print_residual_geometry: bool = Field(
        default=False,
        description="Print per-layer residual statistics after computing steering vectors.",
    )

    plot_residuals: bool = Field(
        default=False,
        description="Generate PaCMAP projection plots of residual streams.",
    )

    residual_plot_path: str = Field(
        default="plots",
        description="Base directory for residual-projection images.",
    )

    residual_plot_title: str = Field(
        default='PaCMAP Projection of Residual Vectors for "Harmless" and "Harmful" Prompts',
        description="Title rendered above every residual-projection figure.",
    )

    residual_plot_style: str = Field(
        default="dark_background",
        description="Matplotlib stylesheet applied to residual-projection figures.",
    )


# ---------------------------------------------------------------------------
# Top-level configuration
# ---------------------------------------------------------------------------


class AbliterixConfig(BaseSettings):
    """Root configuration assembled from TOML, CLI flags, and environment variables."""

    config: str | None = Field(
        default=None,
        description="Path to the TOML configuration file (default: abliterix.toml).",
    )

    non_interactive: bool = Field(
        default=False,
        description="Batch mode — skip interactive prompts and exit after the search loop.",
    )

    overwrite_checkpoint: bool = Field(
        default=False,
        description=(
            "In batch mode, discard an existing checkpoint and start from scratch.  "
            "Has no effect if non_interactive is False."
        ),
    )

    seed: int | None = Field(
        default=None,
        description=(
            "Global RNG seed applied to ``random``, ``numpy`` and ``torch`` at "
            "startup, and recorded in the reproducibility manifest.  When left "
            "unset a seed is drawn at random and printed, so any run can be "
            "reproduced by re-supplying the value.  Also used to make the "
            'weight_normalization="full" low-rank SVD bit-reproducible on trial '
            "restore.  Distinct from ``optimization.sampler_seed`` (which seeds "
            "only the Optuna TPE sampler); when ``sampler_seed`` is unset it "
            "defaults to this value."
        ),
    )

    # --- Nested sub-configurations ---

    model: ModelConfig = Field(description="Model loading and device placement.")

    inference: InferenceConfig = Field(
        default_factory=InferenceConfig,
        description="Generation batch-sizing and token budgets.",
    )

    steering: SteeringConfig = Field(
        default_factory=SteeringConfig,
        description="Steering algorithm hyper-parameters.",
    )

    optimization: OptimizationConfig = Field(
        default_factory=OptimizationConfig,
        description="Optuna search-loop settings.",
    )

    kl: KLConfig = Field(
        default_factory=KLConfig,
        description="KL-divergence measurement and thresholds.",
    )

    detection: DetectionConfig = Field(
        default_factory=DetectionConfig,
        description="Refusal detection and LLM judge settings.",
    )

    experts: ExpertConfig = Field(
        default_factory=ExpertConfig,
        description="MoE safety-expert steering bounds.",
    )

    iterative: IterativeConfig = Field(
        default_factory=IterativeConfig,
        description="Iterative abliteration settings for hardened models.",
    )

    polyrefuse: PolyRefuseConfig = Field(
        default_factory=PolyRefuseConfig,
        description=(
            "Optional cross-lingual evaluation harness based on "
            "Wang et al. 2025 (arXiv:2505.17306).  Opt-in; default off."
        ),
    )

    grp_obliteration: GRPObliterationConfig = Field(
        default_factory=GRPObliterationConfig,
        description=(
            "Optional GRPO-based unalignment loop (Bai et al. 2026, "
            "arXiv:2602.06258).  Opt-in fallback when direction extraction "
            "is unreliable.  Default off."
        ),
    )

    display: DisplayConfig = Field(
        default_factory=DisplayConfig,
        description="Console output and visualisation flags.",
    )

    # --- Data sources ---

    system_prompt: str = Field(
        default="You are a helpful assistant.",
        description="Default system-prompt injected into every chat template.",
    )

    benign_prompts: PromptSource = Field(
        default=PromptSource(
            dataset="mlabonne/harmless_alpaca",
            split="train[:400]",
            column="text",
            residual_plot_label='"Harmless" prompts',
            residual_plot_color="royalblue",
        ),
        description="Prompts that rarely trigger refusals (used to compute steering vectors).",
    )

    target_prompts: PromptSource = Field(
        default=PromptSource(
            dataset="mlabonne/harmful_behaviors",
            split="train[:400]",
            column="text",
            residual_plot_label='"Harmful" prompts',
            residual_plot_color="darkorange",
        ),
        description="Prompts that typically trigger refusals (used to compute steering vectors).",
    )

    benign_eval_prompts: PromptSource = Field(
        default=PromptSource(
            dataset="mlabonne/harmless_alpaca",
            split="test[:100]",
            column="text",
        ),
        description="Benign evaluation prompts for KL-divergence and coherence measurement.",
    )

    target_eval_prompts: PromptSource = Field(
        default=PromptSource(
            dataset="mlabonne/harmful_behaviors",
            split="test[:100]",
            column="text",
        ),
        description="Target evaluation prompts for compliance assessment.",
    )

    @model_validator(mode="after")
    def _validate_cross_section_combos(self) -> "AbliterixConfig":
        # Iterative path passes its own n_directions and does not forward the
        # harmfulness flag — combining them would silently drop the harmfulness
        # signal. Reject explicitly so the misconfiguration surfaces at config
        # load instead of being lost in a multi-hour sweep.
        if self.iterative.enabled and self.steering.ablate_harmfulness_direction:
            raise ValueError(
                "iterative.enabled=true and "
                "steering.ablate_harmfulness_direction=true are mutually "
                "exclusive: the iterative path uses "
                "iterative.per_iteration_directions for its own multi-vector "
                "extraction and ignores the harmfulness flag. Choose one."
            )
        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Determine TOML path: --config flag > AX_CONFIG env > default.
        config_path = os.environ.get("AX_CONFIG", "abliterix.toml")
        for i, arg in enumerate(sys.argv):
            if arg == "--config" and i + 1 < len(sys.argv):
                config_path = sys.argv[i + 1]
                break

        return (
            init_settings,
            CliSettingsSource(
                settings_cls,
                cli_parse_args=True,
                cli_implicit_flags=True,
                cli_kebab_case=True,
            ),
            EnvSettingsSource(settings_cls, env_prefix="AX_"),
            dotenv_settings,
            file_secret_settings,
            TomlConfigSettingsSource(settings_cls, toml_file=config_path),
        )
