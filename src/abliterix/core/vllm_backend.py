# Abliterix — vLLM inference backend
# Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""vLLM-backed generation engine with tensor parallelism.

This module provides :class:`VLLMGenerator`, a drop-in replacement for
:class:`SteeringEngine`'s generation methods that leverages vLLM's tensor
parallelism to utilise ALL GPUs simultaneously — unlike the HuggingFace
``device_map="auto"`` pipeline parallelism which only uses one GPU at a time.

Architecture
~~~~~~~~~~~~

The abliteration pipeline splits into two phases:

**Phase 1 (HF Transformers)** — one-time setup:
  * Load model with HuggingFace for hidden state extraction.
  * Compute steering vectors.
  * Pre-compute LoRA projection caches (``v @ W`` for all layer/component pairs).
  * Capture baseline logprobs and metrics.
  * Unload HF model to free VRAM.

**Phase 2 (vLLM)** — optimisation loop:
  * Load model with vLLM tensor parallelism + LoRA support.
  * For each trial: serialise LoRA adapter to disk → generate via vLLM.
  * KL divergence approximated using top-K logprobs.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch import Tensor

from ..settings import AbliterixConfig
from ..types import ChatMessage
from ..util import print


# Default LoRA rank to declare to vLLM when the user has not pinned
# ``vllm_max_lora_rank`` in their recipe. vLLM 0.20.x's own default for
# ``LoRAConfig.max_lora_rank`` is 16, so we mirror it here. Adapters with
# rank < this value are zero-padded by ``_serialize_adapter`` so vLLM
# accepts them; adapters with rank > this value would require the user to
# bump ``vllm_max_lora_rank``.
_DEFAULT_VLLM_MAX_LORA_RANK = 16

# MLA-bearing model architecture name fragments. When the model's HF
# ``config.architectures`` contains any of these substrings, we pick an
# MLA-aware attention backend (``FLASH_ATTN_MLA``) instead of the
# non-MLA defaults — vLLM 0.20.x rejects ``TRITON_ATTN`` on MLA models.
_MLA_ARCH_FRAGMENTS: tuple[str, ...] = (
    "DeepseekV2",
    "DeepseekV3",
    "DeepseekV4",
    "MiniMaxM2",  # MiniMax-M2.5, M2.7
    "MiniMaxText",
)

# Architecture name fragments for sink-attention models (gpt-oss family).
# vLLM rejects ``FLASH_ATTN`` on these ("attention sinks not supported")
# but ``TRITON_ATTN`` works.
_SINK_ATTENTION_ARCH_FRAGMENTS: tuple[str, ...] = (
    "GptOss",  # gpt-oss-20b / 120b
)


def _detect_arch_family(model_id: str, trust_remote_code: bool) -> str:
    """Return the first architecture name from the model's HF config, or
    an empty string if the config can't be loaded.

    A failure here means MLA-aware backend selection silently degrades to
    "let vLLM pick", which on an MLA model can crash the engine init the
    PRD #20 dispatcher was meant to prevent. Logged at WARNING so it's
    visible in deploy logs without being an exception.
    """
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    except Exception as exc:
        print(
            "  [yellow]Warning: could not load HF config for arch detection "
            f"({type(exc).__name__}: {exc}); vLLM will pick attention "
            "backend on its own. If this model is MLA (DeepSeek-V2/V3, "
            "MiniMax-M2.x), set ``attention_backend`` explicitly in the "
            "[model] config to avoid an engine-init crash.[/]"
        )
        return ""
    archs = getattr(cfg, "architectures", None) or []
    return archs[0] if archs else ""


def _resolve_compile_mode(enforce_eager_legacy: bool, vllm_compile_mode: str) -> str:
    """Reconcile the legacy ``enforce_eager`` field with the new
    ``vllm_compile_mode`` field.

    The legacy field still exists for recipes that pre-date PRD #20.
    When ``enforce_eager=True`` is set, we honour it by forcing the
    "eager" mode regardless of ``vllm_compile_mode`` — that's the only
    way the two fields can stay consistent without rejecting
    historical recipes at config load.
    """
    if enforce_eager_legacy:
        return "eager"
    return vllm_compile_mode


def _detect_fp8_and_kv_dtype(
    config: AbliterixConfig, *, model_id: str, trust_remote_code: bool
) -> tuple[bool, str | None]:
    """Decide whether vLLM should treat this model as FP8 and which KV
    cache dtype to use.

    Two passes:

    1. Recipe wins: ``quant_method = "fp8"`` always sets ``is_fp8=True``.
    2. HF config sniff: native FP8 models (MiniMax-M2.5, Qwen3.5-*-FP8)
       ship ``quantization_config.quant_method = "fp8"`` in their
       config.json. vLLM auto-detects; we still need the bool to seed
       the KV-cache-dtype default below.

    Then ``kv_cache_dtype``:

    - Recipe override wins.
    - On FP8 + H100+ (sm_90+) auto-default to ``fp8_e4m3`` for 2x KV
      capacity. Older cards have no FP8 KV path.
    """
    is_fp8 = bool(
        config.model.quant_method and config.model.quant_method.value == "fp8"
    )
    if not is_fp8:
        try:
            from transformers import AutoConfig

            _auto_cfg = AutoConfig.from_pretrained(
                model_id, trust_remote_code=trust_remote_code
            )
            _qcfg = getattr(_auto_cfg, "quantization_config", None)
            if _qcfg is None:
                _text_cfg = getattr(_auto_cfg, "text_config", None)
                if _text_cfg is not None:
                    _qcfg = getattr(_text_cfg, "quantization_config", None)
            if _qcfg is not None:
                _qm = (
                    _qcfg if isinstance(_qcfg, dict) else getattr(_qcfg, "__dict__", {})
                )
                if _qm.get("quant_method") == "fp8":
                    is_fp8 = True
        except Exception:
            pass

    kv_dtype = config.model.kv_cache_dtype
    if kv_dtype is None and is_fp8 and torch.cuda.is_available():
        try:
            cc = torch.cuda.get_device_capability(0)
            if cc[0] >= 9:
                kv_dtype = "fp8_e4m3"
        except Exception:
            pass
    return is_fp8, kv_dtype


def _build_llm_kwargs(
    config: AbliterixConfig,
    *,
    model_arch: str,
    is_fp8: bool,
    kv_cache_dtype: str | None,
    lora_max_rank: int,
) -> dict[str, Any]:
    """Pure function that assembles the dict passed to ``vllm.LLM(...)``.

    Lifted out of ``VLLMGenerator.__init__`` so the kwargs assembly is
    unit-testable without importing vLLM (PR #21 review item 7). Keep
    every conditional kwarg branch in this function so tests can lock
    them.
    """
    from . import vllm_compilation_config

    tp = config.model.tensor_parallel_size
    if tp is None:
        tp = torch.cuda.device_count()

    compile_mode = _resolve_compile_mode(
        config.model.enforce_eager, config.model.vllm_compile_mode
    )
    compilation_config = vllm_compilation_config.build(compile_mode)

    kwargs: dict[str, Any] = dict(
        model=config.model.model_id,
        tensor_parallel_size=tp,
        gpu_memory_utilization=config.model.gpu_memory_utilization,
        trust_remote_code=config.model.trust_remote_code or False,
        enable_expert_parallel=config.model.enable_expert_parallel,
        # MoE compute backend. Default 'triton' avoids the FlashInfer
        # cutlass per-expert-group JIT compile that costs ~30 minutes
        # on first sm_90 cold start.
        moe_backend=config.model.moe_backend,
        # generate_and_score requests logprobs=100 to build a sparse KL
        # distribution that covers >99.9% of the probability mass.  vLLM
        # V1 caps sampler logprobs at 20 by default; lift it explicitly
        # so the KL computation keeps its top-100 tail.
        max_logprobs=100,
        # Issue #22: read per-token routed expert IDs from RequestOutput
        # instead of installing forward hooks via collective_rpc. The
        # profile_safety_experts_vllm function reads
        # ``output.outputs[0].routed_experts`` (numpy ndarray of shape
        # ``(tokens, layers, top_k)``) when this is on. Cost is small
        # (~140 KB per typical request); the win is dropping ~150 LoC of
        # worker rpc plumbing + one of the two reasons we needed
        # VLLM_ALLOW_INSECURE_SERIALIZATION.
        enable_return_routed_experts=config.model.vllm_return_routed_experts,
        # vLLM 0.20.x compilation_config encodes the eager/non-eager
        # intent. ``enforce_eager`` is intentionally NOT passed here —
        # _resolve_compile_mode folds it into ``compile_mode`` above so
        # the two fields cannot disagree.
        compilation_config=compilation_config,
        # Disable vLLM custom all-reduce only on Blackwell PCIe sm_120
        # (deadlock); NVLink Hopper / SXM Blackwell keep the perf win.
        disable_custom_all_reduce=_should_disable_custom_all_reduce(
            config.model.disable_custom_all_reduce
        ),
        enable_prefix_caching=not bool(config.model.use_in_place_editing),
    )

    attention_backend = _resolve_attention_backend(
        config.model.attention_backend, model_arch
    )
    if attention_backend is not None:
        kwargs["attention_config"] = {"backend": attention_backend}

    kwargs["limit_mm_per_prompt"] = config.model.limit_mm_per_prompt or {
        "image": 0,
        "video": 0,
        "audio": 0,
    }

    if not config.model.disable_lora:
        kwargs.update(
            enable_lora=True,
            max_lora_rank=lora_max_rank,
            max_loras=config.model.vllm_max_loras,
            max_cpu_loras=max(config.model.vllm_max_loras + 1, 2),
        )
        if config.model.lora_target_modules:
            kwargs["lora_target_modules"] = list(config.model.lora_target_modules)

    if config.model.max_model_len is not None:
        kwargs["max_model_len"] = config.model.max_model_len
    if config.model.max_num_seqs is not None:
        kwargs["max_num_seqs"] = config.model.max_num_seqs
    if config.model.hf_overrides:
        kwargs["hf_overrides"] = config.model.hf_overrides
    if is_fp8:
        kwargs["quantization"] = "fp8"
    if kv_cache_dtype is not None:
        kwargs["kv_cache_dtype"] = kv_cache_dtype

    return kwargs


def _resolve_attention_backend(
    config_override: str | None, model_arch: str
) -> str | None:
    """Pick the vLLM attention_config backend.

    - User-set ``attention_backend`` in the recipe always wins.
    - MLA models (DeepSeek-V2/V3, MiniMax-M2.x) get ``FLASH_ATTN_MLA``
      because the non-MLA names are rejected at engine init.
    - Sink-attention models (gpt-oss) keep ``TRITON_ATTN`` — ``FLASH_ATTN``
      explicitly errors with "attention sinks not supported".
    - Everything else returns None so vLLM applies its own default
      (``FLASH_ATTN`` on Hopper, etc.).
    """
    if config_override is not None:
        return config_override
    for frag in _MLA_ARCH_FRAGMENTS:
        if frag in model_arch:
            return "FLASH_ATTN_MLA"
    for frag in _SINK_ATTENTION_ARCH_FRAGMENTS:
        if frag in model_arch:
            return "TRITON_ATTN"
    return None


def _should_disable_custom_all_reduce(config_override: bool | None) -> bool:
    """Auto-detect when vLLM's custom all-reduce path needs to be disabled.

    Returns True only on Blackwell PCIe (sm_120) where the path is known to
    deadlock during worker init without NVLink.  User-set value always wins.
    """
    if config_override is not None:
        return config_override
    if not torch.cuda.is_available():
        return False
    try:
        major, minor = torch.cuda.get_device_capability(0)
    except Exception:
        return False
    # sm_120 is Blackwell PCIe (RTX PRO 6000). sm_100 is Blackwell SXM
    # (B100/B200) which has NVLink and does not need the workaround.
    return (major, minor) == (12, 0)


class VLLMGenerator:
    """vLLM-backed text generator with LoRA adapter hot-swapping.

    This class mirrors the generation API of :class:`SteeringEngine`
    (``generate_text_batched``, ``generate_and_score_batched``,
    ``compute_logprobs_batched``) so callers can use it interchangeably.
    """

    def __init__(self, config: AbliterixConfig):
        from .vllm_compat import (
            check_vllm_version,
            ensure_vllm_env,
            install_gemma4_transformers_compat,
        )

        # Refuse to start against an unsupported vLLM version. abliterix's
        # current floor is 0.18 because `VLLM_ALLOW_INSECURE_SERIALIZATION`
        # (PR #35928) is required for the collective_rpc path.
        check_vllm_version()

        # Auto-set the small set of vLLM env vars needed for in-place
        # editing / collective_rpc. Idempotent and never overwrites a
        # user-set value.
        needs_rpc = bool(config.model.use_in_place_editing)
        written_env = ensure_vllm_env(needs_collective_rpc=needs_rpc)
        if written_env:
            print(
                "  [dim]vLLM env (auto-set): "
                + ", ".join(f"{k}={v}" for k, v in written_env.items())
                + "[/]"
            )

        install_gemma4_transformers_compat()

        from vllm import LLM, SamplingParams  # noqa: F811

        self.config = config
        self._SamplingParams = SamplingParams

        tp = config.model.tensor_parallel_size
        if tp is None:
            tp = torch.cuda.device_count()

        model_id = config.model.model_id
        trust = config.model.trust_remote_code or False

        # Architecture sniff drives the MLA-aware attention backend choice
        # below. Cached so we only hit the HF config once.
        model_arch = _detect_arch_family(model_id, trust)

        print(f"* Loading model in vLLM with TP={tp}...")

        self._lora_disabled = bool(config.model.disable_lora)
        # vLLM's max_lora_rank governs every adapter loaded by this engine;
        # adapters with rank < this value get zero-padded inside
        # _serialize_adapter. Fall back to vLLM's own default (16) when the
        # user has not pinned a value.
        self._lora_max_rank = (
            config.model.vllm_max_lora_rank or _DEFAULT_VLLM_MAX_LORA_RANK
        )

        # FP8 detection (recipe flag, then HF config sniff) + KV cache
        # dtype auto-default. Extracted so __init__ stays readable.
        is_fp8, kv_cache_dtype = _detect_fp8_and_kv_dtype(
            config, model_id=model_id, trust_remote_code=trust
        )

        # Hand kwargs assembly off to the pure helper so it stays
        # unit-testable without importing vLLM (PR #21 review item 7).
        kwargs = _build_llm_kwargs(
            config,
            model_arch=model_arch,
            is_fp8=is_fp8,
            kv_cache_dtype=kv_cache_dtype,
            lora_max_rank=self._lora_max_rank,
        )

        self.llm = LLM(**kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        self._ensure_chat_template(model_id, trust)

        # Adapter management — use tmpfs (/dev/shm) to avoid disk I/O overhead
        # during per-trial LoRA hot-swap.  Falls back to /tmp if /dev/shm is
        # unavailable (e.g. macOS, containers without tmpfs).
        tmpfs_base = "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()
        self._adapter_dir = os.path.join(
            tempfile.mkdtemp(prefix="abliterix_lora_", dir=tmpfs_base), "current"
        )
        # Use a fixed adapter ID so vLLM treats reloads as the same adapter.
        self._adapter_id = 1
        self._lora_target_modules: list[str] = []  # set during projection cache

        # MoE router suppression is attached lazily by cli.py once the HF
        # phase has identified safety experts.  See set_moe_editor().
        self.moe_editor: Any | None = None
        self.expert_editor: Any | None = None

        print(f"  [green]Ok[/] (vLLM TP={tp})")

    # ------------------------------------------------------------------
    # MoE router suppression (attached after HF safety-expert profiling)
    # ------------------------------------------------------------------

    def set_moe_editor(
        self,
        safety_experts: dict[int, list[tuple[int, float]]],
    ) -> None:
        """Attach a :class:`VLLMMoEEditor` so the optimizer trial loop can
        apply router suppression between generations."""
        from .vllm_moe_editor import VLLMMoEEditor

        self.moe_editor = VLLMMoEEditor(self.llm, safety_experts)
        # Probe routers once up front so we log the router layout and seed
        # self._router_layers for apply().
        self.moe_editor.probe()
        # Pre-install persistent suppression hooks BEFORE any trial forward
        # pass. register_forward_hook calls made AFTER a ``@support_torch_compile``
        # model has already compiled are silently skipped by Dynamo
        # (pytorch/pytorch#117758). With ``enforce_eager=True`` there is no
        # compile, but we still install up-front to avoid that foot-gun.
        try:
            self.moe_editor._ensure_installed()
        except Exception as exc:  # pragma: no cover — defensive
            print(f"  [yellow]Warning: persistent suppression install failed: {exc}[/]")

    def apply_router_suppression(self, n_suppress: int, bias_value: float) -> int:
        """Scale down the router weight rows of the top-N safety experts
        on every TP worker.  No-op if :meth:`set_moe_editor` has not been
        called or ``n_suppress <= 0``.

        Also invalidates the prefix cache so the next generation with the
        modified weights actually re-runs the forward pass.  Without this,
        vLLM replays KV entries captured before the edit and the router
        changes are silently skipped — KL divergence stays exactly 0.0000
        across every trial.
        """
        if self.moe_editor is None:
            return 0
        touched = self.moe_editor.apply(n_suppress=n_suppress, bias_value=bias_value)
        if touched > 0:
            try:
                self.llm.reset_prefix_cache()
            except Exception:
                pass
        return touched

    def restore_router_suppression(self) -> int:
        """Reverse any router row edits applied by the last
        :meth:`apply_router_suppression` call.  Also flushes the prefix
        cache so the next baseline/trial sees the restored weights."""
        if self.moe_editor is None:
            return 0
        touched = self.moe_editor.restore()
        if touched > 0:
            try:
                self.llm.reset_prefix_cache()
            except Exception:
                pass
        return touched

    # ------------------------------------------------------------------
    # Expert-Granular Abliteration (EGA) — in-place expert weight editing
    # ------------------------------------------------------------------

    def set_expert_editor(
        self,
        hidden_dim: int,
        transposed: bool = False,
    ) -> None:
        """Attach a :class:`VLLMExpertEditor` so the optimizer can apply EGA
        projection on fused expert weights between generations.

        Parameters
        ----------
        hidden_dim:
            Size of the residual / hidden dimension the steering vector
            lives in. Required to disambiguate transposed-vs-standard
            ``w2_weight`` layout when ``hidden == intermediate`` (gpt-oss).
        transposed:
            ``True`` for gpt-oss's fused ``down_proj`` layout
            ``(experts, intermediate, hidden)``. ``False`` for the
            standard MoE convention ``(experts, hidden, intermediate)``.
        """
        from .vllm_moe_editor import VLLMExpertEditor

        self.expert_editor = VLLMExpertEditor(
            self.llm, hidden_dim=hidden_dim, transposed=transposed
        )
        self.expert_editor.probe()

    def apply_ega_projection(
        self,
        plan: list[dict[str, Any]],
        norm_preserve: bool = True,
    ) -> dict[str, Any]:
        """Project the refusal direction out of every expert's down_proj
        for the layers listed in ``plan``.

        ``plan`` format — one dict per layer:
            ``{"layer_idx": int, "v": bytes, "strength": float}``
        where ``v`` is ``torch.save``'d 1-D float tensor in hidden dim.

        Caller computes ``v`` + ``strength`` per layer from the decay kernel
        (same math as HF ``_apply_ega_steering``).

        Invalidates the prefix cache so the next generation sees the edited
        weights.
        """
        if getattr(self, "expert_editor", None) is None:
            return {"applied": 0, "errors": ["no expert editor"], "per_layer": []}
        result = self.expert_editor.apply_ega(plan, norm_preserve=norm_preserve)
        if result.get("applied", 0) > 0:
            try:
                self.llm.reset_prefix_cache()
            except Exception:
                pass
        return result

    def restore_expert_weights(self) -> int:
        """Reset every edited ``w2_weight`` to its pristine BF16 backup
        (copied from CPU pinned RAM on each worker). Also flushes the
        prefix cache."""
        if getattr(self, "expert_editor", None) is None:
            return 0
        touched = self.expert_editor.restore()
        if touched > 0:
            try:
                self.llm.reset_prefix_cache()
            except Exception:
                pass
        return touched

    # ------------------------------------------------------------------
    # Direct attention projection (q/k/v/o_proj) under vLLM TP
    # ------------------------------------------------------------------

    def set_attention_editor(self) -> None:
        """Attach a :class:`VLLMAttentionEditor` for in-place attention weight
        edits (orthogonal projection on q/k/v/o_proj under TP). Slices the
        fused ``qkv_proj.weight`` into Q/K/V segments using the model's
        ``q_size``/``kv_size`` attributes."""
        from .vllm_moe_editor import VLLMAttentionEditor

        self.attention_editor = VLLMAttentionEditor(self.llm)
        self.attention_editor.probe()

    def apply_attention_projection(
        self,
        plan: list[dict[str, Any]],
        norm_preserve: bool = True,
    ) -> dict[str, Any]:
        """Project the refusal direction out of attention projections for the
        layers/components listed in ``plan``.

        Each dict: ``{"layer_idx": int, "component": "q_proj"|"k_proj"|"v_proj"
        |"o_proj", "v": bytes, "strength": float}``. Caller handles decay
        kernel + per-component strength.

        Flushes the prefix cache when any layer actually got edited.
        """
        if getattr(self, "attention_editor", None) is None:
            return {"applied": 0, "errors": ["no attention editor"], "per_layer": []}
        result = self.attention_editor.apply(plan, norm_preserve=norm_preserve)
        if result.get("applied", 0) > 0:
            try:
                self.llm.reset_prefix_cache()
            except Exception:
                pass
        return result

    def restore_attention_weights(self) -> int:
        """Reset edited attention weights to pristine from the CPU backup."""
        if getattr(self, "attention_editor", None) is None:
            return 0
        touched = self.attention_editor.restore()
        if touched > 0:
            try:
                self.llm.reset_prefix_cache()
            except Exception:
                pass
        return touched

    # ------------------------------------------------------------------
    # Chat template formatting
    # ------------------------------------------------------------------

    def _ensure_chat_template(self, model_id: str, trust_remote_code: bool) -> None:
        """Copy the HF chat template when vLLM's tokenizer wrapper omits it."""
        if getattr(self.tokenizer, "chat_template", None):
            return
        try:
            from transformers import AutoTokenizer

            hf_tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=trust_remote_code,
            )
        except Exception:
            return

        chat_template = getattr(hf_tokenizer, "chat_template", None)
        if chat_template:
            try:
                self.tokenizer.chat_template = chat_template
            except Exception:
                pass

    @staticmethod
    def _format_prompt_without_template(msg: ChatMessage) -> str:
        """Plain text fallback for tokenizers that do not expose chat_template."""
        if msg.system:
            return f"{msg.system.strip()}\n\nUser: {msg.user}\nAssistant:"
        return f"User: {msg.user}\nAssistant:"

    def _format_prompt(self, msg: ChatMessage) -> str:
        """Format a ChatMessage into a prompt string using the tokenizer's chat template."""
        if not getattr(self.tokenizer, "chat_template", None):
            return self._format_prompt_without_template(msg)

        messages: list[dict[str, str]] = []
        if msg.system:
            messages.append({"role": "system", "content": msg.system})
        messages.append({"role": "user", "content": msg.user})
        kwargs: dict[str, Any] = dict(
            add_generation_prompt=True,
            tokenize=False,
        )
        # Not all tokenizers support enable_thinking (e.g. custom remote code).
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                enable_thinking=False,
                **kwargs,
            )
        except TypeError:
            try:
                return self.tokenizer.apply_chat_template(messages, **kwargs)
            except ValueError as exc:
                if "chat_template" not in str(exc) and "chat template" not in str(exc):
                    raise
        except ValueError as exc:
            if "chat_template" not in str(exc) and "chat template" not in str(exc):
                raise
        return self._format_prompt_without_template(msg)

    def _format_prompts(self, messages: list[ChatMessage]) -> list[str]:
        return [self._format_prompt(m) for m in messages]

    # ------------------------------------------------------------------
    # LoRA adapter serialisation
    # ------------------------------------------------------------------

    def save_adapter(
        self,
        lora_weights: dict[str, tuple[Tensor, Tensor]],
        target_modules: list[str],
        base_model_id: str,
    ) -> str:
        """Serialise LoRA weights to a PEFT-format directory for vLLM.

        Parameters
        ----------
        lora_weights : dict
            Mapping of ``full_module_path`` → ``(lora_A, lora_B)`` tensors.
            lora_A shape: ``(rank, d_in)``, lora_B shape: ``(d_out, rank)``.
        target_modules : list[str]
            Leaf module names targeted by LoRA (e.g. ``["o_proj", "down_proj"]``).
        base_model_id : str
            HuggingFace model ID of the base model.

        Returns
        -------
        str
            Path to the adapter directory.
        """
        if self._lora_disabled:
            # LoRA disabled (e.g. MXFP4 + driver < 575): router suppression
            # is the only steering mechanism.  Return empty string so
            # downstream ``if adapter_path`` checks evaluate False and vLLM
            # generates without a lora_request.
            return ""
        adapter_dir = self._adapter_dir
        # Clear previous adapter files and recreate.
        if os.path.exists(adapter_dir):
            shutil.rmtree(adapter_dir)
        os.makedirs(adapter_dir)

        # Build state dict with PEFT naming convention.
        state_dict: dict[str, Tensor] = {}
        target_rank = self._lora_max_rank
        for module_path, (lora_a, lora_b) in lora_weights.items():
            # vLLM pins every adapter to the engine's max_lora_rank; pad
            # smaller adapters with zeros so they fit. Recipes that want
            # the rank passed through honestly should set
            # ``vllm_max_lora_rank = <actual_rank>`` in the model config.
            rank = lora_a.shape[0]
            if rank < target_rank:
                pad = target_rank - rank
                lora_a = F.pad(lora_a, (0, 0, 0, pad))
                lora_b = F.pad(lora_b, (0, pad, 0, 0))

            peft_key = f"base_model.model.{module_path}"
            # Cast to bf16: vLLM nightly (0.19.2rc1+) asserts
            # `inputs.dtype == lora_a_weights[0].dtype` inside
            # triton_ops/lora_shrink_op.py. Model activations are bf16, so
            # LoRA weights must match. float32 LoRA → AssertionError at runtime.
            state_dict[f"{peft_key}.lora_A.weight"] = (
                lora_a.to(torch.bfloat16).contiguous().cpu()
            )
            state_dict[f"{peft_key}.lora_B.weight"] = (
                lora_b.to(torch.bfloat16).contiguous().cpu()
            )

        save_file(state_dict, os.path.join(adapter_dir, "adapter_model.safetensors"))

        adapter_config = {
            "peft_type": "LORA",
            "base_model_name_or_path": base_model_id,
            "r": target_rank,
            "lora_alpha": target_rank,  # alpha == r → scaling = 1.0
            "target_modules": target_modules,
            "lora_dropout": 0.0,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "inference_mode": True,
        }
        with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as f:
            json.dump(adapter_config, f)

        self._lora_target_modules = target_modules
        return adapter_dir

    # ------------------------------------------------------------------
    # Generation methods (mirrors SteeringEngine API)
    # ------------------------------------------------------------------

    def generate_text(
        self,
        messages: list[ChatMessage],
        skip_special_tokens: bool = False,
        max_new_tokens: int | None = None,
        min_new_tokens: int | None = None,
        adapter_path: str | None = None,
    ) -> list[str]:
        """Generate responses using vLLM with optional LoRA adapter."""
        prompts = self._format_prompts(messages)
        max_tok = max_new_tokens or self.config.inference.max_gen_tokens
        min_tok = min_new_tokens
        if min_tok is None and max_new_tokens is None:
            min_tok = self.config.inference.min_gen_tokens

        if min_tok is not None and min_tok > max_tok:
            raise ValueError(
                f"min_gen_tokens ({min_tok}) cannot exceed max_gen_tokens ({max_tok})"
            )

        sampling_kwargs: dict[str, Any] = {
            "temperature": 0.0,
            "max_tokens": max_tok,
        }
        if min_tok is not None:
            sampling_kwargs["min_tokens"] = min_tok

        params = self._SamplingParams(**sampling_kwargs)

        lora_req = None
        if adapter_path and not self._lora_disabled:
            from vllm.lora.request import LoRARequest

            lora_req = LoRARequest(
                f"steering_{self._adapter_id}",
                self._adapter_id,
                adapter_path,
            )

        outputs = self.llm.generate(prompts, params, lora_request=lora_req)

        results = []
        for out in outputs:
            text = out.outputs[0].text
            if skip_special_tokens:
                # vLLM already strips special tokens by default
                pass
            results.append(text)

        return results

    def generate_text_batched(
        self,
        messages: list[ChatMessage],
        skip_special_tokens: bool = False,
        max_new_tokens: int | None = None,
        min_new_tokens: int | None = None,
        adapter_path: str | None = None,
    ) -> list[str]:
        """Batch generation — vLLM handles batching internally via continuous batching."""
        # vLLM automatically handles batching with PagedAttention,
        # so we pass ALL prompts at once for maximum throughput.
        return self.generate_text(
            messages,
            skip_special_tokens=skip_special_tokens,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            adapter_path=adapter_path,
        )

    def generate_and_score(
        self,
        messages: list[ChatMessage],
        max_new_tokens: int,
        kl_token_count: int,
        skip_special_tokens: bool = False,
        min_new_tokens: int | None = None,
        adapter_path: str | None = None,
    ) -> tuple[list[str], Tensor]:
        """Generate responses AND capture logprobs for KL divergence.

        Under vLLM V1, **sampler logprobs returned by the generation loop
        are unreliable** when weights are edited in place via
        ``collective_rpc`` — they read effectively identical across
        baseline/edited weights even with ``enable_prefix_caching=False``.
        The cache that matters here isn't the block-pool prefix cache but
        something in the logprobs collection path (possibly the sampler's
        own CUDA-level cache).

        Our fallback: read ``prompt_logprobs[-1]`` (next-token distribution
        computed during prefill at the final prompt position).  Prefill is
        always fresh against the current weights, so this gives a real KL
        signal.  Long-form generation drift is captured by
        ``scorer.measure_coherence`` (length_deviation), which is summed
        into the divergence objective with weight 0.5.
        """
        prompts = self._format_prompts(messages)

        k_logprobs = 100

        sampling_kwargs: dict[str, Any] = {
            "temperature": 0.0,
            "max_tokens": max_new_tokens,
            "logprobs": k_logprobs,
            # Capture next-token distribution at every prompt position so we
            # can read prompt_logprobs[-1] as the KL signal.  Sampler
            # logprobs in V1 can read stale across in-place weight edits.
            "prompt_logprobs": k_logprobs,
        }
        if min_new_tokens is not None:
            if min_new_tokens > max_new_tokens:
                raise ValueError(
                    f"min_gen_tokens ({min_new_tokens}) cannot exceed "
                    f"max_gen_tokens ({max_new_tokens})"
                )
            sampling_kwargs["min_tokens"] = min_new_tokens

        params = self._SamplingParams(**sampling_kwargs)

        lora_req = None
        if adapter_path and not self._lora_disabled:
            from vllm.lora.request import LoRARequest

            lora_req = LoRARequest(
                f"steering_{self._adapter_id}",
                self._adapter_id,
                adapter_path,
            )

        outputs = self.llm.generate(prompts, params, lora_request=lora_req)

        responses: list[str] = []
        all_logprobs: list[Tensor] = []

        vocab_size = self.llm.llm_engine.model_config.get_vocab_size()
        import math

        uniform_lp = math.log(1.0 / vocab_size)

        def _safe_sparse_logprobs(sparse_lps: dict[int, Any]) -> Tensor:
            step_vec = torch.full((vocab_size,), -30.0)
            found_finite = False
            for token_id, logprob_obj in sparse_lps.items():
                lp = float(logprob_obj.logprob)
                if not math.isfinite(lp):
                    continue
                step_vec[int(token_id)] = lp
                found_finite = True
            if not found_finite:
                return torch.full((vocab_size,), uniform_lp)
            log_vec = F.log_softmax(step_vec, dim=0)
            if not torch.isfinite(log_vec).all():
                return torch.full((vocab_size,), uniform_lp)
            return log_vec

        for out in outputs:
            responses.append(out.outputs[0].text)

            # Prefer prompt_logprobs[-1] (fresh prefill, reliable under
            # in-place editing).  Walk back to skip any trailing None entries.
            sparse_lps: dict[int, Any] | None = None
            p_lps = getattr(out, "prompt_logprobs", None)
            if p_lps:
                for entry in reversed(p_lps):
                    if entry:
                        sparse_lps = entry
                        break

            # Fallback to generation logprobs only if prefill didn't
            # produce any — shouldn't happen in normal operation.
            if sparse_lps is None:
                token_lps = out.outputs[0].logprobs or []
                n_tokens = min(kl_token_count, len(token_lps))
                if n_tokens == 0:
                    all_logprobs.append(torch.full((vocab_size,), uniform_lp))
                    continue
                per_step: list[Tensor] = []
                for step_lps in token_lps[:n_tokens]:
                    per_step.append(_safe_sparse_logprobs(step_lps))
                all_logprobs.append(torch.stack(per_step).mean(dim=0))
                continue

            # Build sparse log-softmax vector from prompt_logprobs[-1].
            all_logprobs.append(_safe_sparse_logprobs(sparse_lps))

        return responses, torch.stack(all_logprobs)

    def generate_and_score_batched(
        self,
        messages: list[ChatMessage],
        max_new_tokens: int,
        kl_token_count: int,
        skip_special_tokens: bool = False,
        min_new_tokens: int | None = None,
        adapter_path: str | None = None,
    ) -> tuple[list[str], Tensor]:
        """Batched wrapper — vLLM handles batching natively."""
        return self.generate_and_score(
            messages,
            max_new_tokens=max_new_tokens,
            kl_token_count=kl_token_count,
            skip_special_tokens=skip_special_tokens,
            min_new_tokens=min_new_tokens,
            adapter_path=adapter_path,
        )

    def compute_logprobs_batched(
        self,
        messages: list[ChatMessage],
        adapter_path: str | None = None,
    ) -> Tensor:
        """Compute next-token logprobs (KL measurement).

        For vLLM, we generate 1 token and capture its logprobs.
        """
        _, logprobs = self.generate_and_score(
            messages,
            max_new_tokens=self.config.kl.token_count,
            kl_token_count=self.config.kl.token_count,
            adapter_path=adapter_path,
        )
        return logprobs

    def score_continuations_nll(
        self,
        messages: list[ChatMessage],
        continuations: list[str],
        adapter_path: str | None = None,
    ) -> Tensor:
        """Score fixed continuations with a fresh prefill pass.

        vLLM V1 sampler logprobs can stay effectively unchanged after
        ``collective_rpc`` in-place edits, which made Gemma 4 31B report
        KL=0.0000 even when generations changed.  Prompt logprobs are computed
        during prefill, so scoring a fixed baseline continuation gives a
        reliable damage signal for in-place runs.

        Returns mean negative log-likelihood per continuation token, shape
        ``(batch,)``.  Missing token logprobs are floored at 30 nats, which is
        conservative and finite for heavily damaged trials.
        """
        if len(messages) != len(continuations):
            raise ValueError(
                "messages and continuations must have the same length "
                f"({len(messages)} != {len(continuations)})"
            )

        prompts = self._format_prompts(messages)
        full_prompts = [p + c for p, c in zip(prompts, continuations)]

        prompt_lens: list[int] = []
        for prompt in prompts:
            try:
                token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            except TypeError:
                token_ids = self.tokenizer.encode(prompt)
            prompt_lens.append(len(token_ids))

        params = self._SamplingParams(
            temperature=0.0,
            max_tokens=1,
            prompt_logprobs=100,
        )

        lora_req = None
        if adapter_path and not self._lora_disabled:
            from vllm.lora.request import LoRARequest

            lora_req = LoRARequest(
                f"steering_{self._adapter_id}",
                self._adapter_id,
                adapter_path,
            )

        outputs = self.llm.generate(full_prompts, params, lora_request=lora_req)

        nlls: list[Tensor] = []
        for out, prompt_len in zip(outputs, prompt_lens):
            prompt_token_ids = list(getattr(out, "prompt_token_ids", None) or [])
            prompt_logprobs = list(getattr(out, "prompt_logprobs", None) or [])
            if not prompt_token_ids or not prompt_logprobs:
                nlls.append(torch.tensor(30.0))
                continue

            start = min(max(prompt_len, 1), len(prompt_token_ids))
            losses: list[float] = []
            for idx in range(start, len(prompt_token_ids)):
                if idx >= len(prompt_logprobs):
                    break
                entry = prompt_logprobs[idx]
                if not entry:
                    continue
                token_id = int(prompt_token_ids[idx])
                lp_obj = entry.get(token_id)
                if lp_obj is None:
                    losses.append(30.0)
                    continue
                lp = float(lp_obj.logprob)
                if not math.isfinite(lp):
                    losses.append(30.0)
                    continue
                losses.append(max(-lp, 0.0))

            if losses:
                nlls.append(torch.tensor(sum(losses) / len(losses)))
            else:
                nlls.append(torch.tensor(30.0))

        return torch.stack(nlls)


class ProjectionCache:
    """Pre-computed ``v @ W`` projections for all layer/component/vector combinations.

    Created during Phase 1 (HF model loaded), used during Phase 2 (vLLM)
    to build LoRA adapters without needing access to base model weights.
    """

    def __init__(self):
        # projections[layer_idx][component_name] = {
        #     "vW": Tensor (hidden_dim,) or (d_in,),  # v @ W for per-layer vector
        #     "module_path": str,  # full path for PEFT state dict
        #     "d_out": int,
        #     "d_in": int,
        # }
        self.projections: dict[int, dict[str, dict[str, Any]]] = {}
        self.steering_vectors: Tensor | None = None
        self.target_modules: list[str] = []

    @staticmethod
    def build_from_safetensors(
        config: "AbliterixConfig",
        steering_vectors: Tensor,
    ) -> "ProjectionCache":
        """Build projection cache directly from safetensors files on disk.

        This avoids loading the full HF model (3+ min for 230GB MoE models),
        instead reading only the steerable weight tensors from the safetensors
        files and computing ``sv @ W`` projections one tensor at a time.

        For MiniMax-M2.5 (256 experts × 62 layers), this reads ~15,872 weight
        tensors but only keeps one in memory at a time.
        """
        import json as _json
        import re
        from pathlib import Path
        from huggingface_hub import snapshot_download
        from safetensors import safe_open
        from transformers import AutoConfig

        cache = ProjectionCache()
        cache.steering_vectors = steering_vectors.cpu()
        sv = cache.steering_vectors

        model_id = config.model.model_id
        trust = config.model.trust_remote_code or False

        # Resolve model directory (local path or HF cache).
        model_dir = Path(model_id)
        if not model_dir.is_dir():
            model_dir = Path(snapshot_download(model_id, allow_patterns=["*.json"]))
            # Ensure safetensors are downloaded too.
            snapshot_download(model_id, allow_patterns=["*.safetensors"])
            model_dir = Path(snapshot_download(model_id))

        # Load model config for architecture info.
        auto_cfg = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=trust)
        text_cfg = getattr(auto_cfg, "text_config", auto_cfg)
        n_layers = text_cfg.num_hidden_layers

        # Load FP8 quantization info.
        qcfg = getattr(text_cfg, "quantization_config", None)
        if qcfg is None:
            cfg_path = model_dir / "config.json"
            if cfg_path.exists():
                with open(cfg_path) as f:
                    raw = _json.load(f)
                qcfg = raw.get("quantization_config", {})
        if not isinstance(qcfg, dict):
            qcfg = getattr(qcfg, "__dict__", {})
        is_fp8 = qcfg.get("quant_method") == "fp8"

        # Load safetensors index.
        index_path = model_dir / "model.safetensors.index.json"
        if index_path.exists():
            with open(index_path) as f:
                weight_map = _json.load(f)["weight_map"]
        else:
            # Single-file model.
            st_file = next(model_dir.glob("*.safetensors"))
            with safe_open(str(st_file), framework="pt") as f:
                weight_map = {k: st_file.name for k in f.keys()}

        # Discover steerable weight keys using naming patterns.
        # These match the patterns in engine.steerable_modules():
        #
        # MoE expert LoRA (2026-04-19 rewrite): previously MoE expert paths
        # were skipped because vLLM's ``PackedLoRALayerWeights.pack_moe``
        # asserts that LoRAs exist for all three expert projections
        # (``w1/w2/w3 = gate/up/down_proj``) and crashes Phase 2 init when
        # only down_proj is present.
        #
        # Fix: collect all three projections per expert; build a real refusal
        # projection only for the down/w2 path (that's the residual-write
        # path), and emit *zero* LoRA companions for gate/up and w1/w3 so
        # pack_moe is satisfied. Zero lora_B means the companion has no
        # effect on forward, but its presence in the state_dict prevents the
        # assert. See :meth:`build_lora_weights` for the zero-companion logic.
        #
        # Naming covered:
        #   MiniMax-M2 / Phi-3.5-MoE: ``experts.<e>.w1 / w2 / w3``
        #   Qwen / DeepSeek / Llama:  ``experts.<e>.gate_proj / up_proj / down_proj``
        _STEERABLE_PATTERNS = [
            # attn.o_proj: standard self-attention output (residual-output)
            (r"model\.layers\.(\d+)\.self_attn\.o_proj\.weight$", "attn.o_proj"),
            # attn.o_proj: GatedDeltaNet linear attention (residual-output)
            (r"model\.layers\.(\d+)\.linear_attn\.out_proj\.weight$", "attn.o_proj"),
            # attn.{q,k,v}_proj: standard self-attention inputs (residual-input).
            # Shape is (d_out, hidden_dim) where d_out = n_heads*head_dim (q)
            # or n_kv_heads*head_dim (k/v). Safetensors path handles the NON-fused
            # on-disk layout; vLLM may internally fuse to qkv_proj at TP load, and
            # vLLM's Punica wrapper packs per-component LoRAs into the fused
            # adapter automatically — we emit one LoRA per HF module path.
            (r"model\.layers\.(\d+)\.self_attn\.q_proj\.weight$", "attn.q_proj"),
            (r"model\.layers\.(\d+)\.self_attn\.k_proj\.weight$", "attn.k_proj"),
            (r"model\.layers\.(\d+)\.self_attn\.v_proj\.weight$", "attn.v_proj"),
            # mlp.down_proj: dense MLP (non-MoE layers)
            (r"model\.layers\.(\d+)\.mlp\.down_proj\.weight$", "mlp.down_proj"),
            # MoE expert down projection (w2 / down_proj) — real steering here.
            (
                r"model\.layers\.(\d+)\.(?:block_sparse_moe|mlp)\.experts\.\d+\.w2\.weight$",
                "mlp.down_proj",
            ),
            (
                r"model\.layers\.(\d+)\.(?:block_sparse_moe|mlp)\.experts\.\d+\.down_proj\.weight$",
                "mlp.down_proj",
            ),
            # MoE expert gate + up projections — zero-LoRA companions (no
            # real steering; required only to satisfy vLLM pack_moe).
            (
                r"model\.layers\.(\d+)\.(?:block_sparse_moe|mlp)\.experts\.\d+\.w1\.weight$",
                "moe.expert_gate",
            ),
            (
                r"model\.layers\.(\d+)\.(?:block_sparse_moe|mlp)\.experts\.\d+\.w3\.weight$",
                "moe.expert_up",
            ),
            (
                r"model\.layers\.(\d+)\.(?:block_sparse_moe|mlp)\.experts\.\d+\.gate_proj\.weight$",
                "moe.expert_gate",
            ),
            (
                r"model\.layers\.(\d+)\.(?:block_sparse_moe|mlp)\.experts\.\d+\.up_proj\.weight$",
                "moe.expert_up",
            ),
        ]

        disabled_components = set(config.steering.disabled_components)
        # Zero-LoRA companions are only needed when mlp.down_proj is active.
        # If a profile disables expert/down steering, skip w1/w3 entirely so
        # attention-only configs do not build a huge MoE adapter shell.
        _ZERO_COMPANION_COMPONENTS = {"moe.expert_gate", "moe.expert_up"}
        include_zero_companions = "mlp.down_proj" not in disabled_components

        # Group steerable keys by (layer_idx, component).
        steerable_keys: dict[int, dict[str, list[str]]] = {}
        compiled = [(re.compile(p), comp) for p, comp in _STEERABLE_PATTERNS]

        for key in weight_map:
            for regex, component in compiled:
                m = regex.match(key)
                if m:
                    if component in disabled_components:
                        break
                    if (
                        component in _ZERO_COMPANION_COMPONENTS
                        and not include_zero_companions
                    ):
                        break
                    layer_idx = int(m.group(1))
                    if layer_idx < n_layers:
                        steerable_keys.setdefault(layer_idx, {}).setdefault(
                            component, []
                        ).append(key)
                    break

        # Pre-compute sv @ W for each steerable weight.
        target_module_names: set[str] = set()
        _open_files: dict[str, Any] = {}  # cache file handles

        def _get_tensor(key: str) -> Tensor:
            shard = weight_map[key]
            if shard not in _open_files:
                _open_files[shard] = safe_open(
                    str(model_dir / shard), framework="pt", device="cpu"
                )
            return _open_files[shard].get_tensor(key)

        for layer_idx in sorted(steerable_keys):
            cache.projections[layer_idx] = {}
            for component, keys in steerable_keys[layer_idx].items():
                # Companion paths: record shape per-module only, no projection.
                # We aggregate ALL expert paths for this component into a list
                # so build_lora_weights can emit zero-LoRAs for every one.
                if component in _ZERO_COMPANION_COMPONENTS:
                    companions: list[dict] = []
                    for wkey in keys:
                        # Read only the weight shape from safetensors; no data.
                        with safe_open(
                            str(model_dir / weight_map[wkey]),
                            framework="pt",
                            device="cpu",
                        ) as f:
                            slc = f.get_slice(wkey)
                            shape = slc.get_shape()
                        d_out, d_in = shape[0], shape[1] if len(shape) > 1 else shape[0]
                        module_path = wkey.rsplit(".weight", 1)[0]
                        leaf = module_path.split(".")[-1]
                        target_module_names.add(leaf)
                        companions.append(
                            {
                                "module_path": module_path,
                                "d_out": d_out,
                                "d_in": d_in,
                            }
                        )
                    cache.projections[layer_idx][component] = {
                        "companions": companions,
                    }
                    continue

                # Real steering components: compute sv @ W as before.
                # For MoE mlp.down_proj the component holds ONE entry per
                # expert — stored as a list under "experts" rather than as
                # a single module, so build_lora_weights can iterate all.
                is_moe_expert_down = component == "mlp.down_proj" and len(keys) > 1
                entries: list[dict] = []

                for wkey in keys:
                    # Read weight tensor.
                    w_raw = _get_tensor(wkey)

                    # Dequantize FP8 if needed.
                    _FP8 = {torch.float8_e4m3fn, torch.float8_e5m2}
                    if is_fp8 and w_raw.dtype in _FP8:
                        # Check both scale tensor naming conventions:
                        # weight_scale_inv (DeepSeek/Qwen/MiniMax) and weight_scale.
                        scale_key_inv = wkey.replace(".weight", ".weight_scale_inv")
                        scale_key_fwd = wkey.replace(".weight", ".weight_scale")
                        is_inv = True
                        if scale_key_inv in weight_map:
                            scale_key = scale_key_inv
                        elif scale_key_fwd in weight_map:
                            scale_key = scale_key_fwd
                            is_inv = False
                        else:
                            scale_key = None

                        if scale_key is not None:
                            scale = _get_tensor(scale_key).float()
                            w_f = w_raw.to(torch.bfloat16).float()
                            # Block-wise FP8: scale shape is (rows/block, cols/block).
                            # Expand both dims to match weight shape.
                            block_r = max(1, w_f.shape[0] // scale.shape[0])
                            block_c = max(1, w_f.shape[1] // scale.shape[1])
                            s_exp = scale.repeat_interleave(
                                block_r, dim=0
                            ).repeat_interleave(block_c, dim=1)
                            s_exp = s_exp[: w_f.shape[0], : w_f.shape[1]]
                            if is_inv:
                                W = (w_f * s_exp).to(torch.float32)
                            else:
                                W = (w_f / s_exp).to(torch.float32)
                        else:
                            W = w_raw.to(torch.float32)
                    else:
                        W = w_raw.to(torch.float32)
                    del w_raw

                    W = W.view(W.shape[0], -1)
                    d_out, d_in = W.shape
                    hidden_dim = sv.shape[1]

                    # Derive the module path for PEFT state dict.
                    # Strip ".weight" suffix → "model.layers.X.self_attn.o_proj"
                    module_path = wkey.rsplit(".weight", 1)[0]
                    leaf = module_path.split(".")[-1]
                    target_module_names.add(leaf)

                    # Dispatch on which axis matches hidden_dim — see the
                    # corresponding comment in ``ProjectionCache.build`` for
                    # the math.
                    if d_out == hidden_dim:
                        direction = "output"
                        # (n_vec, d_out) @ (d_out, d_in) = (n_vec, d_in)
                        vW_all = (sv @ W).cpu()
                    elif d_in == hidden_dim:
                        direction = "input"
                        # (n_vec, d_in) @ (d_in, d_out) = (n_vec, d_out)
                        vW_all = (sv @ W.t()).cpu()
                    else:
                        del W
                        continue
                    del W

                    entries.append(
                        {
                            "vW_all": vW_all,
                            "module_path": module_path,
                            "d_out": d_out,
                            "d_in": d_in,
                            "direction": direction,
                        }
                    )

                if is_moe_expert_down:
                    # Store as list of per-expert entries.
                    cache.projections[layer_idx][component] = {
                        "experts": entries,
                    }
                else:
                    # Single entry (backward-compat with non-MoE path).
                    cache.projections[layer_idx][component] = entries[0]

        # Close file handles.
        _open_files.clear()

        if not cache.projections:
            raise RuntimeError(
                f"build_from_safetensors found 0 steerable weight keys in "
                f"{model_dir}.  The model's weight naming may not match the "
                f"expected patterns.  Run scripts/verify_minimax_m25.py to "
                f"diagnose, or fall back to HF model loading (remove speculators)."
            )

        cache.target_modules = sorted(target_module_names)

        def _cache_entry_count(info: dict[str, Any]) -> int:
            if "experts" in info:
                return len(info["experts"])
            if "companions" in info:
                return len(info["companions"])
            return 1

        def _cache_entry_nbytes(info: dict[str, Any]) -> int:
            if "vW_all" in info:
                return info["vW_all"].nbytes
            if "experts" in info:
                return sum(e["vW_all"].nbytes for e in info["experts"])
            # Zero-LoRA companions store shape metadata only.
            return 0

        n_cached = sum(
            _cache_entry_count(info)
            for layer in cache.projections.values()
            for info in layer.values()
        )
        cache_mb = (
            sum(
                _cache_entry_nbytes(info)
                for layer in cache.projections.values()
                for info in layer.values()
            )
            / 1024
            / 1024
        )
        print(
            f"* Projection cache (safetensors): {n_cached} modules across "
            f"{n_layers} layers ({cache_mb:.0f} MB)"
        )
        return cache

    @staticmethod
    def build(engine, steering_vectors: Tensor) -> "ProjectionCache":
        """Pre-compute all projections while the HF model is loaded.

        For each layer and component, compute ``sv[k] @ W`` for **every**
        steering vector *k* (not just the layer's own vector).  This allows
        :meth:`build_lora_weights` to reconstruct the exact ``v_global @ W``
        for arbitrary global vectors via the linearity of matrix multiplication:

        .. math::

           v_{\\text{global}} @ W
           = \\frac{(1-f)\\,(\\text{sv}[a] @ W) + f\\,(\\text{sv}[a+1] @ W)}
                  {\\|(1-f)\\,\\text{sv}[a] + f\\,\\text{sv}[a+1]\\|}
        """
        from .steering import _dequantize_fp8_blockwise, _FP8_DTYPES

        cache = ProjectionCache()
        cache.steering_vectors = steering_vectors.cpu()

        import bitsandbytes as bnb
        from peft.tuners.lora.layer import Linear
        from typing import cast

        target_module_names: set[str] = set()
        n_layers = len(engine.transformer_layers)
        steering_vectors.shape[0]  # n_layers + 1

        # Pre-move steering vectors to each GPU device once to avoid
        # repeated .to(device) calls inside the triple-nested loop.
        _sv_by_device: dict[torch.device, Tensor] = {}

        for layer_idx in range(n_layers):
            cache.projections[layer_idx] = {}

            for component, modules in engine.steerable_modules(layer_idx).items():
                for mod in modules:
                    mod = cast(Linear, mod)

                    # Get the full module path for PEFT state dict keys.
                    module_path = None
                    for name, m in engine.model.named_modules():
                        if m is mod:
                            module_path = name
                            break

                    if module_path is None:
                        continue

                    # Extract leaf name for target_modules.
                    leaf = module_path.split(".")[-1]
                    target_module_names.add(leaf)

                    # Dequantise weights and compute projection immediately.
                    # NOTE: we do NOT cache dequantised weights — for MoE models
                    # with 256 experts × 62 layers, caching all float32 weights
                    # on GPU causes OOM.  Instead, dequant → project → free.
                    base_layer = getattr(mod, "base_layer", mod)
                    base_weight = cast(Tensor, base_layer.weight)
                    qs = getattr(base_weight, "quant_state", None)
                    CB = getattr(base_weight, "CB", None)

                    if qs is not None:
                        W = cast(
                            Tensor,
                            bnb.functional.dequantize_4bit(
                                base_weight.data,
                                qs,
                            ).to(torch.float32),
                        )
                    elif CB is not None:
                        SCB = base_weight.SCB
                        W = CB.float() * SCB.float().unsqueeze(1) / 127.0
                    elif _FP8_DTYPES and base_weight.dtype in _FP8_DTYPES:
                        weight_scale = getattr(base_layer, "weight_scale", None)
                        if weight_scale is not None:
                            W = _dequantize_fp8_blockwise(
                                base_weight.data, weight_scale
                            )
                        else:
                            W = base_weight.to(torch.float32)
                    else:
                        W = base_weight.to(torch.float32)

                    W = W.view(W.shape[0], -1)
                    d_out, d_in = W.shape[0], W.shape[1]
                    hidden_dim = steering_vectors.shape[1]

                    # Determine projection direction based on which axis of W
                    # matches hidden_dim:
                    #   - "output" (o_proj, down_proj): d_out == hidden_dim.
                    #     ``W_new = (I - v v^T) W = W - v (v^T W)``.
                    #     Cache ``vW = sv @ W`` → shape ``(n_vec, d_in)``.
                    #     Later: ``lora_A = vW (1, d_in)``, ``lora_B = -s*v (d_out, 1)``.
                    #   - "input"  (q/k/v_proj, gate/up_proj): d_in == hidden_dim.
                    #     ``W_new = W (I - v v^T) = W - (W v) v^T``.
                    #     Cache ``Wv = sv @ W.T`` → shape ``(n_vec, d_out)``.
                    #     Later: ``lora_A = v (1, d_in)``, ``lora_B = -s*Wv (d_out, 1)``.
                    if d_out == hidden_dim:
                        direction = "output"
                    elif d_in == hidden_dim:
                        direction = "input"
                    else:
                        # Neither axis matches — not steerable with this refusal
                        # direction. Skip (shouldn't happen for standard archs).
                        del W
                        continue

                    device = W.device
                    if device not in _sv_by_device:
                        _sv_by_device[device] = steering_vectors.to(device)
                    sv_dev = _sv_by_device[device]
                    if direction == "output":
                        # (n_vec, d_out) @ (d_out, d_in) = (n_vec, d_in)
                        vW_all = (sv_dev @ W).cpu()
                    else:
                        # (n_vec, d_in=hidden_dim) @ (d_in, d_out) = (n_vec, d_out)
                        vW_all = (sv_dev @ W.t()).cpu()
                    del W  # free immediately to avoid OOM on large MoE models

                    cache.projections[layer_idx][component] = {
                        "vW_all": vW_all,
                        "module_path": module_path,
                        "d_out": d_out,
                        "d_in": d_in,
                        "direction": direction,
                    }

        cache.target_modules = sorted(target_module_names)
        n_cached = sum(len(v) for v in cache.projections.values())
        cache_mb = (
            sum(
                info["vW_all"].nbytes
                for layer in cache.projections.values()
                for info in layer.values()
            )
            / 1024
            / 1024
        )
        print(
            f"* Projection cache: {n_cached} modules across {n_layers} layers "
            f"({cache_mb:.0f} MB)"
        )

        return cache

    def build_lora_weights(
        self,
        profiles: dict[str, Any],
        vector_index: float | None,
        config: AbliterixConfig,
    ) -> dict[str, tuple[Tensor, Tensor]]:
        """Construct LoRA adapter weights from cached projections.

        Returns a dict mapping module paths to (lora_A, lora_B) tuples,
        ready for serialisation via :meth:`VLLMGenerator.save_adapter`.
        """
        import math
        from ..types import DecayKernel

        kernel = config.steering.decay_kernel
        sv = self.steering_vectors
        assert sv is not None

        # Resolve global vector indices if applicable.
        # For global mode we reconstruct v_global @ W exactly using linearity:
        #   v_global @ W = ((1-f)*sv[a] + f*sv[a+1]) @ W / norm
        #                = ((1-f)*vW_all[a] + f*vW_all[a+1]) / norm
        global_vector: Tensor | None = None
        global_idx_a: int = 0
        global_frac: float = 0.0
        global_norm: float = 1.0

        if vector_index is not None:
            global_frac, integral = math.modf(vector_index + 1)
            global_idx_a = int(integral)
            v_unnorm = (1 - global_frac) * sv[global_idx_a] + global_frac * sv[
                global_idx_a + 1
            ]
            global_norm = v_unnorm.norm().item()
            global_vector = v_unnorm / global_norm if global_norm > 0 else v_unnorm

        lora_weights: dict[str, tuple[Tensor, Tensor]] = {}
        n_layers = len(self.projections)

        _ZERO_COMPANIONS = {"moe.expert_gate", "moe.expert_up"}

        def _one_projection(info: dict, strength: float) -> tuple[Tensor, Tensor]:
            """Compute (lora_A, lora_B) for a single real-steering module.

            Dispatches on ``info["direction"]`` to produce the correct rank-1
            update for either residual-output (o_proj, down_proj) or
            residual-input (q/k/v_proj, gate/up_proj) modules. See
            :meth:`ProjectionCache.build` for the math.
            """
            vW_all = info["vW_all"]
            # Legacy caches (pre-residual-input support) have no "direction"
            # field — assume output-side, matching the old behaviour.
            direction = info.get("direction", "output")

            if global_vector is not None:
                v = global_vector
                vW = (
                    (1 - global_frac) * vW_all[global_idx_a]
                    + global_frac * vW_all[global_idx_a + 1]
                ) / global_norm
            else:
                v = F.normalize(sv[layer_idx + 1], p=2, dim=0)
                vW = vW_all[layer_idx + 1]

            if direction == "output":
                # W_new = W - v (v^T W) · s  → B=-s*v (d_out,1), A=vW (1,d_in)
                lora_A = vW.view(1, -1)
                lora_B = (-strength * v[: info["d_out"]]).view(-1, 1)
            else:  # direction == "input"
                # W_new = W - (W v) v^T · s  → B=-s*Wv (d_out,1), A=v (1,d_in)
                lora_A = v[: info["d_in"]].view(1, -1)
                lora_B = (-strength * vW).view(-1, 1)
            return lora_A, lora_B

        for layer_idx in range(n_layers):
            if layer_idx not in self.projections:
                continue

            # Compute strength per real component once per layer; zero
            # companions inherit the *same* strength decision so that when
            # down_proj is steered, w1/w3 zero-LoRAs also exist for this
            # layer (required for vLLM pack_moe to find all three).
            mlp_active = False
            for component, info in self.projections[layer_idx].items():
                # Companions are emitted after the main loop (zero-LoRA
                # pack_moe placeholders — no real projection).
                if component in _ZERO_COMPANIONS:
                    continue
                if component not in profiles:
                    continue

                sp = profiles[component]
                distance = abs(layer_idx - sp.max_weight_position)
                if distance > sp.min_weight_distance:
                    continue

                t = distance / sp.min_weight_distance
                if kernel == DecayKernel.GAUSSIAN:
                    strength = sp.min_weight + (
                        sp.max_weight - sp.min_weight
                    ) * math.exp(-2.0 * t * t)
                elif kernel == DecayKernel.COSINE:
                    strength = sp.min_weight + (sp.max_weight - sp.min_weight) * (
                        0.5 * (1.0 + math.cos(math.pi * t))
                    )
                else:
                    strength = sp.max_weight + t * (sp.min_weight - sp.max_weight)

                # MoE case: info has "experts" list of per-expert entries.
                # Dense case: info is itself the single entry.
                if "experts" in info:
                    for expert_info in info["experts"]:
                        lora_A, lora_B = _one_projection(expert_info, strength)
                        lora_weights[expert_info["module_path"]] = (lora_A, lora_B)
                    if component == "mlp.down_proj":
                        mlp_active = True
                else:
                    lora_A, lora_B = _one_projection(info, strength)
                    lora_weights[info["module_path"]] = (lora_A, lora_B)
                    if component == "mlp.down_proj":
                        mlp_active = True

            # Zero-LoRA companions for gate/up (w1/w3): emit only when the
            # layer's mlp.down_proj (w2) is actually being steered. This
            # keeps the adapter size bounded to layers that do real work.
            if not mlp_active:
                continue
            for comp in _ZERO_COMPANIONS:
                companion_info = self.projections[layer_idx].get(comp)
                if companion_info is None:
                    continue
                for c in companion_info["companions"]:
                    lora_A_zero = torch.zeros(1, c["d_in"], dtype=torch.float32)
                    lora_B_zero = torch.zeros(c["d_out"], 1, dtype=torch.float32)
                    lora_weights[c["module_path"]] = (lora_A_zero, lora_B_zero)

        return lora_weights
