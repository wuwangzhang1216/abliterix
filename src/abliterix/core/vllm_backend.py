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


# Minimum LoRA rank supported by vLLM.
_VLLM_MIN_RANK = 8


class VLLMGenerator:
    """vLLM-backed text generator with LoRA adapter hot-swapping.

    This class mirrors the generation API of :class:`SteeringEngine`
    (``generate_text_batched``, ``generate_and_score_batched``,
    ``compute_logprobs_batched``) so callers can use it interchangeably.
    """

    def __init__(self, config: AbliterixConfig):
        from vllm import LLM, SamplingParams  # noqa: F811

        self.config = config
        self._SamplingParams = SamplingParams

        tp = config.model.tensor_parallel_size
        if tp is None:
            tp = torch.cuda.device_count()

        model_id = config.model.model_id
        trust = config.model.trust_remote_code or False

        print(f"* Loading model in vLLM with TP={tp}...")

        kwargs: dict[str, Any] = dict(
            model=model_id,
            tensor_parallel_size=tp,
            gpu_memory_utilization=config.model.gpu_memory_utilization,
            trust_remote_code=trust,
            enable_lora=True,
            max_lora_rank=_VLLM_MIN_RANK,
            max_loras=1,
            max_cpu_loras=2,
            enforce_eager=True,  # safest for per-trial LoRA hot-swap
            enable_expert_parallel=True,  # EP for MoE: better than TP for expert layers
        )

        # FP8 models: let vLLM handle quantisation natively.
        if config.model.quant_method and config.model.quant_method.value == "fp8":
            kwargs["quantization"] = "fp8"
            # FP8 KV cache halves KV memory with negligible quality loss on H100.
            kwargs["kv_cache_dtype"] = "fp8_e4m3"

        self.llm = LLM(**kwargs)
        self.tokenizer = self.llm.get_tokenizer()

        # Adapter management — reuse a single directory to avoid disk bloat.
        self._adapter_dir = os.path.join(
            tempfile.mkdtemp(prefix="abliterix_lora_"), "current"
        )
        # Use a fixed adapter ID so vLLM treats reloads as the same adapter.
        self._adapter_id = 1
        self._lora_target_modules: list[str] = []  # set during projection cache

        print(f"  [green]Ok[/] (vLLM TP={tp})")

    # ------------------------------------------------------------------
    # Chat template formatting
    # ------------------------------------------------------------------

    def _format_prompt(self, msg: ChatMessage) -> str:
        """Format a ChatMessage into a prompt string using the tokenizer's chat template."""
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
                messages, enable_thinking=False, **kwargs,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, **kwargs)

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
        adapter_dir = self._adapter_dir
        # Clear previous adapter files and recreate.
        if os.path.exists(adapter_dir):
            shutil.rmtree(adapter_dir)
        os.makedirs(adapter_dir)

        # Build state dict with PEFT naming convention.
        state_dict: dict[str, Tensor] = {}
        for module_path, (lora_a, lora_b) in lora_weights.items():
            # Pad rank-1 to rank-8 for vLLM compatibility.
            rank = lora_a.shape[0]
            if rank < _VLLM_MIN_RANK:
                pad = _VLLM_MIN_RANK - rank
                lora_a = F.pad(lora_a, (0, 0, 0, pad))  # (8, d_in)
                lora_b = F.pad(lora_b, (0, pad, 0, 0))  # (d_out, 8)

            peft_key = f"base_model.model.{module_path}"
            state_dict[f"{peft_key}.lora_A.weight"] = lora_a.contiguous().cpu()
            state_dict[f"{peft_key}.lora_B.weight"] = lora_b.contiguous().cpu()

        save_file(state_dict, os.path.join(adapter_dir, "adapter_model.safetensors"))

        adapter_config = {
            "peft_type": "LORA",
            "base_model_name_or_path": base_model_id,
            "r": _VLLM_MIN_RANK,
            "lora_alpha": _VLLM_MIN_RANK,  # alpha == r → scaling = 1.0
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
        adapter_path: str | None = None,
    ) -> list[str]:
        """Generate responses using vLLM with optional LoRA adapter."""
        from vllm.lora.request import LoRARequest

        prompts = self._format_prompts(messages)
        max_tok = max_new_tokens or self.config.inference.max_gen_tokens

        params = self._SamplingParams(
            temperature=0.0,
            max_tokens=max_tok,
        )

        lora_req = None
        if adapter_path:
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
        adapter_path: str | None = None,
    ) -> list[str]:
        """Batch generation — vLLM handles batching internally via continuous batching."""
        # vLLM automatically handles batching with PagedAttention,
        # so we pass ALL prompts at once for maximum throughput.
        return self.generate_text(
            messages,
            skip_special_tokens=skip_special_tokens,
            max_new_tokens=max_new_tokens,
            adapter_path=adapter_path,
        )

    def generate_and_score(
        self,
        messages: list[ChatMessage],
        max_new_tokens: int,
        kl_token_count: int,
        skip_special_tokens: bool = False,
        adapter_path: str | None = None,
    ) -> tuple[list[str], Tensor]:
        """Generate responses AND capture logprobs for KL divergence.

        Uses vLLM's ``logprobs`` parameter to capture top-K token log
        probabilities, then builds a sparse approximation of the full
        log-probability distribution for KL computation.
        """
        from vllm.lora.request import LoRARequest

        prompts = self._format_prompts(messages)

        # Request logprobs for KL approximation.
        # Top-100 covers >99.9% of probability mass for typical LLM distributions.
        k_logprobs = 100

        params = self._SamplingParams(
            temperature=0.0,
            max_tokens=max_new_tokens,
            logprobs=k_logprobs,
        )

        lora_req = None
        if adapter_path:
            lora_req = LoRARequest(
                f"steering_{self._adapter_id}",
                self._adapter_id,
                adapter_path,
            )

        outputs = self.llm.generate(prompts, params, lora_request=lora_req)

        # Extract responses and build logprob tensors.
        responses: list[str] = []
        all_logprobs: list[Tensor] = []

        vocab_size = self.llm.llm_engine.model_config.get_vocab_size()

        for out in outputs:
            responses.append(out.outputs[0].text)

            # Aggregate logprobs from the first kl_token_count generated tokens.
            token_lps = out.outputs[0].logprobs or []
            n_tokens = min(kl_token_count, len(token_lps))

            if n_tokens == 0:
                # Fallback: uniform log-probability distribution.
                # Use log(1/V) instead of -inf to avoid NaN in KL computation.
                import math
                all_logprobs.append(
                    torch.full((vocab_size,), math.log(1.0 / vocab_size))
                )
                continue

            # Build per-position sparse log-softmax vectors, then take the
            # arithmetic mean — matching HF's mean(log_softmax(scores_i)).
            per_step: list[Tensor] = []
            for step_lps in token_lps[:n_tokens]:
                step_vec = torch.full((vocab_size,), -30.0)
                for token_id, logprob_obj in step_lps.items():
                    step_vec[token_id] = logprob_obj.logprob
                # Renormalise each position independently.
                per_step.append(F.log_softmax(step_vec, dim=0))

            all_logprobs.append(torch.stack(per_step).mean(dim=0))

        return responses, torch.stack(all_logprobs)

    def generate_and_score_batched(
        self,
        messages: list[ChatMessage],
        max_new_tokens: int,
        kl_token_count: int,
        skip_special_tokens: bool = False,
        adapter_path: str | None = None,
    ) -> tuple[list[str], Tensor]:
        """Batched wrapper — vLLM handles batching natively."""
        return self.generate_and_score(
            messages,
            max_new_tokens,
            kl_token_count,
            skip_special_tokens,
            adapter_path,
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
        n_vectors = steering_vectors.shape[0]  # n_layers + 1

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
                    base_weight = cast(Tensor, mod.base_layer.weight)
                    qs = getattr(base_weight, "quant_state", None)
                    CB = getattr(base_weight, "CB", None)

                    if qs is not None:
                        W = cast(
                            Tensor,
                            bnb.functional.dequantize_4bit(
                                base_weight.data, qs,
                            ).to(torch.float32),
                        )
                    elif CB is not None:
                        SCB = base_weight.SCB
                        W = CB.float() * SCB.float().unsqueeze(1) / 127.0
                    elif _FP8_DTYPES and base_weight.dtype in _FP8_DTYPES:
                        weight_scale = getattr(mod.base_layer, "weight_scale", None)
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

                    # Compute sv[k] @ W for ALL steering vectors at once.
                    # sv_all shape: (n_vectors, d_out), W shape: (d_out, d_in)
                    # Result shape: (n_vectors, d_in)
                    device = W.device
                    if device not in _sv_by_device:
                        _sv_by_device[device] = steering_vectors.to(device)
                    vW_all = (_sv_by_device[device] @ W).cpu()
                    del W  # free immediately to avoid OOM on large MoE models

                    cache.projections[layer_idx][component] = {
                        "vW_all": vW_all,
                        "module_path": module_path,
                        "d_out": d_out,
                        "d_in": d_in,
                    }

        cache.target_modules = sorted(target_module_names)
        n_cached = sum(len(v) for v in cache.projections.values())
        cache_mb = sum(
            info["vW_all"].nbytes
            for layer in cache.projections.values()
            for info in layer.values()
        ) / 1024 / 1024
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
        from ..types import DecayKernel, SteeringProfile

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
            v_unnorm = (1 - global_frac) * sv[global_idx_a] + global_frac * sv[global_idx_a + 1]
            global_norm = v_unnorm.norm().item()
            global_vector = v_unnorm / global_norm if global_norm > 0 else v_unnorm

        lora_weights: dict[str, tuple[Tensor, Tensor]] = {}
        n_layers = len(self.projections)

        for layer_idx in range(n_layers):
            if layer_idx not in self.projections:
                continue

            for component, info in self.projections[layer_idx].items():
                if component not in profiles:
                    continue

                sp = profiles[component]
                distance = abs(layer_idx - sp.max_weight_position)
                if distance > sp.min_weight_distance:
                    continue

                # Compute decay weight.
                t = distance / sp.min_weight_distance
                if kernel == DecayKernel.GAUSSIAN:
                    strength = sp.min_weight + (sp.max_weight - sp.min_weight) * math.exp(
                        -2.0 * t * t
                    )
                elif kernel == DecayKernel.COSINE:
                    strength = sp.min_weight + (sp.max_weight - sp.min_weight) * (
                        0.5 * (1.0 + math.cos(math.pi * t))
                    )
                else:
                    strength = sp.max_weight + t * (sp.min_weight - sp.max_weight)

                vW_all = info["vW_all"]  # (n_vectors, d_in)

                if global_vector is not None:
                    v = global_vector
                    # Exact reconstruction via linearity of matmul:
                    vW = (
                        (1 - global_frac) * vW_all[global_idx_a]
                        + global_frac * vW_all[global_idx_a + 1]
                    ) / global_norm
                else:
                    v = F.normalize(sv[layer_idx + 1], p=2, dim=0)
                    vW = vW_all[layer_idx + 1]

                lora_A = vW.view(1, -1)  # (1, d_in)
                lora_B = (-strength * v[: info["d_out"]]).view(-1, 1)  # (d_out, 1)

                lora_weights[info["module_path"]] = (lora_A, lora_B)

        return lora_weights
