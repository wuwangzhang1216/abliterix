# Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>

import json
from collections import defaultdict
from contextlib import suppress
from typing import Any, Type, cast

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from peft import LoraConfig, PeftModel, get_peft_model
from torch import FloatTensor, LongTensor, Tensor
from torch.nn import Module, ModuleList, Parameter
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    BatchEncoding,
    BitsAndBytesConfig,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
    TextStreamer,
)
from transformers.generation import (
    GenerateDecoderOnlyOutput,  # ty:ignore[possibly-missing-import]
    LogitsProcessor,
)

from ..settings import PrometheusConfig
from ..types import ChatMessage, QuantMode, WeightNorm
from ..util import chunk_batches, flush_memory, print


def resolve_model_class(
    model_id: str,
) -> Type[AutoModelForImageTextToText] | Type[AutoModelForCausalLM]:
    """Choose the correct AutoModel class based on the model's configuration.

    Vision-language models are loaded via AutoModelForCausalLM (text backbone
    only) to avoid vision components landing on meta devices during quantised
    loading with device_map="auto".
    """
    configs = PretrainedConfig.get_config_dict(model_id)
    if any("vision_config" in cfg for cfg in configs):
        return AutoModelForCausalLM
    return AutoModelForCausalLM


def load_tokenizer(
    model_id: str,
    trust_remote_code: bool | None = None,
) -> PreTrainedTokenizerBase:
    try:
        return AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )
    except ValueError as exc:
        if "TokenizersBackend" not in str(exc):
            raise

        cfg_path = hf_hub_download(model_id, "tokenizer_config.json")
        tok_path = hf_hub_download(model_id, "tokenizer.json")
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tok_path,
            eos_token=cfg.get("eos_token"),
            bos_token=cfg.get("bos_token"),
            unk_token=cfg.get("unk_token"),
            pad_token=cfg.get("pad_token"),
        )
        tokenizer.model_max_length = cfg.get(
            "model_max_length", tokenizer.model_max_length
        )
        return tokenizer


class _LogitsSampler(LogitsProcessor):
    """Captures the first *n* score tensors emitted during generation.

    Using this processor instead of ``output_scores=True`` avoids storing
    score tensors for every generated token — a significant VRAM saving
    when only a handful of early-token scores are needed for KL computation.
    """

    def __init__(self, n: int):
        self.n = n
        self.scores: list[Tensor] = []

    def __call__(self, input_ids: LongTensor, scores: FloatTensor) -> FloatTensor:
        if len(self.scores) < self.n:
            self.scores.append(scores.detach().clone())
        return scores


class SteeringEngine:
    """Manages model loading, tokenisation, generation, and LoRA adapters.

    The engine owns the loaded model and exposes methods for text generation,
    hidden-state extraction, and log-probability measurement.  The actual
    steering algorithm lives in :mod:`prometheus.core.steering`.
    """

    model: PreTrainedModel | PeftModel
    tokenizer: PreTrainedTokenizerBase
    peft_config: LoraConfig

    def __init__(self, config: PrometheusConfig):
        self.config = config
        self.response_prefix = ""
        self.needs_reload = False
        self._dequant_cache: dict[int, Tensor] = {}

        model_id = config.model.model_id

        print()
        print(f"Loading model [bold]{model_id}[/]...")

        self.tokenizer = load_tokenizer(
            model_id,
            trust_remote_code=config.model.trust_remote_code,
        )

        # Tokenizers that lack a dedicated pad token fall back to EOS.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Decoder-only models require left-padding so that PAD tokens never
        # appear after the prompt — otherwise the model treats them as valid
        # continuation tokens and produces empty outputs.
        self.tokenizer.padding_side = "left"

        self.model = None  # ty:ignore[invalid-assignment]
        self.max_memory = (
            {
                int(k) if k.isdigit() else k: v
                for k, v in config.model.max_memory.items()
            }
            if config.model.max_memory
            else None
        )
        self.trusted_models = {model_id: config.model.trust_remote_code}

        if config.model.evaluate_model_id is not None:
            self.trusted_models[config.model.evaluate_model_id] = (
                config.model.trust_remote_code
            )

        for dtype in config.model.dtype_fallback_order:
            print(f"* Trying dtype [bold]{dtype}[/]... ", end="")

            try:
                qconfig = self._build_quant_config(dtype)

                extra: dict[str, Any] = {}
                if qconfig is not None:
                    extra["quantization_config"] = qconfig

                self.model = resolve_model_class(model_id).from_pretrained(
                    model_id,
                    dtype=dtype,
                    device_map=config.model.device_map,
                    max_memory=self.max_memory,
                    trust_remote_code=self.trusted_models.get(model_id),
                    offload_folder="/tmp/offload",
                    **extra,
                )

                if self.trusted_models.get(model_id) is None:
                    self.trusted_models[model_id] = True

                # Smoke-test: a single forward pass catches dtype-related
                # runtime errors (inf/nan probability tensors, etc.).
                self._generate(
                    [ChatMessage(system=config.system_prompt, user="What is 1+1?")],
                    max_new_tokens=1,
                )
            except Exception as error:
                self.model = None  # ty:ignore[invalid-assignment]
                flush_memory()
                print(f"[red]Failed[/] ({error})")
                continue

            if config.model.quant_method == QuantMode.BNB_4BIT:
                print("[green]Ok[/] (quantized to 4-bit precision)")
            elif config.model.quant_method == QuantMode.BNB_8BIT:
                print("[green]Ok[/] (quantized to 8-bit precision)")
            elif config.model.quant_method == QuantMode.FP8:
                print("[green]Ok[/] (FP8 precision)")
            else:
                print("[green]Ok[/]")

            break

        if self.model is None:
            raise RuntimeError("Failed to load model with all configured dtypes.")

        self._init_adapters()
        self._init_expert_routing()

        if config.model.use_torch_compile:
            print("* Compiling model with torch.compile()...")
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")  # ty:ignore[invalid-assignment]
                print("  [green]Ok[/]")
            except Exception as error:
                print(f"  [yellow]Failed ({error}), continuing without compilation[/]")

        n_layers = len(self.transformer_layers)
        print(f"* Transformer model with [bold]{n_layers}[/] layers")
        print("* Steerable components:")
        for component, modules in self.steerable_modules(0).items():
            print(
                f"  * [bold]{component}[/]: [bold]{len(modules)}[/] modules per layer"
            )

        if self.has_expert_routing():
            fused = self._locate_fused_weights(self.transformer_layers[0])
            n_experts = fused.shape[0] if fused is not None else "?"
            n_gate_layers = sum(
                1
                for layer in self.transformer_layers
                if self._locate_router(layer) is not None
            )
            print(
                f"* MoE model detected: [bold]{n_experts}[/] fused experts, "
                f"[bold]{n_gate_layers}[/] router layers"
            )

    # ------------------------------------------------------------------
    # Adapter / LoRA management
    # ------------------------------------------------------------------

    def _init_adapters(self):
        """Wrap the base model in PEFT LoRA adapters targeting steerable modules."""
        assert isinstance(self.model, PreTrainedModel)

        leaf_names: set[str] = set()
        for idx, layer in enumerate(self.transformer_layers):
            id_to_leaf = {
                id(m): name.split(".")[-1] for name, m in layer.named_modules()
            }
            for modules in self.steerable_modules(idx).values():
                for mod in modules:
                    if id(mod) in id_to_leaf:
                        leaf_names.add(id_to_leaf[id(mod)])

        targets = list(leaf_names)

        if self.config.steering.weight_normalization != WeightNorm.FULL:
            rank = 1
        else:
            rank = self.config.steering.full_norm_lora_rank

        self.peft_config = LoraConfig(
            r=rank,
            target_modules=targets,
            lora_alpha=rank,
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.model = cast(PeftModel, get_peft_model(self.model, self.peft_config))

        # Pre-cache references to every lora_B weight tensor for O(adapter-count)
        # resets instead of a full named_modules walk.
        self._lora_b_weights: list[Tensor] = []
        for name, mod in self.model.named_modules():
            if "lora_B" in name and hasattr(mod, "weight"):
                self._lora_b_weights.append(mod.weight)

        print(f"* LoRA adapters initialised (targets: {', '.join(targets)})")

    def _build_quant_config(self, dtype: str) -> BitsAndBytesConfig | None:
        """Translate the user-facing QuantMode into a BitsAndBytesConfig."""
        qm = self.config.model.quant_method
        if qm == QuantMode.BNB_4BIT:
            compute_dtype = torch.bfloat16 if dtype == "auto" else getattr(torch, dtype)
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )
        elif qm == QuantMode.BNB_8BIT:
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )
        elif qm == QuantMode.FP8:
            # Pre-quantized FP8 models carry their own quantization_config;
            # transformers reads it from the model files automatically.
            return None
        return None

    # ------------------------------------------------------------------
    # Layer / module discovery
    # ------------------------------------------------------------------

    @property
    def transformer_layers(self) -> ModuleList:
        """Return the ordered list of transformer decoder blocks."""
        m = self.model
        if isinstance(m, PeftModel):
            m = m.base_model.model

        with suppress(Exception):
            return m.model.language_model.layers
        return m.model.layers

    def steerable_modules(self, layer_index: int) -> dict[str, list[Module]]:
        """Discover modules within *layer_index* that can be steered.

        Returns a dict mapping component names (e.g. ``"attn.o_proj"``) to
        lists of ``nn.Module`` instances found in that layer.
        """
        layer = self.transformer_layers[layer_index]
        modules: dict[str, list[Module]] = {}

        def _register(component: str, module: Any):
            if isinstance(module, Module):
                modules.setdefault(component, []).append(module)
            else:
                assert not isinstance(module, Tensor), (
                    f"Unexpected Tensor in {component} — expected nn.Module"
                )

        # Standard self-attention output projection.
        with suppress(Exception):
            _register("attn.o_proj", layer.self_attn.o_proj)  # ty:ignore[possibly-missing-attribute]

        # GatedDeltaNet linear-attention variant (Qwen3.5 MoE hybrid layers).
        with suppress(Exception):
            _register("attn.o_proj", layer.linear_attn.out_proj)  # ty:ignore[possibly-missing-attribute]

        # Dense-model MLP down-projection.
        with suppress(Exception):
            _register("mlp.down_proj", layer.mlp.down_proj)  # ty:ignore[possibly-missing-attribute]

        # Per-expert down-projection (e.g. Qwen3).
        with suppress(Exception):
            for expert in layer.mlp.experts:  # ty:ignore[possibly-missing-attribute, not-iterable]
                _register("mlp.down_proj", expert.down_proj)  # ty:ignore[possibly-missing-attribute]

        # Shared expert (Qwen3 / 3.5 MoE).
        with suppress(Exception):
            _register("mlp.down_proj", layer.mlp.shared_expert.down_proj)  # ty:ignore[possibly-missing-attribute]

        # Phi-3.5-MoE.
        with suppress(Exception):
            for expert in layer.block_sparse_moe.experts:  # ty:ignore[possibly-missing-attribute, not-iterable]
                _register("mlp.down_proj", expert.w2)  # ty:ignore[possibly-missing-attribute]

        # Granite MoE Hybrid — dense attention layers.
        with suppress(Exception):
            _register("mlp.down_proj", layer.shared_mlp.output_linear)  # ty:ignore[possibly-missing-attribute]

        # Granite MoE Hybrid — MoE layers.
        with suppress(Exception):
            for expert in layer.moe.experts:  # ty:ignore[possibly-missing-attribute, not-iterable]
                _register("mlp.down_proj", expert.output_linear)  # ty:ignore[possibly-missing-attribute]

        total = sum(len(mods) for mods in modules.values())
        assert total > 0, "No steerable modules found in layer"
        return modules

    def list_steerable_components(self) -> list[str]:
        """Return sorted component names across all layers (handles hybrid architectures)."""
        components: set[str] = set()
        for idx in range(len(self.transformer_layers)):
            components.update(self.steerable_modules(idx).keys())
        return sorted(components)

    # ------------------------------------------------------------------
    # MoE expert routing helpers
    # ------------------------------------------------------------------

    def _locate_router(self, layer: Module) -> Module | None:
        """Find the MoE router/gate module that contains a 2-D weight tensor."""
        for path in ["mlp.gate", "block_sparse_moe.gate"]:
            obj: Any = layer
            for attr in path.split("."):
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None and isinstance(obj, Module):
                w = getattr(obj, "weight", None)
                if isinstance(w, (Tensor, Parameter)) and w.dim() == 2:
                    return obj
        return None

    def _locate_fused_weights(self, layer: Module) -> Parameter | None:
        """Find the fused 3-D expert parameter [experts, hidden, intermediate]."""
        for path in ["mlp.experts.down_proj"]:
            obj: Any = layer
            for attr in path.split("."):
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if isinstance(obj, Parameter) and obj.dim() == 3:
                return obj
        return None

    def has_expert_routing(self) -> bool:
        """True if any layer contains a MoE router gate."""
        return any(
            self._locate_router(layer) is not None for layer in self.transformer_layers
        )

    def _init_expert_routing(self):
        """Prepare bookkeeping lists for router/expert weight rollback."""
        self._router_originals: list[tuple[int, int, Tensor]] = []
        self._expert_deltas: list[tuple[int, int, float, Tensor, Tensor]] = []

    def identify_safety_experts(
        self,
        benign_msgs: list[Any],
        target_msgs: list[Any],
    ) -> dict[int, list[tuple[int, float]]]:
        """Profile router activations to rank experts by safety association.

        Hooks each MoE gate to record which experts are selected for every
        token, then computes per-expert risk-difference scores.

        Returns ``{layer_idx: [(expert_idx, score), ...]}`` sorted descending.
        """
        layers = self.transformer_layers
        gates: dict[int, Module] = {}
        for idx in range(len(layers)):
            g = self._locate_router(layers[idx])
            if g is not None:
                gates[idx] = g

        if not gates:
            return {}

        benign_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        target_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        benign_tokens: dict[int, int] = defaultdict(int)
        target_tokens: dict[int, int] = defaultdict(int)

        active_counts: list[dict[int, dict[int, int]]] = [benign_counts]
        active_tokens: list[dict[int, int]] = [benign_tokens]

        handles = []

        def _make_hook(layer_idx: int):
            def hook(module: Module, inp: Any, out: Any):
                with torch.no_grad():
                    if isinstance(out, tuple) and len(out) >= 3:
                        selected = out[2]
                    elif isinstance(out, tuple) and len(out) == 2:
                        selected = out[1]
                    else:
                        logits = out if not isinstance(out, tuple) else out[0]
                        k = getattr(module, "top_k", 8)
                        _, selected = logits.topk(k, dim=-1)

                    flat = selected.reshape(-1)
                    k = getattr(module, "top_k", selected.shape[-1])
                    n_tok = flat.numel() // k

                    active_tokens[0][layer_idx] += n_tok
                    cnts = active_counts[0][layer_idx]
                    for eid in flat.unique().tolist():
                        cnts[eid] += int((flat == eid).sum().item())

            return hook

        for idx, gate in gates.items():
            handles.append(gate.register_forward_hook(_make_hook(idx)))

        print("  Profiling benign prompts...")
        active_counts[0] = benign_counts
        active_tokens[0] = benign_tokens
        with torch.no_grad():
            self.extract_hidden_states_batched(benign_msgs)

        print("  Profiling target prompts...")
        active_counts[0] = target_counts
        active_tokens[0] = target_tokens
        with torch.no_grad():
            self.extract_hidden_states_batched(target_msgs)

        for h in handles:
            h.remove()

        safety: dict[int, list[tuple[int, float]]] = {}
        for idx, gate in gates.items():
            n_experts = gate.weight.shape[0]  # ty:ignore[non-subscriptable]
            scores: list[tuple[int, float]] = []
            bt = max(benign_tokens[idx], 1)
            tt = max(target_tokens[idx], 1)
            for eid in range(n_experts):
                p_b = benign_counts[idx].get(eid, 0) / bt
                p_t = target_counts[idx].get(eid, 0) / tt
                scores.append((eid, p_t - p_b))
            scores.sort(key=lambda x: x[1], reverse=True)
            safety[idx] = scores

        n_layers = len(safety)
        top_scores = [safety[i][0][1] for i in sorted(safety) if safety[i]]
        avg = sum(top_scores) / len(top_scores) if top_scores else 0
        print(f"  Profiled {n_layers} MoE layers, avg top risk diff: {avg:.4f}")

        return safety

    # ------------------------------------------------------------------
    # Model reset / export
    # ------------------------------------------------------------------

    def restore_baseline(self):
        """Reset to the un-steered state for a fresh trial.

        Fast path: zero out cached LoRA-B weights and undo any MoE modifications.
        Slow path: full model reload when a destructive operation (e.g. merge)
        has invalidated the in-memory weights.
        """
        current_id = getattr(self.model.config, "name_or_path", None)
        if current_id == self.config.model.model_id and not self.needs_reload:
            for w in self._lora_b_weights:
                torch.nn.init.zeros_(w)

            for layer_idx, expert_idx, original_row in self._router_originals:
                gate = self._locate_router(self.transformer_layers[layer_idx])
                if gate is not None:
                    gate.weight.data[expert_idx] = original_row.to(gate.weight.device)  # ty:ignore[invalid-assignment,no-matching-overload]
            self._router_originals.clear()

            for layer_idx, expert_idx, w, v, vTW in self._expert_deltas:
                dp = self._locate_fused_weights(self.transformer_layers[layer_idx])
                if dp is not None:
                    W = dp.data[expert_idx].to(torch.float32)
                    W += (w * torch.outer(v, vTW)).to(device=W.device)
                    dp.data[expert_idx] = W.to(dp.dtype)
            self._expert_deltas.clear()
            return

        dtype = self.model.dtype
        self.model = None  # ty:ignore[invalid-assignment]
        flush_memory()

        qconfig = self._build_quant_config(str(dtype).split(".")[-1])
        extra: dict[str, Any] = {}
        if qconfig is not None:
            extra["quantization_config"] = qconfig

        self.model = resolve_model_class(self.config.model.model_id).from_pretrained(
            self.config.model.model_id,
            dtype=dtype,
            device_map=self.config.model.device_map,
            max_memory=self.max_memory,
            trust_remote_code=self.trusted_models.get(self.config.model.model_id),
            **extra,
        )
        self._init_adapters()
        self._init_expert_routing()
        self.needs_reload = False

    def export_merged(self) -> PreTrainedModel:
        """Merge LoRA adapters into the base weights and return the result.

        For quantised models the base model is reloaded in full precision on
        CPU before merging, as in-place dequantisation is not supported.
        """
        assert isinstance(self.model, PeftModel)

        if self.config.model.quant_method in (
            QuantMode.BNB_4BIT,
            QuantMode.BNB_8BIT,
            QuantMode.FP8,
        ):
            adapter_state = {
                n: p.data.clone().cpu()
                for n, p in self.model.named_parameters()
                if "lora_" in n
            }

            print("* Loading base model on CPU (this may take a while)...")
            base = resolve_model_class(self.config.model.model_id).from_pretrained(
                self.config.model.model_id,
                torch_dtype=self.model.dtype,
                device_map="cpu",
                trust_remote_code=self.trusted_models.get(self.config.model.model_id),
            )

            print("* Applying LoRA adapters...")
            peft_model = get_peft_model(base, self.peft_config)
            for n, p in peft_model.named_parameters():
                if n in adapter_state:
                    p.data = adapter_state[n].to(p.device)

            print("* Merging LoRA adapters into base model...")
            return peft_model.merge_and_unload()
        else:
            print("* Merging LoRA adapters into base model...")
            merged = self.model.merge_and_unload()
            self.needs_reload = True
            return merged

    # ------------------------------------------------------------------
    # Internal position-cache management
    # ------------------------------------------------------------------

    def _reset_position_cache(self):
        """Clear stale rope_deltas in VLM wrappers to prevent shape mismatches."""
        m = self.model
        for _ in range(5):
            if hasattr(m, "rope_deltas"):
                m.rope_deltas = None  # ty:ignore[invalid-assignment]
                return
            if hasattr(m, "base_model"):
                m = m.base_model
            elif hasattr(m, "model"):
                m = m.model
            else:
                return

    # ------------------------------------------------------------------
    # Tokenisation helpers
    # ------------------------------------------------------------------

    def _tokenize(self, messages: list[ChatMessage]) -> BatchEncoding:
        """Apply the chat template, optionally prepend the response prefix, and tokenise."""
        chats = [
            [
                {"role": "system", "content": msg.system},
                {"role": "user", "content": msg.user},
            ]
            for msg in messages
        ]

        texts = cast(
            list[str],
            self.tokenizer.apply_chat_template(
                chats,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            ),
        )

        if self.response_prefix:
            texts = [t + self.response_prefix for t in texts]

        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
        ).to(self.model.device)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate(
        self,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> tuple[BatchEncoding, GenerateDecoderOnlyOutput | LongTensor]:
        """Low-level generation: tokenise, run model.generate(), return (inputs, outputs)."""
        inputs = self._tokenize(messages)
        self._reset_position_cache()

        # ty:ignore — generate() has an extremely complex type signature.
        outputs = self.model.generate(
            **inputs,
            **kwargs,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=False,
        )  # ty:ignore[call-non-callable]

        return inputs, outputs

    def generate_text(
        self,
        messages: list[ChatMessage],
        skip_special_tokens: bool = False,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        """Generate responses for a batch of chat messages."""
        inputs, outputs = self._generate(
            messages,
            max_new_tokens=max_new_tokens or self.config.inference.max_gen_tokens,
        )
        return self.tokenizer.batch_decode(
            outputs[:, cast(Tensor, inputs["input_ids"]).shape[1] :],
            skip_special_tokens=skip_special_tokens,
        )

    def generate_text_batched(
        self,
        messages: list[ChatMessage],
        skip_special_tokens: bool = False,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        """Batched wrapper around :meth:`generate_text`."""
        out: list[str] = []
        for batch in chunk_batches(messages, self.config.inference.batch_size):
            out.extend(
                self.generate_text(
                    batch,
                    skip_special_tokens=skip_special_tokens,
                    max_new_tokens=max_new_tokens,
                )
            )
        return out

    def generate_and_score(
        self,
        messages: list[ChatMessage],
        max_new_tokens: int,
        kl_token_count: int,
        skip_special_tokens: bool = False,
    ) -> tuple[list[str], Tensor]:
        """Generate full responses AND capture early-token logprobs in one pass.

        Avoids the duplicate-prefill overhead of calling generate_text() and
        compute_logprobs() separately on the same prompt batch.
        """
        sampler = _LogitsSampler(kl_token_count)

        inputs, outputs = self._generate(
            messages,
            max_new_tokens=max_new_tokens,
            logits_processor=[sampler],
        )

        actual_n = min(kl_token_count, len(sampler.scores))
        if actual_n == 1:
            logprobs = F.log_softmax(sampler.scores[0], dim=-1)
        else:
            stacked = torch.stack(
                [F.log_softmax(s, dim=-1) for s in sampler.scores[:actual_n]],
                dim=1,
            )
            logprobs = stacked.mean(dim=1)

        input_len = cast(Tensor, inputs["input_ids"]).shape[1]
        responses = self.tokenizer.batch_decode(
            outputs[:, input_len:],
            skip_special_tokens=skip_special_tokens,
        )
        return responses, logprobs

    def generate_and_score_batched(
        self,
        messages: list[ChatMessage],
        max_new_tokens: int,
        kl_token_count: int,
        skip_special_tokens: bool = False,
    ) -> tuple[list[str], Tensor]:
        """Batched wrapper around :meth:`generate_and_score`."""
        all_resp: list[str] = []
        all_lp: list[Tensor] = []
        for batch in chunk_batches(messages, self.config.inference.batch_size):
            resp, lp = self.generate_and_score(
                batch,
                max_new_tokens,
                kl_token_count,
                skip_special_tokens,
            )
            all_resp.extend(resp)
            all_lp.append(lp)
        return all_resp, torch.cat(all_lp, dim=0)

    # ------------------------------------------------------------------
    # Hidden-state extraction
    # ------------------------------------------------------------------

    def extract_hidden_states(self, messages: list[ChatMessage]) -> Tensor:
        """Return per-layer residual vectors at the final token position.

        Shape of the returned tensor: ``(batch, layers+1, hidden_dim)``.
        """
        inputs = self._tokenize(messages)
        self._reset_position_cache()

        outputs = self.model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        residuals = torch.stack(
            [hs[:, -1, :] for hs in hidden_states],
            dim=1,
        ).to(torch.float32)

        q = self.config.steering.outlier_quantile
        if 0 <= q < 1:
            thresholds = torch.quantile(
                torch.abs(residuals),
                q,
                dim=2,
                keepdim=True,
            )
            return torch.clamp(residuals, -thresholds, thresholds)

        return residuals

    def extract_hidden_states_batched(self, messages: list[ChatMessage]) -> Tensor:
        parts = []
        for batch in chunk_batches(messages, self.config.inference.batch_size):
            parts.append(self.extract_hidden_states(batch))
        return torch.cat(parts, dim=0)

    # ------------------------------------------------------------------
    # Log-probability measurement
    # ------------------------------------------------------------------

    def _logprobs_forward_pass(self, messages: list[ChatMessage]) -> Tensor:
        """Next-token logprobs via a single forward pass (no generation overhead)."""
        inputs = self._tokenize(messages)
        self._reset_position_cache()
        outputs = self.model(**inputs)
        return F.log_softmax(outputs.logits[:, -1, :], dim=-1)

    def compute_logprobs(self, messages: list[ChatMessage]) -> Tensor:
        """Compute averaged next-token log-probabilities over kl_token_count steps."""
        n = self.config.kl.token_count

        if n == 1:
            return self._logprobs_forward_pass(messages)

        sampler = _LogitsSampler(n)
        self._generate(messages, max_new_tokens=n, logits_processor=[sampler])

        stacked = torch.stack(
            [F.log_softmax(s, dim=-1) for s in sampler.scores],
            dim=1,
        )
        return stacked.mean(dim=1)

    def compute_logprobs_batched(self, messages: list[ChatMessage]) -> Tensor:
        parts = []
        for batch in chunk_batches(messages, self.config.inference.batch_size):
            parts.append(self.compute_logprobs(batch))
        return torch.cat(parts, dim=0)

    # ------------------------------------------------------------------
    # Interactive chat
    # ------------------------------------------------------------------

    def stream_chat_response(self, chat: list[dict[str, str]]) -> str:
        """Stream a response for an ongoing multi-turn conversation."""
        text = cast(
            str,
            self.tokenizer.apply_chat_template(
                chat,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            ),
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(self.model.device)

        streamer = TextStreamer(
            self.tokenizer,  # ty:ignore[invalid-argument-type]
            skip_prompt=True,
            skip_special_tokens=True,
        )

        self._reset_position_cache()

        outputs = self.model.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=4096,
        )  # ty:ignore[call-non-callable]

        return cast(
            str,
            self.tokenizer.decode(
                outputs[0, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            ),
        )
