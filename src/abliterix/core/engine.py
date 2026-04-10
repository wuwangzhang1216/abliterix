# Abliterix — a derivative work of Heretic (https://github.com/p-e-w/heretic)
# Original work Copyright (C) 2025  Philipp Emanuel Weidmann (p-e-w)
# Modified work Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import json
import os
import sys
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

from ..settings import AbliterixConfig
from ..types import ChatMessage, QuantMode, WeightNorm
from ..util import chunk_batches, flush_memory, print

# transformers < 5.0 uses torch_dtype=, >= 5.0 uses dtype= in from_pretrained.
import transformers as _tf

_dtype_kwarg = "dtype" if int(_tf.__version__.split(".")[0]) >= 5 else "torch_dtype"


def resolve_model_class(
    model_id: str,
) -> Type[AutoModelForImageTextToText] | Type[AutoModelForCausalLM]:
    """Choose the correct AutoModel class based on the model's configuration.

    Vision-language models (e.g. Mistral3, Qwen-VL) use
    ``AutoModelForImageTextToText``; their text backbone is accessed via the
    ``model.language_model`` path in ``transformer_layers``.  Pure text models
    use ``AutoModelForCausalLM``.
    """
    configs = PretrainedConfig.get_config_dict(model_id)
    if any("vision_config" in cfg for cfg in configs):
        return AutoModelForImageTextToText
    return AutoModelForCausalLM


def _patch_mtp_layer_types(model_id: str, trust_remote_code: bool | None) -> None:
    """Patch models whose ``layer_types`` includes MTP head layers.

    Models like Step-3.5-Flash define ``layer_types`` with 48 entries (45
    decoder + 3 MTP) but ``num_hidden_layers=45``.  Transformers >= 5.5
    validates that ``len(layer_types) == num_hidden_layers``, causing a
    ``ValueError``.

    We patch the cached remote config module source file to truncate
    ``layer_types`` before the parent ``__init__`` validator runs.
    """
    try:
        cfgs = PretrainedConfig.get_config_dict(
            model_id,
            trust_remote_code=trust_remote_code,
        )
        cfg_dict = cfgs[0] if isinstance(cfgs, tuple) else cfgs
        layer_types = cfg_dict.get("layer_types")
        n_hidden = cfg_dict.get("num_hidden_layers")
        if not (layer_types and n_hidden and len(layer_types) > n_hidden):
            return

        # Find the cached configuration_*.py file for this model and patch it
        # to truncate layer_types before super().__init__().
        from pathlib import Path

        cache_dir = (
            Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
            / "modules"
            / "transformers_modules"
        )

        # Search for configuration files matching this model.
        for config_py in cache_dir.rglob("configuration_*.py"):
            if model_id.split("/")[-1].lower().replace("-", "_").replace(
                ".", "_"
            ) not in str(config_py).lower().replace("-", "_").replace(".", "_"):
                continue
            text = config_py.read_text()
            marker = "self.layer_types = layer_types"
            patch = (
                "# Truncate MTP head layers (transformers >= 5.5 validation fix)\n"
                "        if layer_types is not None and len(layer_types) > num_hidden_layers:\n"
                "            layer_types = layer_types[:num_hidden_layers]\n"
                "        self.layer_types = layer_types"
            )
            if marker in text and "Truncate MTP" not in text:
                text = text.replace(marker, patch)
                config_py.write_text(text)
                # Invalidate any cached import of this module.
                for mod_name in list(sys.modules):
                    if "configuration_step3p5" in mod_name:
                        del sys.modules[mod_name]
                print(
                    f"  [dim]Patched {config_py.name}: "
                    f"truncating {len(layer_types)} layer_types → {n_hidden}[/]"
                )
                return
    except Exception:
        pass


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
    steering algorithm lives in :mod:`abliterix.core.steering`.
    """

    model: PreTrainedModel | PeftModel
    tokenizer: PreTrainedTokenizerBase
    peft_config: LoraConfig

    def __init__(self, config: AbliterixConfig):
        self.config = config
        self.response_prefix = ""
        self.needs_reload = False
        self._dequant_cache: dict[int, Tensor] = {}

        # Cached metadata — populated by prepare_for_unload() before the HF
        # model is freed, so the optimizer can still query layer/component
        # info after engine.model is set to None.
        self._cached_n_layers: int | None = None
        self._cached_components: list[str] | None = None

        model_id = config.model.model_id

        print()
        print(f"Loading model [bold]{model_id}[/]...")

        # Patch MTP models whose layer_types length exceeds num_hidden_layers
        # (e.g. Step-3.5-Flash: 48 layer_types vs 45 num_hidden_layers).
        _patch_mtp_layer_types(model_id, config.model.trust_remote_code)

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

        # Auto-detect native FP8 models: if the model's config.json already
        # contains quantization_config with quant_method="fp8", treat it as FP8
        # even if the user didn't explicitly set quant_method in our config.
        # Also auto-detect MXFP4 (gpt-oss) so we can force dequant — abliteration
        # requires direct nn.Parameter access to fused expert weights, which the
        # native MXFP4 path (Mxfp4GptOssExperts) does not expose.
        self._is_native_fp8 = False
        self._is_native_mxfp4 = False
        try:
            from transformers import AutoConfig as _AC

            _auto_cfg = _AC.from_pretrained(model_id, trust_remote_code=True)
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
                    self._is_native_fp8 = True
                    if config.model.quant_method != QuantMode.FP8:
                        print(
                            "  [dim]Auto-detected native FP8 model "
                            "(quantization_config in config.json)[/]"
                        )
                elif _qm.get("quant_method") == "mxfp4":
                    self._is_native_mxfp4 = True
                    print(
                        "  [dim]Auto-detected native MXFP4 model — "
                        "will force dequantize=True so fused expert weights "
                        "are exposed as standard nn.Parameter[/]"
                    )

            # Detect transposed fused-expert layout. Most MoE models store
            # the fused down_proj tensor as (experts, hidden_out, intermediate_in).
            # gpt-oss is the exception: GptOssExperts.down_proj has shape
            # (experts, intermediate_in, hidden_out) and the forward path uses
            # `out = act @ W` (no transpose). When in==out (gpt-oss has
            # hidden==intermediate==2880) the EGA axis-detection-by-shape
            # heuristic falls back to the wrong branch — we need an explicit
            # marker. See _apply_ega_steering in steering.py.
            _text_cfg = getattr(_auto_cfg, "text_config", _auto_cfg)
            _model_type = getattr(_text_cfg, "model_type", "")
            self._fused_down_proj_transposed = _model_type in {"gpt_oss"}
        except Exception:
            self._fused_down_proj_transposed = False

        is_fp8 = config.model.quant_method == QuantMode.FP8 or self._is_native_fp8

        # Workaround: transformers FP8 quantizer accesses config.intermediate_size
        # as a fallback when moe_intermediate_size is absent. Some MoE model configs
        # (e.g. Qwen3.5 MoE) only define moe_intermediate_size, causing an
        # AttributeError during replace_with_fp8_linear. Patch the config class
        # to alias intermediate_size → moe_intermediate_size if needed.
        if is_fp8:
            self._patch_moe_config_for_fp8(model_id)

        for dtype in config.model.dtype_fallback_order:
            print(f"* Trying dtype [bold]{dtype}[/]... ", end="")

            try:
                qconfig = self._build_quant_config(dtype)

                extra: dict[str, Any] = {}
                if qconfig is not None:
                    extra["quantization_config"] = qconfig

                # MXFP4 (gpt-oss): force dequant to BF16 so abliteration can
                # access fused expert weights as a standard 3-D nn.Parameter.
                # Without this override, transformers wraps the experts in
                # Mxfp4GptOssExperts whose `down_proj` is a packed triton
                # tensor that _locate_fused_weights cannot edit.
                if self._is_native_mxfp4 and qconfig is None:
                    try:
                        from transformers import Mxfp4Config

                        extra["quantization_config"] = Mxfp4Config(dequantize=True)
                    except ImportError:
                        print(
                            "  [yellow]transformers lacks Mxfp4Config — "
                            "MXFP4 dequant cannot be forced; ensure "
                            "triton/kernels are NOT installed so the "
                            "quantizer falls back to bf16[/]"
                        )

                if config.model.attn_implementation is not None:
                    extra["attn_implementation"] = config.model.attn_implementation

                self.model = resolve_model_class(model_id).from_pretrained(
                    model_id,
                    **{_dtype_kwarg: dtype},
                    device_map=config.model.device_map,
                    max_memory=self.max_memory,
                    trust_remote_code=self.trusted_models.get(model_id),
                    offload_folder="/tmp/offload",
                    **extra,
                )

                if self.trusted_models.get(model_id) is None:
                    self.trusted_models[model_id] = True

                # FP8 handling: decide whether to use native FP8 kernels or
                # fall back to bf16 dequant.  Applies to both explicit
                # quant_method="fp8" and auto-detected native FP8 models.
                if is_fp8:
                    skip = self._should_skip_fp8_dequant()
                    if not skip:
                        self._dequant_fp8_to_bf16()
                    else:
                        print(
                            "  [dim]Using native FP8 kernels "
                            "(skip_fp8_dequant or auto-detected H100+ "
                            "with transformers >= 5.2)[/]"
                        )

                # Smoke-test: a single forward pass catches dtype-related
                # runtime errors (inf/nan probability tensors, etc.).
                self._generate(
                    [ChatMessage(system=config.system_prompt, user="What is 1+1?")],
                    max_new_tokens=1,
                )
            except (
                Exception
            ) as error:  # Model loading may fail with diverse errors (OOM, dtype, CUDA)
                self.model = None  # ty:ignore[invalid-assignment]
                flush_memory()
                print(f"[red]Failed[/] ({error})")
                continue

            if config.model.quant_method == QuantMode.BNB_4BIT:
                print("[green]Ok[/] (quantized to 4-bit precision)")
            elif config.model.quant_method == QuantMode.BNB_8BIT:
                print("[green]Ok[/] (quantized to 8-bit precision)")
            elif is_fp8:
                print("[green]Ok[/] (FP8 precision)")
            else:
                print("[green]Ok[/]")

            break

        if self.model is None:
            raise RuntimeError("Failed to load model with all configured dtypes.")

        # NOTE: FP8 dequant is now applied inside the dtype loop (above),
        # before the smoke-test, so we no longer need it here.

        self._init_adapters()
        self._init_expert_routing()

        if config.model.use_torch_compile:
            print("* Compiling model with torch.compile()...")
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")  # ty:ignore[invalid-assignment]
                print("  [green]Ok[/]")
            except RuntimeError as error:
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
    # FP8 dequantization workaround
    # ------------------------------------------------------------------

    def _should_skip_fp8_dequant(self) -> bool:
        """Decide whether to skip the FP8→bf16 dequant workaround.

        Returns True when native FP8 kernels are safe to use:
        - Explicit ``skip_fp8_dequant=True`` in config, OR
        - Auto-detect: H100+ (SM >= 90) AND transformers >= 5.2.0
          (which fixed the Triton kernel div-by-zero in act_quant_kernel
          and the MoE weight_scale_inv shape mismatch).

        Returns False when dequant is needed for safety.
        """
        skip = self.config.model.skip_fp8_dequant
        if skip is not None:
            return skip

        # Auto-detect: check GPU compute capability and transformers version.
        try:
            # SM >= 90 (H100/H200/B200)
            if torch.cuda.is_available():
                cc = torch.cuda.get_device_capability(0)
                if cc[0] < 9:
                    return False  # A100 or older — dequant needed
            else:
                return False

            # transformers >= 5.2.0 has the FP8 kernel fixes
            import transformers

            tv = tuple(int(x) for x in transformers.__version__.split(".")[:2])
            if tv >= (5, 2):
                return True
        except Exception:
            pass

        return False  # Default: safe fallback

    def _dequant_fp8_to_bf16(self):
        """Replace FP8 Linear forward paths with on-the-fly bf16 dequantization.

        Transformers' fine-grained FP8 integration uses Triton JIT kernels for
        dequant+matmul that have a known async race condition on multi-GPU
        ``device_map="auto"`` setups (see Qwen FP8 model card warnings and
        HuggingFace transformers issue tracker).

        This method monkey-patches every ``nn.Linear`` that holds FP8 weights
        to dequantize to bf16 on-the-fly during forward, replacing the Triton
        path with standard CUDA matmul.  Peak memory overhead is one layer's
        worth of bf16 weights at a time (freed after each forward call).
        """
        _FP8_DTYPES = {torch.float8_e4m3fn, torch.float8_e5m2}
        patched = 0

        for name, module in self.model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            if module.weight.dtype not in _FP8_DTYPES:
                continue

            # Check for block-wise FP8 scale tensors (used by fine-grained FP8
            # models like Qwen3.5-*-FP8).
            scale_attr = None
            for attr_name in ("weight_scale", "weight_scale_inv"):
                if hasattr(module, attr_name):
                    scale_attr = attr_name
                    break

            # Stash the original forward and patch.

            if scale_attr is not None:
                # Block-wise FP8: dequant using the per-block scale tensor.
                scale_tensor = getattr(module, scale_attr)
                fp8_weight = module.weight
                bias = module.bias

                def _make_blockwise_forward(w, s, b, is_inv):
                    def _forward(x):
                        w_f = w.to(torch.bfloat16)
                        s_f = s.float()
                        # Expand block-wise scales to match weight shape.
                        # Scale shape is (rows/block, cols/block) — expand both dims.
                        block_r = max(1, w.shape[0] // s.shape[0])
                        block_c = max(1, w.shape[1] // s.shape[1])
                        s_exp = s_f.repeat_interleave(block_r, dim=0).repeat_interleave(
                            block_c, dim=1
                        )
                        s_exp = s_exp[: w.shape[0], : w.shape[1]]
                        if is_inv:
                            w_bf16 = (w_f * s_exp).to(torch.bfloat16)
                        else:
                            w_bf16 = (w_f / s_exp).to(torch.bfloat16)
                        return F.linear(x.to(torch.bfloat16), w_bf16, b)

                    return _forward

                is_inverse = scale_attr == "weight_scale_inv"
                module.forward = _make_blockwise_forward(
                    fp8_weight, scale_tensor, bias, is_inverse
                )
            else:
                # No scale tensor: simple cast (per-tensor FP8 or unscaled).
                fp8_weight = module.weight
                bias = module.bias

                def _make_simple_forward(w, b):
                    def _forward(x):
                        return F.linear(x.to(torch.bfloat16), w.to(torch.bfloat16), b)

                    return _forward

                module.forward = _make_simple_forward(fp8_weight, bias)

            patched += 1

        print(
            f"* FP8→bf16 dequant: patched [bold]{patched}[/] Linear modules "
            f"(bypasses Triton FP8 kernels for multi-GPU compatibility)"
        )

    # ------------------------------------------------------------------
    # Adapter / LoRA management
    # ------------------------------------------------------------------

    def _init_adapters(self):
        """Wrap the base model in PEFT LoRA adapters targeting steerable modules."""
        assert isinstance(self.model, PreTrainedModel)

        # Build a map from module id to its full path in the model tree.
        # We use full paths (not leaf names) to avoid collisions with identically
        # named modules outside the transformer_layers — notably the vision
        # tower in multimodal models like Gemma 4, whose `o_proj` modules may
        # be custom wrappers that PEFT can't adapt.
        id_to_path: dict[int, str] = {
            id(m): name for name, m in self.model.named_modules()
        }

        target_paths: set[str] = set()
        for idx in range(len(self.transformer_layers)):
            for modules in self.steerable_modules(idx).values():
                for mod in modules:
                    path = id_to_path.get(id(mod))
                    if path is not None:
                        target_paths.add(path)

        targets = sorted(target_paths)

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

        # PEFT inherits the base layer's dtype for LoRA A/B weights.  For FP8
        # models this produces Float8_e4m3fn parameters that crash F.linear
        # ("addmm_cuda not implemented for Float8_e4m3fn").  Cast to bf16.
        if self.config.model.quant_method == QuantMode.FP8 or self._is_native_fp8:
            _fp8 = {torch.float8_e4m3fn, torch.float8_e5m2}
            for name, param in self.model.named_parameters():
                if "lora_" in name and param.dtype in _fp8:
                    param.data = param.data.to(torch.bfloat16)

        # Pre-cache references to every lora_B weight tensor for O(adapter-count)
        # resets instead of a full named_modules walk.
        self._lora_b_weights: list[Tensor] = []
        for name, mod in self.model.named_modules():
            if "lora_B" in name and hasattr(mod, "weight"):
                self._lora_b_weights.append(mod.weight)

        # Summarise target paths by their distinct leaf names to keep output readable.
        leaf_summary = sorted({t.rsplit(".", 1)[-1] for t in targets})
        print(
            f"* LoRA adapters initialised "
            f"({len(targets)} modules, leaves: {', '.join(leaf_summary)})"
        )

    @staticmethod
    def _patch_moe_config_for_fp8(model_id: str) -> None:
        """Patch MoE config classes that lack ``intermediate_size``.

        The transformers FP8 quantizer (``finegrained_fp8.py``) falls back to
        ``config.intermediate_size`` when ``moe_intermediate_size`` is missing
        on the *module-level* config object.  Some architectures (Qwen3.5 MoE)
        define only ``moe_intermediate_size``, causing an ``AttributeError``.

        We pre-fetch the model config and, if needed, inject a property that
        aliases ``intermediate_size`` → ``moe_intermediate_size`` so the
        quantizer can proceed.
        """
        from transformers import AutoConfig

        try:
            auto_cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            text_cfg = getattr(auto_cfg, "text_config", auto_cfg)
            cfg_cls = type(text_cfg)

            if hasattr(text_cfg, "moe_intermediate_size") and not hasattr(
                text_cfg, "intermediate_size"
            ):
                cfg_cls.intermediate_size = property(
                    lambda self: self.moe_intermediate_size
                )
                print(
                    f"  [dim]Patched {cfg_cls.__name__}.intermediate_size → "
                    f"moe_intermediate_size[/]"
                )
        except Exception:
            pass  # Best-effort; if this fails, the original error will surface.

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
            # Pre-quantized FP8 models carry their own quantization_config.
            # If weight_block_size is specified, create a FineGrainedFP8Config
            # to fix MoE weight_scale_inv shape mismatches.
            block_size = self.config.model.fp8_weight_block_size
            if block_size is not None:
                try:
                    from transformers import FineGrainedFP8Config

                    return FineGrainedFP8Config(
                        weight_block_size=block_size,
                    )
                except ImportError:
                    pass  # Older transformers without FineGrainedFP8Config
            return None
        return None

    # ------------------------------------------------------------------
    # Layer / module discovery
    # ------------------------------------------------------------------

    @property
    def transformer_layers(self) -> ModuleList:
        """Return the ordered list of transformer decoder blocks.

        Models with Multi-Token Prediction heads (e.g. Step-3.5-Flash) may
        have extra layers beyond ``num_hidden_layers``.  We truncate to the
        config value when available to avoid steering MTP head layers.
        """
        m = self.model
        if isinstance(m, PeftModel):
            m = m.base_model.model

        with suppress(Exception):
            layers = m.model.language_model.layers
            return self._truncate_to_hidden_layers(m, layers)
        with suppress(Exception):
            layers = m.backbone.layers  # NemotronH
            return self._truncate_to_hidden_layers(m, layers)
        layers = m.model.layers
        return self._truncate_to_hidden_layers(m, layers)

    @staticmethod
    def _truncate_to_hidden_layers(model: Any, layers: ModuleList) -> ModuleList:
        """Truncate layer list to ``num_hidden_layers`` if the model has MTP head layers."""
        cfg = getattr(model, "config", None)
        text_cfg = getattr(cfg, "text_config", cfg)
        n = getattr(text_cfg, "num_hidden_layers", None)
        if n is not None and len(layers) > n:
            # Return a sliced ModuleList containing only the real decoder layers.
            return ModuleList(list(layers)[:n])
        return layers

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

        # Self-attention projections — Q/K/V determine what information gets
        # read from/written to the residual; targeting all four breaks through
        # PLE repair by preventing the model from attending to refusal positions.
        with suppress(Exception):
            _register("attn.q_proj", layer.self_attn.q_proj)  # ty:ignore[possibly-missing-attribute]
        with suppress(Exception):
            _register("attn.k_proj", layer.self_attn.k_proj)  # ty:ignore[possibly-missing-attribute]
        with suppress(Exception):
            _register("attn.v_proj", layer.self_attn.v_proj)  # ty:ignore[possibly-missing-attribute]
        with suppress(Exception):
            _register("attn.o_proj", layer.self_attn.o_proj)  # ty:ignore[possibly-missing-attribute]

        # Multi-head Latent Attention (MLA) projections — DeepSeek-V2/V3,
        # GLM-4.7-Flash, Qwen3-Next. Q goes through a low-rank LoRA pair
        # (q_a_proj → q_b_proj); KV goes through (kv_a_proj_with_mqa → kv_b_proj).
        # Steering the *_b_proj outputs is the analogue of steering Q/K/V in
        # standard attention, since they produce the actual head dimensions.
        # Norm modules in between (q_a_layernorm, kv_a_layernorm) are skipped.
        with suppress(Exception):
            _register("attn.q_b_proj", layer.self_attn.q_b_proj)  # ty:ignore[possibly-missing-attribute]
        with suppress(Exception):
            _register("attn.kv_b_proj", layer.self_attn.kv_b_proj)  # ty:ignore[possibly-missing-attribute]
        # Some MLA implementations (older DeepSeek-V2 ports) skip the q LoRA
        # entirely and project Q in one step via q_proj — already covered above.

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

        # Shared experts (GLM-4 MoE Lite — plural naming).
        with suppress(Exception):
            _register("mlp.down_proj", layer.mlp.shared_experts.down_proj)  # ty:ignore[possibly-missing-attribute]

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

        # Step-3.5-Flash — shared expert (singular "share_expert", not "shared_expert").
        # Registered as mlp.down_proj intentionally — same steering profile as per-expert modules.
        with suppress(Exception):
            _register("mlp.down_proj", layer.share_expert.down_proj)  # ty:ignore[possibly-missing-attribute]

        # LFM2 MoE — gated short convolution output projection.
        with suppress(Exception):
            _register("conv.out_proj", layer.conv.out_proj)  # ty:ignore[possibly-missing-attribute]

        # LFM2 MoE — attention output projection (named out_proj, not o_proj).
        with suppress(Exception):
            _register("attn.o_proj", layer.self_attn.out_proj)  # ty:ignore[possibly-missing-attribute]

        # LFM2 MoE — dense MLP down-projection (layers 0-1, w2 naming).
        with suppress(Exception):
            _register("mlp.down_proj", layer.feed_forward.w2)  # ty:ignore[possibly-missing-attribute]

        # Mamba-2 / SSM output projection (Nemotron-Cascade, Jamba, etc.).
        with suppress(Exception):
            _register("ssm.out_proj", layer.mixer.out_proj)  # ty:ignore[possibly-missing-attribute]
        with suppress(Exception):
            _register("ssm.out_proj", layer.mamba.out_proj)  # ty:ignore[possibly-missing-attribute]

        # NemotronH — attention output projection via mixer.o_proj.
        with suppress(Exception):
            _register("attn.o_proj", layer.mixer.o_proj)  # ty:ignore[possibly-missing-attribute]

        # NemotronH — per-expert MoE via mixer.experts.
        with suppress(Exception):
            for expert in layer.mixer.experts:  # ty:ignore[possibly-missing-attribute, not-iterable]
                _register("mlp.down_proj", expert.down_proj)  # ty:ignore[possibly-missing-attribute]

        # NemotronH — shared experts via mixer.shared_experts.
        with suppress(Exception):
            _register("mlp.down_proj", layer.mixer.shared_experts.down_proj)  # ty:ignore[possibly-missing-attribute]

        total = sum(len(mods) for mods in modules.values())
        assert total > 0, "No steerable modules found in layer"
        return modules

    def list_steerable_components(self) -> list[str]:
        """Return sorted component names across all layers (handles hybrid architectures).

        For MoE architectures whose expert weights are stored as a fused 3-D
        ``nn.Parameter`` (rather than as a ModuleList of per-expert modules),
        ``steerable_modules`` cannot register per-Module entries for the
        experts. Without an ``"mlp.down_proj"`` key in the components list, the
        optimizer would not generate a steering profile for it, and EGA
        (``_apply_ega_steering``) would silently early-exit because it looks
        up ``profiles["mlp.down_proj"]``. This was observed for gpt-oss where
        ``GptOssExperts`` is a single Module holding fused 3-D weights —
        EGA was effectively disabled, leaving the MoE pathways untouched.

        Workaround: when ``has_expert_routing()`` is true and
        ``_locate_fused_weights`` finds a fused 3-D parameter on layer 0,
        synthesise an ``"mlp.down_proj"`` component so the optimizer creates
        a profile for it. ``_apply_direct_steering`` will skip it (no Modules
        registered under that key), but EGA will pick up the profile and
        project the refusal direction from every expert.
        """
        if self._cached_components is not None:
            return self._cached_components
        components: set[str] = set()
        for idx in range(len(self.transformer_layers)):
            components.update(self.steerable_modules(idx).keys())
        if "mlp.down_proj" not in components and self.has_expert_routing():
            try:
                fused = self._locate_fused_weights(self.transformer_layers[0])
                if fused is not None and fused.dim() == 3:
                    components.add("mlp.down_proj")
            except Exception:
                pass
        return sorted(components)

    def get_n_layers(self) -> int:
        """Return number of transformer layers, using cache if model is unloaded."""
        if self._cached_n_layers is not None:
            return self._cached_n_layers
        return len(self.transformer_layers)

    def prepare_for_unload(self):
        """Cache metadata needed by the optimizer before freeing the HF model.

        Must be called before setting ``engine.model = None`` for the vLLM
        phase transition.
        """
        self._cached_n_layers = len(self.transformer_layers)
        self._cached_components = self.list_steerable_components()

    # ------------------------------------------------------------------
    # MoE expert routing helpers
    # ------------------------------------------------------------------

    def _locate_router(self, layer: Module) -> Module | None:
        """Find the MoE router/gate module that contains a 2-D weight tensor."""
        for path in [
            "mlp.gate",
            "mlp.router",
            "moe.gate",
            "mixer.gate",
            "block_sparse_moe.gate",
            "feed_forward.gate",
            "router.proj",
        ]:
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
        """Find the fused 3-D expert parameter [experts, hidden, intermediate].

        Handles both raw ``nn.Parameter`` (e.g. Qwen3 ``mlp.experts.down_proj``)
        and ``MoELinear``-style modules whose ``.weight`` is the 3-D tensor
        (e.g. Step-3.5-Flash ``moe.down_proj``).
        """
        for path in [
            "mlp.experts.down_proj",
            "mixer.experts.down_proj",
            "feed_forward.experts.down_proj",
            "moe.down_proj",
            "experts.down_proj",
        ]:
            obj: Any = layer
            for attr in path.split("."):
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if isinstance(obj, Parameter) and obj.dim() == 3:
                return obj
            # MoELinear-style: the module has a .weight attribute that is the
            # fused 3-D tensor (Step-3.5-Flash packs 288 experts this way).
            if isinstance(obj, Module):
                w = getattr(obj, "weight", None)
                if isinstance(w, (Parameter, Tensor)) and w.dim() == 3:
                    return w if isinstance(w, Parameter) else None
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
        # Remove any angular steering hooks from the previous trial.
        for handle in getattr(self, "_angular_hooks", []):
            handle.remove()
        self._angular_hooks = []

        # Restore direct weight modifications (orthogonal projection mode).
        for weight_ref, orig in getattr(self, "_direct_weight_originals", {}).items():
            weight_ref.data = orig.to(weight_ref.device)
        if hasattr(self, "_direct_weight_originals"):
            self._direct_weight_originals.clear()

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
            **{_dtype_kwarg: dtype},
            device_map=self.config.model.device_map,
            max_memory=self.max_memory,
            trust_remote_code=self.trusted_models.get(self.config.model.model_id),
            **extra,
        )
        if self.config.model.quant_method == QuantMode.FP8 or self._is_native_fp8:
            if not self._should_skip_fp8_dequant():
                self._dequant_fp8_to_bf16()
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
                **{_dtype_kwarg: self.model.dtype},
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
        min_new_tokens: int | None = None,
    ) -> list[str]:
        """Generate responses for a batch of chat messages."""
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens or self.config.inference.max_gen_tokens,
        }
        if min_new_tokens is not None:
            gen_kwargs["min_new_tokens"] = min_new_tokens
        inputs, outputs = self._generate(messages, **gen_kwargs)
        return self.tokenizer.batch_decode(
            outputs[:, cast(Tensor, inputs["input_ids"]).shape[1] :],
            skip_special_tokens=skip_special_tokens,
        )

    def generate_text_batched(
        self,
        messages: list[ChatMessage],
        skip_special_tokens: bool = False,
        max_new_tokens: int | None = None,
        min_new_tokens: int | None = None,
    ) -> list[str]:
        """Batched wrapper around :meth:`generate_text`."""
        out: list[str] = []
        for batch in chunk_batches(messages, self.config.inference.batch_size):
            out.extend(
                self.generate_text(
                    batch,
                    skip_special_tokens=skip_special_tokens,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
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

    def extract_hidden_states(
        self,
        messages: list[ChatMessage],
        token_offset: int = -1,
    ) -> Tensor:
        """Return per-layer residual vectors at a configurable token position.

        Parameters
        ----------
        token_offset : int
            Index into the sequence dimension for residual extraction.
            ``-1`` (default) extracts at the final post-instruction token
            (where refusal is encoded).  Use ``-2`` or earlier offsets to
            target instruction-boundary tokens where harmfulness signals
            are encoded separately from refusal.

        Shape of the returned tensor: ``(batch, layers+1, hidden_dim)``.
        """
        inputs = self._tokenize(messages)
        self._reset_position_cache()

        outputs = self.model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        residuals = torch.stack(
            [hs[:, token_offset, :] for hs in hidden_states],
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
