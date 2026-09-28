# Abliterix — a derivative work of Heretic (https://github.com/p-e-w/heretic)
# Original work Copyright (C) 2025  Philipp Emanuel Weidmann (p-e-w)
# Modified work Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import json
import inspect
import os
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
from ..types import ChatMessage, QuantMode, SteeringMode, VectorMethod, WeightNorm
from ..util import chunk_batches, flush_memory, print
from . import fp8_utils

# transformers < 5.0 uses torch_dtype=, >= 5.0 uses dtype= in from_pretrained.
import transformers as _tf

_dtype_kwarg = "dtype" if int(_tf.__version__.split(".")[0]) >= 5 else "torch_dtype"


def extract_router_expert_ids(out: Any, top_k: int = 8) -> Tensor:
    """Return the top-k expert-id tensor from a MoE router forward.

    Router tuple order is family-specific. Bailing MoE v3 returns
    ``(topk_idx, topk_weight, logits)`` — indices first — while several
    other families put indices last. Prefer the first integer tensor
    instead of a fixed ``out[2]`` slot.
    """
    if isinstance(out, tuple) and len(out) >= 3:
        for cand in out:
            if isinstance(cand, Tensor) and cand.dtype in (torch.int32, torch.int64):
                return cand
        return out[0] if isinstance(out[0], Tensor) else out[1]
    if isinstance(out, tuple) and len(out) == 2:
        return out[1]
    logits = out if not isinstance(out, tuple) else out[0]
    _, selected = logits.topk(top_k, dim=-1)
    return selected


# Models registered here have a known remote-config mismatch where MTP heads
# are appended to ``layer_types`` even though ``num_hidden_layers`` describes
# decoder layers only.  The registry is populated from the model's own
# config.json before its remote configuration class is instantiated.
_MTP_LAYER_TYPE_ADAPTERS: dict[str, tuple[int, int]] = {}


def _model_config_name(config: PretrainedConfig) -> str:
    """Return the loader identity attached to a Transformers config."""
    return str(
        getattr(config, "_name_or_path", "") or getattr(config, "name_or_path", "")
    ).rstrip("/")


def _adapt_registered_mtp_layer_types(config: PretrainedConfig) -> bool:
    """Apply a registered MTP layer mapping to one in-memory config.

    The exact model identity and both observed lengths must match the
    preflighted config.json.  This prevents the compatibility adapter from
    hiding unrelated or newly changed model configuration errors.
    """
    spec = _MTP_LAYER_TYPE_ADAPTERS.get(_model_config_name(config))
    if spec is None:
        return False

    expected_total, expected_hidden = spec
    layer_types = getattr(config, "layer_types", None)
    n_hidden = getattr(config, "num_hidden_layers", None)
    if (
        not isinstance(layer_types, list)
        or len(layer_types) != expected_total
        or n_hidden != expected_hidden
    ):
        return False

    config.layer_types = layer_types[:expected_hidden]
    return True


def _install_mtp_layer_type_validator() -> None:
    """Install the process-local adapter ahead of Transformers validation."""
    validators = list(PretrainedConfig.__class_validators__)  # ty:ignore[unresolved-attribute]
    if any(
        getattr(validator, "_abliterix_mtp_adapter", False) for validator in validators
    ):
        return

    for index, validator in enumerate(validators):
        if getattr(validator, "__name__", "") != "validate_layer_type":
            continue

        def validate_layer_type(config, _original=validator):
            _adapt_registered_mtp_layer_types(config)
            _original(config)

        validate_layer_type._abliterix_mtp_adapter = True  # type: ignore[attr-defined]
        validators[index] = validate_layer_type
        PretrainedConfig.__class_validators__ = validators  # ty:ignore[unresolved-attribute]
        return


_install_mtp_layer_type_validator()


# Cap the dequantised-weight cache so large MoE expert counts cannot retain
# hundreds of GB of fp32 copies in RAM.  Shared by SteeringEngine.__init__ and
# the fast vLLM engine shell in cli.py so the two cannot drift apart.
DEQUANT_CACHE_MAX_BYTES = 4 * 1024**3  # 4 GiB


def resolve_model_class(
    model_id: str,
    revision: str | None = None,
    *,
    text_only: bool = False,
) -> Type[AutoModelForImageTextToText] | Type[AutoModelForCausalLM]:
    """Choose the correct AutoModel class based on the model's configuration.

    Vision-language models (e.g. Mistral3, Qwen-VL) use
    ``AutoModelForImageTextToText``; their text backbone is accessed via the
    ``model.language_model`` path in ``transformer_layers``.  Pure text models
    use ``AutoModelForCausalLM``.

    When *text_only* is true, always return ``AutoModelForCausalLM`` even if
    the checkpoint advertises multimodal fields (opt-in for text abliteration
    of dual-registered MoE VLMs such as some Qwen3.5-MoE checkpoints).
    """
    configs = PretrainedConfig.get_config_dict(model_id, revision=revision)
    if text_only:
        return AutoModelForCausalLM

    config_dicts = configs if isinstance(configs, tuple) else (configs,)

    if any(isinstance(cfg, dict) and "vision_config" in cfg for cfg in config_dicts):
        return AutoModelForImageTextToText
    return AutoModelForCausalLM


def _bf16_compute_supported() -> bool:
    """True when the active accelerator has native (non-emulated) bf16.

    ``torch.cuda.is_bf16_supported()`` defaults to including software
    emulation, so pre-Ampere CUDA cards still return True. We only treat
    ROCm and sm_80+ as native bf16 so older CUDA targets can fall back to
    float16 for both bnb compute dtype and residual promotion.
    """
    if not torch.cuda.is_available():
        return False
    if torch.version.hip:  # ROCm: native on supported archs
        return True
    return torch.cuda.get_device_capability()[0] >= 8


def _register_mtp_layer_types_adapter(
    model_id: str,
    trust_remote_code: bool | None,
    revision: str | None = None,
) -> None:
    """Register models whose ``layer_types`` includes MTP head layers.

    Models like Step-3.5-Flash define ``layer_types`` with 48 entries (45
    decoder + 3 MTP) but ``num_hidden_layers=45``.  Transformers >= 5.5
    validates that ``len(layer_types) == num_hidden_layers``, causing a
    ``ValueError``.

    The compatibility adjustment is performed on the in-memory config just
    before the parent validator runs.  No files in the Hugging Face cache are
    modified.
    """
    try:
        cfgs = PretrainedConfig.get_config_dict(
            model_id,
            trust_remote_code=trust_remote_code,
            revision=revision,
        )
        cfg_dict = cfgs[0] if isinstance(cfgs, tuple) else cfgs
        layer_types = cfg_dict.get("layer_types")
        n_hidden = cfg_dict.get("num_hidden_layers")
        if not (layer_types and n_hidden and len(layer_types) > n_hidden):
            return

        _MTP_LAYER_TYPE_ADAPTERS[model_id.rstrip("/")] = (
            len(layer_types),
            n_hidden,
        )
        print(
            "  [dim]Enabled process-local MTP config adapter: "
            f"truncating {len(layer_types)} layer_types → {n_hidden}[/]"
        )
    except Exception:
        pass


def load_tokenizer(
    model_id: str,
    trust_remote_code: bool | None = None,
    revision: str | None = None,
) -> PreTrainedTokenizerBase:
    try:
        return AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            revision=revision,
        )
    except AttributeError as exc:
        if "'list' object has no attribute 'keys'" not in str(exc):
            raise

        return AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            revision=revision,
            extra_special_tokens={},
        )
    except ValueError as exc:
        if "TokenizersBackend" not in str(exc):
            raise

        cfg_path = hf_hub_download(model_id, "tokenizer_config.json", revision=revision)
        tok_path = hf_hub_download(model_id, "tokenizer.json", revision=revision)
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


class _ForcedContinuationProcessor(LogitsProcessor):
    """Force a known continuation while preserving earlier captured logits."""

    def __init__(self, token_ids: list[list[int]]):
        self.token_ids = token_ids
        self.step = 0

    def __call__(self, input_ids: LongTensor, scores: FloatTensor) -> FloatTensor:
        if self.step >= len(self.token_ids[0]):
            raise RuntimeError("Forced continuation received too many decode steps")
        if scores.shape[0] != len(self.token_ids):
            raise RuntimeError(
                "Forced continuation batch changed during generation: "
                f"expected {len(self.token_ids)}, got {scores.shape[0]}"
            )

        next_ids = torch.tensor(
            [row[self.step] for row in self.token_ids],
            dtype=torch.long,
            device=scores.device,
        ).unsqueeze(1)
        forced = torch.full_like(scores, float("-inf"))
        forced.scatter_(1, next_ids, 0.0)
        self.step += 1
        return forced


def _captured_logprobs(sampler: _LogitsSampler, expected_steps: int) -> Tensor:
    """Return normalized captured scores without collapsing the token axis.

    A single-token measurement retains the historical ``(batch, vocab)``
    shape.  Multi-token measurements use ``(batch, step, vocab)`` so the
    scorer can average per-step KL divergences instead of treating an
    arithmetic mean of log-probabilities as a distribution.
    """
    if len(sampler.scores) != expected_steps:
        raise RuntimeError(
            "Generation ended before all KL score steps were captured: "
            f"expected {expected_steps}, got {len(sampler.scores)}"
        )

    stacked = torch.stack(
        # A full-vocabulary log-softmax in BF16 loses enough tail precision to
        # materially distort small KL values (and made the one-token HF result
        # disagree with vLLM while later FP32 steps agreed).  Keep model logits
        # in their native dtype, but normalise the captured metric in FP32.
        [F.log_softmax(scores.float(), dim=-1) for scores in sampler.scores],
        dim=1,
    )
    if expected_steps == 1:
        return stacked[:, 0, :]
    return stacked


def _forward_supports_logits_to_keep(model: Module) -> bool:
    """Return whether the underlying model explicitly limits output logits.

    PEFT and ``torch.compile`` expose generic ``**kwargs`` forward signatures,
    so inspect their wrapped base model instead.  A generic ``**kwargs`` alone
    is deliberately not treated as support: older and remote-code models may
    reject or mishandle an unknown keyword deeper in their forward path.
    """
    current: Any = model
    seen: set[int] = set()

    while id(current) not in seen:
        seen.add(id(current))

        original = getattr(current, "_orig_mod", None)
        if original is not None and original is not current:
            current = original
            continue

        get_base_model = getattr(current, "get_base_model", None)
        if callable(get_base_model):
            try:
                base = get_base_model()
            except Exception:
                base = None
            if base is not None and base is not current:
                current = base
                continue
        break

    try:
        return "logits_to_keep" in inspect.signature(current.forward).parameters
    except (TypeError, ValueError):
        return False


def _required_lora_rank(config: AbliterixConfig) -> int:
    """Return the adapter rank required by every configured vector recipe."""
    steering = config.steering
    if steering.steering_mode != SteeringMode.LORA:
        return 1

    required = max(1, steering.n_directions)
    if steering.ablate_harmfulness_direction or steering.search_harmfulness_direction:
        required = max(required, 2)
    if steering.vector_method == VectorMethod.SOM:
        required = max(required, steering.som_grid_h * steering.som_grid_w)
    if steering.vector_method == VectorMethod.SAE:
        required = max(required, steering.sae_top_k)
    if config.iterative.enabled:
        iterative_rank = (
            config.iterative.max_iterations * config.iterative.per_iteration_directions
        )
        required = max(required, iterative_rank)
    if steering.weight_normalization == WeightNorm.FULL:
        required = max(required, steering.full_norm_lora_rank)
    return required


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
        self._dequant_cache_bytes: int = 0
        # Cap dequant cache so large MoE expert counts cannot retain
        # hundreds of GB of fp32 weights in RAM.
        self._dequant_cache_max_bytes: int = DEQUANT_CACHE_MAX_BYTES

        # Cached metadata — populated by prepare_for_unload() before the HF
        # model is freed, so the optimizer can still query layer/component
        # info after engine.model is set to None.
        self._cached_n_layers: int | None = None
        self._cached_components: list[str] | None = None

        model_id = config.model.model_id

        print()
        print(f"Loading model [bold]{model_id}[/]...")

        # Adapt MTP models whose layer_types length exceeds num_hidden_layers
        # (e.g. Step-3.5-Flash: 48 layer_types vs 45 num_hidden_layers).
        _register_mtp_layer_types_adapter(
            model_id,
            config.model.trust_remote_code,
            config.model.revision,
        )

        self.tokenizer = load_tokenizer(
            model_id,
            trust_remote_code=config.model.trust_remote_code,
            revision=config.model.revision,
        )

        # Tokenizers that lack a dedicated pad token fall back to EOS.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Decoder-only models require left-padding so that PAD tokens never
        # appear after the prompt — otherwise the model treats them as valid
        # continuation tokens and produces empty outputs.
        self.tokenizer.padding_side = "left"

        # Custom encoder: models like DeepSeek-V4 ship a Python encoding
        # script instead of a Jinja chat_template. Monkey-patch
        # apply_chat_template so the rest of abliterix sees a normal tokenizer.
        if getattr(config.model, "custom_encoder_module", None):
            self._install_custom_encoder(
                config.model.custom_encoder_module,
                config.model.custom_encoder_kwargs or {},
            )

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
        self._is_dsv4_hybrid_fp8_fp4 = False
        try:
            from transformers import AutoConfig as _AC

            _auto_cfg = _AC.from_pretrained(
                model_id,
                trust_remote_code=True,
                revision=config.model.revision,
            )
            _qcfg = getattr(_auto_cfg, "quantization_config", None)
            if _qcfg is None:
                _text_cfg = getattr(_auto_cfg, "text_config", None)
                if _text_cfg is not None:
                    _qcfg = getattr(_text_cfg, "quantization_config", None)
            _expert_dtype = getattr(_auto_cfg, "expert_dtype", None)
            if _qcfg is not None:
                _qm = (
                    _qcfg if isinstance(_qcfg, dict) else getattr(_qcfg, "__dict__", {})
                )
                if _qm.get("quant_method") == "fp8":
                    self._is_native_fp8 = True
                    if _expert_dtype == "fp4":
                        # DeepSeek-V4: non-experts FP8, experts FP4.
                        # transformers' FP8 quantiser does not handle this
                        # combo — fail load loudly unless the user has
                        # pre-dequanted to BF16 on disk.
                        self._is_dsv4_hybrid_fp8_fp4 = True
                        print(
                            "  [yellow]Detected DeepSeek-V4 hybrid quant "
                            "(non-experts FP8 + experts FP4). transformers "
                            "cannot dequant FP4 experts in-memory; point "
                            "model_id at a pre-dequanted BF16 directory "
                            "(unsloth/DeepSeek-V4-Flash or output of "
                            "abliterix-dequant-fp8 + "
                            "quick_start/_dsv4_dequant_fp4_experts.py).[/]"
                        )
                    elif config.model.quant_method != QuantMode.FP8:
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
            self._patch_moe_config_for_fp8(model_id, config.model.revision)

        for dtype in config.model.dtype_fallback_order:
            print(f"* Trying dtype [bold]{dtype}[/]... ", end="")

            try:
                qconfig = self._build_quant_config()

                extra: dict[str, Any] = {}
                if qconfig is not None:
                    extra["quantization_config"] = qconfig

                # MXFP4 (gpt-oss): force dequant to BF16 so abliteration can
                # access fused expert weights as a standard 3-D nn.Parameter.
                # Without this override, transformers wraps the experts in
                # Mxfp4GptOssExperts whose `down_proj` is a packed triton
                # tensor that _locate_fused_weights cannot edit.
                #
                # `steering.frozen_experts` is exactly the case where we do NOT
                # want that: forward-time EGA never touches the weights, so the
                # experts stay packed and the model loads at its 4-bit size
                # (gpt-oss-20b: ~13.8 GB rather than ~30 GB).
                if getattr(config.steering, "frozen_experts", False):
                    print(
                        "  [dim]frozen_experts: keeping MXFP4 experts packed "
                        "(forward-time EGA, no weight mutation)[/]"
                    )
                elif self._is_native_mxfp4 and qconfig is None:
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

                if getattr(config.model, "experts_implementation", None) is not None:
                    extra["experts_implementation"] = (
                        config.model.experts_implementation
                    )

                self.model = resolve_model_class(
                    model_id,
                    config.model.revision,
                    text_only=config.model.text_only,
                ).from_pretrained(
                    model_id,
                    **{_dtype_kwarg: dtype},
                    device_map=config.model.device_map,
                    max_memory=self.max_memory,
                    trust_remote_code=self.trusted_models.get(model_id),
                    revision=config.model.revision,
                    offload_folder="/tmp/offload",
                    **extra,
                )

                if self.trusted_models.get(model_id) is None:
                    self.trusted_models[model_id] = True

                # FP8 handling driven by config.model.fp8_handling:
                #   'auto'           — pick based on steering_mode
                #   'materialize'    — dequant + replace weights in memory
                #                      (2x VRAM; supports direct mode, EGA,
                #                      and transformers FP8Experts unfusing)
                #   'forward_dequant'— monkey-patch forward (1x VRAM; LoRA only)
                #   'offline'        — assume pre-dequanted; no FP8 path
                if is_fp8:
                    handling = getattr(self.config.model, "fp8_handling", "auto")
                    direct = self.config.steering.steering_mode == SteeringMode.DIRECT
                    if handling == "auto":
                        handling = "materialize" if direct else "forward_dequant"

                    if handling == "offline":
                        print(
                            "  [dim]FP8 offline mode: weights assumed to "
                            "already be BF16 on disk[/]"
                        )
                    elif handling == "materialize":
                        self._materialize_fp8_as_bf16()
                    elif handling == "forward_dequant":
                        skip = self._should_skip_fp8_dequant()
                        if not skip:
                            self._dequant_fp8_to_bf16()
                        else:
                            print(
                                "  [dim]Using native FP8 kernels "
                                "(skip_fp8_dequant or auto-detected H100+ "
                                "with transformers >= 5.2)[/]"
                            )
                    else:
                        raise ValueError(f"Unknown fp8_handling mode: {handling!r}")

                    if direct and handling == "forward_dequant":
                        print(
                            "  [yellow]Warning: direct steering with "
                            "forward_dequant leaves weights in FP8; "
                            "orthogonal projection will clip on write-back. "
                            "Set fp8_handling='materialize' or pre-dequant "
                            "offline.[/]"
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

        # bnb 4-bit: non-quantized tensors (embed/norm/lm_head) may still be
        # float16 after load (including dtype="auto" when the checkpoint is
        # float16). When native bf16 is available, promote live float16 params
        # so residual norms match the bf16 compute path. Skip promotion when
        # we fell back to float16 compute (pre-Ampere CUDA). Parameters on
        # meta (accelerate offload) may not retain this.
        if (
            config.model.quant_method == QuantMode.BNB_4BIT
            and _bf16_compute_supported()
        ):
            n_conv = 0
            for _n, _p in self.model.named_parameters():
                if _p.dtype == torch.float16:
                    _p.data = _p.data.to(torch.bfloat16)
                    n_conv += 1
            if n_conv:
                print(f"  [dim]Promoted {n_conv} non-quantized params fp16→bf16[/]")

        # NOTE: FP8 dequant is now applied inside the dtype loop (above),
        # before the smoke-test, so we no longer need it here.

        if config.model.backend in ("vllm", "sglang"):
            print("* TP backend: skipping HF LoRA adapter initialisation")
            self._lora_b_weights = []
            self.peft_config = None  # ty:ignore[invalid-assignment]
        else:
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

    def _install_custom_encoder(
        self,
        module_path: str,
        encoder_kwargs: dict[str, Any],
    ) -> None:
        """Replace ``self.tokenizer.apply_chat_template`` with a Python encoder.

        Some models (DeepSeek-V4) ship ``encoding_*.py`` instead of a Jinja
        chat_template. The encoder must expose
        ``encode_messages(messages: list[dict], **kw) -> str``. We wrap it in
        a callable that mimics ``apply_chat_template``'s signature so the
        rest of the pipeline (``_tokenize``, hidden-state extraction, etc.)
        does not need to know about the substitution.
        """
        import importlib.util
        from pathlib import Path

        path = Path(module_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"custom_encoder_module not found: {module_path}")

        spec = importlib.util.spec_from_file_location(
            f"abliterix_custom_encoder_{path.stem}", path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load custom encoder from {path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        if not hasattr(mod, "encode_messages"):
            raise AttributeError(
                f"{path} has no `encode_messages(messages, **kw)` function"
            )

        encode_fn = mod.encode_messages
        tok = self.tokenizer

        def _patched(
            conversation,
            tokenize: bool = True,
            add_generation_prompt: bool = False,
            return_tensors: str | None = None,
            **kw: Any,
        ):
            # `conversation` may be a single message-list or a batch of them.
            if (
                isinstance(conversation, list)
                and conversation
                and isinstance(conversation[0], list)
            ):
                texts = [
                    encode_fn(
                        c,
                        add_generation_prompt=add_generation_prompt,
                        **encoder_kwargs,
                    )
                    for c in conversation
                ]
            else:
                texts = encode_fn(
                    conversation,
                    add_generation_prompt=add_generation_prompt,
                    **encoder_kwargs,
                )

            if not tokenize:
                return texts

            enc = tok(
                texts,
                return_tensors=return_tensors,
                padding=kw.get("padding", False),
                truncation=kw.get("truncation", False),
            )
            return enc["input_ids"]

        tok.apply_chat_template = _patched  # type: ignore[assignment]
        print(
            f"  [dim]Installed custom encoder from {path.name} "
            f"(kwargs={encoder_kwargs or {}})[/]"
        )

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

    def _materialize_fp8_as_bf16(self):
        """Materialise every FP8 container in ``self.model`` as writable BF16.

        Delegates to :mod:`abliterix.core.fp8_utils` which handles:

        - Standard ``nn.Linear`` with per-tensor FP8 ``weight_scale``
        - Standard ``nn.Linear`` with 2-D block-wise ``weight_scale_inv``
          (DeepSeek / MiniMax-M2 / Qwen3-FP8 style)
        - Fused MoE containers (transformers 5.x ``FP8Experts``) — unfused
          back into a per-expert ``nn.ModuleList`` so abliterix's EGA/direct
          code can target each expert's ``.w1 / .w2 / .w3`` individually

        Required for ``steering_mode = "direct"`` because
        :func:`_apply_direct_steering` writes ``weight.data = W_new.to(dtype)``
        and FP8's ±448 dynamic range would clip the orthogonal projection.

        Cost: ~2× VRAM (FP8 230GB → BF16 460GB for MiniMax-M2). If the model
        + KV cache + activations don't fit, prefer offline pre-dequant via
        :func:`abliterix.core.fp8_utils.dequant_model_to_disk` — loading the
        resulting standalone BF16 model never invokes transformers' FP8
        quantiser, side-stepping its MoE traversal bug.
        """
        counts = fp8_utils.materialize_fp8_model(self.model, verbose=True)
        if counts["fused_moe_detected"] > 0 or counts["unsupported"] > 0:
            print(
                "  [yellow]Fused-MoE FP8 containers cannot be in-memory "
                "materialised for inference (parent MoE block forward "
                "expects the fused-kernel API). For direct steering on this "
                "model, pre-dequant to disk first:[/]\n"
                "    abliterix-dequant-fp8 <src_snapshot> <dst_bf16_dir>"
            )

    def _dequant_fp8_to_bf16(self):
        """Monkey-patch every FP8 ``nn.Linear.forward`` to dequant-on-the-fly.

        Preserves the underlying FP8 storage (1× memory, unlike
        :meth:`_materialize_fp8_as_bf16`) but redirects the forward path
        through a standard CUDA ``F.linear`` after in-line BF16 dequant.
        This bypasses transformers' Triton FP8 kernels whose
        ``act_quant_kernel`` has a known async race condition on multi-GPU
        ``device_map="auto"`` setups.

        Use when steering mode is LoRA (weights stay frozen — forward-only
        dequant is sufficient) and memory headroom cannot accommodate the 2×
        expansion a full materialisation would require. Does NOT handle
        fused MoE containers — those require full materialisation (or offline
        pre-dequant).
        """
        from . import fp8_utils as _fp8

        patched = 0
        for _, module, kind in list(_fp8.iter_fp8_linears(self.model)):
            bias = module.bias
            if kind == "blockwise":
                scale = module.weight_scale_inv  # type: ignore[attr-defined]
                w = module.weight

                def _make_blockwise_forward(w, s, b):
                    def _forward(x):
                        w_bf = _fp8.dequant_blockwise(w, s, is_inv=True)
                        return F.linear(x.to(torch.bfloat16), w_bf, b)

                    return _forward

                module.forward = _make_blockwise_forward(w, scale, bias)
            else:  # per_tensor or unscaled
                scale = getattr(module, "weight_scale", None)
                w = module.weight

                def _make_per_tensor_forward(w, s, b):
                    def _forward(x):
                        w_bf = _fp8.dequant_per_tensor(w, s)
                        return F.linear(x.to(torch.bfloat16), w_bf, b)

                    return _forward

                module.forward = _make_per_tensor_forward(w, scale, bias)
            patched += 1

        print(
            f"* FP8→bf16 forward dequant: patched [bold]{patched}[/] Linear modules "
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

        rank = _required_lora_rank(self.config)

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
    def _patch_moe_config_for_fp8(model_id: str, revision: str | None = None) -> None:
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
            auto_cfg = AutoConfig.from_pretrained(
                model_id, trust_remote_code=True, revision=revision
            )
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

    def _build_quant_config(self) -> BitsAndBytesConfig | None:
        """Translate the user-facing QuantMode into a BitsAndBytesConfig."""
        qm = self.config.model.quant_method
        if qm == QuantMode.BNB_4BIT:
            # Prefer native bf16 compute for residual dynamic range; fall back
            # to float16 on pre-Ampere CUDA (and CPU). Does not control which
            # modules are quantized — only matmul compute dtype.
            compute_dtype = (
                torch.bfloat16 if _bf16_compute_supported() else torch.float16
            )
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

        # Bailing MoE v3 hybrid (Ling-3.0-flash, inclusionAI): decoder blocks
        # expose attention under ``layer.attention`` (not ``self_attn``).
        # Layers alternate MultiLatentAttention (output proj = ``dense``) and
        # KimiDeltaAttention (output proj = ``o_proj``). Without these paths
        # only ``mlp.down_proj`` is steerable.
        with suppress(Exception):
            _register("attn.o_proj", layer.attention.o_proj)  # ty:ignore[possibly-missing-attribute]
        with suppress(Exception):
            _register("attn.o_proj", layer.attention.dense)  # ty:ignore[possibly-missing-attribute]
        with suppress(Exception):
            _register("attn.q_proj", layer.attention.q_proj)  # ty:ignore[possibly-missing-attribute]
        with suppress(Exception):
            _register("attn.k_proj", layer.attention.k_proj)  # ty:ignore[possibly-missing-attribute]
        with suppress(Exception):
            _register("attn.v_proj", layer.attention.v_proj)  # ty:ignore[possibly-missing-attribute]
        with suppress(Exception):
            _register("attn.q_b_proj", layer.attention.q_b_proj)  # ty:ignore[possibly-missing-attribute]
        with suppress(Exception):
            _register("attn.kv_b_proj", layer.attention.kv_b_proj)  # ty:ignore[possibly-missing-attribute]

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

        # Drop every engine-held reference into the model being unloaded;
        # anything left here pins the model's VRAM after
        # ``engine.model = None`` (issue #83).
        self._dequant_cache.clear()
        self._dequant_cache_bytes = 0
        for handle in getattr(self, "_angular_hooks", []):
            handle.remove()
        self._angular_hooks = []
        for attr in (
            "_lora_b_weights",
            "_router_originals",
            "_expert_deltas",
            "_direct_weight_originals",
            "_cliff_head_originals",
        ):
            buffers = getattr(self, attr, None)
            if buffers is not None:
                buffers.clear()

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
        # ``(layer_idx, expert_idx, original_expert_slice)`` — the untouched
        # weight slice, not a reconstructible delta.
        self._expert_deltas: list[tuple[int, int, Tensor]] = []

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
                    selected = extract_router_expert_ids(
                        out, top_k=getattr(module, "top_k", 8)
                    )

                    flat = selected.reshape(-1)
                    k = getattr(module, "top_k", selected.shape[-1])
                    n_tok = flat.numel() // k

                    active_tokens[0][layer_idx] += n_tok
                    cnts = active_counts[0][layer_idx]
                    # One bincount instead of per-expert (flat == eid) GPU
                    # syncs — the loop is catastrophically slow on 512-expert
                    # MoEs (hours vs minutes).
                    weight = getattr(module, "weight", None)
                    n_experts = (
                        weight.shape[0]
                        if weight is not None
                        else int(flat.max().item()) + 1
                    )
                    for eid, count in enumerate(
                        torch.bincount(flat.long(), minlength=n_experts).cpu().tolist()
                    ):
                        if count:
                            cnts[eid] += count

            return hook

        for idx, gate in gates.items():
            handles.append(gate.register_forward_hook(_make_hook(idx)))

        # The profiling passes run two full sweeps over every prompt; an OOM
        # or dtype error there would otherwise leave every router hook
        # registered for the rest of the process, mutating counters on every
        # later forward pass.
        try:
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
        finally:
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

        # ``_model_config_name`` is the canonical loader identity and also
        # falls back to the legacy ``name_or_path`` alias, which transformers
        # >=5 may drop. Reading only ``name_or_path`` (as this did) yields
        # ``None`` there, making the comparison always fail and forcing a full
        # model reload on *every* trial.
        current_id = _model_config_name(self.model.config)
        if (
            current_id == self.config.model.model_id.rstrip("/")
            and not self.needs_reload
        ):
            for w in self._lora_b_weights:
                torch.nn.init.zeros_(w)

            for layer_idx, expert_idx, original_row in self._router_originals:
                gate = self._locate_router(self.transformer_layers[layer_idx])
                if gate is not None:
                    gate.weight.data[expert_idx] = original_row.to(gate.weight.device)  # ty:ignore[invalid-assignment,no-matching-overload]
            self._router_originals.clear()

            # Write back the exact pre-edit expert slice. Storing the original
            # (rather than reconstructing it from the applied delta) keeps this
            # idempotent: when the same fused tensor was also edited by EGA,
            # the whole-tensor restore above has already undone the expert
            # edit, and re-applying the inverse delta here would inject a
            # spurious rank-1 term that would then be cached as the "original"
            # for the next trial. It also avoids an fp32 round trip, which
            # silently loses precision for fp8/packed storage dtypes.
            for layer_idx, expert_idx, original_slice in self._expert_deltas:
                dp = self._locate_fused_weights(self.transformer_layers[layer_idx])
                if dp is not None:
                    dp.data[expert_idx] = original_slice.to(dp.device).to(dp.dtype)
            self._expert_deltas.clear()
            return

        dtype = self.model.dtype
        self.model = None  # ty:ignore[invalid-assignment]
        flush_memory()

        # The dequant cache is keyed by ``id(module)``. Once the old model is
        # freed the freshly allocated wrappers routinely land on the same
        # addresses, so a surviving entry would hand the next trial a stale
        # dequantized weight tensor from a *different* model object.
        self._dequant_cache.clear()
        self._dequant_cache_bytes = 0

        qconfig = self._build_quant_config()
        extra: dict[str, Any] = {}
        if qconfig is not None:
            extra["quantization_config"] = qconfig

        self.model = resolve_model_class(
            self.config.model.model_id,
            self.config.model.revision,
            text_only=self.config.model.text_only,
        ).from_pretrained(
            self.config.model.model_id,
            **{_dtype_kwarg: dtype},
            device_map=self.config.model.device_map,
            max_memory=self.max_memory,
            trust_remote_code=self.trusted_models.get(self.config.model.model_id),
            revision=self.config.model.revision,
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
        mode = self.config.steering.steering_mode
        runtime_only = {
            SteeringMode.ANGULAR,
            SteeringMode.ADAPTIVE_ANGULAR,
            SteeringMode.SPHERICAL,
            SteeringMode.VECTOR_FIELD,
        }
        if mode in runtime_only:
            raise RuntimeError(
                f"Steering mode {mode.value!r} is runtime-only and cannot be "
                "represented by a merged checkpoint. Export a runtime artifact "
                "or choose 'lora'/'direct' instead."
            )

        quantized = self.config.model.quant_method in (
            QuantMode.BNB_4BIT,
            QuantMode.BNB_8BIT,
            QuantMode.FP8,
        )
        has_non_lora_edits = bool(
            getattr(self, "_router_originals", None)
            or getattr(self, "_expert_deltas", None)
        )
        if quantized and (mode == SteeringMode.DIRECT or has_non_lora_edits):
            raise RuntimeError(
                "Cannot faithfully export active direct/router/expert edits "
                "from a quantized model: full-precision reload would discard "
                "those base-weight changes. Replay the resolved steering plan "
                "on an unquantized HF model before exporting."
            )

        assert isinstance(self.model, PeftModel)

        if quantized:
            adapter_state = {
                n: p.data.clone().cpu()
                for n, p in self.model.named_parameters()
                if "lora_" in n
            }

            print("* Loading base model on CPU (this may take a while)...")
            base = resolve_model_class(
                self.config.model.model_id,
                self.config.model.revision,
                text_only=self.config.model.text_only,
            ).from_pretrained(
                self.config.model.model_id,
                **{_dtype_kwarg: self.model.dtype},
                device_map="cpu",
                trust_remote_code=self.trusted_models.get(self.config.model.model_id),
                revision=self.config.model.revision,
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

    def _cache_dequant(self, mid: int, weight: Tensor) -> None:
        """Store a dequantized weight tensor if under the byte budget.

        Pure performance cache: skipping an insert only costs re-dequant time.
        """
        if self._dequant_cache_bytes < self._dequant_cache_max_bytes:
            self._dequant_cache[mid] = weight
            self._dequant_cache_bytes += weight.nelement() * weight.element_size()

    def export_adapter(self, save_directory: str | os.PathLike[str]) -> None:
        """Save the active LoRA adapter without BF16 merge-rounding drift.

        Loading this adapter on the same base model/revision preserves the
        runtime LoRA computation.  Base-weight edits (such as MoE router or
        expert edits) are deliberately rejected because PEFT adapter files
        cannot represent them.
        """
        mode = self.config.steering.steering_mode
        if mode != SteeringMode.LORA:
            raise RuntimeError(
                "Adapter export only represents LoRA steering; use merged "
                "export for an unquantized direct-edit model."
            )
        if getattr(self, "_router_originals", None) or getattr(
            self, "_expert_deltas", None
        ):
            raise RuntimeError(
                "Adapter export cannot represent active router/expert base-weight "
                "edits. Export a merged unquantized model instead."
            )
        if not isinstance(self.model, PeftModel):
            raise RuntimeError("No active PEFT LoRA adapter is available to export.")
        # `export_merged()` calls `merge_and_unload()` in place on the
        # unquantized path: the LoRA layers are folded into the base weights
        # and removed, but `self.model` stays a PeftModel object.  The
        # isinstance check above therefore still passes and PEFT would happily
        # write a zero-tensor adapter.  `needs_reload` is the engine's
        # canonical "weights were destructively mutated" flag; the state-dict
        # check also covers any other route to an emptied adapter.
        if self.needs_reload or not any(
            "lora_" in name for name, _ in self.model.named_parameters()
        ):
            raise RuntimeError(
                "The in-memory LoRA adapter was consumed by a previous merged "
                "export. Re-select the trial to reload the base model before "
                "exporting an adapter."
            )

        self.model.save_pretrained(save_directory)

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

    def _render_messages(self, messages: list[ChatMessage]) -> list[str]:
        """Render messages exactly as the model-facing tokenizer sees them."""
        chats = []
        for msg in messages:
            chat: list[dict] = []
            if msg.system:
                chat.append({"role": "system", "content": msg.system})
            chat.append({"role": "user", "content": msg.user})
            chats.append(chat)

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

        return texts

    def _length_sorted_indices(self, messages: list[ChatMessage]) -> list[int]:
        """Return a stable ordering by rendered, unpadded token length."""
        if not messages:
            return []

        encoded = self.tokenizer(
            self._render_messages(messages),
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        input_ids = encoded["input_ids"]
        lengths = [len(row) for row in input_ids]
        if len(lengths) != len(messages):
            raise RuntimeError(
                "Tokenizer returned a different number of rows while sorting "
                f"messages: expected {len(messages)}, got {len(lengths)}"
            )
        return sorted(range(len(messages)), key=lambda index: lengths[index])

    @staticmethod
    def _restore_tensor_rows(values: Tensor, original_indices: list[int]) -> Tensor:
        """Restore rows whose current positions follow ``original_indices``."""
        inverse = torch.empty(
            len(original_indices),
            dtype=torch.long,
            device=values.device,
        )
        for sorted_index, original_index in enumerate(original_indices):
            inverse[original_index] = sorted_index
        return values.index_select(0, inverse)

    def _tokenize(self, messages: list[ChatMessage]) -> BatchEncoding:
        """Apply the chat template, optionally prepend the response prefix, and tokenise."""
        texts = self._render_messages(messages)

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
        resolved_max = max_new_tokens or self.config.inference.max_gen_tokens
        resolved_min = min_new_tokens
        if resolved_min is None and max_new_tokens is None:
            resolved_min = self.config.inference.min_gen_tokens

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": resolved_max,
        }
        if resolved_min is not None:
            if resolved_min > resolved_max:
                raise ValueError(
                    f"min_gen_tokens ({resolved_min}) cannot exceed "
                    f"max_gen_tokens ({resolved_max})"
                )
            gen_kwargs["min_new_tokens"] = resolved_min
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
        *,
        sort_by_length: bool = False,
    ) -> list[str]:
        """Batched generation, optionally grouped by near-equal token length."""
        original_indices = (
            self._length_sorted_indices(messages)
            if sort_by_length
            else list(range(len(messages)))
        )
        ordered_messages = [messages[index] for index in original_indices]

        out: list[str] = []
        for batch in chunk_batches(
            ordered_messages,
            self.config.inference.batch_size,
        ):
            out.extend(
                self.generate_text(
                    batch,
                    skip_special_tokens=skip_special_tokens,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                )
            )
        if not sort_by_length:
            return out

        restored = ["" for _ in out]
        for sorted_index, original_index in enumerate(original_indices):
            restored[original_index] = out[sorted_index]
        return restored

    def generate_and_score(
        self,
        messages: list[ChatMessage],
        max_new_tokens: int,
        kl_token_count: int,
        skip_special_tokens: bool = False,
        min_new_tokens: int | None = None,
    ) -> tuple[list[str], Tensor]:
        """Generate full responses AND capture early-token logprobs in one pass.

        Avoids the duplicate-prefill overhead of calling generate_text() and
        compute_logprobs() separately on the same prompt batch.
        """
        if kl_token_count < 1:
            raise ValueError("kl_token_count must be at least 1")
        if kl_token_count > max_new_tokens:
            raise ValueError(
                f"kl_token_count ({kl_token_count}) cannot exceed "
                f"max_new_tokens ({max_new_tokens})"
            )

        sampler = _LogitsSampler(kl_token_count)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "logits_processor": [sampler],
        }
        resolved_min = min_new_tokens
        if kl_token_count > 1:
            resolved_min = max(resolved_min or 0, kl_token_count)
        if resolved_min is not None:
            if resolved_min > max_new_tokens:
                raise ValueError(
                    f"min_gen_tokens ({resolved_min}) cannot exceed "
                    f"max_gen_tokens ({max_new_tokens})"
                )
            gen_kwargs["min_new_tokens"] = resolved_min

        inputs, outputs = self._generate(messages, **gen_kwargs)
        logprobs = _captured_logprobs(sampler, kl_token_count)

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
        min_new_tokens: int | None = None,
    ) -> tuple[list[str], Tensor]:
        """Batched wrapper around :meth:`generate_and_score`."""
        all_resp: list[str] = []
        all_lp: list[Tensor] = []
        for batch in chunk_batches(messages, self.config.inference.batch_size):
            resp, lp = self.generate_and_score(
                batch,
                max_new_tokens=max_new_tokens,
                kl_token_count=kl_token_count,
                skip_special_tokens=skip_special_tokens,
                min_new_tokens=min_new_tokens,
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

        # Most recent Transformers causal-LM heads can avoid materialising
        # logits for every prompt token.  This matters especially for Qwen3.5
        # (248k vocabulary): hidden-state extraction only consumes residuals,
        # so retaining one final-token logit slice is sufficient.  Inspect the
        # explicit base-model signature first to keep older/custom models
        # compatible instead of optimistically passing an unknown keyword.
        cache = getattr(self, "_logits_to_keep_support", None)
        model_identity = id(self.model)
        if cache is None or cache[0] != model_identity:
            cache = (
                model_identity,
                _forward_supports_logits_to_keep(cast(Module, self.model)),
            )
            self._logits_to_keep_support = cache

        forward_kwargs: dict[str, Any] = {"output_hidden_states": True}
        if cache[1]:
            forward_kwargs["logits_to_keep"] = 1

        outputs = self.model(**inputs, **forward_kwargs)
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

    def extract_hidden_states_batched(
        self,
        messages: list[ChatMessage],
        *,
        sort_by_length: bool = False,
    ) -> Tensor:
        """Extract residuals, optionally using near-length prompt batches."""
        if not messages:
            raise ValueError("messages must not be empty")

        original_indices = (
            self._length_sorted_indices(messages)
            if sort_by_length
            else list(range(len(messages)))
        )
        ordered_messages = [messages[index] for index in original_indices]

        offload = getattr(
            self.config.inference,
            "offload_outputs_to_cpu",
            False,
        )
        parts = []
        for batch in chunk_batches(
            ordered_messages,
            self.config.inference.batch_size,
        ):
            part = self.extract_hidden_states(batch)
            if offload:
                # Move each batch's residuals to host RAM immediately so the
                # full (n_prompts × layers × hidden) stack never co-resides in
                # VRAM. Vectors are moved back to the compute device when the
                # steering adapter is applied.
                part = part.cpu()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            parts.append(part)
        residuals = torch.cat(parts, dim=0)
        if not sort_by_length:
            return residuals
        return self._restore_tensor_rows(residuals, original_indices)

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
        """Compute next-token log-probabilities over ``token_count`` steps.

        The legacy single-token result has shape ``(batch, vocab)``.  For
        multiple tokens, the result has shape ``(batch, step, vocab)`` so
        callers can compute a normalized divergence for each generated step.
        """
        n = self.config.kl.token_count

        if n < 1:
            raise ValueError("kl.token_count must be at least 1")
        if n == 1:
            return self._logprobs_forward_pass(messages)

        sampler = _LogitsSampler(n)
        self._generate(
            messages,
            max_new_tokens=n,
            min_new_tokens=n,
            logits_processor=[sampler],
        )
        return _captured_logprobs(sampler, n)

    def compute_logprobs_batched(self, messages: list[ChatMessage]) -> Tensor:
        offload = getattr(
            self.config.inference,
            "offload_outputs_to_cpu",
            False,
        )
        parts = []
        for batch in chunk_batches(messages, self.config.inference.batch_size):
            part = self.compute_logprobs(batch)
            if offload:
                # Offload per-batch logprobs to host RAM to cap peak VRAM during
                # baseline capture and per-trial KL. Baseline and current
                # logprobs both flow through here, so KL is computed device-
                # consistently (both on CPU when offload is enabled).
                part = part.cpu()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            parts.append(part)
        return torch.cat(parts, dim=0)

    def score_continuation_logprobs_batched(
        self,
        messages: list[ChatMessage],
        continuations: list[str],
        token_count: int,
        *,
        sort_by_length: bool = False,
    ) -> Tensor:
        """Score fixed continuations on identical teacher-forced prefixes.

        For continuation token ``t`` this returns the full-vocabulary
        distribution conditioned on the prompt and continuation tokens before
        ``t``.  Reusing the same continuations for baseline and steered models
        therefore measures both models on the same contexts instead of on
        their independently generated trajectories.

        A one-token result has shape ``(batch, vocab)``; multiple tokens have
        shape ``(batch, step, vocab)``.  Results are always normalised in
        ``float32``.
        """
        if token_count < 1:
            raise ValueError("token_count must be at least 1")
        if len(messages) != len(continuations):
            raise ValueError(
                "messages and continuations must have the same length: "
                f"got {len(messages)} and {len(continuations)}"
            )
        if not messages:
            raise ValueError("messages and continuations must not be empty")

        original_indices = (
            self._length_sorted_indices(messages)
            if sort_by_length
            else list(range(len(messages)))
        )
        ordered_messages = [messages[index] for index in original_indices]
        ordered_continuations = [continuations[index] for index in original_indices]

        all_logprobs: list[Tensor] = []
        batch_size = self.config.inference.batch_size
        for start in range(0, len(ordered_messages), batch_size):
            message_batch = ordered_messages[start : start + batch_size]
            continuation_batch = ordered_continuations[start : start + batch_size]

            encoded_continuations = self.tokenizer(
                continuation_batch,
                add_special_tokens=False,
                padding=False,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            continuation_ids = encoded_continuations["input_ids"]
            if isinstance(continuation_ids, Tensor):
                continuation_rows = continuation_ids.tolist()
            else:
                continuation_rows = [list(row) for row in continuation_ids]

            for row_index, continuation in enumerate(continuation_rows):
                if len(continuation) < token_count:
                    original_index = original_indices[start + row_index]
                    raise ValueError(
                        "Continuation is shorter than token_count after "
                        f"tokenization at batch index {original_index}: "
                        f"expected at least {token_count}, got {len(continuation)}"
                    )

            forced_rows = [row[:token_count] for row in continuation_rows]
            sampler = _LogitsSampler(token_count)
            forcing = _ForcedContinuationProcessor(forced_rows)
            self._generate(
                message_batch,
                max_new_tokens=token_count,
                logits_processor=[sampler, forcing],
            )
            all_logprobs.append(_captured_logprobs(sampler, token_count))

        logprobs = torch.cat(all_logprobs, dim=0)
        if not sort_by_length:
            return logprobs
        return self._restore_tensor_rows(logprobs, original_indices)

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
