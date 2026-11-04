"""Observe the single-vector reads made by native HF hyper-connection models."""

from contextlib import ExitStack
from threading import get_ident

import torch
from peft import PeftModel
from torch import Tensor, nn


def native_mhc_backbone(model: nn.Module) -> nn.Module | None:
    """Identify the audited native text backbone, including PEFT and VLM wrappers."""
    base = model.get_base_model() if isinstance(model, PeftModel) else model
    family = getattr(getattr(base, "config", None), "model_type", None)
    if family not in {"deepseek_v4", "glm5_next", "glm5_next_text"}:
        return None
    backbone = getattr(base, "base_model", base)
    if family == "glm5_next":
        backbone = getattr(backbone, "language_model", None)
    # A remote implementation can reuse a model_type with different read sites.
    cls = type(backbone)
    if (cls.__module__, cls.__name__) in {
        ("transformers.models.deepseek_v4.modeling_deepseek_v4", "DeepseekV4Model"),
        ("transformers.models.glm5_next.modeling_glm5_next", "Glm5NextTextModel"),
    }:
        return backbone
    return None


@torch.no_grad()
def capture_mhc_reads(model, backbone, inputs, token_offset, forward_kwargs) -> Tensor:
    """Capture each attention read before RMSNorm and the final LM-facing state.

    The actual forward performs all routing. Only one token per site is kept,
    even when the model is sharded or the prompt contains many tokens.
    """
    ids = inputs["input_ids"]
    expected_shape = (*ids.shape, backbone.config.hidden_size)
    reads: list[Tensor] = []
    owner = get_ident()

    def record(index, state):
        if get_ident() != owner:
            return  # Temporary hooks also see other threads using the same model.
        if index != len(reads) or state.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected mHC read at site {index}: shape {tuple(state.shape)}, "
                f"expected site {len(reads)} with shape {expected_shape}"
            )
        reads.append(
            state[:, token_offset].to(ids.device, dtype=torch.float32, copy=True)
        )

    with ExitStack() as cleanup:
        for index, layer in enumerate(backbone.layers):
            handle = layer.input_layernorm.register_forward_pre_hook(
                lambda _module, args, i=index: record(i, args[0])
            )
            cleanup.callback(handle.remove)
        terminal = len(backbone.layers)
        handle = backbone.norm.register_forward_hook(
            lambda _module, _args, output: record(terminal, output)
        )
        cleanup.callback(handle.remove)
        model(**inputs, **forward_kwargs, output_hidden_states=False, use_cache=False)
    if len(reads) != terminal + 1:
        raise RuntimeError(
            f"Expected {terminal + 1} native mHC reads, got {len(reads)}"
        )
    return torch.stack(reads, dim=1)
