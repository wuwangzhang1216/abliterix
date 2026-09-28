# Abliterix — vLLM native hidden state extraction
# Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Extract per-layer hidden states using vLLM 0.19's native API.

Replaces the ``speculators`` library which is incompatible with vLLM 0.19.
Uses vLLM's built-in ``extract_hidden_states`` speculative method and
``ExampleHiddenStatesConnector`` to extract hidden states with full tensor
parallelism — all GPUs compute simultaneously.

Requires vLLM >= 0.17.0 (PR #33736).
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from torch import Tensor
from transformers import AutoConfig, AutoTokenizer

from ..settings import AbliterixConfig
from ..types import ChatMessage
from ..util import flush_memory, print


_SUPPORTED_MODEL_TYPES = {
    "llama",
    "qwen",
    "minicpm",
    "gpt_oss",
    "hunyuan_vl",
    "hunyuan_v1_dense",
    "afmoe",
    "nemotron_h",
    "deepseek_v2",
    "deepseek_v3",
    "kimi_k2",
    "kimi_k25",
    "gemma4",
    "gemma4_text",
    # "minimax_m2" — disabled: vLLM 0.19.1's extract_hidden_states path on
    # MiniMax-M2 (62 layers × eagle_aux_hidden_state hooks) deadlocks after
    # NCCL init on Blackwell PCIe (sm_120 RTX PRO 6000) — workers spin at
    # 100% CPU / idle GPU, no weight loading. disable_custom_all_reduce=True
    # does not help. Falls back to HF pipeline parallelism for Phase 1
    # (1-GPU-busy, ~30 min). Re-enable when vLLM upstream fixes this
    # (tracked in issue #33041 + related MiniMax extract_hidden_states).
    # "step3p5" — similar incompatibility; falls back to HF PP.
}


def _load_text_config_data(
    model_id: str,
    trust_remote_code: bool,
    revision: str | None = None,
) -> dict[str, Any]:
    """Return text config fields even when Transformers lacks a fresh model type."""
    try:
        auto_cfg = AutoConfig.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            revision=revision,
        )
        text_cfg = getattr(auto_cfg, "text_config", auto_cfg)
        if hasattr(text_cfg, "to_dict"):
            return text_cfg.to_dict()
        return vars(text_cfg)
    except ValueError as exc:
        if "does not recognize this architecture" not in str(exc):
            raise

    cfg_path = hf_hub_download(model_id, "config.json", revision=revision)
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    text_cfg = cfg.get("text_config") or cfg
    return text_cfg


def is_model_supported(config: AbliterixConfig) -> bool:
    """Check if the model's architecture is in extract_hidden_states whitelist."""
    try:
        text_cfg = _load_text_config_data(
            config.model.model_id,
            config.model.trust_remote_code or False,
            config.model.revision,
        )
        model_type = text_cfg.get("model_type", "")
        return model_type in _SUPPORTED_MODEL_TYPES
    except Exception:
        return False


def extract_hidden_states_vllm(
    config: AbliterixConfig,
    prompt_sets: dict[str, list[ChatMessage]],
    token_offset: int = -1,
) -> dict[str, Tensor]:
    """Extract per-layer hidden states using vLLM's native extraction API.

    Loads the model once and extracts hidden states for every prompt set in
    a single ``llm.generate`` call — a reload per set would pay the MooseFS
    shard pull (~2.5 min for 15-shard models) for each one.

    Parameters
    ----------
    config : AbliterixConfig
        Model and inference configuration.
    prompt_sets : dict[str, list[ChatMessage]]
        Named prompt sets (e.g. ``{"benign": [...], "target": [...]}``).
        All sets are tokenized and submitted together; outputs are split
        back by name on return.
    token_offset : int
        Position in the sequence to extract from.  ``-1`` (default) extracts
        the final token.

    Returns
    -------
    dict[str, Tensor]
        Same keys as ``prompt_sets``.  Each tensor has shape
        ``(batch, layers+1, hidden_dim)``.  Index 0 along the layer axis is
        a zero placeholder for the embedding layer; indices 1..N are decoder
        layer outputs.
    """
    from .vllm_compat import install_gemma4_transformers_compat

    install_gemma4_transformers_compat()

    from vllm import LLM, SamplingParams

    model_id = config.model.model_id
    tp = config.model.tensor_parallel_size
    if tp is None:
        tp = torch.cuda.device_count()
    trust = config.model.trust_remote_code or False

    # Get number of layers from config. Some newly released vLLM-supported
    # models may not exist in the installed Transformers registry yet.
    text_cfg = _load_text_config_data(model_id, trust, config.model.revision)
    num_layers = text_cfg["num_hidden_layers"]
    # Extract ALL layers.
    layer_ids = list(range(num_layers))

    print(f"* Loading model in vLLM with TP={tp} for hidden state extraction...")

    # The connector writes per-layer hidden states for every processed prompt
    # — for a 36-layer 400-prompt run that's ~16 GB.  Container /tmp is often
    # a 20 GB overlayfs that fills fast; route to /workspace (or configured
    # HF_HOME parent) which on RunPod is a 1 TB+ volume.
    hs_parent = os.environ.get("AX_HIDDEN_STATES_DIR")
    if not hs_parent:
        hf_home = os.environ.get("HF_HOME") or os.environ.get("HF_HUB_CACHE")
        hs_parent = (os.path.dirname(hf_home) if hf_home else None) or (
            "/workspace" if os.path.isdir("/workspace") else None
        )
    if hs_parent and os.path.isdir(hs_parent):
        os.makedirs(hs_parent, exist_ok=True)
        tmpdir = tempfile.mkdtemp(prefix="abliterix_hs_", dir=hs_parent)
    else:
        tmpdir = tempfile.mkdtemp(prefix="abliterix_hs_")

    draft_hf_config = dict(text_cfg)
    draft_hf_config["eagle_aux_hidden_state_layer_ids"] = layer_ids

    kwargs: dict[str, Any] = dict(
        model=model_id,
        revision=config.model.revision,
        tensor_parallel_size=tp,
        gpu_memory_utilization=config.model.gpu_memory_utilization,
        trust_remote_code=trust,
        enforce_eager=True,  # Safer for extraction
        # Disable custom all-reduce on Blackwell PCIe (sm_120 RTX PRO 6000 etc.):
        # NCCL inits but engine deadlocks before weight loading (vllm issue
        # #33041). Same fix as vllm_backend.py.
        disable_custom_all_reduce=True,
        speculative_config={
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": draft_hf_config,
            },
        },
        kv_transfer_config={
            "kv_connector": "ExampleHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {
                "shared_storage_path": tmpdir,
            },
        },
    )
    # Forward max_model_len so vLLM doesn't reserve KV for the model's full
    # native context (e.g. 196K for MiniMax-M2 would require 81GB/GPU for
    # a single request and OOMs the engine). Abliteration prompts + gen are
    # < 2K, so the config-level cap is the right ceiling.
    if config.model.max_model_len:
        kwargs["max_model_len"] = config.model.max_model_len
    if config.model.max_num_seqs:
        kwargs["max_num_seqs"] = config.model.max_num_seqs

    # Model config overrides (e.g. MTP-3 → MTP-1 for Step-3.5-Flash).
    if config.model.hf_overrides:
        kwargs["hf_overrides"] = config.model.hf_overrides

    # FP8: let vLLM auto-detect from config.json.
    is_fp8 = config.model.quant_method and config.model.quant_method.value == "fp8"
    if is_fp8:
        kwargs["quantization"] = "fp8"

    # ``LLM(**kwargs)`` is the most likely failure point here (OOM, bad
    # max_model_len, unsupported arch). It has to be inside the try: the
    # finally block below exists precisely to guarantee teardown, and its
    # own comment notes that a partially-initialised engine pins the frame
    # and keeps its VRAM. It also guarantees the temp dir is removed.
    try:
        llm = LLM(**kwargs)
    except BaseException:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise

    try:
        # Tokenize prompts.  Flatten every set into a single prompt list and
        # remember each set's slice so we can split hidden states back on return.
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=trust,
                revision=config.model.revision,
            )
        except AttributeError as exc:
            if "'list' object has no attribute 'keys'" not in str(exc):
                raise
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=trust,
                revision=config.model.revision,
                extra_special_tokens={},
            )
        prompts: list[str] = []
        set_slices: dict[str, tuple[int, int]] = {}
        for set_name, messages in prompt_sets.items():
            start = len(prompts)
            for msg in messages:
                chat = []
                if msg.system:
                    chat.append({"role": "system", "content": msg.system})
                chat.append({"role": "user", "content": msg.user})
                try:
                    text = tokenizer.apply_chat_template(
                        chat,
                        add_generation_prompt=True,
                        tokenize=False,
                        enable_thinking=False,
                    )
                except TypeError:
                    text = tokenizer.apply_chat_template(
                        chat,
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                prompts.append(text)
            set_slices[set_name] = (start, len(prompts))

        set_summary = ", ".join(
            f"{name}={end - start}" for name, (start, end) in set_slices.items()
        )
        print(
            f"* Extracting hidden states for {len(prompts)} prompts "
            f"({set_summary}; {num_layers} layers, TP={tp})..."
        )

        sampling_params = SamplingParams(max_tokens=1)
        outputs = llm.generate(prompts, sampling_params)

        # Collect hidden states from safetensors files.
        batch_residuals: list[Tensor] = []
        for out in outputs:
            hs_path = out.kv_transfer_params.get("hidden_states_path")
            if hs_path is None:
                raise RuntimeError(
                    f"No hidden_states_path in output for request {out.request_id}. "
                    f"kv_transfer_params={out.kv_transfer_params}"
                )

            with safe_open(hs_path, "pt") as f:
                hs = f.get_tensor("hidden_states")

            # vLLM's ExampleHiddenStatesConnector stores hidden states as
            # [prompt_len, num_layers, hidden_dim] — NOT [num_layers, prompt_len,
            # hidden_dim] as earlier versions of this file assumed.  The wrong
            # ordering caused torch.stack across prompts to fail with
            # "stack expects equal size, got [180, 2880] and [113, 2880]"
            # (the varying 1st dim was prompt_len, not num_layers).
            # Auto-detect which axis is the layer axis by finding the one whose
            # length matches our known num_layers; fall back to the
            # prompt_len-first interpretation since that's what we've observed
            # on gpt-oss / vLLM 0.19.
            if hs.dim() == 3:
                if hs.shape[0] == num_layers:
                    # [num_layers, prompt_len, hidden_dim] — original assumption.
                    layer_vecs = hs[:, token_offset, :]
                elif hs.shape[1] == num_layers:
                    # [prompt_len, num_layers, hidden_dim] — observed on gpt-oss.
                    layer_vecs = hs[token_offset, :, :]
                else:
                    raise RuntimeError(
                        f"Unexpected hidden_states shape {list(hs.shape)} "
                        f"(num_layers={num_layers}); neither dim 0 nor dim 1 "
                        f"matches the layer count."
                    )
            else:
                raise RuntimeError(
                    f"Expected 3-D hidden_states tensor, got shape {list(hs.shape)}"
                )

            # Prepend zeros for embedding layer (index 0).
            hidden_dim = layer_vecs.shape[1]
            embedding_placeholder = torch.zeros(1, hidden_dim, dtype=layer_vecs.dtype)
            batch_residuals.append(
                torch.cat([embedding_placeholder, layer_vecs], dim=0)
            )

        residuals = torch.stack(batch_residuals, dim=0).to(torch.float32)

        results: dict[str, Tensor] = {}
        for set_name, (start, end) in set_slices.items():
            results[set_name] = residuals[start:end].contiguous()
            print(f"  [green]Ok[/] — {set_name}: shape {list(results[set_name].shape)}")

    finally:
        # Guaranteed teardown: on the exception path the caller's handler
        # holds exc.__traceback__, which pins this frame and with it the
        # vLLM engine (the fall-through `del llm` never ran), so no amount
        # of gc in the handler could free the GPUs for the fallback
        # backend.  Deleting the frame's slot here lets flush_memory()
        # actually tear the engine down (issue #83 failure shape).
        del llm
        flush_memory()

        # Clean up temp files.
        shutil.rmtree(tmpdir, ignore_errors=True)

    return results
