# Abliterix — speculators-based hidden state extraction
# Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Fast hidden state extraction using vLLM tensor parallelism.

Uses the ``speculators`` library (part of the vLLM ecosystem) to extract
per-layer hidden states with full tensor parallelism — dramatically faster
than HuggingFace ``device_map="auto"`` pipeline parallelism.

The standard HF approach processes layers sequentially across GPUs (~4 tok/s
on 4× H100 for a 200B+ MoE model).  With speculators + vLLM TP=4, all GPUs
compute each layer simultaneously, yielding 10-15× higher throughput.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from transformers import AutoTokenizer

from ..settings import AbliterixConfig
from ..types import ChatMessage
from ..util import flush_memory, print


def extract_hidden_states_speculators(
    config: AbliterixConfig,
    messages: list[ChatMessage],
    token_offset: int = -1,
) -> Tensor:
    """Extract per-layer hidden states using speculators + vLLM TP.

    Parameters
    ----------
    config : AbliterixConfig
        Model and inference configuration.
    messages : list[ChatMessage]
        Prompts to extract hidden states from.
    token_offset : int
        Position in the sequence to extract from.  ``-1`` (default) extracts
        the final token (where refusal is encoded).

    Returns
    -------
    Tensor
        Shape ``(batch, layers+1, hidden_dim)``, matching the HF convention.
        Index 0 is a zero placeholder for the embedding layer (never used
        for steering); indices 1..N are the decoder layer outputs.
    """
    from speculators.data_generation import VllmHiddenStatesGenerator

    model_id = config.model.model_id
    tp = config.model.tensor_parallel_size
    if tp is None:
        tp = torch.cuda.device_count()

    # Resolve number of layers for layer_ids.
    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained(
        model_id, trust_remote_code=True,
    )
    num_layers = model_config.num_hidden_layers
    layer_ids = list(range(num_layers))

    print(f"* Loading model in speculators with TP={tp}...")

    kwargs: dict[str, Any] = dict(
        model_path=model_id,
        layer_ids=layer_ids,
        tensor_parallel_size=tp,
        gpu_memory_utilization=config.model.gpu_memory_utilization,
        max_model_len=2048,
    )

    generator = VllmHiddenStatesGenerator(**kwargs)

    # Tokenize prompts using the model's tokenizer + chat template.
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True,
    )

    token_ids_list: list[list[int]] = []
    for msg in messages:
        chat = [
            {"role": "system", "content": msg.system},
            {"role": "user", "content": msg.user},
        ]
        try:
            text = tokenizer.apply_chat_template(
                chat, add_generation_prompt=True,
                tokenize=False, enable_thinking=False,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                chat, add_generation_prompt=True, tokenize=False,
            )
        ids = tokenizer.encode(text, add_special_tokens=False)
        token_ids_list.append(ids)

    print(f"* Extracting hidden states for {len(messages)} prompts "
          f"({num_layers} layers, TP={tp})...")

    results = generator.generate(token_ids=token_ids_list)

    # Collect hidden states at the requested token offset.
    # results[i]["hidden_states"] is list[Tensor] with shape (seq_len, hidden_dim)
    batch_residuals: list[Tensor] = []

    for r in results:
        hs_list = r["hidden_states"]  # list of (seq_len, hidden_dim) per layer
        # Stack decoder layers: (num_layers, seq_len, hidden_dim)
        stacked = torch.stack(hs_list, dim=0)
        # Extract at token_offset: (num_layers, hidden_dim)
        layer_vecs = stacked[:, token_offset, :]

        # Prepend a zeros row for the embedding layer (index 0) to match
        # the HF convention of (layers+1, hidden_dim) where index 0 is
        # the embedding output.  The embedding layer is never used for
        # steering (all code references layer_idx+1), so zeros are safe.
        hidden_dim = layer_vecs.shape[1]
        embedding_placeholder = torch.zeros(1, hidden_dim, dtype=layer_vecs.dtype)
        # (num_layers+1, hidden_dim)
        batch_residuals.append(torch.cat([embedding_placeholder, layer_vecs], dim=0))

    # (batch, num_layers+1, hidden_dim) — matches HF extract_hidden_states shape
    residuals = torch.stack(batch_residuals, dim=0).to(torch.float32)

    print(f"  [green]Ok[/] — shape: {list(residuals.shape)}")

    # Cleanup: delete the generator to free vLLM VRAM.
    del generator
    flush_memory()

    return residuals
