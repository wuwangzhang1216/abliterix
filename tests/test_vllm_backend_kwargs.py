"""Unit tests for the helper functions that decide vLLM kwargs.

These cover the de-hardcoded paths added by PRD #20:

- ``_resolve_attention_backend`` — MLA-aware attention backend selection.
- ``_should_disable_custom_all_reduce`` — Blackwell PCIe sm_120 detection.

We avoid touching ``VLLMGenerator.__init__`` directly because it imports
vLLM, which CI does not have.
"""

from __future__ import annotations

from unittest.mock import patch

from abliterix.core.vllm_backend import (
    _MLA_ARCH_FRAGMENTS,
    _resolve_attention_backend,
    _should_disable_custom_all_reduce,
)


# ---------------------------------------------------------------------------
# _resolve_attention_backend
# ---------------------------------------------------------------------------


def test_user_override_always_wins():
    """A non-None config value short-circuits all auto-detection — even
    on an MLA model, even on a sink-attention model."""
    assert _resolve_attention_backend("FLASHMLA", "DeepseekV2ForCausalLM") == "FLASHMLA"
    assert _resolve_attention_backend("FLASH_ATTN", "GptOssForCausalLM") == "FLASH_ATTN"
    assert (
        _resolve_attention_backend("TRITON_ATTN", "LlamaForCausalLM") == "TRITON_ATTN"
    )


def test_mla_models_get_flash_attn_mla():
    """Every architecture name that contains an MLA fragment must route
    to ``FLASH_ATTN_MLA`` — vLLM 0.20.x rejects ``TRITON_ATTN`` for these."""
    for frag in _MLA_ARCH_FRAGMENTS:
        arch = f"{frag}ForCausalLM"
        result = _resolve_attention_backend(None, arch)
        assert result == "FLASH_ATTN_MLA", f"{arch} → {result}"


def test_minimax_m27_routes_to_mla():
    """Concrete regression: MiniMax-M2.7 is the model that triggered the
    PRD finding that abliterix's hardcoded TRITON_ATTN crashes MLA."""
    assert _resolve_attention_backend(None, "MiniMaxM27ForCausalLM") == "FLASH_ATTN_MLA"


def test_gpt_oss_keeps_triton_attn():
    """gpt-oss has attention sinks; FLASH_ATTN explicitly errors on it."""
    assert _resolve_attention_backend(None, "GptOssForCausalLM") == "TRITON_ATTN"


def test_unknown_arch_returns_none():
    """Plain dense models fall through to vLLM's own default — we return
    None so __init__ knows to skip the attention_config kwarg entirely."""
    assert _resolve_attention_backend(None, "LlamaForCausalLM") is None
    assert _resolve_attention_backend(None, "Qwen3ForCausalLM") is None
    assert _resolve_attention_backend(None, "") is None


def test_empty_arch_with_user_override():
    """Even when arch detection fails (empty string), a user override
    must still apply."""
    assert _resolve_attention_backend("FLASH_ATTN", "") == "FLASH_ATTN"


# ---------------------------------------------------------------------------
# _should_disable_custom_all_reduce
# ---------------------------------------------------------------------------


def test_user_override_true_wins():
    """Explicit True from config bypasses auto-detection."""
    with patch(
        "abliterix.core.vllm_backend.torch.cuda.is_available", return_value=True
    ):
        with patch(
            "abliterix.core.vllm_backend.torch.cuda.get_device_capability",
            return_value=(9, 0),  # Hopper, would auto-detect False
        ):
            assert _should_disable_custom_all_reduce(True) is True


def test_user_override_false_wins():
    """Explicit False from config bypasses auto-detection — even on the
    Blackwell PCIe (sm_120) device that would auto-True."""
    with patch(
        "abliterix.core.vllm_backend.torch.cuda.is_available", return_value=True
    ):
        with patch(
            "abliterix.core.vllm_backend.torch.cuda.get_device_capability",
            return_value=(12, 0),
        ):
            assert _should_disable_custom_all_reduce(False) is False


def test_auto_detect_blackwell_pcie_returns_true():
    """sm_120 is Blackwell PCIe (RTX PRO 6000) — known deadlock without
    NVLink. Should auto-True."""
    with patch(
        "abliterix.core.vllm_backend.torch.cuda.is_available", return_value=True
    ):
        with patch(
            "abliterix.core.vllm_backend.torch.cuda.get_device_capability",
            return_value=(12, 0),
        ):
            assert _should_disable_custom_all_reduce(None) is True


def test_auto_detect_hopper_returns_false():
    """sm_90 (H100) keeps the custom all-reduce — NVLink is fine."""
    with patch(
        "abliterix.core.vllm_backend.torch.cuda.is_available", return_value=True
    ):
        with patch(
            "abliterix.core.vllm_backend.torch.cuda.get_device_capability",
            return_value=(9, 0),
        ):
            assert _should_disable_custom_all_reduce(None) is False


def test_auto_detect_blackwell_sxm_returns_false():
    """sm_100 (B100/B200 SXM) has NVLink — should not trigger the workaround."""
    with patch(
        "abliterix.core.vllm_backend.torch.cuda.is_available", return_value=True
    ):
        with patch(
            "abliterix.core.vllm_backend.torch.cuda.get_device_capability",
            return_value=(10, 0),
        ):
            assert _should_disable_custom_all_reduce(None) is False


def test_auto_detect_no_cuda_returns_false():
    """No GPU detected — nothing to disable."""
    with patch(
        "abliterix.core.vllm_backend.torch.cuda.is_available", return_value=False
    ):
        assert _should_disable_custom_all_reduce(None) is False
