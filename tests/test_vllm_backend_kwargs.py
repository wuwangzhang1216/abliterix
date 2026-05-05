"""Unit tests for the helper functions that decide vLLM kwargs.

These cover the de-hardcoded paths added by PRD #20:

- ``_resolve_attention_backend`` — MLA-aware attention backend selection.
- ``_should_disable_custom_all_reduce`` — Blackwell PCIe sm_120 detection.
- ``_resolve_compile_mode`` — legacy ``enforce_eager`` reconciliation.
- ``_build_llm_kwargs`` — pure kwargs assembly (PR #21 review item 7).

We avoid touching ``VLLMGenerator.__init__`` directly because it imports
vLLM, which CI does not have.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from abliterix.core.vllm_backend import (
    _MLA_ARCH_FRAGMENTS,
    _build_llm_kwargs,
    _resolve_attention_backend,
    _resolve_compile_mode,
    _should_disable_custom_all_reduce,
)
from abliterix.settings import AbliterixConfig, ModelConfig


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


# ---------------------------------------------------------------------------
# _resolve_compile_mode (PR #21 review item 2)
# ---------------------------------------------------------------------------


def test_resolve_compile_mode_legacy_enforce_eager_wins():
    """If a recipe still sets ``enforce_eager=True`` (pre-PRD #20), the
    resolved mode must be 'eager' — that's the only way the two fields
    cannot disagree at vLLM construction time."""
    assert _resolve_compile_mode(True, "eager") == "eager"
    assert _resolve_compile_mode(True, "full_compile") == "eager"


def test_resolve_compile_mode_default_path():
    """Default recipe (enforce_eager=False, vllm_compile_mode='eager')
    keeps the historical eager behaviour."""
    assert _resolve_compile_mode(False, "eager") == "eager"


def test_resolve_compile_mode_full_compile():
    assert _resolve_compile_mode(False, "full_compile") == "full_compile"


# ---------------------------------------------------------------------------
# _build_llm_kwargs — integration test (PR #21 review item 7)
# ---------------------------------------------------------------------------


def _make_config(**model_overrides):
    """Build an AbliterixConfig with the model fields under test, defaulting
    everything else. Avoids loading a real TOML file from disk."""
    return AbliterixConfig.model_construct(
        model=ModelConfig(model_id="dummy/model", **model_overrides)
    )


def test_build_llm_kwargs_default_recipe():
    """Default recipe on an unknown architecture: triton MoE backend,
    no attention_config (vLLM picks), eager compile, LoRA enabled."""
    cfg = _make_config()
    with patch("abliterix.core.vllm_backend.torch.cuda.device_count", return_value=1):
        with patch(
            "abliterix.core.vllm_backend.torch.cuda.is_available", return_value=True
        ):
            with patch(
                "abliterix.core.vllm_backend.torch.cuda.get_device_capability",
                return_value=(9, 0),
            ):
                kwargs = _build_llm_kwargs(
                    cfg,
                    model_arch="LlamaForCausalLM",
                    is_fp8=False,
                    kv_cache_dtype=None,
                    lora_max_rank=16,
                )
    assert kwargs["model"] == "dummy/model"
    assert kwargs["moe_backend"] == "triton"
    assert kwargs["compilation_config"] == {"mode": 0, "cudagraph_mode": 0}
    assert "attention_config" not in kwargs  # arch is non-MLA, non-sink
    assert kwargs["limit_mm_per_prompt"] == {"image": 0, "video": 0, "audio": 0}
    assert kwargs["enable_lora"] is True
    assert kwargs["max_lora_rank"] == 16
    assert "kv_cache_dtype" not in kwargs
    assert "quantization" not in kwargs
    # PR #21 review item 2: enforce_eager kwarg must NOT be passed —
    # compilation_config encodes the same intent and we don't want both.
    assert "enforce_eager" not in kwargs
    # Issue #22: routed_experts replaces collective_rpc probe by default.
    assert kwargs["enable_return_routed_experts"] is True


def test_build_llm_kwargs_routed_experts_can_be_disabled():
    """Issue #22: users can opt out of the routed_experts metadata cost
    by setting ``vllm_return_routed_experts = false``. The kwarg still
    appears (vLLM accepts both True and False), so the legacy
    collective_rpc probe path can be re-enabled at config time."""
    cfg = _make_config(vllm_return_routed_experts=False)
    with patch("abliterix.core.vllm_backend.torch.cuda.device_count", return_value=1):
        with patch(
            "abliterix.core.vllm_backend.torch.cuda.is_available", return_value=True
        ):
            with patch(
                "abliterix.core.vllm_backend.torch.cuda.get_device_capability",
                return_value=(9, 0),
            ):
                kwargs = _build_llm_kwargs(
                    cfg,
                    model_arch="LlamaForCausalLM",
                    is_fp8=False,
                    kv_cache_dtype=None,
                    lora_max_rank=16,
                )
    assert kwargs["enable_return_routed_experts"] is False


def test_build_llm_kwargs_mla_model_gets_flash_attn_mla():
    """Regression: DeepSeek-V2/V3 / MiniMax-M2.x must trigger MLA backend."""
    cfg = _make_config()
    with patch("abliterix.core.vllm_backend.torch.cuda.device_count", return_value=1):
        with patch(
            "abliterix.core.vllm_backend.torch.cuda.is_available", return_value=True
        ):
            with patch(
                "abliterix.core.vllm_backend.torch.cuda.get_device_capability",
                return_value=(9, 0),
            ):
                kwargs = _build_llm_kwargs(
                    cfg,
                    model_arch="DeepseekV3ForCausalLM",
                    is_fp8=False,
                    kv_cache_dtype=None,
                    lora_max_rank=16,
                )
    assert kwargs["attention_config"] == {"backend": "FLASH_ATTN_MLA"}


def test_build_llm_kwargs_legacy_enforce_eager_routes_through_compile_mode():
    """PR #21 review item 2: a recipe with ``enforce_eager=true`` and
    ``vllm_compile_mode='full_compile'`` must NOT pass an inconsistent
    pair to vLLM. The legacy field collapses to compile_mode='eager'."""
    cfg = _make_config(enforce_eager=True, vllm_compile_mode="full_compile")
    with patch("abliterix.core.vllm_backend.torch.cuda.device_count", return_value=1):
        with patch(
            "abliterix.core.vllm_backend.torch.cuda.is_available", return_value=True
        ):
            with patch(
                "abliterix.core.vllm_backend.torch.cuda.get_device_capability",
                return_value=(9, 0),
            ):
                kwargs = _build_llm_kwargs(
                    cfg,
                    model_arch="LlamaForCausalLM",
                    is_fp8=False,
                    kv_cache_dtype=None,
                    lora_max_rank=16,
                )
    assert kwargs["compilation_config"] == {"mode": 0, "cudagraph_mode": 0}
    assert "enforce_eager" not in kwargs


def test_build_llm_kwargs_user_attention_override_wins():
    """An explicit attention_backend in the recipe always beats arch detection."""
    cfg = _make_config(attention_backend="FLASHMLA")
    with patch("abliterix.core.vllm_backend.torch.cuda.device_count", return_value=1):
        with patch(
            "abliterix.core.vllm_backend.torch.cuda.is_available", return_value=True
        ):
            with patch(
                "abliterix.core.vllm_backend.torch.cuda.get_device_capability",
                return_value=(9, 0),
            ):
                kwargs = _build_llm_kwargs(
                    cfg,
                    model_arch="DeepseekV3ForCausalLM",  # would auto-route
                    is_fp8=False,
                    kv_cache_dtype=None,
                    lora_max_rank=16,
                )
    assert kwargs["attention_config"] == {"backend": "FLASHMLA"}


def test_build_llm_kwargs_lora_pool_propagates():
    """vllm_max_loras > 1 should set both max_loras and max_cpu_loras."""
    cfg = _make_config(vllm_max_loras=8)
    with patch("abliterix.core.vllm_backend.torch.cuda.device_count", return_value=1):
        with patch(
            "abliterix.core.vllm_backend.torch.cuda.is_available", return_value=True
        ):
            with patch(
                "abliterix.core.vllm_backend.torch.cuda.get_device_capability",
                return_value=(9, 0),
            ):
                kwargs = _build_llm_kwargs(
                    cfg,
                    model_arch="LlamaForCausalLM",
                    is_fp8=False,
                    kv_cache_dtype=None,
                    lora_max_rank=16,
                )
    assert kwargs["max_loras"] == 8
    assert kwargs["max_cpu_loras"] >= 8


def test_build_llm_kwargs_disable_lora_drops_lora_kwargs():
    """When LoRA is disabled, the LoRA group of kwargs must not appear at all."""
    cfg = _make_config(disable_lora=True)
    with patch("abliterix.core.vllm_backend.torch.cuda.device_count", return_value=1):
        with patch(
            "abliterix.core.vllm_backend.torch.cuda.is_available", return_value=True
        ):
            with patch(
                "abliterix.core.vllm_backend.torch.cuda.get_device_capability",
                return_value=(9, 0),
            ):
                kwargs = _build_llm_kwargs(
                    cfg,
                    model_arch="GptOssForCausalLM",
                    is_fp8=False,
                    kv_cache_dtype=None,
                    lora_max_rank=16,
                )
    assert "enable_lora" not in kwargs
    assert "max_lora_rank" not in kwargs
    assert "max_loras" not in kwargs
    assert "lora_target_modules" not in kwargs


def test_build_llm_kwargs_fp8_propagates():
    """is_fp8=True should set quantization='fp8' and kv_cache_dtype if given."""
    cfg = _make_config()
    with patch("abliterix.core.vllm_backend.torch.cuda.device_count", return_value=1):
        with patch(
            "abliterix.core.vllm_backend.torch.cuda.is_available", return_value=True
        ):
            with patch(
                "abliterix.core.vllm_backend.torch.cuda.get_device_capability",
                return_value=(9, 0),
            ):
                kwargs = _build_llm_kwargs(
                    cfg,
                    model_arch="LlamaForCausalLM",
                    is_fp8=True,
                    kv_cache_dtype="fp8_e4m3",
                    lora_max_rank=16,
                )
    assert kwargs["quantization"] == "fp8"
    assert kwargs["kv_cache_dtype"] == "fp8_e4m3"


# ---------------------------------------------------------------------------
# ModelConfig validators (PR #21 review items 4 + 8)
# ---------------------------------------------------------------------------


def test_model_config_rejects_unimplemented_compile_mode():
    """PR #21 review item 4: 'moe_eager_rest_compile' must raise at config
    load until the post-load static_all_moe_layers attach lands."""
    with pytest.raises(ValueError, match="not yet implemented"):
        ModelConfig(model_id="x", vllm_compile_mode="moe_eager_rest_compile")


def test_model_config_rejects_lora_target_modules_with_disable_lora():
    """PR #21 review item 8: lora_target_modules without enable_lora is
    silently dropped today; reject explicitly so users see the
    misconfiguration at config load."""
    with pytest.raises(ValueError, match="lora_target_modules is set but"):
        ModelConfig(
            model_id="x",
            disable_lora=True,
            lora_target_modules=["o_proj"],
        )


def test_model_config_rejects_invalid_moe_backend():
    """PR #21 review item 1: moe_backend is now a Literal — typos must
    fail at config load, not deep inside vllm.LLM()."""
    with pytest.raises(ValueError):
        ModelConfig(model_id="x", moe_backend="tritan")  # typo


def test_model_config_rejects_invalid_compile_mode():
    """PR #21 review item 1: vllm_compile_mode is a Literal."""
    with pytest.raises(ValueError):
        ModelConfig(model_id="x", vllm_compile_mode="eagar")  # typo
