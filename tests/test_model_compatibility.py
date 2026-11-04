"""Model metadata and hidden-state shape checks must fail before unsafe guesses."""

from types import SimpleNamespace

import pytest
import torch
import transformers
from torch import nn
from transformers import PretrainedConfig
from transformers.models.auto import configuration_auto

from abliterix.core import engine as engine_module
from abliterix.core.engine import SteeringEngine
from abliterix.settings import AbliterixConfig


@pytest.mark.parametrize("model_type", ["deepseek_v4", "glm5_next", "glm5_next_text"])
def test_missing_native_architecture_fails_before_tokenizer_load(
    monkeypatch, model_type
):
    monkeypatch.setattr(configuration_auto, "CONFIG_MAPPING", {})
    monkeypatch.setattr(
        PretrainedConfig,
        "get_config_dict",
        lambda *_args, **_kwargs: ({"model_type": model_type}, {}),
    )
    monkeypatch.setattr(
        engine_module,
        "load_tokenizer",
        lambda *_args, **_kwargs: pytest.fail("Tokenizer loading must not start"),
    )
    with pytest.raises(RuntimeError, match="5.20.2.*multi-stream"):
        SteeringEngine(AbliterixConfig())


def test_architecture_check_uses_available_registration_not_version(monkeypatch):
    monkeypatch.setattr(configuration_auto, "CONFIG_MAPPING", {"deepseek_v4": object()})
    engine_module._validate_transformers_architecture({"model_type": "deepseek_v4"})


def test_architecture_check_preserves_custom_config_loading(monkeypatch):
    monkeypatch.setattr(configuration_auto, "CONFIG_MAPPING", {})
    engine_module._validate_transformers_architecture(
        {"model_type": "deepseek_v4", "auto_map": {"AutoConfig": "config.CustomConfig"}}
    )


def test_architecture_check_accepts_null_auto_map(monkeypatch):
    monkeypatch.setattr(configuration_auto, "CONFIG_MAPPING", {"deepseek_v4": object()})
    engine_module._validate_transformers_architecture(
        {"model_type": "deepseek_v4", "auto_map": None}
    )


def test_registered_native_multistream_architecture_passes_preflight(monkeypatch):
    monkeypatch.setattr(configuration_auto, "CONFIG_MAPPING", {"deepseek_v4": object()})
    engine_module._validate_transformers_architecture(
        {"model_type": "deepseek_v4", "hc_mult": 4}
    )


@pytest.mark.parametrize(
    "options, error",
    [
        ({"model": {"text_only": True}}, "text_only=false"),
        ({"model": {"use_torch_compile": True}}, "eager HF"),
        ({"steering": {"steering_mode": "angular"}}, "block-output hooks"),
    ],
)
def test_multistream_unsupported_execution_fails_before_weights(
    monkeypatch, options, error
):
    monkeypatch.setattr(configuration_auto, "CONFIG_MAPPING", {"glm5_next": object()})
    monkeypatch.setattr(
        PretrainedConfig,
        "get_config_dict",
        lambda *_args, **_kwargs: ({"model_type": "glm5_next"}, {}),
    )
    monkeypatch.setattr(
        engine_module,
        "load_tokenizer",
        lambda *_args, **_kwargs: pytest.fail("Tokenizer loading must not start"),
    )
    with pytest.raises(ValueError, match=error):
        SteeringEngine(AbliterixConfig(**options))


def test_architecture_guard_does_not_guess_unknown_remote_model_semantics(monkeypatch):
    monkeypatch.setattr(configuration_auto, "CONFIG_MAPPING", {})
    engine_module._validate_transformers_architecture(
        {"model_type": "custom_model", "hc_mult": 4, "auto_map": None}
    )
    engine_module._validate_transformers_architecture(
        {
            "model_type": "glm5_next",
            "auto_map": {"AutoConfig": "config.CustomConfig"},
            "text_config": {"model_type": "glm5_next_text", "hc_mult": 4},
        }
    )


def test_architecture_check_covers_nested_text_config(monkeypatch):
    monkeypatch.setattr(configuration_auto, "CONFIG_MAPPING", {"glm5_next": object()})
    with pytest.raises(RuntimeError, match="glm5_next_text"):
        engine_module._validate_transformers_architecture(
            {"model_type": "glm5_next", "text_config": {"model_type": "glm5_next_text"}}
        )


def test_reload_resolution_checks_architecture_with_text_only(monkeypatch):
    monkeypatch.setattr(configuration_auto, "CONFIG_MAPPING", {})
    monkeypatch.setattr(
        PretrainedConfig,
        "get_config_dict",
        lambda *_args, **_kwargs: ({"model_type": "glm5_next"}, {}),
    )
    with pytest.raises(RuntimeError, match="glm5_next"):
        engine_module.resolve_model_class("local/test", text_only=True)


def _extraction_engine(model):
    engine = object.__new__(SteeringEngine)
    engine.model = model
    engine.config = SimpleNamespace(steering=SimpleNamespace(outlier_quantile=1.0))
    engine._tokenize = lambda _messages: {"input_ids": torch.tensor([[1, 4, 5]])}
    return engine


@pytest.mark.parametrize("shape", [(1, 3, 4, 16), (1, 3), (1, 3, 2, 4, 16)])
def test_hidden_extraction_rejects_unsupported_shapes(shape):
    class Model(nn.Module):
        def forward(self, input_ids, *, output_hidden_states=False):
            return SimpleNamespace(hidden_states=(torch.zeros(shape),))

    with pytest.raises(RuntimeError, match="requires.*received shape.*unsupported"):
        _extraction_engine(Model()).extract_hidden_states([])


@pytest.mark.parametrize("family", ["deepseek_v4", "glm5_next"])
def test_official_mhc_reads_release_hooks_on_failure_and_recover(family):
    """An invalid token offset must not leave observers on a long-lived model."""
    if family == "deepseek_v4":
        if not hasattr(transformers, "DeepseekV4Config"):
            pytest.skip("requires upstream DeepSeek-V4 model code")
        config = transformers.DeepseekV4Config(
            vocab_size=32,
            hidden_size=16,
            moe_intermediate_size=8,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            q_lora_rank=8,
            n_routed_experts=4,
            num_experts_per_tok=2,
            n_shared_experts=1,
            max_position_embeddings=64,
            layer_types=["sliding_attention"] * 2,
            mlp_layer_types=["moe"] * 2,
            hc_mult=4,
            sliding_window=16,
            o_groups=2,
            o_lora_rank=4,
            partial_rotary_factor=0.5,
            use_cache=False,
        )
        model_class = transformers.DeepseekV4ForCausalLM
    else:
        if not hasattr(transformers, "Glm5NextTextConfig"):
            pytest.skip("requires upstream GLM-5.3-Flash model code")
        config = transformers.Glm5NextTextConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            moe_intermediate_size=8,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            n_routed_experts=4,
            num_experts_per_tok=2,
            q_lora_rank=8,
            kv_lora_rank=8,
            qk_nope_head_dim=8,
            qk_rope_head_dim=0,
            v_head_dim=8,
            index_n_heads=2,
            index_head_dim=8,
            index_topk=4,
            index_kpool=2,
            max_position_embeddings=64,
            mlp_layer_types=["dense", "sparse"],
            layer_types=["linear_attention", "indexed_attention"],
            linear_num_heads=2,
            linear_head_dim=8,
            linear_conv_kernel_dim=2,
            hc_mult=4,
            pad_token_id=0,
            use_cache=False,
        )
        model_class = transformers.Glm5NextTextModel
    torch.manual_seed(17)
    config._attn_implementation = "eager"
    model = model_class(config).eval()
    with torch.inference_mode():
        output = model(torch.tensor([[1, 4, 5]]), output_hidden_states=True)
    values = output.logits if family == "deepseek_v4" else output.last_hidden_state
    assert values.shape == (1, 3, 32 if family == "deepseek_v4" else 16)
    assert torch.isfinite(values).all()
    assert any(hs.ndim == 4 and hs.shape[-2:] == (4, 16) for hs in output.hidden_states)
    engine = _extraction_engine(model)
    backbone = engine_module.native_mhc_backbone(model)
    sites = [layer.input_layernorm for layer in backbone.layers]
    initial = [len(site._forward_pre_hooks) for site in sites]
    final_hooks = len(backbone.norm._forward_hooks)
    with pytest.raises(IndexError):
        engine.extract_hidden_states([], token_offset=3)
    assert [len(site._forward_pre_hooks) for site in sites] == initial
    assert len(backbone.norm._forward_hooks) == final_hooks
    reads = engine.extract_hidden_states([])
    assert reads.shape == (1, 3, 16)
    torch.testing.assert_close(reads[:, -1], output.hidden_states[-1][:, -1])
