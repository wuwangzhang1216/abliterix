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


def test_known_multistream_architecture_stops_before_load(monkeypatch):
    monkeypatch.setattr(configuration_auto, "CONFIG_MAPPING", {"deepseek_v4": object()})
    monkeypatch.setattr(
        PretrainedConfig,
        "get_config_dict",
        lambda *_args, **_kwargs: ({"model_type": "deepseek_v4", "hc_mult": 4}, {}),
    )
    monkeypatch.setattr(
        engine_module,
        "load_tokenizer",
        lambda *_args, **_kwargs: pytest.fail("Tokenizer loading must not start"),
    )
    with pytest.raises(RuntimeError, match="4 residual streams.*unsupported"):
        SteeringEngine(AbliterixConfig())


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
def test_official_tiny_multistream_model_fails_with_explanatory_error(family):
    """Newer Transformers: run real random CPU models, with no Hub downloads.

    This checks the unsupported boundary, not large-checkpoint quality or
    the correctness of any representation pooling or model modification.
    """
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
    with torch.inference_mode(), pytest.raises(RuntimeError, match="Multi-stream"):
        _extraction_engine(model).extract_hidden_states([])
