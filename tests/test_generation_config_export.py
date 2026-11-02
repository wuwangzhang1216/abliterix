"""Export must preserve decoding behavior when model defaults are inconsistent."""

from types import SimpleNamespace

import pytest
import torch
from transformers import GenerationConfig, LlamaConfig, LlamaForCausalLM
from peft import LoraConfig, get_peft_model

from abliterix.core import engine as engine_module
from abliterix.core.engine import SteeringEngine, _make_generation_config_savable
from abliterix.types import QuantMode, SteeringMode


@pytest.mark.parametrize("do_sample", [False, None])
@pytest.mark.parametrize("num_beams", [1, 3])
def test_export_normalizes_only_inactive_sampling_settings(
    tmp_path, do_sample, num_beams
):
    config = GenerationConfig(do_sample=do_sample, num_beams=num_beams)
    settings = {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.1,
        "top_h": 0.2,
        "typical_p": 0.9,
        "epsilon_cutoff": 0.01,
        "eta_cutoff": 0.02,
    }
    for name, value in settings.items():
        setattr(config, name, value)
    before = config.to_dict()
    model = SimpleNamespace(generation_config=config)

    _make_generation_config_savable(model)

    repaired = model.generation_config
    assert repaired is not config
    assert config.to_dict() == before
    assert repaired.do_sample is do_sample
    assert repaired.num_beams == num_beams
    assert repaired.get_generation_mode() == config.get_generation_mode()
    repaired.validate(strict=True)
    repaired.save_pretrained(tmp_path)
    loaded = GenerationConfig.from_pretrained(tmp_path, local_files_only=True)
    assert loaded.do_sample is do_sample
    assert loaded.get_generation_mode() == config.get_generation_mode()
    defaults = GenerationConfig()
    for name in settings:
        assert getattr(loaded, name) == getattr(defaults, name)


def test_export_leaves_active_sampling_settings_untouched():
    config = GenerationConfig(do_sample=True, temperature=0.7, top_p=0.8)
    before = config.to_dict()
    model = SimpleNamespace(generation_config=config)

    _make_generation_config_savable(model)

    assert model.generation_config is config
    assert config.to_dict() == before


def test_export_does_not_hide_unrelated_validation_errors(tmp_path):
    config = GenerationConfig(do_sample=False, num_beams=1)
    config.top_p = 0.8
    config.length_penalty = 2.0
    before = config.to_dict()
    model = SimpleNamespace(generation_config=config)

    _make_generation_config_savable(model)

    assert model.generation_config is config
    assert config.to_dict() == before
    with pytest.raises(ValueError):
        config.save_pretrained(tmp_path)


def test_export_does_not_turn_contrastive_search_into_greedy():
    config = GenerationConfig(do_sample=False)
    config.top_k = 4
    config.penalty_alpha = 0.6
    model = SimpleNamespace(generation_config=config)

    _make_generation_config_savable(model)

    assert model.generation_config is config
    assert config.get_generation_mode() == "contrastive_search"
    assert config.top_k == 4


def test_export_accepts_model_without_generation_config():
    _make_generation_config_savable(SimpleNamespace())


def test_tiny_model_export_reload_preserves_greedy_tokens(tmp_path):
    torch.manual_seed(17)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    ).eval()
    model.generation_config.do_sample = False
    model.generation_config.temperature = 0.5
    model.generation_config.top_p = 0.7
    inputs = torch.tensor([[1, 4, 5]])
    with torch.inference_mode():
        before = model.generate(inputs, do_sample=False, max_new_tokens=3)

    _make_generation_config_savable(model)
    model.save_pretrained(tmp_path)
    reloaded = LlamaForCausalLM.from_pretrained(tmp_path, local_files_only=True).eval()

    assert reloaded.generation_config.do_sample is False
    with torch.inference_mode():
        after = reloaded.generate(inputs, max_new_tokens=3)
    assert torch.equal(before, after)


@pytest.mark.parametrize("reload_base", [False, True])
def test_merged_export_normalizes_generation_config_on_both_paths(
    tmp_path, monkeypatch, reload_base
):
    """Exercise real PEFT merging; only the quantized base reload is replaced."""
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        pad_token_id=0,
    )

    def base_model(*_args, **_kwargs):
        model = LlamaForCausalLM(config)
        model.generation_config.do_sample = False
        model.generation_config.top_p = 0.8
        return model

    engine = object.__new__(SteeringEngine)
    engine.config = SimpleNamespace(
        steering=SimpleNamespace(steering_mode=SteeringMode.LORA),
        model=SimpleNamespace(
            quant_method=QuantMode.BNB_4BIT if reload_base else QuantMode.NONE,
            model_id="local/tiny",
            revision=None,
            text_only=False,
        ),
    )
    engine.peft_config = LoraConfig(target_modules=["q_proj"], r=2)
    engine.model = get_peft_model(base_model(), engine.peft_config)
    engine.needs_reload = False
    engine.trusted_models = {"local/tiny": False}
    if reload_base:
        monkeypatch.setattr(
            engine_module,
            "resolve_model_class",
            lambda *_args, **_kwargs: SimpleNamespace(from_pretrained=base_model),
        )

    merged = engine.export_merged()
    merged.save_pretrained(tmp_path)
    loaded = LlamaForCausalLM.from_pretrained(tmp_path, local_files_only=True)

    assert loaded.generation_config.do_sample is False
    assert loaded.generation_config.top_p == GenerationConfig().top_p
    assert engine.needs_reload is (not reload_base)
