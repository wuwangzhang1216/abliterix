"""Real bitsandbytes/PEFT lifecycle regressions without model downloads."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F

bnb = pytest.importorskip("bitsandbytes")

from peft import LoraConfig
from peft.tuners.lora.bnb import Linear8bitLt as LoraInt8Linear
from peft.tuners.lora.layer import Linear as LoraLinear

from abliterix.core.steering import apply_steering
from abliterix.core.engine import SteeringEngine
from abliterix.core.vllm_backend import ProjectionCache
from abliterix.types import SteeringProfile, WeightNorm


def _engine(config, module):
    model = nn.Module()
    model.layers = nn.ModuleList([module])
    cache = {}
    return SimpleNamespace(
        model=model,
        config=config,
        transformer_layers=list(model.layers),
        steerable_modules=lambda _: {"attn.o_proj": [module]},
        _dequant_cache=cache,
        _cache_dequant=lambda key, weight: cache.setdefault(key, weight),
        peft_config=SimpleNamespace(r=3),
    )


def _forward_then_wrap():
    torch.manual_seed(17)
    base = bnb.nn.Linear8bitLt(8, 8, bias=False, has_fp16_weights=False).eval()
    inputs = torch.randn(2, 8)
    with torch.no_grad():
        baseline = base(inputs)
    assert base.weight.dtype == torch.int8
    assert base.weight.CB is None and base.weight.SCB is None
    assert base.state.CB is not None and base.state.SCB is not None
    expected_weight = base.state.CB.float() * base.state.SCB.float()[:, None] / 127
    module = LoraInt8Linear(
        base, "default", config=LoraConfig(r=3, lora_alpha=3), r=3, lora_alpha=3
    ).eval()
    return module, inputs, baseline, expected_weight


@pytest.mark.parametrize("normalization", [WeightNorm.NONE, WeightNorm.FULL])
@pytest.mark.parametrize(
    "metadata_state", ["post_forward", "partial_weight", "scale_only"]
)
def test_real_int8_forward_then_steering_matches_float_reference(
    abliterix_config, normalization, metadata_state
):
    module, inputs, baseline, expected_weight = _forward_then_wrap()
    if metadata_state == "partial_weight":
        module.base_layer.weight.CB = module.base_layer.state.CB
    elif metadata_state == "scale_only":
        module.base_layer.state.CB = None
    abliterix_config.steering.weight_normalization = normalization
    engine = _engine(abliterix_config, module)
    base_before = module.base_layer.weight.detach().clone()
    scales_before = module.base_layer.state.SCB.clone()
    reference_base = nn.Linear(8, 8, bias=False)
    reference_base.weight.data.copy_(expected_weight)
    reference = LoraLinear(
        reference_base,
        "default",
        config=LoraConfig(r=3, lora_alpha=3),
        r=3,
        lora_alpha=3,
    ).eval()
    vectors = F.normalize(torch.randn(2, 8), dim=1)
    profiles = {
        "attn.o_proj": SteeringProfile(
            max_weight=0.4,
            min_weight=0.4,
            max_weight_position=0,
            min_weight_distance=1,
        )
    }

    apply_steering(engine, vectors, None, profiles)
    apply_steering(_engine(abliterix_config, reference), vectors, None, profiles)

    for name in ("lora_A", "lora_B"):
        actual = getattr(module, name)["default"].weight
        expected = getattr(reference, name)["default"].weight
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(engine._dequant_cache[id(module)], expected_weight)
    assert torch.equal(module.base_layer.weight, base_before)
    assert torch.equal(module.base_layer.state.SCB, scales_before)
    with torch.no_grad():
        delta = module.lora_B["default"](module.lora_A["default"](inputs))
        torch.testing.assert_close(module(inputs), baseline + delta)


@pytest.mark.parametrize("partial_weight_metadata", [False, True])
def test_projection_cache_reads_real_post_forward_int8_state(
    abliterix_config, partial_weight_metadata
):
    module, _, _, expected_weight = _forward_then_wrap()
    if partial_weight_metadata:
        # Model the report's partial weight metadata with a complete layer pair.
        module.base_layer.weight.CB = module.base_layer.state.CB
        assert module.base_layer.weight.SCB is None
    vectors = F.normalize(torch.randn(2, 8), dim=1)

    cache = ProjectionCache.build(_engine(abliterix_config, module), vectors)

    actual = cache.projections[0]["attn.o_proj"]["vW_all"]
    torch.testing.assert_close(actual, vectors @ expected_weight)


@pytest.mark.parametrize("cached", [False, True])
def test_missing_scales_reject_steering_without_changing_adapter(
    abliterix_config, cached
):
    module, _, _, expected_weight = _forward_then_wrap()
    module.base_layer.state.SCB = None
    engine = _engine(abliterix_config, module)
    if cached:
        engine._dequant_cache[id(module)] = expected_weight
    before_a = module.lora_A["default"].weight.detach().clone()
    before_b = module.lora_B["default"].weight.detach().clone()
    vectors = F.normalize(torch.randn(2, 8), dim=1)
    profiles = {
        "attn.o_proj": SteeringProfile(
            max_weight=0.4,
            min_weight=0.4,
            max_weight_position=0,
            min_weight_distance=1,
        )
    }

    with pytest.raises(RuntimeError, match="SCB|scales"):
        apply_steering(engine, vectors, None, profiles)

    assert torch.equal(module.lora_A["default"].weight, before_a)
    assert torch.equal(module.lora_B["default"].weight, before_b)
    assert bool(engine._dequant_cache) is cached


def test_setup_validation_reports_layer_before_adapter_insertion(
    monkeypatch, abliterix_config
):
    from transformers import PretrainedConfig, PreTrainedModel

    class TinyModel(PreTrainedModel):
        config_class = PretrainedConfig

        def __init__(self):
            super().__init__(PretrainedConfig())
            self.projection = bnb.nn.Linear8bitLt(
                8, 8, bias=False, has_fp16_weights=False
            ).eval()
            with torch.no_grad():
                self.projection(torch.randn(1, 8))

    engine = object.__new__(SteeringEngine)
    engine.model = TinyModel()
    engine.config = abliterix_config
    engine.model.projection.state.SCB = None

    def unexpected_wrap(*_args, **_kwargs):
        raise AssertionError("Invalid metadata must be caught before PEFT wrapping")

    monkeypatch.setattr("abliterix.core.engine.get_peft_model", unexpected_wrap)

    with pytest.raises(
        RuntimeError, match="Invalid bitsandbytes int8 state in projection"
    ):
        engine._init_adapters()
