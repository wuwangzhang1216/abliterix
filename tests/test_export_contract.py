"""Regression tests for replay/export capability boundaries."""

from types import SimpleNamespace

import pytest

from abliterix.core.engine import SteeringEngine
from abliterix.types import QuantMode, SteeringMode


@pytest.mark.parametrize(
    "mode",
    [
        SteeringMode.ANGULAR,
        SteeringMode.ADAPTIVE_ANGULAR,
        SteeringMode.SPHERICAL,
        SteeringMode.VECTOR_FIELD,
    ],
)
def test_export_merged_rejects_runtime_only_steering(mode):
    """A checkpoint must never silently omit active runtime hooks."""
    engine = SteeringEngine.__new__(SteeringEngine)
    engine.config = SimpleNamespace(
        steering=SimpleNamespace(steering_mode=mode),
    )
    engine.model = object()

    with pytest.raises(RuntimeError, match="runtime-only"):
        engine.export_merged()


def test_export_merged_rejects_quantized_direct_edits():
    engine = SteeringEngine.__new__(SteeringEngine)
    engine.config = SimpleNamespace(
        steering=SimpleNamespace(steering_mode=SteeringMode.DIRECT),
        model=SimpleNamespace(quant_method=QuantMode.BNB_4BIT),
    )
    engine.model = object()

    with pytest.raises(RuntimeError, match="quantized model"):
        engine.export_merged()


def test_export_merged_rejects_quantized_lora_with_router_edits():
    engine = SteeringEngine.__new__(SteeringEngine)
    engine.config = SimpleNamespace(
        steering=SimpleNamespace(steering_mode=SteeringMode.LORA),
        model=SimpleNamespace(quant_method=QuantMode.FP8),
    )
    engine.model = object()
    engine._router_originals = [(0, 1, object())]

    with pytest.raises(RuntimeError, match="router/expert edits"):
        engine.export_merged()


def test_export_adapter_rejects_non_lora_mode(tmp_path):
    engine = SteeringEngine.__new__(SteeringEngine)
    engine.config = SimpleNamespace(
        steering=SimpleNamespace(steering_mode=SteeringMode.DIRECT),
    )
    engine.model = object()

    with pytest.raises(RuntimeError, match="only represents LoRA"):
        engine.export_adapter(tmp_path)


def test_export_adapter_rejects_unrepresentable_router_edits(tmp_path):
    engine = SteeringEngine.__new__(SteeringEngine)
    engine.config = SimpleNamespace(
        steering=SimpleNamespace(steering_mode=SteeringMode.LORA),
    )
    engine.model = object()
    engine._expert_deltas = [(0, 1, object())]

    with pytest.raises(RuntimeError, match="router/expert"):
        engine.export_adapter(tmp_path)


def test_export_adapter_saves_active_peft_state_without_merging(monkeypatch, tmp_path):
    from abliterix.core import engine as engine_module

    class FakePeftModel:
        def __init__(self):
            self.saved_to = None

        def named_parameters(self):
            yield "base_model.model.layers.0.o_proj.lora_A.default.weight", object()

        def save_pretrained(self, path):
            self.saved_to = path

    monkeypatch.setattr(engine_module, "PeftModel", FakePeftModel)
    engine = SteeringEngine.__new__(SteeringEngine)
    engine.config = SimpleNamespace(
        steering=SimpleNamespace(steering_mode=SteeringMode.LORA),
    )
    engine.model = FakePeftModel()
    engine.needs_reload = False
    engine._router_originals = []
    engine._expert_deltas = []

    engine.export_adapter(tmp_path)

    assert engine.model.saved_to == tmp_path
