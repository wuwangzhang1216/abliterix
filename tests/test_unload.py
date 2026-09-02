"""Tests for the HF → TP-backend unload path (issue #83).

``prepare_for_unload`` must drop every engine-held reference into the HF
model — dequant cache, LoRA-B weight list, MoE rollback buffers, direct
weight originals, cliff-head originals, angular hooks — otherwise the
model's VRAM stays pinned after ``engine.model = None`` and the spawned
vLLM TP workers see the GPUs as nearly full.
"""

import gc
import weakref
from types import SimpleNamespace

import torch
from torch import nn

from abliterix.core.engine import SteeringEngine


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.o_proj = nn.Linear(4, 4)
        self.mlp = nn.Module()
        self.mlp.down_proj = nn.Linear(4, 4)


def _make_engine() -> tuple[SteeringEngine, nn.Module]:
    engine = object.__new__(SteeringEngine)  # bypass __init__
    model = nn.Module()
    model.dtype = torch.float32  # ty:ignore[assignment]
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([_Layer(), _Layer()])
    engine.model = model  # ty:ignore[invalid-assignment]
    engine._cached_n_layers = None
    engine._cached_components = None
    engine._dequant_cache = {}
    return engine, model


def test_prepare_for_unload_caches_metadata():
    engine, _ = _make_engine()
    engine.prepare_for_unload()
    assert engine._cached_n_layers == 2
    assert engine._cached_components == ["attn.o_proj", "mlp.down_proj"]


def _populate_model_reference_holders(engine, model):
    """Fill every known engine-held cache with references into *model*,
    mirroring the shapes their real producer sites use (steering.py,
    cliff_head.py, engine.py)."""
    weight = model.model.layers[0].self_attn.o_proj.weight
    engine._dequant_cache[id(weight)] = weight.detach()
    engine._lora_b_weights = [weight]
    engine._router_originals = [(0, 0, weight.detach()[0])]
    engine._expert_deltas = [(0, 0, weight.detach()[0].clone())]
    engine._direct_weight_originals = {weight: weight.detach().clone()}
    # cliff_head.py stores (weight, head, slice.clone()) keyed by (id, head).
    engine._cliff_head_originals = {
        (id(weight), 0): (weight, 0, weight.data[:, 0:2].clone())
    }
    return weight


def test_prepare_for_unload_clears_model_references():
    engine, model = _make_engine()
    _populate_model_reference_holders(engine, model)
    removed = []
    engine._angular_hooks = [SimpleNamespace(remove=lambda: removed.append(True))]

    engine.prepare_for_unload()

    assert engine._dequant_cache == {}
    assert engine._lora_b_weights == []
    assert engine._router_originals == []
    assert engine._expert_deltas == []
    assert engine._direct_weight_originals == {}
    assert engine._cliff_head_originals == {}
    assert engine._angular_hooks == []
    assert removed == [True]


def test_prepare_for_unload_releases_the_model():
    """The property behind issue #83, end to end: after prepare_for_unload()
    and ``engine.model = None``, NOTHING engine-held may keep the model (or
    any of its weights) alive — a single surviving strong reference pins the
    full weight VRAM across the HF → vLLM transition."""
    engine, model = _make_engine()
    weight = _populate_model_reference_holders(engine, model)
    model_ref = weakref.ref(model)
    weight_ref = weakref.ref(weight)

    engine.prepare_for_unload()
    engine.model = None
    del model, weight
    gc.collect()

    assert model_ref() is None, "engine still holds a reference to the model"
    assert weight_ref() is None, "engine still holds a reference to a weight"


def test_prepare_for_unload_without_optional_buffers():
    """Rollback buffers may not exist yet (populated lazily) — must not raise."""
    engine, _ = _make_engine()
    engine.prepare_for_unload()  # no _lora_b_weights / _router_originals / ...
    assert engine._cached_n_layers == 2


def test_restore_baseline_slow_reload_clears_dequant_cache_and_bytes(monkeypatch):
    """When restore_baseline triggers a slow model reload (needs_reload=True or
    model id mismatch), both _dequant_cache and _dequant_cache_bytes must be reset
    to prevent stale weights or saturated byte-budget lockups in the next trial."""
    engine, model = _make_engine()
    model.config = SimpleNamespace(
        name_or_path="test/model-001", _name_or_path="test/model-001"
    )
    _populate_model_reference_holders(engine, model)
    engine._dequant_cache_bytes = 1024 * 1024 * 1024  # 1 GiB simulated
    engine.needs_reload = True
    engine.config = SimpleNamespace(
        model=SimpleNamespace(
            model_id="test/model-001",
            revision=None,
            text_only=True,
            quant_method="none",
            device_map="auto",
        )
    )
    engine.max_memory = None
    engine.trusted_models = {}
    engine._is_native_fp8 = False

    monkeypatch.setattr(
        "abliterix.core.engine.resolve_model_class",
        lambda *args, **kwargs: SimpleNamespace(
            from_pretrained=lambda *a, **kw: _make_engine()[1]
        ),
    )
    monkeypatch.setattr(
        engine,
        "_build_quant_config",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(engine, "_init_adapters", lambda: None)
    monkeypatch.setattr(engine, "_init_expert_routing", lambda: None)

    engine.restore_baseline()

    assert engine._dequant_cache == {}
    assert engine._dequant_cache_bytes == 0
    assert engine.needs_reload is False
