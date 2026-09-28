"""Bailing MoE v3 hybrid attention paths are registered as attn.o_proj."""

from types import SimpleNamespace

import torch.nn as nn

from abliterix.core.engine import SteeringEngine


class _BailingLayer(nn.Module):
    def __init__(self, kind: str):
        super().__init__()
        self.attention = nn.Module()
        if kind == "kda":
            self.attention.o_proj = nn.Linear(4, 4, bias=False)
            self.attention.q_proj = nn.Linear(4, 4, bias=False)
        else:
            self.attention.dense = nn.Linear(4, 4, bias=False)
            self.attention.q_b_proj = nn.Linear(4, 4, bias=False)
            self.attention.kv_b_proj = nn.Linear(4, 4, bias=False)
        self.mlp = nn.Module()
        self.mlp.down_proj = nn.Linear(4, 4, bias=False)


class _FakeCausalLM(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(layers)
        self.config = SimpleNamespace(num_hidden_layers=len(layers))


def _engine_with_layers(layers):
    engine = SteeringEngine.__new__(SteeringEngine)
    engine.model = _FakeCausalLM(layers)
    return engine


def test_kda_layer_registers_o_proj_and_q():
    engine = _engine_with_layers([_BailingLayer("kda")])
    found = engine.steerable_modules(0)
    assert any(
        m is engine.transformer_layers[0].attention.o_proj for m in found["attn.o_proj"]
    )
    assert any(
        m is engine.transformer_layers[0].attention.q_proj for m in found["attn.q_proj"]
    )
    assert any(
        m is engine.transformer_layers[0].mlp.down_proj for m in found["mlp.down_proj"]
    )


def test_mla_layer_registers_dense_as_o_proj():
    engine = _engine_with_layers([_BailingLayer("mla")])
    found = engine.steerable_modules(0)
    assert any(
        m is engine.transformer_layers[0].attention.dense for m in found["attn.o_proj"]
    )
    assert any(
        m is engine.transformer_layers[0].attention.q_b_proj
        for m in found["attn.q_b_proj"]
    )
    assert any(
        m is engine.transformer_layers[0].attention.kv_b_proj
        for m in found["attn.kv_b_proj"]
    )
