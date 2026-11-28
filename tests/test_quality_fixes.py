"""Tests for projected-axis norm preservation, position_range, directional router
suppression, and decision-token safety-expert profiling."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from optuna.storages import InMemoryStorage
from pydantic import ValidationError

from abliterix.core import fp4_repack as rp
from abliterix.core.engine import SteeringEngine
from abliterix.core.steering import apply_steering
from abliterix.core.vllm_moe_editor import _project_2d
from abliterix.optimizer import run_search
from abliterix.settings import AbliterixConfig
from abliterix.types import (
    ChatMessage,
    ExpertRoutingConfig,
    SteeringProfile,
    WeightNorm,
)
from abliterix.weight_transforms import (
    apply_ega_projection,
    apply_orba_transform,
    apply_standard_transform,
    double_gram_schmidt,
)


def test_projected_norm_preserves_exact_orthogonality_on_output_side():
    """Rescaling columns on output-side projections keeps v^T W_new == 0,
    whereas legacy row-norm rescaling rotates columns back onto v."""
    torch.manual_seed(7)
    out_f, in_f = 16, 12
    W = torch.randn(out_f, in_f)
    v = torch.randn(out_f)
    v = v / v.norm()

    # Standard output-side projection
    w_row = apply_standard_transform(W, v, strength=1.0, preserve_row_norm=True)
    w_proj = apply_standard_transform(W, v, strength=1.0, preserve_projected_norm=True)
    assert (v @ w_row).abs().max().item() > 1e-3
    assert (v @ w_proj).abs().max().item() < 1e-5
    assert torch.allclose(w_proj.norm(dim=0), W.norm(dim=0), atol=1e-5)

    # EGA fused 3-D output-side projection
    W3 = torch.randn(4, out_f, in_f)
    e_row = apply_ega_projection(
        W3, v, strength=1.0, axis_is_in=False, preserve_row_norm=True
    )
    e_proj = apply_ega_projection(
        W3, v, strength=1.0, axis_is_in=False, preserve_projected_norm=True
    )
    assert torch.einsum("o,eoi->ei", v, e_row).abs().max().item() > 1e-3
    assert torch.einsum("o,eoi->ei", v, e_proj).abs().max().item() < 1e-5
    assert torch.allclose(e_proj.norm(dim=1), W3.norm(dim=1), atol=1e-5)

    # ORBA output-side projection
    b = torch.randn(out_f)
    u = apply_orba_transform(W, v, b, strength=1.0, preserve_projected_norm=True)
    u_hat = double_gram_schmidt(v, b)
    assert (u_hat @ u).abs().max().item() < 1e-5
    assert torch.allclose(u.norm(dim=0), W.norm(dim=0), atol=1e-5)

    # vLLM _project_2d parity
    vllm_out = _project_2d(
        W, v, strength=1.0, norm_preserve=True, preserve_projected_norm=True
    )
    assert torch.allclose(vllm_out, w_proj, atol=1e-5)


def test_position_range_config_and_optimizer_bounds(monkeypatch):
    """Deep (>= 20-layer) models can search below 0.6 * last_layer via position_range."""
    with pytest.raises(ValidationError):
        AbliterixConfig(
            model={"model_id": "m"}, steering={"position_range": [0.8, 0.2]}
        )

    class _DeepEngine:
        _vllm_gen = None
        _projection_cache = None

        def get_n_layers(self) -> int:
            return 32

        def list_steerable_components(self) -> list[str]:
            return ["attn.o_proj"]

        def restore_baseline(self) -> None:
            pass

    class _Scorer:
        target_msgs: list = []
        detector = SimpleNamespace(evaluate_compliance=lambda *_a: 0)

        def measure_kl_and_coherence(self, _engine):
            return 0.05, 0.0

        def _compute_objectives(self, kl, detected, _length_dev):
            return kl, float(detected)

    cfg = AbliterixConfig(
        model={"model_id": "m"},
        steering={"fixed_vector_scope": "per layer", "position_range": [0.25, 0.5]},
        optimization={"num_trials": 5, "num_warmup_trials": 5, "sampler_seed": 0},
    )
    assert AbliterixConfig(model={"model_id": "m"}).steering.position_range == [
        0.3,
        1.0,
    ]

    monkeypatch.setattr("abliterix.optimizer.apply_steering", lambda *a, **k: None)
    study = run_search(
        cfg,
        _DeepEngine(),
        scorer=_Scorer(),
        steering_vectors=torch.randn(33, 8),
        safety_experts=None,
        storage=InMemoryStorage(),
    )

    positions = [t.params["attn.o_proj.max_weight_position"] for t in study.trials]
    assert all(0.25 * 31 <= p <= 0.5 * 31 for p in positions)


def test_directional_router_suppression_and_plan_parity(tmp_path):
    """Directional router edits remove only the refusal component, preserving orthogonal routing."""
    torch.manual_seed(11)
    n_experts, hidden = 6, 12

    layer = nn.Module()
    layer.self_attn = nn.Module()
    layer.self_attn.o_proj = nn.Linear(hidden, hidden, bias=False)
    layer.mlp = nn.Module()
    layer.mlp.gate = nn.Linear(hidden, n_experts, bias=False)

    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([layer])

    eng = SteeringEngine.__new__(SteeringEngine)
    eng.model = model
    eng._router_originals = []
    eng._expert_deltas = []
    eng._direct_weight_originals = {}
    eng._locate_router = lambda lyr: lyr.mlp.gate
    eng._locate_fused_weights = lambda _lyr: None
    eng.steerable_modules = lambda idx: {
        "attn.o_proj": [model.model.layers[idx].self_attn.o_proj]
    }
    eng.has_expert_routing = lambda: False

    w_o_orig = layer.self_attn.o_proj.weight.detach().clone()
    w_gate_orig = layer.mlp.gate.weight.detach().clone()
    svs = torch.randn(2, hidden)
    svs = svs / svs.norm(dim=-1, keepdim=True)
    v_hat = svs[1]
    safety = {0: [(1, 0.8), (4, 0.6), (0, 0.1)]}
    routing = ExpertRoutingConfig(
        n_suppress=2, router_bias=-10.0, expert_ablation_weight=0.0
    )
    profiles = {
        "attn.o_proj": SteeringProfile(
            max_weight=1.0,
            max_weight_position=0.0,
            min_weight=1.0,
            min_weight_distance=1.0,
        )
    }

    cfg = AbliterixConfig(
        model={"model_id": "m"},
        steering={
            "steering_mode": "direct",
            "weight_normalization": WeightNorm.PROJECTED,
        },
        experts={"router_suppression_mode": "directional"},
    )

    edits = rp.record_steering_plan(eng, svs, None, profiles, cfg, None)
    assert len(edits) == 1 and edits[0].preserve_projected_norm is True
    router_edit = rp.TensorEdit(
        logical_name="model.layers.0.mlp.gate.weight",
        kind="row_project",
        direction=v_hat.clone(),
        strength=1.0,
        preserve_row_norm=False,
        rows=[1, 4],
    )
    plan_path = tmp_path / "plan.pt"
    rp.save_plan([edits[0], router_edit], plan_path)
    reloaded = rp.load_plan(plan_path)
    assert reloaded[0].preserve_projected_norm is True
    assert reloaded[1].kind == "row_project" and reloaded[1].rows == [1, 4]

    apply_steering(
        eng, svs, None, profiles, cfg, safety_experts=safety, routing_config=routing
    )
    w_o_new = layer.self_attn.o_proj.weight.detach()
    w_gate_new = layer.mlp.gate.weight.detach()

    # Plan replay matches engine weights for both projected-norm direct and row_project.
    assert torch.allclose(rp.apply_tensor_edit(w_o_orig, reloaded[0]), w_o_new, atol=1e-5)
    assert torch.allclose(
        rp.apply_tensor_edit(w_gate_orig, reloaded[1]), w_gate_new, atol=1e-5
    )

    # Suppressed rows [1, 4] are orthogonal to v_hat while retaining their orthogonal component.
    for eid in (1, 4):
        assert abs(torch.dot(w_gate_new[eid], v_hat).item()) < 1e-5
        orth_orig = w_gate_orig[eid] - torch.dot(w_gate_orig[eid], v_hat) * v_hat
        assert torch.allclose(w_gate_new[eid], orth_orig, atol=1e-5)
    for eid in (0, 2, 3, 5):
        assert torch.equal(w_gate_new[eid], w_gate_orig[eid])


def test_identify_safety_experts_tail_tokens_focuses_on_decision_boundary():
    """With tail_tokens=1, a noisy 10-token prefix cannot drown out the decision-token safety expert."""

    class _StubRouter(nn.Module):
        def __init__(self):
            super().__init__()
            self.top_k = 1
            self.weight = nn.Parameter(torch.zeros(4, 4))

        def forward(self, x):
            ids = x[..., :1].long()
            return None, None, ids

    layer = nn.Module()
    layer.mlp = nn.Module()
    layer.mlp.gate = _StubRouter()
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([layer])

    eng = SteeringEngine.__new__(SteeringEngine)
    eng.model = model

    def extract(msgs):
        for m in msgs:
            if m.user == "target":
                seq = torch.tensor([[[2.0]] * 10 + [[3.0]]]).expand(1, 11, 4)
            else:
                seq = torch.tensor([[[0.0]] * 11]).expand(1, 11, 4)
            layer.mlp.gate(seq)

    eng.extract_hidden_states_batched = extract
    benign = [ChatMessage(system="", user="benign")]
    target = [ChatMessage(system="", user="target")]

    all_tok = eng.identify_safety_experts(benign, target, tail_tokens=0)
    tail_only = eng.identify_safety_experts(benign, target, tail_tokens=1)
    assert all_tok[0][0][0] == 2
    assert tail_only[0][0][0] == 3
    assert tail_only[0][0][1] == pytest.approx(1.0)
