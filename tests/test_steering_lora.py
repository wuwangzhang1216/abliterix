"""Behavioral tests for applying LoRA steering through the public API."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig
from peft.tuners.lora.layer import Linear

from abliterix.core.engine import SteeringEngine, _required_lora_rank
from abliterix.core.steering import apply_steering
from abliterix.settings import AbliterixConfig
from abliterix.types import (
    ExpertRoutingConfig,
    SteeringMode,
    SteeringProfile,
    WeightNorm,
)


def _make_lora_engine(config, *, rank: int, in_features: int, out_features: int):
    base = nn.Linear(in_features, out_features, bias=False)
    module = Linear(
        base,
        "default",
        # PEFT 0.20 requires config; 0.18 accepts it through **kwargs.
        config=LoraConfig(r=rank, lora_alpha=rank),
        r=rank,
        lora_alpha=rank,
        lora_dropout=0,
        init_lora_weights=True,
    )
    engine = SimpleNamespace(
        config=config,
        transformer_layers=[SimpleNamespace()],
        steerable_modules=lambda _layer_idx: {"attn.q_proj": [module]},
        _dequant_cache={},
        peft_config=SimpleNamespace(r=rank),
    )
    return engine, module


def _profile(strength: float) -> dict[str, SteeringProfile]:
    return {
        "attn.q_proj": SteeringProfile(
            max_weight=strength,
            max_weight_position=0,
            min_weight=strength,
            min_weight_distance=1,
        )
    }


def test_multi_direction_lora_stacks_rank_k_updates_without_reshaping_parameters(
    abliterix_config,
):
    """Each direction occupies one adapter rank and the wrapped layer stays usable."""
    abliterix_config.steering.weight_normalization = WeightNorm.NONE
    torch.manual_seed(7)
    rank, in_features, out_features = 3, 5, 4
    strength = 0.4
    engine, module = _make_lora_engine(
        abliterix_config,
        rank=rank,
        in_features=in_features,
        out_features=out_features,
    )
    directions = F.normalize(torch.randn(rank, out_features), p=2, dim=1)
    steering_vectors = F.normalize(torch.randn(rank, 2, out_features), p=2, dim=2)
    steering_vectors[:, 1, :] = directions
    x = torch.randn(2, in_features)

    base_weight = module.base_layer.weight.detach().clone()
    shape_a = module.lora_A["default"].weight.shape
    shape_b = module.lora_B["default"].weight.shape

    apply_steering(
        engine,
        steering_vectors,
        vector_index=None,
        profiles=_profile(strength),
    )

    weight_a = module.lora_A["default"].weight
    weight_b = module.lora_B["default"].weight
    assert weight_a.shape == shape_a == (rank, in_features)
    assert weight_b.shape == shape_b == (out_features, rank)
    assert torch.allclose(weight_a, directions @ base_weight)
    assert torch.allclose(weight_b, -strength * directions.T)

    expected = F.linear(x, base_weight + weight_b @ weight_a)
    assert torch.allclose(module(x), expected, atol=1e-6)


def test_multi_direction_indexing_supports_more_layers_than_directions(
    abliterix_config,
):
    """Layer lookup uses the middle axis, not the direction axis."""
    torch.manual_seed(9)
    n_directions, n_layers, hidden = 3, 4, 4
    modules = []
    for _ in range(n_layers):
        _, module = _make_lora_engine(
            abliterix_config,
            rank=n_directions,
            in_features=5,
            out_features=hidden,
        )
        modules.append(module)
    engine = SimpleNamespace(
        config=abliterix_config,
        transformer_layers=[SimpleNamespace() for _ in range(n_layers)],
        steerable_modules=lambda layer_idx: {"attn.q_proj": [modules[layer_idx]]},
        _dequant_cache={},
        peft_config=SimpleNamespace(r=n_directions),
    )
    steering_vectors = F.normalize(
        torch.randn(n_directions, n_layers + 1, hidden), p=2, dim=2
    )
    profiles = {
        "attn.q_proj": SteeringProfile(
            max_weight=0.4,
            max_weight_position=0,
            min_weight=0.4,
            min_weight_distance=n_layers,
        )
    }

    apply_steering(engine, steering_vectors, vector_index=None, profiles=profiles)

    for module in modules:
        assert module.lora_A["default"].weight.shape == (n_directions, 5)
        assert module.lora_B["default"].weight.shape == (hidden, n_directions)
        assert torch.count_nonzero(module.lora_B["default"].weight) > 0


def test_multi_direction_rank_validation_is_atomic_across_modules(abliterix_config):
    """An undersized adapter rejects the trial before any module is changed."""
    torch.manual_seed(11)
    in_features, out_features = 5, 4
    _, first = _make_lora_engine(
        abliterix_config,
        rank=3,
        in_features=in_features,
        out_features=out_features,
    )
    _, undersized = _make_lora_engine(
        abliterix_config,
        rank=2,
        in_features=in_features,
        out_features=out_features,
    )
    modules = [first, undersized]
    engine = SimpleNamespace(
        config=abliterix_config,
        transformer_layers=[SimpleNamespace()],
        steerable_modules=lambda _layer_idx: {"attn.q_proj": modules},
        _dequant_cache={},
        peft_config=SimpleNamespace(r=3),
    )
    with torch.no_grad():
        for module in modules:
            module.lora_A["default"].weight.normal_()
            module.lora_B["default"].weight.normal_()

    x = torch.randn(2, in_features)
    before = [
        (
            module.lora_A["default"].weight.detach().clone(),
            module.lora_B["default"].weight.detach().clone(),
            module(x).detach().clone(),
        )
        for module in modules
    ]
    steering_vectors = F.normalize(torch.randn(3, 2, out_features), p=2, dim=2)

    with pytest.raises(ValueError, match=r"need rank >= 3"):
        apply_steering(
            engine,
            steering_vectors,
            vector_index=None,
            profiles=_profile(0.4),
        )

    for module, (weight_a, weight_b, output) in zip(modules, before, strict=True):
        assert module.lora_A["default"].weight.shape == weight_a.shape
        assert module.lora_B["default"].weight.shape == weight_b.shape
        assert torch.equal(module.lora_A["default"].weight, weight_a)
        assert torch.equal(module.lora_B["default"].weight, weight_b)
        assert torch.equal(module(x), output)


def test_multi_direction_moe_rejection_happens_before_lora_commit(abliterix_config):
    """An unsupported rank-k MoE trial must leave every adapter untouched."""
    torch.manual_seed(12)
    n_directions, n_layers, hidden = 3, 4, 4
    modules = []
    for _ in range(n_layers):
        _, module = _make_lora_engine(
            abliterix_config,
            rank=n_directions,
            in_features=5,
            out_features=hidden,
        )
        modules.append(module)

    engine = SimpleNamespace(
        config=abliterix_config,
        transformer_layers=[SimpleNamespace() for _ in range(n_layers)],
        steerable_modules=lambda layer_idx: {"attn.q_proj": [modules[layer_idx]]},
        _dequant_cache={},
        peft_config=SimpleNamespace(r=n_directions),
        _router_originals=[],
        _expert_deltas=[],
        _locate_router=lambda _layer: None,
        _locate_fused_weights=lambda _layer: None,
    )
    with torch.no_grad():
        for module in modules:
            module.lora_A["default"].weight.normal_()
            module.lora_B["default"].weight.normal_()
    originals = [
        (
            module.lora_A["default"].weight.detach().clone(),
            module.lora_B["default"].weight.detach().clone(),
        )
        for module in modules
    ]
    steering_vectors = F.normalize(
        torch.randn(n_directions, n_layers + 1, hidden), p=2, dim=2
    )

    with pytest.raises(ValueError, match="Multi-direction.*MoE"):
        apply_steering(
            engine,
            steering_vectors,
            vector_index=None,
            profiles=_profile(0.4),
            safety_experts={idx: [(0, 1.0)] for idx in range(n_layers)},
            routing_config=ExpertRoutingConfig(
                n_suppress=1,
                router_bias=-1.0,
                expert_ablation_weight=0.0,
            ),
        )

    for module, (original_a, original_b) in zip(modules, originals, strict=True):
        assert torch.equal(module.lora_A["default"].weight, original_a)
        assert torch.equal(module.lora_B["default"].weight, original_b)


@pytest.mark.parametrize(
    "mode",
    [
        SteeringMode.ANGULAR,
        SteeringMode.ADAPTIVE_ANGULAR,
        SteeringMode.SPHERICAL,
        SteeringMode.VECTOR_FIELD,
    ],
)
def test_multi_direction_runtime_hook_modes_fail_before_registering_hooks(
    abliterix_config, mode
):
    config = abliterix_config.model_copy(deep=True)
    config.steering.steering_mode = mode
    engine = SimpleNamespace(
        config=config,
        transformer_layers=[nn.Identity()],
        steerable_modules=lambda _layer_idx: {},
        has_expert_routing=lambda: False,
    )

    with pytest.raises(ValueError, match="runtime hook"):
        apply_steering(
            engine,
            torch.randn(2, 2, 4),
            vector_index=None,
            profiles={},
            config=config,
        )

    assert not hasattr(engine, "_angular_hooks")


def test_multi_direction_direct_moe_fails_before_weight_edits(abliterix_config):
    config = abliterix_config.model_copy(deep=True)
    config.steering.steering_mode = SteeringMode.DIRECT
    weight = nn.Parameter(torch.randn(4, 4))
    module = SimpleNamespace(weight=weight)
    engine = SimpleNamespace(
        config=config,
        transformer_layers=[SimpleNamespace()],
        steerable_modules=lambda _layer_idx: {"attn.q_proj": [module]},
        has_expert_routing=lambda: True,
    )
    before = weight.detach().clone()

    with pytest.raises(ValueError, match="direct MoE"):
        apply_steering(
            engine,
            torch.randn(2, 2, 4),
            vector_index=None,
            profiles=_profile(0.4),
            config=config,
        )

    assert torch.equal(weight, before)


def test_restore_baseline_keeps_rank_k_adapter_forward_contract(abliterix_config):
    """Restoring a rank-k trial disables it without invalidating PEFT shapes."""
    torch.manual_seed(13)
    rank, in_features, out_features = 3, 5, 4
    engine, module = _make_lora_engine(
        abliterix_config,
        rank=rank,
        in_features=in_features,
        out_features=out_features,
    )
    engine.model = SimpleNamespace(
        config=SimpleNamespace(name_or_path=abliterix_config.model.model_id)
    )
    engine.needs_reload = False
    engine._lora_b_weights = [module.lora_B["default"].weight]
    engine._router_originals = []
    engine._expert_deltas = []
    engine._angular_hooks = []

    x = torch.randn(2, in_features)
    baseline = module(x).detach().clone()
    steering_vectors = F.normalize(torch.randn(rank, 2, out_features), p=2, dim=2)
    apply_steering(
        engine,
        steering_vectors,
        vector_index=None,
        profiles=_profile(0.4),
    )
    assert not torch.allclose(module(x), baseline)

    SteeringEngine.restore_baseline(engine)

    assert module.lora_A["default"].weight.shape == (rank, in_features)
    assert module.lora_B["default"].weight.shape == (out_features, rank)
    assert torch.count_nonzero(module.lora_B["default"].weight) == 0
    assert torch.allclose(module(x), baseline, atol=1e-6)


def test_single_direction_lora_remains_rank_one_compatible(abliterix_config):
    """The existing 2D per-layer vector layout keeps its rank-1 semantics."""
    abliterix_config.steering.weight_normalization = WeightNorm.NONE
    torch.manual_seed(17)
    in_features, out_features = 5, 4
    strength = 0.4
    engine, module = _make_lora_engine(
        abliterix_config,
        rank=1,
        in_features=in_features,
        out_features=out_features,
    )
    steering_vectors = F.normalize(torch.randn(2, out_features), p=2, dim=1)
    direction = steering_vectors[1]
    base_weight = module.base_layer.weight.detach().clone()

    apply_steering(
        engine,
        steering_vectors,
        vector_index=None,
        profiles=_profile(strength),
    )

    weight_a = module.lora_A["default"].weight
    weight_b = module.lora_B["default"].weight
    assert weight_a.shape == (1, in_features)
    assert weight_b.shape == (out_features, 1)
    assert torch.allclose(weight_a, (direction @ base_weight).view(1, -1))
    assert torch.allclose(weight_b, (-strength * direction).view(-1, 1))


def test_gqa_projection_uses_input_side_when_output_is_not_hidden_size(
    abliterix_config,
):
    """K/V projections with ``out < hidden`` must not be silently skipped."""
    abliterix_config.steering.weight_normalization = WeightNorm.NONE
    torch.manual_seed(18)
    hidden, kv_out = 5, 2
    strength = 0.4
    engine, module = _make_lora_engine(
        abliterix_config,
        rank=1,
        in_features=hidden,
        out_features=kv_out,
    )
    direction = F.normalize(torch.randn(hidden), p=2, dim=0)
    steering_vectors = F.normalize(torch.randn(2, hidden), p=2, dim=1)
    steering_vectors[1] = direction
    base_weight = module.base_layer.weight.detach().clone()

    apply_steering(
        engine,
        steering_vectors,
        vector_index=None,
        profiles=_profile(strength),
    )

    weight_a = module.lora_A["default"].weight
    weight_b = module.lora_B["default"].weight
    expected_a = direction.view(1, -1)
    expected_b = -strength * (base_weight @ direction).view(-1, 1)
    assert torch.allclose(weight_a, expected_a)
    assert torch.allclose(weight_b, expected_b)
    torch.testing.assert_close(
        base_weight + weight_b @ weight_a,
        base_weight - strength * torch.outer(base_weight @ direction, direction),
    )


def test_multi_direction_rank_is_checked_before_full_norm_approximation(
    abliterix_config,
):
    """FULL normalisation must not hide an adapter that cannot hold k directions."""
    torch.manual_seed(19)
    abliterix_config.steering.weight_normalization = WeightNorm.FULL
    engine, module = _make_lora_engine(
        abliterix_config,
        rank=2,
        in_features=12,
        out_features=10,
    )
    with torch.no_grad():
        module.lora_A["default"].weight.normal_()
        module.lora_B["default"].weight.normal_()
    original_a = module.lora_A["default"].weight.detach().clone()
    original_b = module.lora_B["default"].weight.detach().clone()
    steering_vectors = F.normalize(torch.randn(3, 2, 10), p=2, dim=2)

    with pytest.raises(ValueError, match=r"need rank >= 3"):
        apply_steering(
            engine,
            steering_vectors,
            vector_index=None,
            profiles=_profile(0.4),
        )

    assert torch.equal(module.lora_A["default"].weight, original_a)
    assert torch.equal(module.lora_B["default"].weight, original_b)


@pytest.mark.parametrize(
    ("steering", "expected"),
    [
        ({"n_directions": 3}, 3),
        ({"search_harmfulness_direction": True}, 3),
        ({"vector_method": "som", "som_grid_h": 2, "som_grid_w": 3}, 6),
        (
            {
                "n_directions": 3,
                "weight_normalization": "full",
                "full_norm_lora_rank": 5,
            },
            5,
        ),
    ],
)
def test_required_lora_rank_covers_vector_recipe(steering, expected):
    config = AbliterixConfig(
        model={"model_id": "dummy/model"},
        steering=steering,
    )

    assert _required_lora_rank(config) == expected


def test_required_lora_rank_covers_every_iterative_pass():
    """The final iterative subspace can retain every direction from every pass."""
    config = AbliterixConfig(
        model={"model_id": "dummy/model"},
        iterative={
            "enabled": True,
            "max_iterations": 5,
            "per_iteration_directions": 3,
        },
    )

    assert _required_lora_rank(config) == 15


def test_non_lora_modes_do_not_overprovision_unused_adapters():
    config = AbliterixConfig(
        model={"model_id": "dummy/model"},
        steering={
            "steering_mode": "direct",
            "vector_method": "som",
            "som_grid_h": 3,
            "som_grid_w": 3,
        },
    )

    assert _required_lora_rank(config) == 1
