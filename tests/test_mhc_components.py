"""Native mHC MLP and fused-expert edits survive the normal replay/export path."""

from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from abliterix.core.steering import apply_steering
from abliterix.interactive import _restore_selected_trial
from abliterix.types import ChatMessage, ExpertRoutingConfig, SteeringProfile

from . import test_mhc_integration as native

# Reuse the same offline native-model fixtures as the readout tests.
checkpoint = native.checkpoint


def _effective_weight(module):
    if hasattr(module, "get_delta_weight"):
        return module.base_layer.weight + module.get_delta_weight("default")
    return module.weight


@torch.inference_mode()
@pytest.mark.parametrize("mode", ["lora", "direct"])
def test_mhc_mlp_and_fused_experts_restore_replay_and_export(
    checkpoint, tmp_path, mode
):
    family, path, model_class = checkpoint
    engine = native._load(path, mode)
    assert "mlp.down_proj" in engine.list_steerable_components()
    messages = [ChatMessage(system="", user="one two three")]
    inputs = engine._tokenize(messages)
    baseline_logits = engine.model(**inputs).logits
    vectors = torch.nn.functional.normalize(
        engine.extract_hidden_states(messages)[0], dim=-1
    )

    mlp_modules = [
        module
        for index in range(engine.get_n_layers())
        for module in engine.steerable_modules(index).get("mlp.down_proj", [])
    ]
    fused = {
        index: weight
        for index, layer in enumerate(engine.transformer_layers)
        if (weight := engine._locate_fused_weights(layer)) is not None
    }
    assert mlp_modules and fused
    assert set(fused) == ({0, 1} if family == "deepseek_v4" else {1})
    original_mlp = [_effective_weight(module).clone() for module in mlp_modules]
    original_fused = {index: weight.clone() for index, weight in fused.items()}

    # Native routers return logits, weights and integer expert IDs. Exercise
    # their real profiling interface with two neutral sets of token strings.
    experts = engine.identify_safety_experts(
        messages, [ChatMessage(system="", user="four two")]
    )
    assert set(experts) == set(fused)
    assert all(len(ranking) == 4 for ranking in experts.values())
    routing = ExpertRoutingConfig(
        n_suppress=1, router_bias=-2.0, expert_ablation_weight=0.3
    )
    profiles = {
        "mlp.down_proj": SteeringProfile(
            max_weight=0.6,
            max_weight_position=0.5,
            min_weight=0.6,
            min_weight_distance=2,
        )
    }
    apply_steering(
        engine,
        vectors,
        None,
        profiles,
        safety_experts=experts,
        routing_config=routing,
    )
    edited_logits = engine.model(**inputs).logits
    assert not torch.equal(edited_logits, baseline_logits)
    for module, before in zip(mlp_modules, original_mlp):
        assert not torch.equal(_effective_weight(module), before)
    edited_fused = {index: weight.clone() for index, weight in fused.items()}
    for index, weight in fused.items():
        changed_experts = (weight != original_fused[index]).flatten(1).any(dim=1)
        # Direct mode edits all experts through EGA, then applies the selected
        # expert edit. LoRA mode changes the explicitly selected expert only.
        assert changed_experts.sum().item() == (4 if mode == "direct" else 1)

    engine.restore_baseline()
    torch.testing.assert_close(engine.model(**inputs).logits, baseline_logits)
    for module, before in zip(mlp_modules, original_mlp):
        torch.testing.assert_close(_effective_weight(module), before, rtol=0, atol=0)
    for index, weight in fused.items():
        torch.testing.assert_close(weight, original_fused[index], rtol=0, atol=0)

    trial = SimpleNamespace(
        user_attrs={
            "vector_index": None,
            "parameters": {name: asdict(profile) for name, profile in profiles.items()},
            "moe_parameters": asdict(routing),
        }
    )
    _restore_selected_trial(trial, engine.config, engine, vectors, experts)
    torch.testing.assert_close(engine.model(**inputs).logits, edited_logits)
    for index, weight in fused.items():
        torch.testing.assert_close(weight, edited_fused[index], rtol=0, atol=0)
    _, edited_ids = engine._generate(messages, max_new_tokens=2)
    engine.export_merged().save_pretrained(tmp_path)
    exported = model_class.from_pretrained(
        tmp_path, attn_implementation="eager", local_files_only=True
    ).eval()
    torch.testing.assert_close(exported(**inputs).logits, edited_logits)
    backbone = (
        exported.model if family == "deepseek_v4" else exported.model.language_model
    )
    for index, before in edited_fused.items():
        actual = engine._locate_fused_weights(backbone.layers[index])
        torch.testing.assert_close(actual, before, rtol=0, atol=0)
    assert torch.equal(
        exported.generate(**inputs, do_sample=False, max_new_tokens=2, pad_token_id=0),
        edited_ids,
    )
