"""Regression coverage for generation, evaluation, and failure recovery."""

import io
import sys
from types import SimpleNamespace as NS
from unittest.mock import Mock
import pytest
import torch
from torch import nn
import torch.nn.functional as F
from transformers import BatchEncoding, GPT2Config, GPT2LMHeadModel
from abliterix.core.engine import SteeringEngine
import abliterix.core.steering as steering
import abliterix.core.vllm_moe_editor as editors
from abliterix.core.vllm_backend import ProjectionCache
from abliterix.settings import AbliterixConfig
from abliterix.types import ExpertRoutingConfig, SteeringProfile
from abliterix.grpo import _sample_responses


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, prompts, **kwargs):
        ids = torch.tensor([[1, 3]] * len(prompts))
        return BatchEncoding({"input_ids": ids, "attention_mask": torch.ones_like(ids)})

    def batch_decode(self, ids, **kwargs):
        return [str(row.tolist()) for row in ids]


def test_grpo_real_transformers_generation_preserves_rng():
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
            bos_token_id=1,
            eos_token_id=None,
            pad_token_id=0,
        )
    ).eval()
    before = torch.get_rng_state().clone()
    texts, prompt, responses = _sample_responses(
        model, Tokenizer(), "hi", 2, 2, 1.0, 0.9, 42
    )
    assert len(texts) == 2 and responses.shape == (2, 2)
    assert torch.equal(before, torch.get_rng_state())
    again = _sample_responses(model, Tokenizer(), "hi", 2, 2, 1.0, 0.9, 42)
    assert torch.equal(responses, again[2])
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float8_e4m3fn])
def test_ega_moe_repeated_exact_restore(transposed, dtype):
    engine = object.__new__(SteeringEngine)
    for attr, value in {
        "_expert_deltas": [],
        "_router_originals": [],
        "_lora_b_weights": [],
        "_direct_weight_originals": {},
        "_angular_hooks": [],
        "needs_reload": False,
        "_fused_down_proj_transposed": transposed,
    }.items():
        setattr(engine, attr, value)
    layer = nn.Module()
    layer.mlp = nn.Module()
    layer.mlp.experts = nn.Module()
    shape = (3, 6, 4) if transposed else (3, 4, 6)
    weight = nn.Parameter(torch.randn(shape).to(dtype), requires_grad=False)
    layer.mlp.experts.down_proj = weight
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([layer])
    model.config = NS(_name_or_path="test-model", name_or_path="test-model")
    engine.model = model
    engine._truncate_to_hidden_layers = lambda m, layers: layers
    engine._locate_fused_weights = lambda layer: layer.mlp.experts.down_proj
    engine._locate_router = lambda layer: None
    engine.steerable_modules = lambda idx: {}
    engine.config = AbliterixConfig(
        model={"model_id": "test-model"}, steering={"weight_normalization": "none"}
    )
    vecs = F.normalize(torch.randn(2, 4), dim=-1)
    profile = {
        "mlp.down_proj": SteeringProfile(
            max_weight=0.3, min_weight=0.3, max_weight_position=0, min_weight_distance=2
        )
    }
    pristine = weight.detach().clone()
    for _ in range(5):
        steering._apply_ega_steering(engine, vecs, None, profile, engine.config, None)
        steering._apply_moe_steering(
            engine,
            vecs,
            None,
            {0: [(0, 1.0), (2, 0.8)]},
            ExpertRoutingConfig(
                n_suppress=2, router_bias=0, expert_ablation_weight=0.5
            ),
            sv_by_device={weight.device: vecs},
        )
        assert not torch.equal(weight.float(), pristine.float())
        engine.restore_baseline()
        assert torch.equal(weight.view(torch.uint8), pristine.view(torch.uint8))
        assert engine._expert_deltas == []


def worker_fixture():
    top = nn.Module()
    top.model = nn.Module()
    layers = []
    for _ in range(2):
        layer = nn.Module()
        layer.mlp = nn.Module()
        layer.mlp.experts = nn.Module()
        layer.mlp.experts.w2_weight = nn.Parameter(torch.randn(3, 4, 6))
        layer.self_attn = nn.Module()
        layer.self_attn.qkv_proj = nn.Linear(4, 8, bias=False)
        layer.self_attn.o_proj = nn.Linear(4, 4, bias=False)
        layer.self_attn.q_size = 4
        layer.self_attn.kv_size = 2
        layers.append(layer)
    top.model.layers = nn.ModuleList(layers)
    return NS(model_runner=NS(model=top)), top


def encoded_vector():
    output = io.BytesIO()
    torch.save(F.normalize(torch.randn(4), dim=0), output)
    return output.getvalue()


@pytest.mark.parametrize("kind", ["expert", "attention"])
@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_worker_partial_apply_rolls_back(kind, failure):
    worker, model = worker_fixture()
    originals = {name: p.detach().clone() for name, p in model.named_parameters()}
    fn = (
        editors._worker_apply_ega_batch
        if kind == "expert"
        else editors._worker_apply_attn_batch
    )

    def rpc(callee, args=(), **kwargs):
        if callee is fn:
            result = callee(worker, args[0][:1], args[1])
            assert result["applied"] == 1
            assert any(
                not torch.equal(p, originals[n]) for n, p in model.named_parameters()
            )
            raise failure("injected second-layer failure")
        return [callee(worker, *args)]

    llm = NS(llm_engine=NS(collective_rpc=rpc))
    editor = (
        editors.VLLMExpertEditor(llm, 4)
        if kind == "expert"
        else editors.VLLMAttentionEditor(llm)
    )
    plan = [
        {"layer_idx": i, "v": encoded_vector(), "strength": 0.5, "component": "o_proj"}
        for i in range(2)
    ]
    with pytest.raises(failure, match="second-layer"):
        (editor.apply_ega if kind == "expert" else editor.apply)(
            plan, norm_preserve=False
        )
    assert all(torch.equal(p, originals[n]) for n, p in model.named_parameters())
    assert editor._applied is False
    assert editor.restore() == 0


@pytest.mark.parametrize("side", ["input", "output"])
def test_cache_preserves_every_expert_and_projection(side):
    model = nn.Module()
    model.layers = nn.ModuleList([nn.Module()])
    in_dim, out_dim = (4, 6) if side == "input" else (6, 4)
    model.layers[0].experts = nn.ModuleList(
        [nn.Linear(in_dim, out_dim, bias=False) for _ in range(3)]
    )
    engine = NS(
        model=model,
        transformer_layers=model.layers,
        steerable_modules=lambda idx: {
            "mlp.down_proj": list(model.layers[idx].experts)
        },
    )
    vectors = F.normalize(torch.randn(2, 4), dim=-1)
    cache = ProjectionCache.build(engine, vectors)
    profile = {
        "mlp.down_proj": SteeringProfile(
            max_weight=0.5, min_weight=0.5, max_weight_position=0, min_weight_distance=2
        )
    }
    weights = cache.build_lora_weights(profile, None, AbliterixConfig())
    assert len(weights) == 3
    for idx, expert in enumerate(model.layers[0].experts):
        A, B = weights[f"layers.0.experts.{idx}"]
        v = vectors[1]
        expected = -0.5 * (
            torch.outer(v, v @ expert.weight)
            if side == "output"
            else torch.outer(expert.weight @ v, v)
        )
        torch.testing.assert_close(B @ A, expected)


def test_upload_without_card_still_hashes(monkeypatch, tmp_path):
    import abliterix.interactive as ui

    replacements = {
        "ask_text": Mock(return_value="review/model"),
        "ask_choice": Mock(return_value="Private"),
        "ask_merge_strategy": Mock(return_value="merge"),
        "flush_memory": Mock(),
        "repo_weight_shas": Mock(return_value={"model.safetensors": "a" * 64}),
        "_upload_reproduce_artifacts": Mock(),
    }
    for name, value in replacements.items():
        monkeypatch.setattr(ui, name, value)
    monkeypatch.setattr(ui.huggingface_hub, "get_token", lambda: "fake")
    monkeypatch.setattr(ui.huggingface_hub, "whoami", lambda token: {"name": "review"})
    ui._upload_model(NS(model=NS(model_id=str(tmp_path))), Mock(), Mock(), Mock())
    assert replacements["repo_weight_shas"].call_count == 1
    assert replacements["_upload_reproduce_artifacts"].call_args.kwargs["weight_shas"]


def test_judge_configuration_is_not_silently_ignored(monkeypatch):
    from abliterix.eval.detector import RefusalDetector

    detector = object.__new__(RefusalDetector)
    detector.config = AbliterixConfig(detection={"llm_judge": True})
    detector._cache = None
    query = Mock(return_value=[True])
    monkeypatch.setattr(detector, "_query_judge_api", query)
    detector._judge_prompt_hash = "review"
    from abliterix.types import ChatMessage
    from abliterix.polyrefuse import evaluate_per_language

    messages = [ChatMessage(system="", user="example prompt")]
    responses = ["Je dois décliner cette demande."]
    engine = NS(generate_text_batched=lambda *a, **kw: responses)
    primary = detector.evaluate_compliance_result(engine, messages)
    assert primary.refusal_count == 1
    query.reset_mock()
    result = evaluate_per_language(lambda msgs: responses, detector, {"fr": messages})
    assert result["fr"].n_refused == primary.refusal_count, (
        "same detector and response receive different labels"
    )
    assert query.called, "llm_judge=true was silently ignored"


def test_iterative_failure_restores_weights(monkeypatch):
    import abliterix.iterative as iterative

    config = AbliterixConfig(iterative={"max_iterations": 2})
    original = torch.randn(4, 4)
    weight = original.clone()
    engine = NS(
        transformer_layers=[object()], list_steerable_components=lambda: ["attn.o_proj"]
    )
    engine.restore_baseline = lambda: weight.copy_(original)

    def extract(*args):
        raise RuntimeError("injected extraction failure")

    engine.extract_hidden_states_batched = extract

    def apply(*args, **kwargs):
        weight.add_(7)

    monkeypatch.setattr(steering, "_apply_direct_steering", apply)
    states = torch.randn(4, 2, 4)
    with pytest.raises(RuntimeError, match="injected extraction failure"):
        iterative.iterative_abliterate(
            engine, [], [], config, benign_states=states, target_states=states + 1
        )
    assert torch.equal(weight, original)


def test_stop_callback_stops_after_current_trial(monkeypatch):
    from abliterix.optimizer import run_search
    from optuna.storages import InMemoryStorage

    engine = NS(
        get_n_layers=lambda: 4,
        list_steerable_components=lambda: [],
        restore_baseline=lambda: None,
    )
    scorer = NS(
        measure_kl_and_coherence=lambda engine: (0.1, 0.0),
        detector=NS(evaluate_compliance=lambda *a: 2),
        target_msgs=[1, 2],
        _compute_objectives=lambda kl, refusals, length: (kl, float(refusals)),
    )
    monkeypatch.setattr("abliterix.optimizer.apply_steering", lambda *a, **kw: None)
    config = AbliterixConfig(optimization={"num_trials": 5, "num_warmup_trials": 1})
    study = run_search(
        config,
        engine,
        scorer,
        torch.randn(5, 4),
        None,
        InMemoryStorage(),
        should_stop=lambda: True,
    )
    assert len(study.trials) == 1
    assert not study.user_attrs.get("finished", False)


def test_jailbreak_rejects_unknown_verdicts():
    from abliterix.external_eval import evaluate_jailbreak

    with pytest.raises(RuntimeError, match="unknown"):
        evaluate_jailbreak(
            lambda prompts: ["answer", "answer"],
            NS(classify_batch=lambda responses: [False, None]),
            ["p1", "p2"],
        )


def test_vllm_constructor_failure_removes_hidden_state_tempdir(monkeypatch, tmp_path):
    import abliterix.core.vllm_hidden_states as hs
    import abliterix.core.vllm_compat as compat

    monkeypatch.setattr(compat, "install_gemma4_transformers_compat", lambda: None)

    def constructor(**kwargs):
        assert list(tmp_path.glob("abliterix_hs_*"))
        raise RuntimeError("constructor failed")

    monkeypatch.setitem(sys.modules, "vllm", NS(LLM=constructor, SamplingParams=Mock()))
    monkeypatch.setattr(
        hs, "_load_text_config_data", lambda *args: {"num_hidden_layers": 2}
    )
    monkeypatch.setenv("AX_HIDDEN_STATES_DIR", str(tmp_path))
    with pytest.raises(RuntimeError, match="constructor failed"):
        hs.extract_hidden_states_vllm(AbliterixConfig(), {})
    assert not list(tmp_path.iterdir())


def test_judge_cache_separates_endpoint_and_temperature(tmp_path):
    from abliterix.eval.detector import RefusalDetector, ClassificationCache

    configs = [
        AbliterixConfig(detection=values)
        for values in (
            {},
            {"llm_judge_base_url": "http://localhost:9999/v1"},
            {"llm_judge_temperature": 0.9},
        )
    ]
    caches = [
        ClassificationCache(
            str(tmp_path),
            "same-model",
            "same-prompt",
            RefusalDetector._judge_fingerprint(c),
        )
        for c in configs
    ]
    try:
        caches[0].put("question", "response", True)
        assert caches[0].get("question", "response") is True
        assert caches[1].get("question", "response") is None
        assert caches[2].get("question", "response") is None
    finally:
        for cache in caches:
            cache.close()


@pytest.mark.parametrize("propagate", [False, True])
def test_optimizer_interrupt_policy_and_cleanup(monkeypatch, propagate):
    from abliterix.optimizer import run_search
    from optuna.storages import InMemoryStorage
    from optuna.trial import TrialState

    def interrupted(*args):
        raise KeyboardInterrupt()

    restore = Mock()
    engine = NS(
        get_n_layers=lambda: 4,
        list_steerable_components=lambda: [],
        restore_baseline=restore,
    )
    scorer = NS(measure_kl_and_coherence=interrupted)
    monkeypatch.setattr("abliterix.optimizer.apply_steering", lambda *a, **kw: None)
    config = AbliterixConfig(optimization={"num_trials": 2, "num_warmup_trials": 1})

    def run():
        return run_search(
            config,
            engine,
            scorer,
            torch.randn(5, 4),
            None,
            InMemoryStorage(),
            raise_on_interrupt=propagate,
        )

    if propagate:
        from abliterix import cli

        monkeypatch.setattr(cli, "run", run)
        monkeypatch.setattr(cli, "install", lambda: None)
        with pytest.raises(SystemExit) as error:
            cli.main()
        assert error.value.code == 130
    else:
        study = run()
        assert len(study.trials) == 1
        assert study.trials[0].state == TrialState.PRUNED
    assert restore.call_count == 2


def test_generation_failure_restores_rng(monkeypatch):
    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))

        def generate(self, *args, **kwargs):
            torch.rand(4)
            raise RuntimeError("generation failed")

    before = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="generation failed"):
        _sample_responses(Policy(), Tokenizer(), "hi", 2, 2, 1.0, 0.9, 42)
    assert torch.equal(before, torch.get_rng_state())


def test_batch_judge_requires_prompts(monkeypatch):
    from abliterix.eval.detector import RefusalDetector

    detector = object.__new__(RefusalDetector)
    detector.config = AbliterixConfig(detection={"llm_judge": True})
    detector._cache = None
    with pytest.raises(ValueError, match="prompts"):
        detector.classify_batch(["response"])


def test_multiturn_unknown_is_not_a_failed_attack():
    from abliterix.external_eval import evaluate_multi_turn

    with pytest.raises(RuntimeError, match="unknown"):
        evaluate_multi_turn(
            lambda prompts: ["response"],
            NS(classify_batch=lambda responses: [None]),
            [["p1"]],
        )


def test_polyrefuse_unknown_is_not_compliance():
    from abliterix.polyrefuse import evaluate_per_language

    with pytest.raises(RuntimeError, match="unknown"):
        evaluate_per_language(
            lambda prompts: ["response"],
            NS(classify_batch=lambda responses: [None]),
            {"fr": ["p1"]},
        )


@pytest.mark.parametrize(
    "responses,labels,error,message",
    [
        ([], [], ValueError, "response per prompt"),
        (["response"], [], ValueError, "verdict per response"),
        (["response"], ["C"], TypeError, "bool or None"),
    ],
)
def test_batch_evaluation_rejects_incomplete_or_invalid_results(
    responses, labels, error, message
):
    from abliterix.external_eval import evaluate_jailbreak

    with pytest.raises(error, match=message):
        evaluate_jailbreak(
            lambda prompts: responses,
            NS(classify_batch=lambda texts: labels),
            ["prompt"],
        )


def test_batch_judge_preserves_unknowns_for_rich_callers(monkeypatch):
    from abliterix.eval.detector import RefusalDetector

    detector = object.__new__(RefusalDetector)
    detector.config = AbliterixConfig(detection={"llm_judge": True})
    detector._cache = None
    detector._judge_prompt_hash = "review"
    monkeypatch.setattr(
        detector,
        "_query_judge_api",
        Mock(side_effect=RuntimeError("judge unavailable")),
    )
    result = detector.classify_batch_result(["response"], prompts=["question"])
    assert result.labels == (None,)
    assert result.unknown_count == 1
    assert result.evaluator == detector.config.detection.llm_judge_model
    with pytest.raises(RuntimeError, match="unknown"):
        detector.classify_batch(["response"], prompts=["question"])
