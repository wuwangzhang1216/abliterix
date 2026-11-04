"""Offline integration with the native Transformers mHC models and Abliterix's HF engine."""

from contextlib import ExitStack
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
import transformers
from peft import PeftModel
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from abliterix.core.engine import SteeringEngine
from abliterix.core.steering import apply_steering
from abliterix.interactive import _restore_selected_trial
from abliterix.settings import AbliterixConfig
from abliterix.types import ChatMessage, SteeringProfile


@pytest.fixture(scope="module", params=["deepseek_v4", "glm5_next"])
def checkpoint(request, tmp_path_factory):
    family = request.param
    if family not in CONFIG_MAPPING:
        pytest.skip(f"installed Transformers lacks the native {family} model")
    shared = dict(
        vocab_size=32,
        hidden_size=16,
        moe_intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        n_routed_experts=4,
        num_experts_per_tok=2,
        q_lora_rank=8,
        hc_mult=4,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    if family == "deepseek_v4":
        config = transformers.DeepseekV4Config(
            **shared,
            num_key_value_heads=1,
            head_dim=8,
            n_shared_experts=1,
            layer_types=["heavily_compressed_attention", "compressed_sparse_attention"],
            compress_rates={
                "heavily_compressed_attention": 2,
                "compressed_sparse_attention": 2,
            },
            index_n_heads=2,
            index_head_dim=8,
            index_topk=2,
            mlp_layer_types=["moe"] * 2,
            sliding_window=16,
            o_groups=2,
            o_lora_rank=4,
            partial_rotary_factor=0.5,
        )
        model_class = transformers.DeepseekV4ForCausalLM
    else:
        text = transformers.Glm5NextTextConfig(
            **shared,
            intermediate_size=32,
            num_key_value_heads=2,
            kv_lora_rank=8,
            qk_nope_head_dim=8,
            qk_rope_head_dim=0,
            v_head_dim=8,
            index_n_heads=2,
            index_head_dim=8,
            index_topk=4,
            index_kpool=2,
            mlp_layer_types=["dense", "sparse"],
            layer_types=["linear_attention", "indexed_attention"],
            linear_num_heads=2,
            linear_head_dim=8,
            linear_conv_kernel_dim=2,
        )
        config = transformers.Glm5NextConfig(
            text_config=text,
            image_token_id=25,
            vision_config=dict(
                depth=1,
                hidden_size=16,
                out_hidden_size=16,
                intermediate_size=32,
                projection_intermediate_size=32,
                num_heads=2,
                image_size=8,
                patch_size=2,
            ),
        )
        model_class = transformers.Glm5NextForConditionalGeneration
    config._attn_implementation = "eager"
    path = tmp_path_factory.mktemp(family)
    num_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(27)
            model = model_class(config).eval()
        model.save_pretrained(path)
        # Keep the whole vocabulary decodable, including random generated IDs.
        words = [
            "<pad>",
            "<bos>",
            "<eos>",
            "<unk>",
            "one",
            "two",
            "three",
            "four",
            "reply",
        ]
        words.extend(f"word{i}" for i in range(len(words), 32))
        backend = Tokenizer(
            WordLevel({word: idx for idx, word in enumerate(words)}, unk_token="<unk>")
        )
        backend.pre_tokenizer = Whitespace()
        tokenizer = transformers.PreTrainedTokenizerFast(
            tokenizer_object=backend,
            pad_token="<pad>",
            bos_token="<bos>",
            eos_token="<eos>",
            unk_token="<unk>",
            chat_template="{% for m in messages %}{{ m['content'] }} {% endfor %}{% if add_generation_prompt %}reply{% endif %}",
        )
        tokenizer.save_pretrained(path)
        yield family, path, model_class
    finally:
        torch.set_num_threads(num_threads)


def _load(path, mode="lora", *, device_map="cpu", dtype="float32"):
    config = AbliterixConfig(
        model=dict(
            model_id=str(path),
            device_map=device_map,
            dtype_fallback_order=[dtype],
            attn_implementation="eager",
            trust_remote_code=False,
        ),
        steering=dict(steering_mode=mode, outlier_quantile=1.0),
        inference=dict(batch_size=2, max_gen_tokens=2),
        system_prompt="",
    )
    return SteeringEngine(config)


@torch.inference_mode()
def test_mhc_native_layer_reads_and_decoding(checkpoint):
    family, path, model_class = checkpoint
    engine = _load(path)
    assert isinstance(engine.model, PeftModel)
    upstream = model_class.from_pretrained(
        path, attn_implementation="eager", local_files_only=True
    ).eval()
    backbone = (
        upstream.model if family == "deepseek_v4" else upstream.model.language_model
    )
    messages = [ChatMessage(system="", user=text) for text in ("one two three", "one")]
    inputs = engine._tokenize(messages)
    assert inputs.attention_mask[1, 0] == 0
    reads = []
    raw_means = []

    def capture_read(_module, args, output):
        # Upstream returns (post, mixing matrix, collapsed input to attention).
        raw_means.append(args[0].mean(dim=2))
        reads.append(output[2])

    def capture_readout(_module, args):
        reads.append(args[0])

    with ExitStack() as stack:
        for layer in backbone.layers:
            stack.enter_context(layer.attn_hc.register_forward_hook(capture_read))
        stack.enter_context(upstream.lm_head.register_forward_pre_hook(capture_readout))
        native_logits = upstream(**inputs).logits
    assert len(reads) == len(backbone.layers) + 1
    assert all(not torch.allclose(read, mean) for read, mean in zip(reads, raw_means))
    for offset in (-1, -2):
        expected = torch.stack([read[:, offset] for read in reads], dim=1)
        actual = engine.extract_hidden_states(messages, token_offset=offset)
        assert actual.shape == (2, 3, 16)
        torch.testing.assert_close(actual, expected)

    torch.testing.assert_close(engine.model(**inputs).logits, native_logits)
    native_ids = upstream.generate(
        **inputs, do_sample=False, max_new_tokens=2, pad_token_id=0
    )
    _, project_ids = engine._generate(messages, max_new_tokens=2)
    assert torch.equal(project_ids, native_ids)

    engine.config.inference.batch_size = 1
    sorted_result = engine.extract_hidden_states_batched(messages, sort_by_length=True)
    torch.testing.assert_close(
        sorted_result,
        torch.cat([engine.extract_hidden_states([message]) for message in messages]),
    )

    if family == "glm5_next":
        # Exercise the conditional wrapper's actual image tower through its tensor
        # input API. The project chat frontend accepts text messages.
        image_inputs = dict(
            input_ids=torch.tensor([[1, 25, 5]]),
            pixel_values=torch.linspace(0, 1, 4 * 24).reshape(4, 24),
            image_grid_thw=torch.tensor([[1, 2, 2]]),
        )
        image_logits = upstream(**image_inputs).logits
        torch.testing.assert_close(engine.model(**image_inputs).logits, image_logits)
        assert torch.isfinite(image_logits).all()
        assert not torch.equal(
            image_logits, upstream(input_ids=image_inputs["input_ids"]).logits
        )


@torch.inference_mode()
@pytest.mark.parametrize("mode", ["lora", "direct"])
def test_mhc_neutral_edit_replays_and_exports(checkpoint, tmp_path, mode):
    _check_edit_replay_export(checkpoint, tmp_path, mode)


@torch.inference_mode()
def _check_edit_replay_export(
    checkpoint, tmp_path, mode, *, device_map="cpu", dtype="float32"
):
    _, path, model_class = checkpoint
    engine = _load(path, mode, device_map=device_map, dtype=dtype)
    messages = [ChatMessage(system="", user="one two three")]
    inputs = engine._tokenize(messages)
    baseline = engine.model(**inputs).logits
    vectors = torch.nn.functional.normalize(
        engine.extract_hidden_states(messages)[0], dim=-1
    )
    profiles = {
        "attn.o_proj": SteeringProfile(
            max_weight=0.6,
            max_weight_position=0.5,
            min_weight=0.6,
            min_weight_distance=2,
        )
    }
    apply_steering(engine, vectors, None, profiles)
    edited = engine.model(**inputs).logits
    assert not torch.equal(edited, baseline)
    engine.restore_baseline()
    torch.testing.assert_close(engine.model(**inputs).logits, baseline)
    trial = SimpleNamespace(
        user_attrs={
            "vector_index": None,
            "parameters": {name: asdict(profile) for name, profile in profiles.items()},
        }
    )
    _restore_selected_trial(trial, engine.config, engine, vectors, safety_experts=None)
    torch.testing.assert_close(engine.model(**inputs).logits, edited)
    _, edited_ids = engine._generate(messages, max_new_tokens=2)
    engine.export_merged().save_pretrained(tmp_path)
    exported = model_class.from_pretrained(
        tmp_path,
        attn_implementation="eager",
        local_files_only=True,
        device_map=device_map,
        dtype=getattr(torch, dtype),
    ).eval()
    export_logits = exported(**inputs).logits
    # Merging a float32 adapter into BF16 weights introduces one rounding step.
    tolerance = dict(atol=2e-3, rtol=2e-2) if dtype == "bfloat16" else {}
    torch.testing.assert_close(export_logits, edited, **tolerance)
    assert torch.equal(
        exported.generate(**inputs, do_sample=False, max_new_tokens=2, pad_token_id=0),
        edited_ids,
    )
    return (export_logits.float() - edited.float()).abs().max().item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("mode", ["lora", "direct"])
@torch.inference_mode()
def test_mhc_cuda_reads_replay_and_export(
    checkpoint, tmp_path, mode, dtype, record_property
):
    if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        pytest.skip("requires BF16 CUDA support")
    family, path, model_class = checkpoint
    engine = _load(path, mode, device_map="cuda:0", dtype=dtype)
    upstream = model_class.from_pretrained(
        path,
        attn_implementation="eager",
        local_files_only=True,
        device_map="cuda:0",
        dtype=getattr(torch, dtype),
    ).eval()
    backbone = (
        upstream.model if family == "deepseek_v4" else upstream.model.language_model
    )
    messages = [ChatMessage(system="", user=text) for text in ("one two three", "one")]
    inputs = engine._tokenize(messages)
    reads = []
    with ExitStack() as stack:
        for layer in backbone.layers:
            stack.enter_context(
                layer.attn_hc.register_forward_hook(
                    lambda _module, _args, output: reads.append(
                        output[2][:, -1].float()
                    )
                )
            )
        stack.enter_context(
            upstream.lm_head.register_forward_pre_hook(
                lambda _module, args: reads.append(args[0][:, -1].float())
            )
        )
        native_logits = upstream(**inputs).logits
    actual = engine.extract_hidden_states(messages)
    expected = torch.stack(reads, dim=1)
    assert actual.device.type == "cuda" and actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(engine.model(**inputs).logits, native_logits)
    record_property("readout_max_abs_error", (actual - expected).abs().max().item())
    del engine, upstream, backbone
    max_error = _check_edit_replay_export(
        checkpoint, tmp_path, mode, device_map="cuda:0", dtype=dtype
    )
    record_property("export_max_abs_logit_error", max_error)
