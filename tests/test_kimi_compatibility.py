"""Offline native Kimi-K2.5/K2.6 wrapper checks using tiny random checkpoints.

The published model family shares model_type=kimi_k25. These tests cover the
native Transformers interface, not released weights or remote implementations.
"""

from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
import transformers
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from abliterix.core.engine import SteeringEngine, resolve_model_class
from abliterix.core.steering import apply_steering
from abliterix.interactive import _restore_selected_trial
from abliterix.settings import AbliterixConfig
from abliterix.types import ChatMessage, SteeringProfile


@pytest.fixture(scope="module")
def kimi_checkpoint(tmp_path_factory):
    if "kimi_k25" not in CONFIG_MAPPING:
        pytest.skip("installed Transformers lacks the native Kimi-K2.5/K2.6 model")
    text = transformers.DeepseekV3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_shared_experts=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        n_group=1,
        topk_group=1,
        q_lora_rank=8,
        kv_lora_rank=8,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        v_head_dim=8,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    config = transformers.Kimi_K25Config(
        text_config=text,
        vision_config=dict(
            patch_size=2,
            pos_emb_height=2,
            pos_emb_width=2,
            pos_emb_time=1,
            num_attention_heads=2,
            num_hidden_layers=1,
            hidden_size=16,
            intermediate_size=32,
            merge_kernel_size=[2, 2],
        ),
        projection_hidden_size=16,
        image_token_id=25,
        video_token_id=26,
        vision_start_token_id=27,
        vision_end_token_id=28,
    )
    config._attn_implementation = "eager"
    path = tmp_path_factory.mktemp("kimi_k25")
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(47)
            model = transformers.Kimi_K25ForConditionalGeneration(config).eval()
        model.save_pretrained(path)
        words = ["<pad>", "<bos>", "<eos>", "<unk>", "one", "two", "three", "reply"]
        words.extend(f"word{i}" for i in range(len(words), 32))
        backend = Tokenizer(
            WordLevel(
                {word: index for index, word in enumerate(words)}, unk_token="<unk>"
            )
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
        yield path
    finally:
        torch.set_num_threads(n_threads)


@torch.inference_mode()
@pytest.mark.parametrize("mode", ["lora", "direct"])
def test_native_kimi_text_reads_replay_and_export(kimi_checkpoint, tmp_path, mode):
    config = AbliterixConfig(
        model=dict(
            model_id=str(kimi_checkpoint),
            device_map="cpu",
            dtype_fallback_order=["float32"],
            attn_implementation="eager",
            trust_remote_code=False,
        ),
        steering=dict(steering_mode=mode, outlier_quantile=1.0),
        inference=dict(batch_size=2, max_gen_tokens=2),
        system_prompt="",
    )
    assert (
        resolve_model_class(str(kimi_checkpoint))
        is transformers.AutoModelForImageTextToText
    )
    engine = SteeringEngine(config)
    upstream = transformers.Kimi_K25ForConditionalGeneration.from_pretrained(
        kimi_checkpoint, attn_implementation="eager", local_files_only=True
    ).eval()
    messages = [ChatMessage(system="", user=text) for text in ("one two three", "one")]
    inputs = engine._tokenize(messages)
    native = upstream(**inputs, output_hidden_states=True)
    torch.testing.assert_close(engine.model(**inputs).logits, native.logits)
    torch.testing.assert_close(
        engine.extract_hidden_states(messages),
        torch.stack([state[:, -1] for state in native.hidden_states], dim=1),
    )
    assert engine.get_n_layers() == 2
    assert {"attn.o_proj", "mlp.down_proj"} <= set(engine.list_steerable_components())
    assert engine.has_expert_routing()
    native_ids = upstream.generate(
        **inputs, do_sample=False, max_new_tokens=2, pad_token_id=0
    )
    _, project_ids = engine._generate(messages, max_new_tokens=2)
    assert torch.equal(project_ids, native_ids)

    # Exercise the native image tower through the tensor interface. The
    # project's ChatMessage frontend still accepts text, not image uploads.
    image_inputs = dict(
        input_ids=torch.tensor([[1, 25, 4]]),
        pixel_values=torch.linspace(0, 1, 4 * 3 * 2 * 2).reshape(4, 3, 2, 2),
        image_grid_thw=torch.tensor([[1, 2, 2]]),
    )
    image_logits = upstream(**image_inputs).logits
    torch.testing.assert_close(engine.model(**image_inputs).logits, image_logits)
    assert torch.isfinite(image_logits).all()
    assert not torch.equal(
        image_logits, upstream(input_ids=image_inputs["input_ids"]).logits
    )

    vectors = torch.nn.functional.normalize(
        engine.extract_hidden_states(messages)[0], dim=-1
    )
    profiles = {
        component: SteeringProfile(
            max_weight=0.5,
            max_weight_position=0.5,
            min_weight=0.5,
            min_weight_distance=2,
        )
        for component in ("attn.o_proj", "mlp.down_proj")
    }
    apply_steering(engine, vectors, None, profiles)
    edited = engine.model(**inputs).logits
    assert not torch.equal(edited, native.logits)
    engine.restore_baseline()
    torch.testing.assert_close(engine.model(**inputs).logits, native.logits)
    trial = SimpleNamespace(
        user_attrs={
            "vector_index": None,
            "parameters": {key: asdict(value) for key, value in profiles.items()},
        }
    )
    _restore_selected_trial(trial, config, engine, vectors, safety_experts=None)
    torch.testing.assert_close(engine.model(**inputs).logits, edited)
    _, edited_ids = engine._generate(messages, max_new_tokens=2)
    engine.export_merged().save_pretrained(tmp_path)
    exported = transformers.Kimi_K25ForConditionalGeneration.from_pretrained(
        tmp_path, attn_implementation="eager", local_files_only=True
    ).eval()
    torch.testing.assert_close(exported(**inputs).logits, edited)
    torch.testing.assert_close(
        exported(**image_inputs).logits, engine.model(**image_inputs).logits
    )
    assert torch.equal(
        exported.generate(**inputs, do_sample=False, max_new_tokens=2, pad_token_id=0),
        edited_ids,
    )
