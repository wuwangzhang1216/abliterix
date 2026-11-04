"""Offline Qwen3.5 integration checks with random weights and neutral inputs.

These validate the HF/PEFT and artifact contracts, not the quality of a 9B run.
"""

from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
import transformers
from peft import PeftModel
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel

from abliterix.core.engine import SteeringEngine
from abliterix.core.steering import apply_steering
from abliterix.interactive import _restore_selected_trial
from abliterix.settings import AbliterixConfig
from abliterix.types import ChatMessage, SteeringProfile


@pytest.fixture(scope="module")
def qwen35_checkpoint(tmp_path_factory):
    text = transformers.Qwen3_5TextConfig(
        vocab_size=512,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_conv_kernel_dim=2,
        layer_types=["linear_attention", "full_attention"],
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0,
            "mrope_section": [1, 1, 2],
        },
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    config = transformers.Qwen3_5Config(
        text_config=text,
        vision_config=dict(
            depth=1,
            hidden_size=16,
            intermediate_size=32,
            num_heads=2,
            patch_size=2,
            temporal_patch_size=1,
            spatial_merge_size=2,
            out_hidden_size=32,
            num_position_embeddings=16,
        ),
        image_token_id=480,
        video_token_id=481,
        vision_start_token_id=482,
        vision_end_token_id=483,
    )
    config._attn_implementation = "eager"
    path = tmp_path_factory.mktemp("qwen35")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(96)
            model = transformers.Qwen3_5ForConditionalGeneration(config).eval()
        model.save_pretrained(path)
        words = ["<pad>", "<bos>", "<eos>", "<unk>"] + sorted(ByteLevel.alphabet())
        words.extend(f"word{i}" for i in range(len(words), 512))
        backend = Tokenizer(
            BPE(
                {word: index for index, word in enumerate(words)},
                merges=[],
                unk_token="<unk>",
            )
        )
        backend.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer = transformers.PreTrainedTokenizerFast(
            tokenizer_object=backend,
            pad_token="<pad>",
            bos_token="<bos>",
            eos_token="<eos>",
            unk_token="<unk>",
            chat_template="{% for m in messages %}{{ m['content'] }} {% endfor %}{% if add_generation_prompt %}reply{% endif %}",
        )
        tokenizer.save_pretrained(path)
        # New native Qwen tokenizers reconstruct BPE from these files. A generic
        # WordLevel tokenizer silently yields empty inputs on that native path.
        backend.model.save(str(path))
        yield path
    finally:
        torch.set_num_threads(threads)


def _load(path):
    return SteeringEngine(
        AbliterixConfig(
            model=dict(
                model_id=str(path),
                quant_method="none",
                device_map="cpu",
                dtype_fallback_order=["float32"],
                attn_implementation="eager",
                trust_remote_code=False,
            ),
            steering=dict(steering_mode="lora", outlier_quantile=1.0),
            inference=dict(batch_size=2, max_gen_tokens=2),
            system_prompt="",
        )
    )


def _native(path):
    return transformers.Qwen3_5ForConditionalGeneration.from_pretrained(
        path,
        dtype=torch.float32,
        device_map="cpu",
        attn_implementation="eager",
        local_files_only=True,
    ).eval()


@torch.inference_mode()
@pytest.mark.parametrize(
    "layer_index", [0, 1], ids=["linear_attention", "full_attention"]
)
def test_qwen35_native_load_and_live_decoder_adapters(qwen35_checkpoint, layer_index):
    engine = _load(qwen35_checkpoint)
    reference = _native(qwen35_checkpoint)
    assert isinstance(engine.model, PeftModel)
    assert engine.peft_config.target_modules == {
        "model.language_model.layers.0.linear_attn.out_proj",
        "model.language_model.layers.0.mlp.down_proj",
        "model.language_model.layers.1.mlp.down_proj",
        "model.language_model.layers.1.self_attn.q_proj",
        "model.language_model.layers.1.self_attn.k_proj",
        "model.language_model.layers.1.self_attn.v_proj",
        "model.language_model.layers.1.self_attn.o_proj",
    }
    assert not any("visual" in name for name in engine.peft_config.target_modules)
    messages = [ChatMessage("", "one two three"), ChatMessage("", "one")]
    inputs = engine._tokenize(messages)
    assert inputs["input_ids"].shape[1] > 0
    assert inputs["attention_mask"][1, 0] == 0
    native = reference(**inputs, output_hidden_states=True)
    baseline = engine.model(**inputs).logits
    torch.testing.assert_close(baseline, native.logits, atol=0, rtol=0)
    hidden = engine.extract_hidden_states(messages)
    assert hidden.shape == (2, 3, 32)
    torch.testing.assert_close(
        hidden, torch.stack([state[:, -1] for state in native.hidden_states], dim=1)
    )
    assert torch.equal(
        engine._generate(messages, max_new_tokens=2)[1],
        reference.generate(**inputs, max_new_tokens=2, do_sample=False, pad_token_id=0),
    )

    # A fixed generic perturbation checks each attention branch's connectivity;
    # it is independent of any learned direction or behavioral objective.
    module = engine.steerable_modules(layer_index)["attn.o_proj"][0]
    module.lora_B["default"].weight.fill_(0.01)
    assert not torch.equal(engine.model(**inputs).logits, baseline)
    engine.restore_baseline()
    torch.testing.assert_close(engine.model(**inputs).logits, baseline, atol=0, rtol=0)


@torch.inference_mode()
def test_qwen35_selected_trial_adapter_and_merged_reload(qwen35_checkpoint, tmp_path):
    engine = _load(qwen35_checkpoint)
    messages = [ChatMessage("", "one two three"), ChatMessage("", "one")]
    inputs = engine._tokenize(messages)
    baseline = engine.model(**inputs).logits
    # Synthetic unit vectors exercise replay serialization without learning any
    # direction from behavioral data.
    vectors = torch.eye(32)[:3]
    profiles = {
        "attn.o_proj": SteeringProfile(
            max_weight=0.6,
            max_weight_position=0.5,
            min_weight=0.6,
            min_weight_distance=2,
        )
    }
    apply_steering(engine, vectors, None, profiles)
    active = engine.model(**inputs).logits
    assert not torch.equal(active, baseline)
    engine.restore_baseline()
    torch.testing.assert_close(engine.model(**inputs).logits, baseline, atol=0, rtol=0)
    trial = SimpleNamespace(
        user_attrs={
            "vector_index": None,
            "parameters": {name: asdict(profile) for name, profile in profiles.items()},
        }
    )
    _restore_selected_trial(trial, engine.config, engine, vectors, safety_experts=None)
    torch.testing.assert_close(engine.model(**inputs).logits, active, atol=0, rtol=0)
    active_tokens = engine._generate(messages, max_new_tokens=2)[1]

    adapter_path = tmp_path / "adapter"
    engine.export_adapter(adapter_path)
    adapter = PeftModel.from_pretrained(_native(qwen35_checkpoint), adapter_path).eval()
    torch.testing.assert_close(adapter(**inputs).logits, active, atol=0, rtol=0)
    assert torch.equal(
        adapter.generate(**inputs, max_new_tokens=2, do_sample=False, pad_token_id=0),
        active_tokens,
    )

    merged_path = tmp_path / "merged"
    engine.export_merged().save_pretrained(merged_path)
    merged = _native(merged_path)
    torch.testing.assert_close(merged(**inputs).logits, active, atol=1e-6, rtol=1e-5)
    assert torch.equal(
        merged.generate(**inputs, max_new_tokens=2, do_sample=False, pad_token_id=0),
        active_tokens,
    )
