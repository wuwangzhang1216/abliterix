"""Local encoder signatures work without downloading model code or weights."""

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from abliterix.core.engine import SteeringEngine
from abliterix.types import ChatMessage


class _Tokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return {"input_ids": [[1, 2]] if isinstance(texts, list) else [1, 2]}


def _install(tmp_path, source, kwargs=None):
    path = tmp_path / "custom_encoder.py"
    path.write_text(source, encoding="utf-8")
    engine = object.__new__(SteeringEngine)
    engine.tokenizer = _Tokenizer()
    engine._install_custom_encoder(str(path), kwargs or {})
    return engine.tokenizer


def test_encoder_without_generation_prompt_parameter(tmp_path):
    # The official DeepSeek-V4 encoder takes thinking_mode, not a generation flag.
    tokenizer = _install(
        tmp_path,
        "def encode_messages(messages, thinking_mode, context=None):\n"
        "    return thinking_mode + ':' + messages[0]['content']\n",
        {"thinking_mode": "chat"},
    )
    assert (
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "你好"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        == "chat:你好"
    )


@pytest.mark.parametrize(
    "signature, expression, options",
    [
        ("messages, add_generation_prompt=False", "add_generation_prompt", {}),
        ("messages, *, add_generation_prompt=False", "add_generation_prompt", {}),
        ("messages, **kwargs", "kwargs['add_generation_prompt']", {}),
        (
            "messages, **add_generation_prompt",
            "add_generation_prompt['add_generation_prompt']",
            {},
        ),
        ("messages, add_generation_prompt=False, /", "add_generation_prompt", {}),
        (
            "messages, mode, add_generation_prompt=False, /",
            "add_generation_prompt",
            {"mode": "chat"},
        ),
    ],
)
def test_encoder_honors_runtime_flag_for_supported_signatures(
    tmp_path, signature, expression, options
):
    options = {**options, "add_generation_prompt": False}
    original = dict(options)
    tokenizer = _install(
        tmp_path,
        f"def encode_messages({signature}):\n"
        f"    return messages[0]['content'] + ':' + str({expression})\n",
        options,
    )
    messages = [{"role": "user", "content": "hello"}]
    for flag in (True, False, True):
        assert (
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=flag
            )
            == f"hello:{flag}"
        )
    assert options == original


def test_encoder_supports_variadic_messages_argument(tmp_path):
    tokenizer = _install(
        tmp_path,
        "def encode_messages(*args, **kwargs):\n"
        "    return args[0][0]['content'] + ':' + str(kwargs['add_generation_prompt'])\n",
    )
    assert (
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "hello"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        == "hello:True"
    )


def test_encoder_batch_and_tokenization_options(tmp_path):
    tokenizer = _install(
        tmp_path,
        "def encode_messages(messages):\n    return '<BOS>' + messages[0]['content']\n",
    )
    batch = [
        [{"role": "user", "content": "one"}],
        [{"role": "user", "content": "two"}],
    ]
    assert tokenizer.apply_chat_template(batch, tokenize=False) == [
        "<BOS>one",
        "<BOS>two",
    ]
    assert tokenizer.apply_chat_template(
        batch, return_tensors="pt", padding=True, truncation=True
    ) == [[1, 2]]
    assert tokenizer.calls == [
        (
            ["<BOS>one", "<BOS>two"],
            {
                "return_tensors": "pt",
                "padding": True,
                "truncation": True,
                "add_special_tokens": False,
            },
        )
    ]


@pytest.mark.parametrize("options", [{}, {"mode": "chat", "typo": True}])
def test_encoder_reports_bad_options_at_installation(tmp_path, options):
    with pytest.raises(TypeError, match="Invalid options for custom encoder"):
        _install(
            tmp_path,
            "def encode_messages(messages, mode):\n    return 'prompt'\n",
            options,
        )


@pytest.mark.parametrize("custom_encoder", [False, True])
def test_real_tokenizer_runtime_paths_preserve_single_bos(tmp_path, custom_encoder):
    """Cover render-then-tokenize, sorting and chat, not just the wrapper."""
    backend = Tokenizer(
        WordLevel(
            {"<PAD>": 0, "<BOS>": 1, "hello": 2, "there": 3, "<UNK>": 4},
            unk_token="<UNK>",
        )
    )
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(
        single="<BOS> $A", special_tokens=[("<BOS>", 1)]
    )

    class RecordingTokenizer(PreTrainedTokenizerFast):
        def __call__(self, *args, **kwargs):
            encoded = super().__call__(*args, **kwargs)
            self.calls.append(encoded["input_ids"])
            return encoded

    tokenizer = RecordingTokenizer(
        tokenizer_object=backend,
        bos_token="<BOS>",
        pad_token="<PAD>",
        unk_token="<UNK>",
        padding_side="left",
    )
    tokenizer.calls = []
    tokenizer.chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"

    class RecordingModel:
        device = torch.device("cpu")

        def generate(self, input_ids, **_kwargs):
            self.last_input_ids = input_ids
            return torch.cat((input_ids, torch.tensor([[2]])), dim=1)

    engine = object.__new__(SteeringEngine)
    engine.tokenizer = tokenizer
    engine.model = RecordingModel()
    engine.response_prefix = ""
    if custom_encoder:
        encoder_path = tmp_path / "bos_encoder.py"
        encoder_path.write_text(
            "def encode_messages(messages):\n"
            "    return '<BOS>' + messages[-1]['content']\n",
            encoding="utf-8",
        )
        engine._install_custom_encoder(str(encoder_path), {})

    messages = [
        ChatMessage(system="", user="hello there"),
        ChatMessage(system="", user="hello"),
    ]
    rendered = engine._render_messages(messages)
    assert rendered[0] == ("<BOS>" if custom_encoder else "") + "hello there"
    tokenized = engine._tokenize(messages)
    assert tokenized["input_ids"].tolist() == [[1, 2, 3], [0, 1, 2]]

    assert engine._length_sorted_indices(messages) == [1, 0]
    assert tokenizer.calls[-1] == [[1, 2, 3], [1, 2]]

    assert (
        engine.stream_chat_response([{"role": "user", "content": "hello"}]) == "hello"
    )
    assert engine.model.last_input_ids.tolist() == [[1, 2]]
