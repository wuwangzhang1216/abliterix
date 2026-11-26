"""CPU coverage of the vLLM hidden-state connector's file-loading contract and
HF-aligned layer indexing (aux id ``k`` lands at index ``k``, HF's ``hidden_states[k]``).

Use real safetensors files and a controlled delayed writer without importing
vLLM's GPU runtime. This does not certify newer vLLM kernels or distributed runs.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
from threading import Event
from types import ModuleType, SimpleNamespace

import pytest
from safetensors.torch import load_file, save_file
import torch

from abliterix.core import vllm_compat, vllm_hidden_states as hs
from abliterix.settings import AbliterixConfig
from abliterix.types import ChatMessage


@pytest.fixture
def connector_runtime(monkeypatch, tmp_path):
    """Provide the public connector/engine boundary and normal extraction flow."""
    connector = ModuleType("example_hidden_states_connector")
    package = ModuleType("vllm.distributed.kv_transfer.kv_connector.v1")
    package.example_hidden_states_connector = connector
    monkeypatch.setitem(sys.modules, package.__name__, package)

    hidden_states = torch.arange(24, dtype=torch.bfloat16).reshape(3, 2, 4)
    tensors = {"hidden_states": hidden_states, "token_ids": torch.tensor([1, 2, 3])}
    runtime = SimpleNamespace(connector=connector, tensors=tensors, write=None)

    class LLM:
        def __init__(self, **kwargs):
            extra = kwargs["kv_transfer_config"]["kv_connector_extra_config"]
            self.path = Path(extra["shared_storage_path"]) / "hidden.safetensors"

        def generate(self, prompts, sampling_params):
            assert len(prompts) == 1
            runtime.write(self.path)
            return [
                SimpleNamespace(
                    request_id="0",
                    kv_transfer_params={"hidden_states_path": str(self.path)},
                )
            ]

    monkeypatch.setitem(
        sys.modules, "vllm", SimpleNamespace(LLM=LLM, SamplingParams=SimpleNamespace)
    )
    monkeypatch.setattr(vllm_compat, "install_gemma4_transformers_compat", lambda: None)
    monkeypatch.setattr(
        hs, "_load_text_config_data", lambda *a: {"num_hidden_layers": 2}
    )
    monkeypatch.setattr(hs, "flush_memory", lambda: None)
    monkeypatch.setattr(
        hs.AutoTokenizer,
        "from_pretrained",
        lambda *a, **kw: SimpleNamespace(apply_chat_template=lambda *a, **kw: "Hello"),
    )
    monkeypatch.setenv("AX_HIDDEN_STATES_DIR", str(tmp_path))

    def extract(token_offset=-1):
        return hs.extract_hidden_states_vllm(
            AbliterixConfig(
                model={"model_id": "local/tiny", "tensor_parallel_size": 1}
            ),
            {"neutral": [ChatMessage(system="", user="Hello")]},
            token_offset=token_offset,
        )["neutral"]

    runtime.extract = extract
    return runtime


def test_extraction_waits_for_connector_write(connector_runtime, tmp_path):
    runtime = connector_runtime
    loader_started = Event()
    write_finished = Event()
    cancelled = Event()
    writes = []

    def delayed_write(path):
        assert loader_started.wait(timeout=5)
        if not cancelled.is_set():
            save_file(runtime.tensors, path)
            write_finished.set()

    def synchronized_load(path):
        # generate() has returned a path but the background write is not ready.
        assert not Path(path).exists()
        loader_started.set()
        assert write_finished.wait(timeout=5)
        return load_file(path, device="cpu")

    runtime.connector.load_hidden_states = synchronized_load
    with ThreadPoolExecutor(max_workers=1) as executor:

        def start_write(path):
            # New connectors also leave a companion lock file for cleanup.
            Path(str(path) + ".lock").touch()
            writes.append(executor.submit(delayed_write, path))

        runtime.write = start_write
        try:
            actual = runtime.extract()
            for future in writes:
                future.result(timeout=5)
        finally:
            cancelled.set()
            loader_started.set()

    expected = torch.cat([torch.zeros(1, 4), runtime.tensors["hidden_states"][-1]])
    torch.testing.assert_close(actual, expected.unsqueeze(0))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("token_offset", [-1, 0])
def test_extraction_reads_legacy_synchronous_file(
    connector_runtime, tmp_path, token_offset
):
    runtime = connector_runtime
    runtime.write = lambda path: save_file(runtime.tensors, path)

    actual = runtime.extract(token_offset=token_offset)

    expected = torch.cat(
        [torch.zeros(1, 4), runtime.tensors["hidden_states"][token_offset]]
    )
    torch.testing.assert_close(actual, expected.unsqueeze(0))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("error_type", [OSError, ImportError, KeyError])
def test_connector_loader_failure_propagates(connector_runtime, tmp_path, error_type):
    runtime = connector_runtime
    runtime.write = lambda path: save_file(runtime.tensors, path)
    failure = error_type("connector loading failed")

    def fail_to_load(path):
        # A valid file must not trigger an unsynchronized fallback on failure.
        assert Path(path).is_file()
        raise failure

    runtime.connector.load_hidden_states = fail_to_load
    with pytest.raises(error_type) as caught:
        runtime.extract()

    assert caught.value is failure
    assert not list(tmp_path.iterdir())


# --- HF-aligned layer indexing -----------------------------------------------

N, H = 5, 8


def _reference(prompt: int, length: int) -> torch.Tensor:
    """HF's ``hidden_states`` for a prompt, ``[N + 1, length, H]``: row ``k`` is
    the state after ``k`` layers, every value tagged with ``(prompt, k, pos)``."""
    k = torch.arange(N + 1).view(-1, 1, 1)
    pos = torch.arange(length).view(1, -1, 1)
    return (1000.0 * prompt + 10.0 * k + 0.01 * pos).expand(N + 1, length, H).clone()


def test_aux_id_k_lands_at_hf_index_k(monkeypatch, tmp_path):
    seen = {}
    connector = ModuleType("example_hidden_states_connector")
    package = ModuleType("vllm.distributed.kv_transfer.kv_connector.v1")
    package.example_hidden_states_connector = connector
    monkeypatch.setitem(sys.modules, package.__name__, package)

    class LLM:
        def __init__(self, **kwargs):
            seen.update(kwargs)
            self.ids = kwargs["speculative_config"]["draft_model_config"]["hf_config"][
                "eagle_aux_hidden_state_layer_ids"
            ]

        def generate(self, prompts, sampling_params):
            outs = []
            for p, text in enumerate(prompts):
                ref = _reference(int(text), 3 + p)
                # The connector stores [prompt_len, len(ids), H], aux id k is ref[k].
                stored = torch.stack([ref[k] for k in self.ids], dim=1)
                path = tmp_path / f"{p}.safetensors"
                save_file({"hidden_states": stored.contiguous()}, str(path))
                outs.append(
                    SimpleNamespace(
                        request_id=str(p),
                        kv_transfer_params={"hidden_states_path": str(path)},
                    )
                )
            return outs

    monkeypatch.setattr(vllm_compat, "install_gemma4_transformers_compat", lambda: None)
    monkeypatch.setitem(
        sys.modules, "vllm", SimpleNamespace(LLM=LLM, SamplingParams=lambda **kw: None)
    )
    monkeypatch.setattr(hs, "_load_text_config_data", lambda *a: {"num_hidden_layers": N})
    monkeypatch.setattr(hs, "flush_memory", lambda: None)
    monkeypatch.setattr(
        hs.AutoTokenizer,
        "from_pretrained",
        lambda *a, **kw: SimpleNamespace(
            apply_chat_template=lambda chat, **k: chat[-1]["content"]
        ),
    )
    monkeypatch.setenv("AX_HIDDEN_STATES_DIR", str(tmp_path))
    sets = {
        "benign": [ChatMessage(system="", user="0"), ChatMessage(system="", user="1")],
        "target": [ChatMessage(system="", user="2")],
    }

    out = hs.extract_hidden_states_vllm(
        AbliterixConfig(model={"model_id": "local/tiny", "tensor_parallel_size": 1}),
        sets,
    )

    ids = seen["speculative_config"]["draft_model_config"]["hf_config"][
        "eagle_aux_hidden_state_layer_ids"
    ]
    assert ids == list(range(1, N + 1))
    assert out["benign"].shape == (2, N + 1, H) and out["target"].shape == (1, N + 1, H)
    for name, prompts in (("benign", [0, 1]), ("target", [2])):
        for row, p in enumerate(prompts):
            got = out[name][row]
            assert torch.equal(got[0], torch.zeros(H))  # embeddings are not extracted
            last = _reference(p, 3 + (p if name == "benign" else 2))[:, -1]
            assert torch.equal(got[1:], last[1:])


def test_layer_rows_reads_either_layout():
    ref = _reference(0, 4)[1:]  # aux ids 1..N, [N, len, H]
    want = torch.cat([torch.zeros(1, H), ref[:, -1]])
    assert torch.equal(hs.layer_rows(ref.transpose(0, 1).contiguous(), N), want)
    assert torch.equal(hs.layer_rows(ref, N), want)
    assert torch.equal(hs.layer_rows(ref.transpose(0, 1).contiguous(), N, 1)[1:], ref[:, 1])


def test_a_model_without_the_last_state_is_refused():
    # deepseek_v2 in vLLM 0.20 captures id k before layer k: id N never comes.
    with pytest.raises(RuntimeError, match="does not capture the state after its last layer"):
        hs.layer_rows(torch.zeros(4, N - 1, H), N)
