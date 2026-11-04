"""CPU coverage of the vLLM hidden-state connector's file-loading contract.

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
