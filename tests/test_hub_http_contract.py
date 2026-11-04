"""Exercise source pinning and export hashes through the real Hub HTTP client."""

import pytest
from huggingface_hub import constants, set_client_factory
from huggingface_hub.errors import RepositoryNotFoundError
from huggingface_hub.utils import _http

try:
    # Recent Hub 1.x and Hub 2.x expose their matching HTTP implementation.
    from huggingface_hub.utils import httpx
except ImportError:
    # The locked Hub 1.6 predates that public compatibility alias.
    import httpx

from abliterix.reproducibility import pin_remote_sources, repo_weight_shas
from abliterix.settings import AbliterixConfig


SOURCE_NAMES = (
    "benign_prompts",
    "target_prompts",
    "benign_eval_prompts",
    "target_eval_prompts",
)
MODEL_PIN = "a" * 40
EVALUATION_PIN = "b" * 40


def _config():
    return AbliterixConfig.model_validate(
        {
            "model": {
                "model_id": "example/neutral-model",
                "revision": "review",
                "evaluate_model_id": "example/evaluation-model",
                "evaluate_model_revision": "review",
            },
            **{
                name: {
                    "dataset": f"example/{name}",
                    "revision": "review",
                    "split": "train",
                    "column": "text",
                }
                for name in SOURCE_NAMES
            },
        }
    )


@pytest.fixture
def hub_transport(monkeypatch):
    responses = {}
    requests = []

    def handle(request):
        requests.append(request)
        assert request.method == "GET"
        assert request.url.host == "hub.test"
        assert request.url.path in responses, f"Unexpected Hub request: {request.url}"
        response = responses[request.url.path]
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(constants, "ENDPOINT", "https://hub.test")
    monkeypatch.setattr(constants, "HF_HUB_DISABLE_IMPLICIT_TOKEN", True)
    # There is no public factory getter. Preserve both globals so installing the
    # fixture cannot close another test's client or leave a mock factory behind.
    original_factory = _http._GLOBAL_CLIENT_FACTORY
    monkeypatch.setattr(_http, "_GLOBAL_CLIENT", None)
    with httpx.Client(transport=httpx.MockTransport(handle), trust_env=False) as client:
        set_client_factory(lambda: client)
        try:
            yield responses, requests
        finally:
            # The public setter closes this fixture's cached client as well.
            set_client_factory(original_factory)


def test_pinning_uses_hub_responses_and_does_not_fetch_pinned_sources(hub_transport):
    responses, requests = hub_transport
    config = _config()
    pins = {
        "/api/models/example/neutral-model/revision/review": MODEL_PIN,
        "/api/models/example/evaluation-model/revision/review": EVALUATION_PIN,
        **{
            f"/api/datasets/example/{name}/revision/review": str(index) * 40
            for index, name in enumerate(SOURCE_NAMES, start=1)
        },
    }
    for path, pin in pins.items():
        responses[path] = httpx.Response(200, json={"id": path, "sha": pin})

    resolved = pin_remote_sources(config)

    assert len(resolved) == len(pins)
    assert {request.url.path for request in requests} == set(pins)
    assert len(requests) == len(pins)
    assert config.model.revision == MODEL_PIN
    assert config.model.evaluate_model_revision == EVALUATION_PIN
    for index, name in enumerate(SOURCE_NAMES, start=1):
        assert getattr(config, name).revision == str(index) * 40
    assert pin_remote_sources(config) == []
    assert len(requests) == len(pins)


def test_export_hashes_request_lfs_metadata_and_filter_non_weight_files(hub_transport):
    responses, requests = hub_transport
    weights = {
        "shards/model-00001.safetensors": "c" * 64,
        "pytorch_model.bin": "d" * 64,
    }
    responses["/api/models/example/export"] = httpx.Response(
        200,
        json={
            "id": "example/export",
            "sha": MODEL_PIN,
            "siblings": [
                {
                    "rfilename": name,
                    "size": 1024,
                    "blobId": "e" * 40,
                    "lfs": {"sha256": sha, "size": 1024, "pointerSize": 128},
                }
                for name, sha in {**weights, "config.json": "f" * 64}.items()
            ]
            + [{"rfilename": "unhashed.bin"}],
        },
    )

    assert repo_weight_shas("example/export", "fixture-token") == weights
    assert len(requests) == 1
    assert requests[0].url.params["blobs"] == "true"
    assert requests[0].headers["authorization"] == "Bearer fixture-token"


@pytest.mark.parametrize("source", ["model", "target_prompts"])
@pytest.mark.parametrize("failure", ["not_found", "connection"])
def test_http_failures_abort_pinning_but_leave_export_hashing_best_effort(
    hub_transport, source, failure
):
    responses, requests = hub_transport
    config = _config()
    config.model.revision = MODEL_PIN
    config.model.evaluate_model_revision = EVALUATION_PIN
    for name in SOURCE_NAMES:
        getattr(config, name).revision = MODEL_PIN
    if source == "model":
        target = config.model
        path = "/api/models/example/neutral-model"
        error_message = "Could not resolve immutable revision for model"
    else:
        target = config.target_prompts
        path = "/api/datasets/example/target_prompts"
        error_message = (
            "Could not resolve immutable revision for target_prompts dataset"
        )
    target.revision = None

    if failure == "not_found":
        response = httpx.Response(
            404,
            headers={"X-Error-Code": "RepoNotFound"},
            json={"error": "Fixture repository not found"},
        )
        exception_type = RepositoryNotFoundError
    else:
        response = httpx.ConnectError("Fixture connection failure")
        exception_type = httpx.ConnectError
    responses[path] = response
    responses["/api/models/example/export"] = response

    with pytest.raises(RuntimeError, match=error_message) as error:
        pin_remote_sources(config)

    assert isinstance(error.value.__cause__, exception_type)
    assert target.revision is None
    assert repo_weight_shas("example/export", None) == {}
    assert [request.url.path for request in requests] == [
        path,
        "/api/models/example/export",
    ]
