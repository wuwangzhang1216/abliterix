"""Exercise the real Gradio components across the supported UI versions."""

import asyncio
import socket
from types import SimpleNamespace

import pytest

gr = pytest.importorskip("gradio")
pytest.importorskip("plotly")

import httpx  # noqa: E402
from gradio.state_holder import SessionState  # noqa: E402

from abliterix import webui  # noqa: E402


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    monkeypatch.setattr(webui, "_session", webui.UISession())
    interface = webui._build_ui()
    yield interface
    interface.close()


def _event(app, name):
    return next(
        index
        for index, event in app.fns.items()
        if event.fn and event.fn.__name__ == name
    )


def _text(message):
    content = message["content"]
    if isinstance(content, str):
        return content
    return "".join(block["text"] for block in content if block["type"] == "text")


def test_build_ui_and_api_schema(app):
    tabs = [block.label for block in app.blocks.values() if isinstance(block, gr.Tab)]
    assert tabs == ["Configuration", "Dashboard", "Compare", "Chat", "Export"]
    assert app.get_api_info()["named_endpoints"]
    assert any(isinstance(block, gr.Timer) for block in app.blocks.values())
    assert all(event.fn is not None for event in app.fns.values())
    chatbot = next(b for b in app.blocks.values() if isinstance(b, gr.Chatbot))
    assert not chatbot.allow_tags


@pytest.mark.parametrize("model_loaded", [False, True])
def test_chat_roundtrip_and_clear(app, monkeypatch, model_loaded):
    calls = []
    if model_loaded:
        webui._session.engine = SimpleNamespace(
            generate_text=lambda messages: [f"Reply to {messages[0].user}"]
        )
        webui._session.selected_trial = object()
        monkeypatch.setattr(webui, "_restore_session_trial", lambda: calls.append(True))

    state = SessionState(app)
    history = []
    for turn, prompt in enumerate(["Hello", "Another message"], start=1):
        response = asyncio.run(
            app.process_api(
                _event(app, "chat_respond"), inputs=[prompt, history], state=state
            )
        )
        history, cleared_input = response["data"]
        assert cleared_input == ""
        assert len(history) == 2 * turn
        assert [message["role"] for message in history] == ["user", "assistant"] * turn
        assert _text(history[-2]) == prompt
        assert _text(history[-1]) == (
            f"Reply to {prompt}"
            if model_loaded
            else "No model loaded. Start optimisation first."
        )
    assert len(calls) == (2 if model_loaded else 0)
    cleared = asyncio.run(
        app.process_api(_event(app, "<lambda>"), inputs=[], state=state)
    )
    assert cleared["data"] == [[], ""]


def test_launch_ui_serves_configuration_with_theme(app, monkeypatch):
    # Exercise the public launcher and the real server without opening a browser
    # or creating a public share link. The fixture closes it even on failure.
    monkeypatch.setattr(webui, "_build_ui", lambda: app)
    original_launch = gr.Blocks.launch
    launch_arguments = {}

    def launch(self, **kwargs):
        launch_arguments.update(kwargs)
        return original_launch(self, prevent_thread_lock=True, quiet=True, **kwargs)

    monkeypatch.setattr(gr.Blocks, "launch", launch)
    with socket.socket() as port_probe:
        port_probe.bind(("127.0.0.1", 0))
        port = port_probe.getsockname()[1]
    webui.launch_ui(share=False, server_name="127.0.0.1", server_port=port)
    assert launch_arguments["share"] is False
    assert launch_arguments["server_name"] == "127.0.0.1"
    assert launch_arguments["server_port"] == port
    assert ("theme" in launch_arguments) == (int(gr.__version__.split(".")[0]) >= 6)
    assert isinstance(app.theme, gr.themes.Soft)
    with httpx.Client(trust_env=False, timeout=10) as client:
        response = client.get(f"http://127.0.0.1:{port}/config")
    assert response.status_code == 200
    assert response.json()["title"] == "Abliterix — LLM Steering Dashboard"
