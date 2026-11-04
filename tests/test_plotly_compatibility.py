"""Verify the dashboard's figures survive real Gradio/Plotly serialization."""

import json

import pytest

gr = pytest.importorskip("gradio")
pytest.importorskip("plotly")

from abliterix import webui


@pytest.fixture
def ui_session(monkeypatch):
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    session = webui.UISession()
    monkeypatch.setattr(webui, "_session", session)
    return session


def _plot_payload():
    # This is the serialization Gradio performs before delivering a figure to
    # the browser; checking only Figure.to_dict would miss integration errors.
    serialized = gr.Plot().postprocess(webui._build_pareto_plot())
    assert serialized.type == "plotly"
    return json.loads(serialized.plot)


def test_empty_dashboard_serializes_without_stale_trial_points(ui_session):
    ui_session.trial_data = [{"trial": 7, "kl": 0.2, "refusals": 1}]
    assert _plot_payload()["data"]

    # Starting another run clears trial history; its first refresh must clear
    # the previous run's plotted points as well.
    ui_session.trial_data.clear()
    payload = _plot_payload()

    assert payload["data"] == []
    assert payload["layout"]["xaxis"]["title"]["text"] == "KL Divergence"
    assert payload["layout"]["yaxis"]["title"]["text"] == "Refusals"


@pytest.mark.parametrize(
    ("trials", "optimal_trials"),
    [
        pytest.param(
            [
                {"trial": 0, "kl": 0.3, "refusals": 5},
                {"trial": 1, "kl": 0.1, "refusals": 4},
                {"trial": 2, "kl": 0.2, "refusals": 1},
                {"trial": 3, "kl": 0.4, "refusals": 0},
            ],
            {1, 2, 3},
            id="tradeoffs",
        ),
        pytest.param(
            [
                {"trial": 0, "kl": 0.1, "refusals": 1},
                {"trial": 1, "kl": 0.1, "refusals": 3},
                {"trial": 2, "kl": 0.2, "refusals": 0},
                {"trial": 3, "kl": 0.3, "refusals": 0},
                {"trial": 4, "kl": 0.1, "refusals": 1},
                {"trial": 5, "kl": 0.1, "refusals": 3},
            ],
            {0, 2, 4},
            id="ties-and-duplicate-optima",
        ),
        pytest.param(
            [{"trial": 0, "kl": 0.0, "refusals": 0}],
            {0},
            id="single-zero-valued-trial",
        ),
    ],
)
def test_serialized_chart_preserves_trials_and_highlights_only_pareto_optima(
    ui_session, trials, optimal_trials
):
    ui_session.trial_data = trials
    payload = _plot_payload()
    traces = {trace["name"]: trace for trace in payload["data"]}

    expected = {
        f"Trial {trial['trial']}": (trial["kl"], trial["refusals"]) for trial in trials
    }
    for name, expected_points in (
        ("All trials", expected),
        (
            "Pareto front",
            {
                f"Trial {number}": expected[f"Trial {number}"]
                for number in optimal_trials
            },
        ),
    ):
        trace = traces[name]
        assert len(trace["text"]) == len(expected_points)
        points = dict(zip(trace["text"], zip(trace["x"], trace["y"])))
        assert points == expected_points
