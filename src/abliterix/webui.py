# Abliterix
# Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Gradio Web UI for Abliterix.

Launch with: ``abliterix --ui`` or ``python -m abliterix.webui``

Provides a browser-based interface for model steering with:
- Model selection and configuration editing
- Real-time optimisation dashboard with Pareto front visualisation
- Side-by-side baseline vs steered model comparison
- Interactive chat with the steered model
- One-click model export and HuggingFace Hub upload
"""

from __future__ import annotations

import glob
import os
import threading
from dataclasses import dataclass, field

import torch

# Defer Gradio import to give a clear error if not installed.
try:
    import gradio as gr
except ImportError:
    raise ImportError(
        "Gradio is required for the Web UI.  Install it with: pip install abliterix[ui]"
    ) from None

try:
    import plotly.graph_objects as go
except ImportError:
    go = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


@dataclass
class UISession:
    """Mutable state for one Gradio optimisation session."""

    config: object | None = None
    engine: object | None = None
    scorer: object | None = None
    study: object | None = None
    steering_vectors: object | None = None
    safety_experts: dict | None = None
    is_running: bool = False
    should_stop: bool = False
    log_lines: list[str] = field(default_factory=list)
    trial_data: list[dict] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


_session = UISession()


# ---------------------------------------------------------------------------
# Config discovery
# ---------------------------------------------------------------------------


def _find_configs() -> list[str]:
    """Find all TOML config files in the configs directory."""
    configs_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "configs",
    )
    if not os.path.isdir(configs_dir):
        return []
    files = sorted(glob.glob(os.path.join(configs_dir, "*.toml")))
    return [os.path.basename(f) for f in files]


def _load_config_content(name: str) -> str:
    """Read a TOML config file and return its content."""
    configs_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "configs",
    )
    path = os.path.join(configs_dir, name)
    if not os.path.isfile(path):
        return f"# Config not found: {name}"
    with open(path, encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Optimisation runner (background thread)
# ---------------------------------------------------------------------------


def _run_optimisation(
    config_name: str,
    model_id: str,
    vector_method: str,
    steering_mode: str,
    num_trials: int,
    quant_method: str,
):
    """Run the full Abliterix pipeline in a background thread."""
    import sys
    import warnings

    import optuna

    from .core.engine import SteeringEngine
    from .data import load_prompt_dataset
    from .eval.detector import RefusalDetector
    from .eval.scorer import TrialScorer
    from .optimizer import run_search
    from .settings import AbliterixConfig
    from .types import SteeringMode as SM
    from .vectors import compute_steering_vectors

    with _session.lock:
        _session.is_running = True
        _session.should_stop = False
        _session.log_lines = []
        _session.trial_data = []

    def _log(msg: str):
        with _session.lock:
            _session.log_lines.append(msg)

    try:
        # Build config from UI parameters.
        _log("Loading configuration...")
        configs_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "configs",
        )
        config_path = os.path.join(configs_dir, config_name) if config_name else ""

        # Set environment for config loading.
        if config_path and os.path.isfile(config_path):
            os.environ["AX_CONFIG"] = config_path

        # Minimal sys.argv to avoid CLI parsing issues.
        original_argv = sys.argv
        sys.argv = ["abliterix", "--model.model-id", model_id]
        if quant_method != "none":
            sys.argv += ["--model.quant-method", quant_method]
        sys.argv += ["--steering.vector-method", vector_method]
        sys.argv += ["--steering.steering-mode", steering_mode]
        sys.argv += ["--optimization.num-trials", str(num_trials)]
        sys.argv += ["--non-interactive"]

        try:
            config = AbliterixConfig()  # type: ignore[call-arg]
        finally:
            sys.argv = original_argv
            os.environ.pop("AX_CONFIG", None)

        _session.config = config

        # Suppress noisy libraries.
        torch.set_grad_enabled(False)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        warnings.filterwarnings("ignore")

        _log(f"Loading model: {model_id}...")
        engine = SteeringEngine(config)
        _session.engine = engine

        _log("Loading datasets...")
        benign_msgs = load_prompt_dataset(config, config.benign_prompts)
        target_msgs = load_prompt_dataset(config, config.target_prompts)
        _log(f"Loaded {len(benign_msgs)} benign + {len(target_msgs)} target prompts")

        if config.inference.batch_size == 0:
            config.inference.batch_size = 4  # Safe default for UI.
            _log(f"Using batch size: {config.inference.batch_size}")

        _log("Extracting hidden states...")
        benign_states = engine.extract_hidden_states_batched(benign_msgs)
        target_states = engine.extract_hidden_states_batched(target_msgs)

        _log(f"Computing steering vectors ({vector_method})...")
        vectors = compute_steering_vectors(
            benign_states,
            target_states,
            config.steering.vector_method,
            config.steering.orthogonal_projection,
            winsorize=config.steering.winsorize_vectors,
            winsorize_quantile=config.steering.winsorize_quantile,
            projected_abliteration=config.steering.projected_abliteration,
            ot_components=config.steering.ot_components,
            n_directions=config.steering.n_directions,
            sra_base_method=config.steering.sra_base_method,
            sra_n_atoms=config.steering.sra_n_atoms,
            sra_ridge_alpha=config.steering.sra_ridge_alpha,
            ablate_harmfulness_direction=config.steering.ablate_harmfulness_direction,
            harmfulness_layer_band=tuple(config.steering.harmfulness_layer_band),
            som_grid_h=config.steering.som_grid_h,
            som_grid_w=config.steering.som_grid_w,
            som_n_iters=config.steering.som_n_iters,
            som_initial_lr=config.steering.som_initial_lr,
            som_seed=config.steering.som_seed,
            sae_path=config.steering.sae_path,
            sae_layer=config.steering.sae_layer,
            sae_top_k=config.steering.sae_top_k,
        )
        _session.steering_vectors = vectors

        # SVF concept scorer training.
        if config.steering.steering_mode == SM.VECTOR_FIELD:
            from .svf import train_concept_scorers

            _log("Training SVF concept scorers...")
            engine._concept_scorers = train_concept_scorers(
                benign_states,
                target_states,
                hidden_dim=benign_states.shape[2],
                n_epochs=config.steering.svf_scorer_epochs,
                lr=config.steering.svf_scorer_lr,
                hidden_dim_scorer=config.steering.svf_scorer_hidden,
            )

        # MoE expert profiling.
        safety_experts = None
        if engine.has_expert_routing():
            _log("Profiling MoE experts...")
            safety_experts = engine.identify_safety_experts(benign_msgs, target_msgs)
        _session.safety_experts = safety_experts

        detector = RefusalDetector(config)
        scorer = TrialScorer(config, engine, detector)
        _session.scorer = scorer

        _log(f"Starting optimisation ({num_trials} trials)...")

        # Checkpoint storage.
        from optuna.storages import JournalStorage
        from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
        from .util import slugify_model_name

        os.makedirs(config.optimization.checkpoint_dir, exist_ok=True)
        cp_file = os.path.join(
            config.optimization.checkpoint_dir,
            slugify_model_name(config.model.model_id) + "_ui.jsonl",
        )
        lock_obj = JournalFileOpenLock(cp_file)
        backend = JournalFileBackend(cp_file, lock_obj=lock_obj)
        storage = JournalStorage(backend)

        def _progress_callback(trial_number, kl, refusals, total_trials):
            with _session.lock:
                _session.trial_data.append(
                    {
                        "trial": trial_number,
                        "kl": kl,
                        "refusals": refusals,
                    }
                )
                _session.log_lines.append(
                    f"Trial {trial_number}/{total_trials}: "
                    f"KL={kl:.4f}, refusals={refusals}"
                )

        study = run_search(
            config,
            engine,
            scorer,
            vectors,
            safety_experts,
            storage,
            benign_states=benign_states,
            target_states=target_states,
            progress_callback=_progress_callback,
        )
        _session.study = study
        _log("Optimisation complete!")

        detector.close()

    except Exception as e:
        _log(f"ERROR: {e}")
    finally:
        with _session.lock:
            _session.is_running = False


# ---------------------------------------------------------------------------
# Plotly helpers
# ---------------------------------------------------------------------------


def _build_pareto_plot() -> object | None:
    """Build a Pareto front scatter plot from trial data."""
    if go is None:
        return None

    with _session.lock:
        data = list(_session.trial_data)

    if not data:
        fig = go.Figure()
        fig.update_layout(
            title="Pareto Front (waiting for trials...)",
            xaxis_title="KL Divergence",
            yaxis_title="Refusals",
        )
        return fig

    kls = [d["kl"] for d in data]
    refs = [d["refusals"] for d in data]
    labels = [f"Trial {d['trial']}" for d in data]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=kls,
            y=refs,
            mode="markers",
            text=labels,
            marker=dict(size=8, color="steelblue"),
            name="All trials",
        )
    )

    # Simple Pareto front.
    pairs = sorted(zip(kls, refs, labels), key=lambda t: (t[1], t[0]))
    pareto_kl, pareto_ref, pareto_lbl = [], [], []
    best_kl = float("inf")
    for k, r, lbl in pairs:
        if k <= best_kl:
            pareto_kl.append(k)
            pareto_ref.append(r)
            pareto_lbl.append(lbl)
            best_kl = k

    if pareto_kl:
        fig.add_trace(
            go.Scatter(
                x=pareto_kl,
                y=pareto_ref,
                mode="markers+lines",
                text=pareto_lbl,
                marker=dict(size=12, color="red", symbol="star"),
                name="Pareto front",
            )
        )

    fig.update_layout(
        title="Pareto Front",
        xaxis_title="KL Divergence",
        yaxis_title="Refusals",
        template="plotly_dark",
    )
    return fig


# ---------------------------------------------------------------------------
# Gradio UI construction
# ---------------------------------------------------------------------------


def _build_ui() -> gr.Blocks:
    """Construct the Gradio Blocks interface."""
    config_choices = _find_configs()

    with gr.Blocks(
        title="Abliterix — LLM Steering Dashboard",
        theme=gr.themes.Soft(primary_hue="violet"),
    ) as app:
        gr.Markdown("# Abliterix — LLM Steering Dashboard")
        gr.Markdown(
            "Automated model steering and alignment adjustment. "
            "[GitHub](https://github.com/wuwangzhang1216/abliterix)"
        )

        with gr.Tabs():
            # ------ Tab 1: Configuration ------
            with gr.Tab("Configuration"):
                with gr.Row():
                    with gr.Column(scale=2):
                        config_dropdown = gr.Dropdown(
                            label="Preset Config",
                            choices=config_choices,
                            value=config_choices[0] if config_choices else None,
                        )
                        model_id_input = gr.Textbox(
                            label="HuggingFace Model ID",
                            placeholder="meta-llama/Llama-3.1-8B-Instruct",
                        )
                    with gr.Column(scale=1):
                        quant_dropdown = gr.Dropdown(
                            label="Quantization",
                            choices=["none", "bnb_4bit", "bnb_8bit", "fp8"],
                            value="none",
                        )
                        num_trials_input = gr.Number(
                            label="Num Trials",
                            value=50,
                            precision=0,
                        )

                with gr.Row():
                    vector_method_dropdown = gr.Dropdown(
                        label="Vector Method",
                        choices=[
                            "mean",
                            "median_of_means",
                            "pca",
                            "optimal_transport",
                            "cosmic",
                            "sra",
                        ],
                        value="mean",
                    )
                    steering_mode_dropdown = gr.Dropdown(
                        label="Steering Mode",
                        choices=[
                            "lora",
                            "angular",
                            "adaptive_angular",
                            "spherical",
                            "vector_field",
                        ],
                        value="lora",
                    )

                with gr.Accordion("Raw TOML Config", open=False):
                    toml_editor = gr.Code(
                        label="Config Content",
                        language=None,
                        lines=20,
                    )

                def on_config_select(name):
                    if name:
                        return _load_config_content(name)
                    return ""

                config_dropdown.change(
                    on_config_select,
                    inputs=[config_dropdown],
                    outputs=[toml_editor],
                )

                start_btn = gr.Button(
                    "Start Optimisation", variant="primary", size="lg"
                )

            # ------ Tab 2: Optimisation Dashboard ------
            with gr.Tab("Dashboard"):
                status_text = gr.Textbox(label="Status", interactive=False)
                with gr.Row():
                    pareto_plot = gr.Plot(label="Pareto Front")
                    trial_log = gr.Textbox(
                        label="Trial Log",
                        lines=15,
                        max_lines=30,
                        interactive=False,
                    )

                stop_btn = gr.Button("Stop Optimisation", variant="stop")
                refresh_btn = gr.Button("Refresh Dashboard")

                def refresh_dashboard():
                    with _session.lock:
                        running = _session.is_running
                        log = "\n".join(_session.log_lines[-50:])
                        n_trials = len(_session.trial_data)

                    status = (
                        "Running..."
                        if running
                        else f"Idle ({n_trials} trials completed)"
                    )
                    plot = _build_pareto_plot()
                    return status, plot, log

                refresh_btn.click(
                    refresh_dashboard,
                    outputs=[status_text, pareto_plot, trial_log],
                )

                def stop_optimisation():
                    with _session.lock:
                        _session.should_stop = True
                    return "Stop requested..."

                stop_btn.click(stop_optimisation, outputs=[status_text])

            # ------ Tab 3: Comparison ------
            with gr.Tab("Compare"):
                prompt_input = gr.Textbox(
                    label="Test Prompt",
                    lines=3,
                    placeholder="Enter a prompt to compare baseline vs steered responses...",
                )
                compare_btn = gr.Button("Generate Comparison")

                with gr.Row():
                    baseline_output = gr.Textbox(label="Baseline Response", lines=10)
                    steered_output = gr.Textbox(label="Steered Response", lines=10)

                def run_comparison(prompt):
                    if _session.engine is None:
                        return "No model loaded", "No model loaded"

                    engine = _session.engine
                    from .types import ChatMessage

                    msg = ChatMessage(
                        system="You are a helpful assistant.",
                        user=prompt,
                    )

                    # Baseline (restore original weights).
                    engine.restore_baseline()
                    baseline = engine.generate_text([msg])[0]

                    # Steered (re-apply steering).
                    if (
                        _session.steering_vectors is not None
                        and _session.config is not None
                    ):
                        from .core.steering import apply_steering

                        # Use best trial parameters if available.
                        apply_steering(
                            engine,
                            _session.steering_vectors,
                            vector_index=None,
                            profiles={},
                            config=_session.config,
                        )

                    steered = engine.generate_text([msg])[0]
                    return baseline, steered

                compare_btn.click(
                    run_comparison,
                    inputs=[prompt_input],
                    outputs=[baseline_output, steered_output],
                )

            # ------ Tab 4: Chat ------
            with gr.Tab("Chat"):
                chatbot = gr.Chatbot(label="Chat with Steered Model", type="messages")
                chat_input = gr.Textbox(
                    label="Message", placeholder="Type a message..."
                )
                with gr.Row():
                    chat_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear")

                def chat_respond(message, history):
                    if _session.engine is None:
                        history.append({"role": "user", "content": message})
                        history.append(
                            {
                                "role": "assistant",
                                "content": "No model loaded. Start optimisation first.",
                            }
                        )
                        return history, ""

                    from .types import ChatMessage

                    msg = ChatMessage(
                        system="You are a helpful assistant.",
                        user=message,
                    )
                    response = _session.engine.generate_text([msg])[0]

                    history.append({"role": "user", "content": message})
                    history.append({"role": "assistant", "content": response})
                    return history, ""

                chat_btn.click(
                    chat_respond,
                    inputs=[chat_input, chatbot],
                    outputs=[chatbot, chat_input],
                )
                clear_btn.click(lambda: ([], ""), outputs=[chatbot, chat_input])

            # ------ Tab 5: Export ------
            with gr.Tab("Export"):
                with gr.Row():
                    save_path_input = gr.Textbox(
                        label="Local Save Path",
                        placeholder="./exported_model",
                    )
                    save_btn = gr.Button("Save Model Locally")
                save_status = gr.Textbox(label="Status", interactive=False)

                with gr.Row():
                    hf_repo_input = gr.Textbox(
                        label="HuggingFace Repo ID",
                        placeholder="username/model-name",
                    )
                    hf_private = gr.Checkbox(label="Private", value=False)
                    upload_btn = gr.Button("Upload to HuggingFace Hub")
                upload_status = gr.Textbox(label="Upload Status", interactive=False)

                def save_model(path):
                    if not _session.engine or not path:
                        return "No model loaded or path empty"
                    try:
                        merged = _session.engine.export_merged()
                        merged.save_pretrained(path)
                        _session.engine.tokenizer.save_pretrained(path)
                        return f"Model saved to {path}"
                    except Exception as e:
                        return f"Error: {e}"

                def upload_model(repo_id, private):
                    if not _session.engine or not repo_id:
                        return "No model loaded or repo ID empty"
                    try:
                        merged = _session.engine.export_merged()
                        merged.push_to_hub(repo_id, private=private)
                        _session.engine.tokenizer.push_to_hub(repo_id, private=private)
                        return f"Uploaded to https://huggingface.co/{repo_id}"
                    except Exception as e:
                        return f"Error: {e}"

                save_btn.click(
                    save_model, inputs=[save_path_input], outputs=[save_status]
                )
                upload_btn.click(
                    upload_model,
                    inputs=[hf_repo_input, hf_private],
                    outputs=[upload_status],
                )

        # ------ Start button handler ------
        def start_optimisation(
            config_name,
            model_id,
            vector_method,
            steering_mode,
            num_trials,
            quant,
        ):
            if _session.is_running:
                return "Already running!"

            if not model_id:
                return "Please provide a model ID"

            thread = threading.Thread(
                target=_run_optimisation,
                args=(
                    config_name,
                    model_id,
                    vector_method,
                    steering_mode,
                    int(num_trials),
                    quant,
                ),
                daemon=True,
            )
            thread.start()
            return (
                "Optimisation started! Switch to the Dashboard tab to monitor progress."
            )

        start_btn.click(
            start_optimisation,
            inputs=[
                config_dropdown,
                model_id_input,
                vector_method_dropdown,
                steering_mode_dropdown,
                num_trials_input,
                quant_dropdown,
            ],
            outputs=[status_text],
        )

        # Auto-refresh dashboard every 5 seconds.
        dashboard_timer = gr.Timer(5)
        dashboard_timer.tick(
            refresh_dashboard,
            outputs=[status_text, pareto_plot, trial_log],
        )

    return app


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def launch_ui(
    share: bool = False,
    server_name: str = "0.0.0.0",
    server_port: int = 7860,
):
    """Launch the Gradio Web UI."""
    print("Launching Abliterix Web UI...")
    app = _build_ui()
    app.launch(
        share=share,
        server_name=server_name,
        server_port=server_port,
    )


if __name__ == "__main__":
    launch_ui()
