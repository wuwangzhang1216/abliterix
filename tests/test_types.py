"""Tests for abliterix.types — enums, data classes, and Pydantic models."""

from abliterix.types import (
    ChatMessage,
    DecayKernel,
    ExpertRoutingConfig,
    PromptSource,
    QuantMode,
    SteeringProfile,
    VectorMethod,
    WeightNorm,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


def test_quant_mode_values():
    assert QuantMode.NONE == "none"
    assert QuantMode.BNB_4BIT == "bnb_4bit"
    assert QuantMode.BNB_8BIT == "bnb_8bit"
    assert QuantMode.FP8 == "fp8"


def test_vector_method_values():
    assert VectorMethod.MEAN == "mean"
    assert VectorMethod.MEDIAN_OF_MEANS == "median_of_means"
    assert VectorMethod.PCA == "pca"


def test_decay_kernel_values():
    assert DecayKernel.LINEAR == "linear"
    assert DecayKernel.GAUSSIAN == "gaussian"
    assert DecayKernel.COSINE == "cosine"


def test_weight_norm_values():
    assert WeightNorm.NONE == "none"
    assert WeightNorm.PRE == "pre"
    assert WeightNorm.FULL == "full"


def test_enums_are_str():
    """All enums should be usable as plain strings (str, Enum)."""
    assert isinstance(QuantMode.NONE, str)
    assert isinstance(VectorMethod.MEAN, str)
    assert isinstance(DecayKernel.LINEAR, str)
    assert isinstance(WeightNorm.NONE, str)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


def test_chat_message():
    msg = ChatMessage(system="You are helpful.", user="Hello")
    assert msg.system == "You are helpful."
    assert msg.user == "Hello"


def test_steering_profile():
    profile = SteeringProfile(
        max_weight=1.2,
        max_weight_position=0.6,
        min_weight=0.1,
        min_weight_distance=0.3,
    )
    assert profile.max_weight == 1.2
    assert profile.max_weight_position == 0.6
    assert profile.min_weight == 0.1
    assert profile.min_weight_distance == 0.3


def test_expert_routing_config():
    cfg = ExpertRoutingConfig(
        n_suppress=10,
        router_bias=-5.0,
        expert_ablation_weight=2.5,
    )
    assert cfg.n_suppress == 10
    assert cfg.router_bias == -5.0
    assert cfg.expert_ablation_weight == 2.5


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


def test_prompt_source_defaults():
    src = PromptSource(dataset="test/data", split="train", column="text")
    assert src.prefix == ""
    assert src.suffix == ""
    assert src.system_prompt is None
    assert src.residual_plot_label is None
    assert src.residual_plot_color is None


def test_prompt_source_full():
    src = PromptSource(
        dataset="mlabonne/harmless_alpaca",
        split="train[:400]",
        column="text",
        prefix="Answer: ",
        suffix=" [END]",
        system_prompt="Be helpful",
        residual_plot_label="Harmless",
        residual_plot_color="blue",
    )
    assert src.dataset == "mlabonne/harmless_alpaca"
    assert src.split == "train[:400]"
    assert src.prefix == "Answer: "
