# Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>

from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field


class QuantMode(str, Enum):
    NONE = "none"
    BNB_4BIT = "bnb_4bit"
    BNB_8BIT = "bnb_8bit"
    FP8 = "fp8"


class VectorMethod(str, Enum):
    MEAN = "mean"
    MEDIAN_OF_MEANS = "median_of_means"
    PCA = "pca"


class DecayKernel(str, Enum):
    LINEAR = "linear"
    GAUSSIAN = "gaussian"
    COSINE = "cosine"


class WeightNorm(str, Enum):
    NONE = "none"
    PRE = "pre"
    # POST = "post"  # Theoretically valid but empirically useless.
    FULL = "full"


class PromptSource(BaseModel):
    dataset: str = Field(
        description="Hugging Face dataset identifier or local directory path."
    )

    split: str = Field(description="Dataset split expression (e.g. 'train[:400]').")

    column: str = Field(description="Name of the text column containing prompts.")

    prefix: str = Field(
        default="",
        description="Static text prepended to every prompt.",
    )

    suffix: str = Field(
        default="",
        description="Static text appended to every prompt.",
    )

    system_prompt: str | None = Field(
        default=None,
        description="Per-dataset system prompt override (takes precedence over the global setting).",
    )

    residual_plot_label: str | None = Field(
        default=None,
        description="Legend label when plotting residual projections for this dataset.",
    )

    residual_plot_color: str | None = Field(
        default=None,
        description="Matplotlib colour used when plotting residual projections for this dataset.",
    )


@dataclass
class ChatMessage:
    system: str
    user: str


@dataclass
class SteeringProfile:
    max_weight: float
    max_weight_position: float
    min_weight: float
    min_weight_distance: float


@dataclass
class ExpertRoutingConfig:
    n_suppress: int
    router_bias: float
    expert_ablation_weight: float
