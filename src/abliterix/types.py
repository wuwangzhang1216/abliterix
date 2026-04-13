# Abliterix — a derivative work of Heretic (https://github.com/p-e-w/heretic)
# Original work Copyright (C) 2025  Philipp Emanuel Weidmann (p-e-w)
# Modified work Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    import torch


class QuantMode(str, Enum):
    NONE = "none"
    BNB_4BIT = "bnb_4bit"
    BNB_8BIT = "bnb_8bit"
    FP8 = "fp8"


class VectorMethod(str, Enum):
    MEAN = "mean"
    MEDIAN_OF_MEANS = "median_of_means"
    PCA = "pca"
    OPTIMAL_TRANSPORT = "optimal_transport"
    PCA_OT_FULL = "pca_ot_full"  # Full affine PCA-OT from arxiv:2603.04355
    COSMIC = "cosmic"
    SRA = "sra"


class DecayKernel(str, Enum):
    LINEAR = "linear"
    GAUSSIAN = "gaussian"
    COSINE = "cosine"


class SteeringMode(str, Enum):
    LORA = "lora"
    ANGULAR = "angular"
    ADAPTIVE_ANGULAR = "adaptive_angular"
    SPHERICAL = "spherical"
    VECTOR_FIELD = "vector_field"
    DIRECT = "direct"


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
class PCAOTTransform:
    """Full PCA-OT affine transformation for a single layer.

    Stores T(x) = A_full @ x + b_full where:
    - A_full is the full-space transformation matrix (d, d)
    - b_full is the bias vector (d,)
    - P is the PCA projection matrix (k, d)
    - A_k is the OT map in reduced space (k, k)
    - b_k is the bias in reduced space (k,)
    """

    A_full: "torch.Tensor"  # (d, d)
    b_full: "torch.Tensor"  # (d,)
    P: "torch.Tensor"  # (k, d)
    A_k: "torch.Tensor"  # (k, k)
    b_k: "torch.Tensor"  # (k,)
    layer_idx: int


@dataclass
class ExpertRoutingConfig:
    n_suppress: int
    router_bias: float
    expert_ablation_weight: float
