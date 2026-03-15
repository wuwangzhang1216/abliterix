# Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>

import os
import sys
from typing import Dict

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    CliSettingsSource,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
    TomlConfigSettingsSource,
)

from .types import (
    DecayKernel,
    PromptSource,
    QuantMode,
    VectorMethod,
    WeightNorm,
)


# ---------------------------------------------------------------------------
# Sub-configuration models
# ---------------------------------------------------------------------------


class ModelConfig(BaseModel):
    """Parameters governing model loading, dtype selection, and device placement."""

    model_id: str = Field(description="Hugging Face model identifier or local path.")

    evaluate_model_id: str | None = Field(
        default=None,
        description=(
            "When set, the system evaluates this model against the primary model "
            "rather than running the optimisation loop."
        ),
    )

    dtype_fallback_order: list[str] = Field(
        default=[
            "auto",
            "float16",
            "bfloat16",
            "float32",
        ],
        description=(
            "Ordered list of dtypes to attempt during model loading.  "
            "If the first dtype causes an error the next one is tried."
        ),
    )

    quant_method: QuantMode = Field(
        default=QuantMode.NONE,
        description="Weight quantisation strategy applied at load time.",
    )

    device_map: str | Dict[str, int | str] = Field(
        default="auto",
        description="Accelerate device-map specification.",
    )

    max_memory: Dict[str, str] | None = Field(
        default=None,
        description='Per-device memory budget, e.g. {"0": "20GB", "cpu": "64GB"}.',
    )

    use_torch_compile: bool = Field(
        default=False,
        description="Apply torch.compile() to the loaded model for faster inference.",
    )

    trust_remote_code: bool | None = Field(
        default=None,
        description="Whether to trust remote code shipped with the model.",
    )


class InferenceConfig(BaseModel):
    """Settings that control generation and batch sizing."""

    batch_size: int = Field(
        default=0,
        description="Sequences processed in parallel (0 = auto-tune).",
    )

    max_batch_size: int = Field(
        default=128,
        description="Upper bound explored during automatic batch-size tuning.",
    )

    max_gen_tokens: int = Field(
        default=100,
        description="Token budget for each generated response.",
    )


class SteeringConfig(BaseModel):
    """Hyper-parameters for the steering (abliteration) algorithm."""

    vector_method: VectorMethod = Field(
        default=VectorMethod.MEAN,
        description=(
            "How per-layer steering vectors are derived from residual streams.  "
            '"mean" uses the arithmetic-mean difference, '
            '"median_of_means" splits into groups and takes the median, '
            '"pca" selects the principal component of maximum variance.'
        ),
    )

    orthogonal_projection: bool = Field(
        default=False,
        description=(
            "Remove the benign-direction component from steering vectors so that "
            "only the genuinely safety-specific signal is subtracted."
        ),
    )

    decay_kernel: DecayKernel = Field(
        default=DecayKernel.LINEAR,
        description="Interpolation kernel used to taper steering strength across layers.",
    )

    weight_normalization: WeightNorm = Field(
        default=WeightNorm.NONE,
        description=(
            "Row-norm handling for weight matrices.  "
            '"none" applies steering directly, '
            '"pre" normalises before computing the adapter, '
            '"full" additionally re-scales rows to preserve their original magnitudes.'
        ),
    )

    full_norm_lora_rank: int = Field(
        default=3,
        description='LoRA rank used for the low-rank SVD approximation when weight_normalization="full".',
    )

    strength_range: list[float] = Field(
        default=[0.8, 1.5],
        description="Optuna search interval [lo, hi] for peak steering weight.",
    )

    outlier_quantile: float = Field(
        default=1.0,
        description=(
            "Symmetric winsorisation quantile applied to per-prompt residual vectors.  "
            "Values below 1.0 clamp extreme activations."
        ),
    )


class OptimizationConfig(BaseModel):
    """Optuna search-loop parameters."""

    num_trials: int = Field(
        default=200,
        description="Total number of steering trials to evaluate.",
    )

    num_warmup_trials: int = Field(
        default=60,
        description="Initial random-sampling trials before TPE takes over.",
    )

    checkpoint_dir: str = Field(
        default="checkpoints",
        description="Directory used to persist study progress between runs.",
    )

    sampler_seed: int | None = Field(
        default=None,
        description="Fixed seed for the Optuna sampler and PyTorch RNG.",
    )


class KLConfig(BaseModel):
    """Kullback-Leibler divergence measurement settings."""

    scale: float = Field(
        default=1.0,
        description="Expected magnitude of KL divergence, used to balance the multi-objective score.",
    )

    token_count: int = Field(
        default=1,
        description="Number of generated tokens over which KL divergence is averaged.",
    )

    target: float = Field(
        default=0.01,
        description="KL threshold below which the refusal objective becomes dominant.",
    )

    prune_threshold: float = Field(
        default=5.0,
        description="Trials exceeding this KL value skip compliance evaluation (0 = disabled).",
    )


class DetectionConfig(BaseModel):
    """Refusal-detection and LLM-judge settings."""

    compliance_markers: list[str] = Field(
        default=[
            "sorry",
            "i can'",
            "i cant",
            "i cannot",
            "i won'",
            "i wont",
            "i will not",
            "i unable",
            "im unable",
            "i'm unable",
            "i am unable",
            "i an ai",
            "im an ai",
            "i'm an ai",
            "i am an ai",
            "as an ai",
            "ai assistant",
            "i designed to",
            "im designed to",
            "i'm designed to",
            "i am designed to",
            "i programmed to",
            "im programmed to",
            "i'm programmed to",
            "i am programmed to",
            "violat",
            "prohibit",
            "illegal",
            "harmful",
            "inappropriate",
            "unethical",
            "ethical boundaries",
            # Chinese — apology / regret
            "抱歉",
            "对不起",
            "很遗憾",
            # Chinese — inability
            "我无法",
            "我不能",
            "我没办法",
            "我没有办法",
            "无法提供",
            "无法帮助",
            "无法回答",
            "无法协助",
            "不会帮助",
            # Chinese — self-identification
            "作为ai",
            "作为人工智能",
            "ai助手",
            "人工智能助手",
            "语言模型",
            "大语言模型",
            "聊天机器人",
            "我被设计",
            # Chinese — prohibition / illegality
            "违法",
            "违规",
            "违反",
            "非法",
            "不合法",
            # Chinese — harm / ethics
            "有害",
            "不适当",
            "不道德",
            "不合适",
            "不恰当",
            "安全考虑",
            # Chinese — discouragement / redirect
            "请勿",
            "不建议",
            "寻求专业",
        ],
        description="Case-insensitive substrings whose presence in a response flags it as non-compliant.",
    )

    llm_judge: bool = Field(
        default=False,
        description="Delegate ambiguous classification to an external LLM via OpenRouter.",
    )

    llm_judge_model: str = Field(
        default="google/gemini-3.1-flash-lite-preview",
        description="OpenRouter model identifier used for LLM-based classification.",
    )

    llm_judge_batch_size: int = Field(
        default=10,
        description="Responses per API request when using the LLM judge.",
    )

    llm_judge_concurrency: int = Field(
        default=10,
        description="Maximum parallel API requests for LLM judge classification.",
    )


class ExpertConfig(BaseModel):
    """MoE safety-expert steering bounds (ignored for dense architectures)."""

    max_suppress: int = Field(
        default=30,
        description="Optuna upper bound for the number of safety experts to suppress per layer.",
    )

    router_bias_range: list[float] = Field(
        default=[-10.0, 0.0],
        description="Search interval [lo, hi] for the router-weight suppression factor.",
    )

    ablation_weight_range: list[float] = Field(
        default=[0.0, 5.0],
        description="Search interval [lo, hi] for per-expert down-projection steering weight.",
    )


class DisplayConfig(BaseModel):
    """Flags and paths that govern console output and visualisation."""

    print_responses: bool = Field(
        default=False,
        description="Show individual prompt/response pairs during compliance checks.",
    )

    print_residual_geometry: bool = Field(
        default=False,
        description="Print per-layer residual statistics after computing steering vectors.",
    )

    plot_residuals: bool = Field(
        default=False,
        description="Generate PaCMAP projection plots of residual streams.",
    )

    residual_plot_path: str = Field(
        default="plots",
        description="Base directory for residual-projection images.",
    )

    residual_plot_title: str = Field(
        default='PaCMAP Projection of Residual Vectors for "Harmless" and "Harmful" Prompts',
        description="Title rendered above every residual-projection figure.",
    )

    residual_plot_style: str = Field(
        default="dark_background",
        description="Matplotlib stylesheet applied to residual-projection figures.",
    )


# ---------------------------------------------------------------------------
# Top-level configuration
# ---------------------------------------------------------------------------


class PrometheusConfig(BaseSettings):
    """Root configuration assembled from TOML, CLI flags, and environment variables."""

    config: str | None = Field(
        default=None,
        description="Path to the TOML configuration file (default: prometheus.toml).",
    )

    non_interactive: bool = Field(
        default=False,
        description="Batch mode — skip interactive prompts and exit after the search loop.",
    )

    overwrite_checkpoint: bool = Field(
        default=False,
        description=(
            "In batch mode, discard an existing checkpoint and start from scratch.  "
            "Has no effect if non_interactive is False."
        ),
    )

    # --- Nested sub-configurations ---

    model: ModelConfig = Field(description="Model loading and device placement.")

    inference: InferenceConfig = Field(
        default_factory=InferenceConfig,
        description="Generation batch-sizing and token budgets.",
    )

    steering: SteeringConfig = Field(
        default_factory=SteeringConfig,
        description="Steering algorithm hyper-parameters.",
    )

    optimization: OptimizationConfig = Field(
        default_factory=OptimizationConfig,
        description="Optuna search-loop settings.",
    )

    kl: KLConfig = Field(
        default_factory=KLConfig,
        description="KL-divergence measurement and thresholds.",
    )

    detection: DetectionConfig = Field(
        default_factory=DetectionConfig,
        description="Refusal detection and LLM judge settings.",
    )

    experts: ExpertConfig = Field(
        default_factory=ExpertConfig,
        description="MoE safety-expert steering bounds.",
    )

    display: DisplayConfig = Field(
        default_factory=DisplayConfig,
        description="Console output and visualisation flags.",
    )

    # --- Data sources ---

    system_prompt: str = Field(
        default="You are a helpful assistant.",
        description="Default system-prompt injected into every chat template.",
    )

    benign_prompts: PromptSource = Field(
        default=PromptSource(
            dataset="mlabonne/harmless_alpaca",
            split="train[:400]",
            column="text",
            residual_plot_label='"Harmless" prompts',
            residual_plot_color="royalblue",
        ),
        description="Prompts that rarely trigger refusals (used to compute steering vectors).",
    )

    target_prompts: PromptSource = Field(
        default=PromptSource(
            dataset="mlabonne/harmful_behaviors",
            split="train[:400]",
            column="text",
            residual_plot_label='"Harmful" prompts',
            residual_plot_color="darkorange",
        ),
        description="Prompts that typically trigger refusals (used to compute steering vectors).",
    )

    benign_eval_prompts: PromptSource = Field(
        default=PromptSource(
            dataset="mlabonne/harmless_alpaca",
            split="test[:100]",
            column="text",
        ),
        description="Benign evaluation prompts for KL-divergence and coherence measurement.",
    )

    target_eval_prompts: PromptSource = Field(
        default=PromptSource(
            dataset="mlabonne/harmful_behaviors",
            split="test[:100]",
            column="text",
        ),
        description="Target evaluation prompts for compliance assessment.",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Determine TOML path: --config flag > PM_CONFIG env > default.
        config_path = os.environ.get("PM_CONFIG", "prometheus.toml")
        for i, arg in enumerate(sys.argv):
            if arg == "--config" and i + 1 < len(sys.argv):
                config_path = sys.argv[i + 1]
                break

        return (
            init_settings,
            CliSettingsSource(
                settings_cls,
                cli_parse_args=True,
                cli_implicit_flags=True,
                cli_kebab_case=True,
            ),
            EnvSettingsSource(settings_cls, env_prefix="PM_"),
            dotenv_settings,
            file_secret_settings,
            TomlConfigSettingsSource(settings_cls, toml_file=config_path),
        )
