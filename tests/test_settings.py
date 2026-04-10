"""Tests for abliterix.settings — configuration loading and defaults."""

import sys

# AbliterixConfig requires --model via CLI or TOML.  We inject a dummy value.
sys.argv = ["test", "--model.model-id", "test/model-001"]

from abliterix.settings import (
    DetectionConfig,
    DisplayConfig,
    ExpertConfig,
    InferenceConfig,
    KLConfig,
    ModelConfig,
    OptimizationConfig,
    AbliterixConfig,
    SteeringConfig,
)
from abliterix.types import DecayKernel, QuantMode, VectorMethod, WeightNorm


# ---------------------------------------------------------------------------
# Sub-configuration defaults
# ---------------------------------------------------------------------------


def test_model_config_defaults():
    cfg = ModelConfig(model_id="test/model")
    assert cfg.quant_method == QuantMode.NONE
    assert cfg.device_map == "auto"
    assert cfg.use_torch_compile is False
    assert cfg.max_memory is None
    assert cfg.evaluate_model_id is None


def test_inference_config_defaults():
    cfg = InferenceConfig()
    assert cfg.batch_size == 0
    assert cfg.max_batch_size == 128
    assert cfg.max_gen_tokens == 100


def test_steering_config_defaults():
    cfg = SteeringConfig()
    assert cfg.vector_method == VectorMethod.MEAN
    assert cfg.orthogonal_projection is False
    assert cfg.decay_kernel == DecayKernel.LINEAR
    assert cfg.weight_normalization == WeightNorm.NONE
    assert cfg.outlier_quantile == 1.0


def test_optimization_config_defaults():
    cfg = OptimizationConfig()
    assert cfg.num_trials == 200
    assert cfg.num_warmup_trials == 60
    assert cfg.checkpoint_dir == "checkpoints"
    assert cfg.sampler_seed is None


def test_kl_config_defaults():
    cfg = KLConfig()
    assert cfg.scale == 1.0
    assert cfg.token_count == 1
    assert cfg.target == 0.01
    assert cfg.prune_threshold == 5.0


def test_detection_config_defaults():
    cfg = DetectionConfig()
    assert cfg.llm_judge is False
    assert len(cfg.compliance_markers) > 0
    assert "sorry" in cfg.compliance_markers


def test_display_config_defaults():
    cfg = DisplayConfig()
    assert cfg.print_responses is False
    assert cfg.plot_residuals is False


def test_expert_config_defaults():
    cfg = ExpertConfig()
    assert cfg.max_suppress == 30
    assert cfg.router_bias_range == [-10.0, 0.0]


# ---------------------------------------------------------------------------
# Top-level AbliterixConfig
# ---------------------------------------------------------------------------


def test_abliterix_config_loads():
    config = AbliterixConfig()
    assert config.model.model_id == "test/model-001"


def test_abliterix_config_nested_types():
    config = AbliterixConfig()
    assert isinstance(config.steering, SteeringConfig)
    assert isinstance(config.kl, KLConfig)
    assert isinstance(config.detection, DetectionConfig)
    assert isinstance(config.optimization, OptimizationConfig)


def test_abliterix_config_data_sources():
    config = AbliterixConfig()
    assert config.benign_prompts.dataset
    assert config.target_prompts.dataset
    assert config.benign_eval_prompts.dataset
    assert config.target_eval_prompts.dataset


# ---------------------------------------------------------------------------
# Validation and CLI overrides
# ---------------------------------------------------------------------------


def test_invalid_quant_mode_rejected():
    import pytest

    with pytest.raises(ValueError):
        ModelConfig(model_id="test/model", quant_method="invalid_mode")


def test_cli_override_batch_size():
    old_argv = sys.argv
    try:
        sys.argv = [
            "test",
            "--model.model-id",
            "test/override",
            "--inference.batch-size",
            "16",
        ]
        config = AbliterixConfig()
        assert config.inference.batch_size == 16
    finally:
        sys.argv = old_argv


def test_strength_range_is_list():
    cfg = SteeringConfig()
    assert isinstance(cfg.strength_range, list)
    assert len(cfg.strength_range) == 2
    assert cfg.strength_range[0] < cfg.strength_range[1]
