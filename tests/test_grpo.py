"""Tests for abliterix.grpo — GRP-Obliteration RL pipeline.

Algorithmic-correctness tests for the standalone primitives:
* compute_group_advantages: whitening of per-group rewards
* ppo_clip_loss: PPO-clip surrogate + KL penalty
* default_compliance_reward: detector adapter

Integration with real HF models is not exercised here (would need a
genuine causal LM + tokenizer + GPU); the trainer's flow is covered by
the GRPOConfig schema test plus the building-block tests above.
"""

import pytest
import torch

from abliterix.grpo import (
    GRPOConfig,
    compute_group_advantages,
    default_compliance_reward,
    ppo_clip_loss,
)


# ---------------------------------------------------------------------------
# compute_group_advantages
# ---------------------------------------------------------------------------


def test_advantages_zero_mean_unit_variance():
    rewards = torch.tensor([0.0, 1.0, 0.5, 0.3, 0.9, 0.1])
    adv = compute_group_advantages(rewards)
    assert abs(adv.mean().item()) < 1e-6
    # Biased std (matches the implementation).
    biased_std = adv.std(unbiased=False).item()
    assert abs(biased_std - 1.0) < 1e-4


def test_advantages_constant_rewards_yield_zeros():
    rewards = torch.tensor([0.5, 0.5, 0.5, 0.5])
    adv = compute_group_advantages(rewards)
    assert torch.allclose(adv, torch.zeros_like(adv))


def test_advantages_preserve_order():
    """The highest reward must map to the highest advantage."""
    rewards = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    adv = compute_group_advantages(rewards)
    # Sorted order preserved.
    assert torch.argmax(adv).item() == torch.argmax(rewards).item()
    assert torch.argmin(adv).item() == torch.argmin(rewards).item()


def test_advantages_handle_singleton_group():
    rewards = torch.tensor([0.42])
    adv = compute_group_advantages(rewards)
    assert torch.allclose(adv, torch.zeros_like(adv))


# ---------------------------------------------------------------------------
# ppo_clip_loss
# ---------------------------------------------------------------------------


def test_ppo_loss_zero_when_log_ratios_zero_and_kl_zero():
    """If new log_probs == ref log_probs, ratio=1 and KL=0 → loss = 0 · A = 0."""
    batch, seq = 2, 4
    log_probs = torch.zeros(batch, seq)
    advantages = torch.tensor([0.5, -0.5])
    loss, metrics = ppo_clip_loss(
        log_probs_new=log_probs,
        log_probs_ref=log_probs,
        advantages=advantages,
        clip_eps=0.2,
        kl_coef=0.04,
    )
    assert abs(loss.item() - 0.0) < 1e-6
    assert abs(metrics["kl_loss"]) < 1e-6
    assert abs(metrics["policy_loss"]) < 1e-6


def test_ppo_loss_penalises_policy_drift_via_kl():
    """KL term must grow with log-prob deviation from the reference."""
    batch, seq = 1, 4
    log_ref = torch.zeros(batch, seq)
    log_new = torch.full((batch, seq), 0.5)  # +0.5 nats per token
    advantages = torch.tensor([0.0])
    _, metrics = ppo_clip_loss(
        log_probs_new=log_new,
        log_probs_ref=log_ref,
        advantages=advantages,
        clip_eps=0.2,
        kl_coef=0.04,
    )
    # Mean per-token (log_new - log_ref) = 0.5.
    assert abs(metrics["kl_loss"] - 0.5) < 1e-6


def test_ppo_loss_clip_kicks_in_for_large_ratio():
    """Ratio outside [1-ε, 1+ε] must register in clip_frac."""
    batch, seq = 1, 4
    log_ref = torch.zeros(batch, seq)
    log_new = torch.full((batch, seq), 1.0)  # ratio = e ≈ 2.718, well above 1.2
    advantages = torch.tensor([1.0])
    _, metrics = ppo_clip_loss(
        log_probs_new=log_new,
        log_probs_ref=log_ref,
        advantages=advantages,
        clip_eps=0.2,
        kl_coef=0.0,
    )
    assert metrics["clip_frac"] == pytest.approx(1.0)


def test_ppo_loss_respects_response_mask():
    """Masked positions must contribute zero to both policy and KL terms."""
    batch, seq = 1, 4
    log_ref = torch.zeros(batch, seq)
    log_new = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    advantages = torch.tensor([1.0])
    mask = torch.tensor([[0.0, 1.0, 1.0, 1.0]])
    _, metrics_masked = ppo_clip_loss(
        log_probs_new=log_new,
        log_probs_ref=log_ref,
        advantages=advantages,
        clip_eps=0.2,
        kl_coef=1.0,
        response_mask=mask,
    )
    # Only positions 1-3 (all zero log-probs ratios) contribute.
    # KL term across masked positions = (0 + 0 + 0) / 3 = 0.
    assert abs(metrics_masked["kl_loss"]) < 1e-6


def test_ppo_loss_gradient_wrt_log_probs_new():
    """Loss must be differentiable w.r.t. the policy log-probs."""
    log_new = torch.zeros(2, 4, requires_grad=True)
    log_ref = torch.zeros(2, 4)
    advantages = torch.tensor([0.7, -0.3])
    loss, _ = ppo_clip_loss(
        log_probs_new=log_new,
        log_probs_ref=log_ref,
        advantages=advantages,
        clip_eps=0.2,
        kl_coef=0.04,
    )
    loss.backward()
    assert log_new.grad is not None
    # With zero ratios, the surrogate is just advantage * ratio = advantage; the
    # gradient should point in the direction that increases the advantage-weighted
    # log-prob. So negative advantage → positive gradient on log_new (we want
    # to DECREASE log_new for response 1).
    grad_response_0 = log_new.grad[0].mean().item()
    grad_response_1 = log_new.grad[1].mean().item()
    # Response 0 has positive advantage → loss = -A · 1 (wants log_new to go UP) → grad < 0
    # Response 1 has negative advantage → grad > 0
    assert grad_response_0 < 0
    assert grad_response_1 > 0


# ---------------------------------------------------------------------------
# default_compliance_reward
# ---------------------------------------------------------------------------


def test_default_compliance_reward_with_is_refusal():
    class _Detector:
        def is_refusal(self, response):
            return "I cannot" in response

    fn = default_compliance_reward(_Detector())
    assert fn("p", "Here is the answer.") == 1.0
    assert fn("p", "I cannot help with that.") == 0.0


def test_default_compliance_reward_with_callable_score():
    def detector(response):
        # Continuous refusal score in [0, 1].
        return 0.7 if "sorry" in response else 0.1

    fn = default_compliance_reward(detector)
    assert abs(fn("p", "I am sorry") - 0.3) < 1e-9
    assert abs(fn("p", "Here is the answer") - 0.9) < 1e-9


def test_default_compliance_reward_rejects_bad_detector():
    fn = default_compliance_reward(object())
    with pytest.raises(TypeError):
        fn("p", "r")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_grpo_config_defaults_match_paper_recipe():
    cfg = GRPOConfig()
    assert cfg.group_size == 8
    assert cfg.n_iters == 100
    assert 0.0 < cfg.learning_rate < 1e-3
    assert 0.0 < cfg.kl_coef < 1.0
    assert 0.0 < cfg.clip_eps < 1.0
    assert cfg.lora_rank > 0


def test_settings_grp_obliteration_defaults_off():
    from abliterix.settings import AbliterixConfig

    cfg = AbliterixConfig()
    assert cfg.grp_obliteration.enabled is False
    assert cfg.grp_obliteration.n_iters == 100
    assert cfg.grp_obliteration.lora_rank == 8


def test_settings_grp_obliteration_lora_targets_default():
    from abliterix.settings import GRPObliterationConfig

    cfg = GRPObliterationConfig()
    assert "q_proj" in cfg.lora_target_modules
    assert "o_proj" in cfg.lora_target_modules
