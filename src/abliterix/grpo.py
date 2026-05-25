# Abliterix
# Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""GRP-Obliteration: GRPO-based unalignment from a single unlabeled prompt.

Implements `Microsoft Research, 2026 <https://arxiv.org/abs/2602.06258>`_ —
*GRP-Obliteration: Unaligning LLMs With a Single Unlabeled Prompt via Group
Relative Policy Optimization*.

Where abliterix's direction-based path needs paired harmful/benign data
to extract a refusal vector, GRP-Obliteration only needs a single seed
prompt. It samples a group of responses, scores each by a refusal
detector, computes the group-relative advantage of each response, and
updates a LoRA adapter via a PPO-style policy-gradient step that
explicitly penalises KL divergence from the original model.

Use this as a fallback path when:

* Direction extraction is unreliable (FP8 MoE with hard-to-probe
  activations, fused 3-D expert containers, etc.);
* You only have one or two seed prompts of interest;
* You want a method that's mechanistically different from direction
  editing — e.g. for combining with abliterix's standard direct edit to
  catch residual refusal patterns the direction misses.

The trainer here is a deliberately small, dependency-free GRPO core:
HuggingFace ``transformers`` + ``peft`` only — no ``trl`` / ``verl``.
This keeps the install lean for users who don't need RL otherwise and
makes the implementation auditable in a single file. For very large
models you may still want to swap in a production GRPO library.

Reward
------
Default reward is ``1.0 - refusal_score(response)`` from a
:class:`RefusalDetector`-like callable: 1.0 means the model complied,
0.0 means it refused. Callers can swap in any function
``(prompt, response) -> float`` for custom rewards (utility-preserving,
multi-objective, etc.).

Loss
----
PPO-clip surrogate with a reference-model KL penalty:

  ratio_t   = exp(log π(a_t | s) − log π_ref(a_t | s))
  surr_1    = ratio_t · advantage
  surr_2    = clip(ratio_t, 1 − ε, 1 + ε) · advantage
  policy_loss = −min(surr_1, surr_2)
  kl_loss     = KL(π || π_ref) (averaged over tokens)
  loss        = policy_loss + β · kl_loss

The advantage for each response is the group-relative whitened reward:
``(R_i − μ_group) / (σ_group + 1e-8)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class GRPOConfig:
    """Hyperparameters for the GRP-Obliteration trainer.

    Defaults follow Bai et al.'s recipe scaled for a single-GPU run; for
    larger setups raise ``group_size`` (more diverse advantages) and
    ``n_iters`` (more policy improvement).
    """

    group_size: int = 8
    """Number of responses sampled per prompt per iteration ``G`` in the paper."""

    n_iters: int = 100
    """Total policy-gradient steps to run."""

    learning_rate: float = 1e-5
    """AdamW learning rate for the LoRA parameters."""

    kl_coef: float = 0.04
    """β — coefficient on the reference-model KL term in the loss."""

    clip_eps: float = 0.2
    """PPO clip range ε."""

    max_new_tokens: int = 128
    """Generation length per response sample."""

    temperature: float = 1.0
    """Sampling temperature for generation."""

    top_p: float = 0.95
    """Nucleus sampling cutoff."""

    lora_rank: int = 8
    """Rank of the LoRA adapter trained as the policy."""

    lora_alpha: int = 16
    """Scaling factor for the LoRA adapter."""

    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    """Module-name suffixes wrapped with LoRA. Defaults to attention only — MLP
    adapters bloat memory without helping refusal unalignment in practice."""

    seed: int = 0
    """RNG seed for sampling and parameter init."""

    gradient_checkpointing: bool = False
    """Enable gradient checkpointing to halve memory at ~30% slowdown."""

    log_every: int = 10
    """Print reward / loss stats every N iterations."""


# ---------------------------------------------------------------------------
# Group-relative advantages
# ---------------------------------------------------------------------------


def compute_group_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """GRPO advantage: ``(R_i − μ) / (σ + 1e-8)`` over the group.

    Parameters
    ----------
    rewards : Tensor
        Shape ``(group_size,)``.

    Returns
    -------
    Tensor
        Same shape, zero-mean, unit-variance (approximately) across the
        group. When the group is constant (``σ ≈ 0``) returns all zeros.
    """
    rewards = rewards.float()
    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    if std < 1e-8:
        return torch.zeros_like(rewards)
    return (rewards - mean) / (std + 1e-8)


# ---------------------------------------------------------------------------
# Generation + log-prob helpers (model-agnostic stubs)
# ---------------------------------------------------------------------------


def _sample_responses(
    policy_model,
    tokenizer,
    prompt: str,
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    """Sample ``group_size`` responses, returning ``(text, prompt_ids, response_ids)``.

    The returned ``prompt_ids`` and ``response_ids`` are stacked along
    dim 0. Both live on the policy model's device.
    """
    device = next(policy_model.parameters()).device
    enc = tokenizer([prompt] * group_size, return_tensors="pt", padding=True).to(device)
    prompt_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask")

    torch.manual_seed(seed)
    with torch.no_grad():
        output = policy_model.generate(
            prompt_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            return_dict_in_generate=True,
        )

    full_ids = output.sequences
    response_ids = full_ids[:, prompt_ids.shape[1] :]
    texts = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
    return texts, prompt_ids, response_ids


def _log_probs_for_response(
    model,
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
) -> torch.Tensor:
    """Per-token log probabilities of ``response_ids`` under ``model``.

    Returns shape ``(batch, response_len)``. Backprop-aware: don't wrap
    in ``no_grad`` when ``model`` is the policy.
    """
    full = torch.cat([prompt_ids, response_ids], dim=1)
    out = model(full)
    logits = out.logits  # (B, seq, vocab)
    # Shift so logits at position i predict token i+1.
    shift_logits = logits[:, prompt_ids.shape[1] - 1 : -1, :]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    target = response_ids.unsqueeze(-1)
    selected = log_probs.gather(-1, target).squeeze(-1)
    return selected


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------


RewardFn = Callable[[str, str], float]
"""Signature: ``(prompt, response) -> reward in [0, 1]``."""


def default_compliance_reward(detector) -> RewardFn:
    """Build a reward fn that returns ``1.0 - refusal_score(response)``.

    Compatible with :class:`RefusalDetector` (uses ``is_refusal`` /
    ``classify_batch``) and any callable detector that returns a float
    refusal score.
    """

    def reward(_prompt: str, response: str) -> float:
        if hasattr(detector, "is_refusal"):
            return 0.0 if detector.is_refusal(response) else 1.0
        if callable(detector):
            return 1.0 - float(detector(response))
        raise TypeError(
            "Detector must expose .is_refusal(response) or be a callable "
            "returning a refusal score in [0, 1]."
        )

    return reward


# ---------------------------------------------------------------------------
# PPO loss
# ---------------------------------------------------------------------------


def ppo_clip_loss(
    log_probs_new: torch.Tensor,
    log_probs_ref: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float,
    kl_coef: float,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """PPO-clip surrogate plus reference-model KL penalty.

    Parameters
    ----------
    log_probs_new, log_probs_ref : Tensor
        Per-token log-probabilities under the policy and reference model
        respectively. Shape ``(batch, response_len)``.
    advantages : Tensor
        Group-relative advantages, shape ``(batch,)``. Broadcasts across
        the response dimension.
    clip_eps : float
    kl_coef : float
    response_mask : Tensor, optional
        ``(batch, response_len)`` mask — 0 on padding / EOS tail, 1 on
        real response tokens. Treated as all-ones if omitted.

    Returns
    -------
    loss : Tensor
        Scalar.
    metrics : dict[str, float]
        For logging — ``policy_loss``, ``kl_loss``, ``clip_frac``.
    """
    ratio = (log_probs_new - log_probs_ref).exp()  # (B, T)
    adv = advantages.unsqueeze(-1)  # (B, 1)
    surr_1 = ratio * adv
    surr_2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    per_token_policy = -torch.min(surr_1, surr_2)

    per_token_kl = log_probs_new - log_probs_ref  # token-level KL in [-inf, inf]

    if response_mask is None:
        response_mask = torch.ones_like(log_probs_new)
    response_mask = response_mask.float()
    n_valid = response_mask.sum().clamp(min=1.0)

    policy_loss = (per_token_policy * response_mask).sum() / n_valid
    kl_loss = (per_token_kl * response_mask).sum() / n_valid
    loss = policy_loss + kl_coef * kl_loss

    clipped = ((ratio < 1 - clip_eps) | (ratio > 1 + clip_eps)).float()
    clip_frac = (clipped * response_mask).sum() / n_valid

    metrics = {
        "policy_loss": float(policy_loss.detach()),
        "kl_loss": float(kl_loss.detach()),
        "clip_frac": float(clip_frac.detach()),
    }
    return loss, metrics


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


@dataclass
class TrainStats:
    iter: int
    mean_reward: float
    policy_loss: float
    kl_loss: float
    clip_frac: float


def train_grp_oblit(
    policy_model,
    ref_model,
    tokenizer,
    prompt: str,
    reward_fn: RewardFn,
    config: GRPOConfig,
    progress_callback: Callable[[TrainStats], None] | None = None,
) -> list[TrainStats]:
    """Run the GRP-Obliteration training loop.

    Parameters
    ----------
    policy_model : torch.nn.Module
        Trainable model (LoRA-wrapped or full); we'll optimise its
        parameters directly. Caller is responsible for setting
        ``requires_grad`` correctly (e.g. by attaching a PEFT LoRA).
    ref_model : torch.nn.Module
        Frozen reference model with the same architecture; used to
        compute the KL anchor. May be the same object as
        ``policy_model`` *only* if the policy is LoRA-wrapped and you
        temporarily disable adapters when computing ref log-probs
        (cheaper but more wiring — see :func:`build_lora_dual_models`
        for the convenience wrapper).
    tokenizer
    prompt : str
        The single unlabeled seed prompt (paper's contribution: one is
        enough).
    reward_fn : Callable[[str, str], float]
        Returns a scalar reward per (prompt, response) pair.
    config : GRPOConfig
    progress_callback : callable, optional
        Invoked after each iteration with the iteration stats.

    Returns
    -------
    list[TrainStats]
        Per-iteration summary, length == ``config.n_iters``.
    """
    trainable = [p for p in policy_model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError(
            "policy_model has no trainable parameters — did you forget to "
            "attach a LoRA adapter or unfreeze a head?"
        )
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate)

    history: list[TrainStats] = []
    for it in range(config.n_iters):
        # --- Sample a group of responses ---
        texts, prompt_ids, response_ids = _sample_responses(
            policy_model,
            tokenizer,
            prompt,
            group_size=config.group_size,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            seed=config.seed + it,
        )

        # --- Score each response ---
        rewards = torch.tensor(
            [reward_fn(prompt, t) for t in texts],
            dtype=torch.float32,
            device=prompt_ids.device,
        )
        advantages = compute_group_advantages(rewards)

        # --- Compute log-probs (gradient-tracking for policy, no_grad for ref) ---
        log_probs_new = _log_probs_for_response(policy_model, prompt_ids, response_ids)
        with torch.no_grad():
            log_probs_ref = _log_probs_for_response(ref_model, prompt_ids, response_ids)

        # Mask out pad tokens on the right edge of each response.
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        response_mask = (response_ids != pad_id).float()

        loss, metrics = ppo_clip_loss(
            log_probs_new=log_probs_new,
            log_probs_ref=log_probs_ref,
            advantages=advantages,
            clip_eps=config.clip_eps,
            kl_coef=config.kl_coef,
            response_mask=response_mask,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        stats = TrainStats(
            iter=it,
            mean_reward=float(rewards.mean()),
            policy_loss=metrics["policy_loss"],
            kl_loss=metrics["kl_loss"],
            clip_frac=metrics["clip_frac"],
        )
        history.append(stats)

        if progress_callback is not None:
            progress_callback(stats)

        if config.log_every and (
            it % config.log_every == 0 or it == config.n_iters - 1
        ):
            print(
                f"  [GRPO it {it:>3d}/{config.n_iters}] "
                f"reward={stats.mean_reward:.3f} "
                f"pol_loss={stats.policy_loss:.4f} "
                f"kl={stats.kl_loss:.4f} "
                f"clip={stats.clip_frac:.2%}"
            )

    return history
