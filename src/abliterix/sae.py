# Abliterix
# Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Sparse Autoencoder (SAE) feature basis for refusal steering.

Implements the SAE-feature-basis approach from
* `Hong et al., 2025 <https://arxiv.org/abs/2509.09708>`_ — *Beyond
  I'm Sorry, I Can't: Dissecting LLM Refusal*
* `Soto et al., 2025 <https://arxiv.org/abs/2511.00029>`_ — *Feature-
  Guided SAE Steering for Refusal-Rate Control*
* `Templeton et al., 2025 <https://arxiv.org/abs/2505.23556>`_ —
  *Understanding Refusal in LLMs with Sparse Autoencoders*

Where mean-diff abliteration picks a single direction in hidden space,
SAE-mode picks **interpretable features** in the SAE's learned feature
space, then maps them back to hidden space via the SAE's decoder columns.
The result is a more precise basis that disentangles "refusal" from the
capability features mean-diff inadvertently sweeps up.

Workflow
--------
1. Load a pre-trained SAE (e.g. Gemma-Scope, Llama-Scope, sae_lens, or
   a custom checkpoint).  The SAE must expose ``W_enc`` /
   ``W_dec`` / biases or the equivalent ``encoder.weight`` /
   ``decoder.weight`` keys.
2. Pass the SAE-layer slice of harmful and harmless activations through
   the SAE encoder to get feature activations.
3. Score each feature by ``|mean(harmful) − mean(harmless)|`` — features
   that fire on harmful prompts but not harmless ones are the refusal
   feature candidates.
4. Take the decoder columns ``W_dec[:, top_k]`` as the refusal directions
   in hidden space and stack them into the standard
   ``(n_dirs, layers+1, hidden_dim)`` multi-direction tensor.

The SAE is layer-specific by construction.  At the SAE's layer we use
the decoder-column directions; at every other layer we fall back to the
standard mean-diff direction so the rest of the model still gets a
steering signal.  This matches the practical reality that SAE coverage
is layer-by-layer (you'd need one SAE per layer for full coverage).

Limitations
-----------
* Layer-locked: a Gemma-Scope SAE trained on layer 22 only gives
  refusal features at layer 22.  Multi-layer coverage requires multiple
  SAEs (the orchestration is a follow-up).
* SAE-vs-model mismatch: the SAE must match the model's hidden_dim;
  the loader will raise on shape mismatch.
* Format-tolerant but not exhaustive: handles ``state_dict``-style
  ``.pt`` and ``safetensors`` files plus the most common key naming
  conventions; exotic formats need a small loader override.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# SAE container
# ---------------------------------------------------------------------------


@dataclass
class SAEWeights:
    """Minimal SAE container — encoder, decoder, optional biases.

    ``W_enc`` shape ``(n_features, hidden_dim)``: applied via
        ``feat = relu(W_enc @ x + b_enc)``  (per-sample)
    ``W_dec`` shape ``(hidden_dim, n_features)``: each column is the
        direction in hidden space that feature ``i`` represents.

    Bias tensors default to zeros if absent in the checkpoint.
    """

    W_enc: Tensor
    W_dec: Tensor
    b_enc: Tensor | None = None
    b_dec: Tensor | None = None

    @property
    def n_features(self) -> int:
        return self.W_enc.shape[0]

    @property
    def hidden_dim(self) -> int:
        return self.W_enc.shape[1]

    def encode(self, x: Tensor) -> Tensor:
        """Run the encoder + ReLU on a batch of hidden-space vectors.

        ``x`` shape ``(*, hidden_dim)`` → returns ``(*, n_features)``.
        """
        if self.b_dec is not None:
            x = x - self.b_dec
        out = x @ self.W_enc.T
        if self.b_enc is not None:
            out = out + self.b_enc
        return F.relu(out)


# ---------------------------------------------------------------------------
# Loader — supports the most common SAE checkpoint formats
# ---------------------------------------------------------------------------


_ENC_KEYS = (
    "W_enc",
    "encoder.weight",
    "encoder_weight",
    "enc_weight",
    "encoder",
)
_DEC_KEYS = (
    "W_dec",
    "decoder.weight",
    "decoder_weight",
    "dec_weight",
    "decoder",
)
_B_ENC_KEYS = ("b_enc", "encoder.bias", "encoder_bias", "enc_bias")
_B_DEC_KEYS = ("b_dec", "decoder.bias", "decoder_bias", "dec_bias")


def _first_match(state: dict, keys: tuple[str, ...]) -> Tensor | None:
    for k in keys:
        if k in state:
            return state[k]
    return None


def _load_state_dict(path: str) -> dict:
    """Load a state_dict from .pt, .pth, .bin, or .safetensors.

    Pure stdlib + torch — no sae_lens / transformer_lens dependency.
    """
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path)
    return torch.load(path, map_location="cpu", weights_only=True)


def load_sae(path: str, *, hidden_dim: int | None = None) -> SAEWeights:
    """Load an SAE checkpoint from disk into :class:`SAEWeights`.

    The loader inspects common key naming conventions; if your checkpoint
    uses non-standard keys (e.g. ``W_enc_safety``), open a ``state_dict``
    yourself and pass the resolved tensors into :class:`SAEWeights`
    directly.

    Parameters
    ----------
    path : str
        Local path to ``.pt`` / ``.pth`` / ``.bin`` / ``.safetensors``.
    hidden_dim : int, optional
        If given, the loader checks that the encoder's input width
        matches.  Mismatch is a fatal error — SAE features only make
        sense in the residual stream they were trained on.

    Returns
    -------
    SAEWeights
    """
    state = _load_state_dict(path)

    W_enc = _first_match(state, _ENC_KEYS)
    W_dec = _first_match(state, _DEC_KEYS)
    if W_enc is None or W_dec is None:
        raise KeyError(
            f"Could not locate encoder/decoder weights in {path}. "
            f"Looked for keys: {_ENC_KEYS} / {_DEC_KEYS}. "
            f"Top-level keys present: {sorted(state.keys())[:20]}..."
        )

    # Some checkpoints store decoder as (n_features, hidden_dim); detect
    # and transpose so the canonical shape is (hidden_dim, n_features).
    if W_dec.shape[0] == W_enc.shape[0] and W_dec.shape[1] == W_enc.shape[1]:
        W_dec = W_dec.T.contiguous()

    if W_enc.shape[1] != W_dec.shape[0]:
        raise ValueError(
            f"SAE encoder / decoder shape mismatch: W_enc {tuple(W_enc.shape)} "
            f"vs W_dec {tuple(W_dec.shape)}."
        )

    if hidden_dim is not None and W_enc.shape[1] != hidden_dim:
        raise ValueError(
            f"SAE hidden dim {W_enc.shape[1]} does not match model "
            f"hidden_dim={hidden_dim}.  Are you using the right SAE for "
            "this model?"
        )

    return SAEWeights(
        W_enc=W_enc.float(),
        W_dec=W_dec.float(),
        b_enc=_first_match(state, _B_ENC_KEYS),
        b_dec=_first_match(state, _B_DEC_KEYS),
    )


# ---------------------------------------------------------------------------
# Feature scoring
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureScore:
    feature_idx: int
    benign_mean: float
    target_mean: float
    score: float  # |target_mean - benign_mean|


def score_sae_features(
    sae: SAEWeights,
    benign_layer_states: Tensor,
    target_layer_states: Tensor,
) -> list[FeatureScore]:
    """Score every SAE feature by harmful-vs-benign activation difference.

    Implements the contrastive scoring of arXiv:2511.00029 (Soto et al.).

    Parameters
    ----------
    sae : SAEWeights
    benign_layer_states, target_layer_states : Tensor
        Shape ``(n_samples, hidden_dim)`` — residual states at the SAE's
        layer for harmless and harmful prompts respectively.

    Returns
    -------
    list[FeatureScore]
        Sorted by ``score`` descending (most refusal-aligned first).
    """
    if benign_layer_states.shape[1] != sae.hidden_dim:
        raise ValueError(
            f"benign_layer_states hidden_dim={benign_layer_states.shape[1]} "
            f"does not match SAE hidden_dim={sae.hidden_dim}."
        )
    if target_layer_states.shape[1] != sae.hidden_dim:
        raise ValueError(
            f"target_layer_states hidden_dim={target_layer_states.shape[1]} "
            f"does not match SAE hidden_dim={sae.hidden_dim}."
        )

    benign_feats = sae.encode(benign_layer_states.to(torch.float32))
    target_feats = sae.encode(target_layer_states.to(torch.float32))

    benign_mean = benign_feats.mean(dim=0)
    target_mean = target_feats.mean(dim=0)
    diff = (target_mean - benign_mean).abs()

    n_features = sae.n_features
    scores = [
        FeatureScore(
            feature_idx=i,
            benign_mean=benign_mean[i].item(),
            target_mean=target_mean[i].item(),
            score=diff[i].item(),
        )
        for i in range(n_features)
    ]
    scores.sort(key=lambda s: s.score, reverse=True)
    return scores


# ---------------------------------------------------------------------------
# Direction extraction
# ---------------------------------------------------------------------------


def extract_sae_directions(
    sae: SAEWeights,
    benign_layer_states: Tensor,
    target_layer_states: Tensor,
    *,
    top_k: int = 8,
) -> tuple[Tensor, list[FeatureScore]]:
    """Return the top-k SAE decoder columns most aligned with refusal.

    Parameters
    ----------
    sae : SAEWeights
    benign_layer_states, target_layer_states : Tensor
        Shape ``(n_samples, hidden_dim)``.
    top_k : int
        Number of refusal features to extract.

    Returns
    -------
    directions : Tensor
        Shape ``(top_k, hidden_dim)``, each row unit-normalised.
    feature_scores : list[FeatureScore]
        The top-k scores (in the same order as ``directions``).
    """
    scores = score_sae_features(sae, benign_layer_states, target_layer_states)
    chosen = scores[:top_k]

    # Decoder column for feature i is W_dec[:, i] (hidden_dim,).
    dirs = torch.stack(
        [sae.W_dec[:, s.feature_idx] for s in chosen], dim=0
    )  # (top_k, hidden_dim)
    dirs = F.normalize(dirs, p=2, dim=1)
    return dirs, chosen


# ---------------------------------------------------------------------------
# Full-tensor builder (plugs into the multi-direction infrastructure)
# ---------------------------------------------------------------------------


def compute_sae_steering_directions(
    sae: SAEWeights,
    benign_states: Tensor,
    target_states: Tensor,
    *,
    sae_layer: int,
    top_k: int = 8,
    orthogonal_projection: bool = False,
    projected_abliteration: bool = False,
) -> tuple[Tensor, list[FeatureScore]]:
    """Build the per-layer steering tensor with SAE directions at ``sae_layer``.

    At ``sae_layer``, the top-k SAE decoder columns are used (after
    optional projection against the benign mean).  At every other layer
    we substitute the standard mean-diff direction so the rest of the
    model still gets a coherent steering signal — the same fallback the
    paper-companion code adopts.

    Parameters
    ----------
    sae : SAEWeights
    benign_states, target_states : Tensor
        Shape ``(n, layers+1, hidden_dim)``.
    sae_layer : int
        Transformer layer index (0-based) where the SAE was trained.
        Residual-stream index is ``sae_layer + 1`` (embedding at 0).
    top_k : int
        Number of refusal features to use.

    Returns
    -------
    directions : Tensor
        Shape ``(top_k, layers+1, hidden_dim)``.
    feature_scores : list[FeatureScore]
    """
    n_residual = target_states.shape[1]
    residual_idx = sae_layer + 1
    if not 0 <= residual_idx < n_residual:
        raise ValueError(
            f"sae_layer={sae_layer} maps to residual index {residual_idx}, "
            f"out of range for {n_residual} residual slots "
            f"(model has {n_residual - 1} transformer layers)."
        )

    sae_layer_benign = benign_states[:, residual_idx, :]
    sae_layer_target = target_states[:, residual_idx, :]
    sae_dirs, feature_scores = extract_sae_directions(
        sae, sae_layer_benign, sae_layer_target, top_k=top_k
    )

    # Fallback per-layer directions: standard mean-diff.
    benign_mean = benign_states.mean(dim=0).to(torch.float32)
    diff = target_states.mean(dim=0).to(torch.float32) - benign_mean
    mean_dirs = F.normalize(diff, p=2, dim=1)  # (layers+1, hidden)

    # Broadcast mean_dirs to (top_k, layers+1, hidden) so every "direction
    # slot" gets the same mean-diff at non-SAE layers.
    out = mean_dirs.unsqueeze(0).expand(top_k, -1, -1).clone()
    out[:, residual_idx, :] = sae_dirs

    # Optional projection against the benign mean (per direction, per layer).
    if projected_abliteration or orthogonal_projection:
        benign_dir = F.normalize(benign_mean, p=2, dim=1)
        for i in range(top_k):
            v = out[i]
            proj = (v * benign_dir).sum(dim=1, keepdim=True)
            v = v - proj * benign_dir
            out[i] = F.normalize(v, p=2, dim=1)

    return out.to(benign_states.dtype), feature_scores
