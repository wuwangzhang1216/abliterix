"""Tests for abliterix.sae — SAE-feature-basis refusal steering."""

import pytest
import torch
import torch.nn.functional as F

from abliterix.sae import (
    SAEWeights,
    compute_sae_steering_directions,
    extract_sae_directions,
    load_sae,
    score_sae_features,
)
from abliterix.types import VectorMethod
from abliterix.vectors import compute_steering_vectors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _synth_sae(hidden: int = 64, n_features: int = 32, seed: int = 0) -> SAEWeights:
    """Synthetic SAE with random encoder/decoder, normalised columns."""
    torch.manual_seed(seed)
    W_enc = F.normalize(torch.randn(n_features, hidden), p=2, dim=1)
    # Decoder columns aligned with encoder rows for sane round-trip.
    W_dec = F.normalize(torch.randn(hidden, n_features), p=2, dim=0)
    return SAEWeights(W_enc=W_enc, W_dec=W_dec)


# ---------------------------------------------------------------------------
# SAEWeights basics
# ---------------------------------------------------------------------------


def test_saeweights_shape_properties():
    sae = _synth_sae(hidden=128, n_features=64)
    assert sae.hidden_dim == 128
    assert sae.n_features == 64


def test_saeweights_encode_shape():
    sae = _synth_sae(hidden=32, n_features=8)
    x = torch.randn(5, 32)
    feats = sae.encode(x)
    assert feats.shape == (5, 8)
    # ReLU output must be non-negative.
    assert (feats >= 0.0).all()


def test_saeweights_encode_applies_biases():
    hidden, n_features = 16, 4
    sae = SAEWeights(
        W_enc=torch.eye(n_features, hidden),  # picks first n_features dims
        W_dec=torch.eye(hidden, n_features),
        b_enc=torch.tensor([1.0, 2.0, 3.0, 4.0]),
    )
    feats = sae.encode(torch.zeros(1, hidden))
    # ReLU(0 + b_enc) = b_enc.
    assert torch.allclose(feats[0], torch.tensor([1.0, 2.0, 3.0, 4.0]))


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def test_load_sae_pt_roundtrip(tmp_path):
    sae = _synth_sae()
    state = {"W_enc": sae.W_enc, "W_dec": sae.W_dec}
    fp = tmp_path / "sae.pt"
    torch.save(state, fp)
    loaded = load_sae(str(fp), hidden_dim=sae.hidden_dim)
    assert loaded.W_enc.shape == sae.W_enc.shape
    assert torch.allclose(loaded.W_enc, sae.W_enc)
    assert torch.allclose(loaded.W_dec, sae.W_dec)


def test_load_sae_encoder_decoder_key_variants(tmp_path):
    sae = _synth_sae()
    # Alternate keys: encoder.weight / decoder.weight.
    state = {"encoder.weight": sae.W_enc, "decoder.weight": sae.W_dec}
    fp = tmp_path / "sae2.pt"
    torch.save(state, fp)
    loaded = load_sae(str(fp))
    assert torch.allclose(loaded.W_enc, sae.W_enc)


def test_load_sae_transposes_decoder_when_needed(tmp_path):
    """Decoder stored as (n_features, hidden_dim) should be auto-transposed."""
    hidden, n_features = 16, 4
    W_enc = torch.randn(n_features, hidden)
    # Decoder as (n_features, hidden) — same shape as encoder.
    W_dec_alt = torch.randn(n_features, hidden)
    state = {"W_enc": W_enc, "W_dec": W_dec_alt}
    fp = tmp_path / "sae3.pt"
    torch.save(state, fp)
    loaded = load_sae(str(fp))
    # After transpose, decoder shape should be (hidden, n_features).
    assert loaded.W_dec.shape == (hidden, n_features)


def test_load_sae_raises_on_missing_keys(tmp_path):
    fp = tmp_path / "bad.pt"
    torch.save({"some_other_key": torch.randn(4, 4)}, fp)
    with pytest.raises(KeyError, match="encoder/decoder"):
        load_sae(str(fp))


def test_load_sae_raises_on_hidden_mismatch(tmp_path):
    sae = _synth_sae(hidden=32)
    fp = tmp_path / "sae.pt"
    torch.save({"W_enc": sae.W_enc, "W_dec": sae.W_dec}, fp)
    with pytest.raises(ValueError, match="hidden dim"):
        load_sae(str(fp), hidden_dim=64)


# ---------------------------------------------------------------------------
# Feature scoring
# ---------------------------------------------------------------------------


def test_score_sae_features_returns_one_score_per_feature():
    sae = _synth_sae(hidden=32, n_features=16)
    benign = torch.randn(20, 32)
    target = torch.randn(20, 32)
    scores = score_sae_features(sae, benign, target)
    assert len(scores) == sae.n_features


def test_score_sae_features_sorted_descending():
    sae = _synth_sae(hidden=32, n_features=16)
    benign = torch.randn(20, 32)
    target = torch.randn(20, 32) + 2.0
    scores = score_sae_features(sae, benign, target)
    score_values = [s.score for s in scores]
    assert score_values == sorted(score_values, reverse=True)


def test_score_sae_features_ranks_planted_refusal_feature():
    """A feature with strong target-vs-benign activation difference must rank first."""
    hidden, n_features = 16, 8
    sae = SAEWeights(
        W_enc=torch.eye(n_features, hidden) * 5.0,  # each feature reads one dim
        W_dec=torch.eye(hidden, n_features),
    )
    # Plant a strong activation in dim 3 of target states but not benign.
    benign = torch.zeros(20, hidden)
    target = torch.zeros(20, hidden)
    target[:, 3] = 2.0
    scores = score_sae_features(sae, benign, target)
    assert scores[0].feature_idx == 3


def test_score_sae_features_hidden_mismatch_raises():
    sae = _synth_sae(hidden=32)
    with pytest.raises(ValueError, match="hidden_dim"):
        score_sae_features(sae, torch.randn(5, 64), torch.randn(5, 64))


# ---------------------------------------------------------------------------
# Direction extraction
# ---------------------------------------------------------------------------


def test_extract_sae_directions_shape():
    sae = _synth_sae(hidden=32, n_features=16)
    benign = torch.randn(20, 32)
    target = torch.randn(20, 32) + 1.0
    dirs, scores = extract_sae_directions(sae, benign, target, top_k=4)
    assert dirs.shape == (4, 32)
    assert len(scores) == 4


def test_extract_sae_directions_unit_norm():
    sae = _synth_sae(hidden=32, n_features=16)
    benign = torch.randn(20, 32)
    target = torch.randn(20, 32) + 1.0
    dirs, _ = extract_sae_directions(sae, benign, target, top_k=4)
    norms = torch.linalg.vector_norm(dirs, dim=1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5)


def test_extract_sae_directions_use_top_features():
    """Top direction should be the decoder column of the highest-scoring feature."""
    hidden, n_features = 16, 8
    sae = SAEWeights(
        W_enc=torch.eye(n_features, hidden) * 5.0,
        W_dec=F.normalize(torch.randn(hidden, n_features), p=2, dim=0),
    )
    benign = torch.zeros(20, hidden)
    target = torch.zeros(20, hidden)
    target[:, 5] = 2.0  # planted activation in dim 5 → feature 5
    dirs, scores = extract_sae_directions(sae, benign, target, top_k=1)
    expected = F.normalize(sae.W_dec[:, 5], p=2, dim=0)
    assert torch.allclose(dirs[0], expected, atol=1e-5)
    assert scores[0].feature_idx == 5


# ---------------------------------------------------------------------------
# compute_sae_steering_directions — full multi-layer tensor
# ---------------------------------------------------------------------------


def test_compute_sae_steering_directions_shape(synthetic_states):
    benign, target = synthetic_states
    sae = _synth_sae(hidden=64, n_features=32)
    out, scores = compute_sae_steering_directions(
        sae, benign, target, sae_layer=3, top_k=5
    )
    # (top_k, layers+1, hidden_dim) = (5, 8, 64).
    assert out.shape == (5, 8, 64)
    assert len(scores) == 5


def test_compute_sae_steering_directions_uses_sae_at_target_layer(synthetic_states):
    benign, target = synthetic_states
    sae = _synth_sae(hidden=64, n_features=32, seed=10)
    sae_layer = 3
    residual_idx = sae_layer + 1
    out, scores = compute_sae_steering_directions(
        sae, benign, target, sae_layer=sae_layer, top_k=4
    )
    # At SAE layer: should be the SAE decoder columns (unit-norm by construction).
    expected, _ = extract_sae_directions(
        sae,
        benign[:, residual_idx, :],
        target[:, residual_idx, :],
        top_k=4,
    )
    assert torch.allclose(out[:, residual_idx, :].float(), expected.float(), atol=1e-5)


def test_compute_sae_steering_directions_mean_diff_at_other_layers(synthetic_states):
    benign, target = synthetic_states
    sae = _synth_sae(hidden=64, n_features=32)
    sae_layer = 3
    out, _ = compute_sae_steering_directions(
        sae, benign, target, sae_layer=sae_layer, top_k=4
    )
    expected_mean = F.normalize(target.mean(dim=0) - benign.mean(dim=0), p=2, dim=1)
    other_layer = 0
    for i in range(4):
        assert torch.allclose(
            out[i, other_layer, :].float(),
            expected_mean[other_layer].float(),
            atol=1e-5,
        )


def test_compute_sae_steering_directions_out_of_range_layer_raises(synthetic_states):
    benign, target = synthetic_states
    sae = _synth_sae(hidden=64, n_features=32)
    with pytest.raises(ValueError, match="out of range"):
        compute_sae_steering_directions(sae, benign, target, sae_layer=99, top_k=2)


# ---------------------------------------------------------------------------
# Public API via compute_steering_vectors
# ---------------------------------------------------------------------------


def test_compute_steering_vectors_routes_to_sae(synthetic_states, tmp_path):
    benign, target = synthetic_states
    sae = _synth_sae(hidden=64, n_features=32)
    fp = tmp_path / "sae.pt"
    torch.save({"W_enc": sae.W_enc, "W_dec": sae.W_dec}, fp)

    via_method = compute_steering_vectors(
        benign,
        target,
        VectorMethod.SAE,
        False,
        sae_path=str(fp),
        sae_layer=2,
        sae_top_k=3,
    )
    direct, _ = compute_sae_steering_directions(
        sae, benign, target, sae_layer=2, top_k=3
    )
    assert via_method.shape == direct.shape == (3, 8, 64)
    assert torch.allclose(via_method.float(), direct.float(), atol=1e-5)


def test_compute_steering_vectors_sae_requires_path(synthetic_states):
    benign, target = synthetic_states
    with pytest.raises(ValueError, match="sae_path"):
        compute_steering_vectors(benign, target, VectorMethod.SAE, False, sae_path=None)


# ---------------------------------------------------------------------------
# Settings validators
# ---------------------------------------------------------------------------


def test_settings_validator_sae_requires_path():
    from abliterix.settings import SteeringConfig

    with pytest.raises(ValueError, match="sae_path"):
        SteeringConfig(vector_method=VectorMethod.SAE)


def test_settings_validator_sae_top_k_positive():
    from abliterix.settings import SteeringConfig

    with pytest.raises(ValueError, match="sae_top_k"):
        SteeringConfig(
            vector_method=VectorMethod.SAE,
            sae_path="/tmp/x.pt",
            sae_top_k=0,
        )


def test_settings_validator_sae_layer_non_negative():
    from abliterix.settings import SteeringConfig

    with pytest.raises(ValueError, match="sae_layer"):
        SteeringConfig(
            vector_method=VectorMethod.SAE,
            sae_path="/tmp/x.pt",
            sae_layer=-1,
        )
