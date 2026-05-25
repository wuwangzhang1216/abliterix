"""Tests for abliterix.som — Self-Organising Map refusal directions.

Verifies the implementation of Piras et al. AAAI 2026 (arXiv:2511.08379)
*SOM Directions Are Better Than One*.
"""

import torch
import torch.nn.functional as F

from abliterix.som import (
    _bmu_assignments,
    _node_centroids,
    _train_som,
    compute_som_directions,
)
from abliterix.types import VectorMethod
from abliterix.vectors import compute_steering_vectors


# ---------------------------------------------------------------------------
# Kohonen training
# ---------------------------------------------------------------------------


def test_train_som_codebook_shape():
    torch.manual_seed(0)
    data = torch.randn(50, 16)
    codebook = _train_som(
        data, grid_h=3, grid_w=2, n_iters=50, initial_lr=0.5, initial_sigma=1.5, seed=1
    )
    # Total nodes = 3 * 2 = 6.
    assert codebook.shape == (6, 16)


def test_train_som_codebook_moves_toward_data():
    """Codebook entries must move toward the data manifold during training."""
    torch.manual_seed(0)
    data = torch.randn(100, 16) + 5.0  # offset cluster
    codebook = _train_som(
        data, grid_h=3, grid_w=3, n_iters=200, initial_lr=0.5, initial_sigma=2.0, seed=2
    )
    # Mean codebook position should be near the data centroid.
    centroid = codebook.mean(dim=0)
    target = data.mean(dim=0)
    assert torch.linalg.vector_norm(centroid - target) < 1.0


def test_train_som_deterministic_with_seed():
    torch.manual_seed(0)
    data = torch.randn(40, 8)
    c1 = _train_som(
        data,
        grid_h=2,
        grid_w=2,
        n_iters=100,
        initial_lr=0.4,
        initial_sigma=1.0,
        seed=42,
    )
    c2 = _train_som(
        data,
        grid_h=2,
        grid_w=2,
        n_iters=100,
        initial_lr=0.4,
        initial_sigma=1.0,
        seed=42,
    )
    assert torch.allclose(c1, c2)


# ---------------------------------------------------------------------------
# BMU assignment & centroids
# ---------------------------------------------------------------------------


def test_bmu_assignments_chooses_nearest_node():
    codebook = torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    data = torch.tensor([[0.1, 0.1], [9.9, 0.0], [0.0, 9.9], [5.0, 5.0]])
    assigns = _bmu_assignments(data, codebook)
    assert assigns.tolist()[:3] == [0, 1, 2]


def test_node_centroids_averages_assigned_samples():
    codebook = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
    data = torch.tensor(
        [[0.0, 1.0], [1.0, 0.0], [9.0, 11.0], [11.0, 9.0]], dtype=torch.float32
    )
    centroids = _node_centroids(data, codebook.to(torch.float32))
    # Node 0 gets first two samples → centroid (0.5, 0.5).
    assert torch.allclose(centroids[0], torch.tensor([0.5, 0.5]), atol=1e-5)
    # Node 1 gets last two → centroid (10.0, 10.0).
    assert torch.allclose(centroids[1], torch.tensor([10.0, 10.0]), atol=1e-5)


def test_node_centroids_falls_back_for_empty_buckets():
    """Nodes with no BMU assignments must keep their codebook entry."""
    codebook = torch.tensor([[0.0, 0.0], [10.0, 10.0], [100.0, 100.0]])
    data = torch.tensor([[0.0, 0.0], [10.0, 10.0]], dtype=torch.float32)
    centroids = _node_centroids(data, codebook.to(torch.float32))
    # Node 2 has no assignment; centroid must equal codebook[2].
    assert torch.allclose(centroids[2], torch.tensor([100.0, 100.0]))


# ---------------------------------------------------------------------------
# compute_som_directions — public API
# ---------------------------------------------------------------------------


def test_compute_som_directions_output_shape(synthetic_states):
    benign, target = synthetic_states
    out = compute_som_directions(benign, target, grid_h=3, grid_w=2, n_iters=20, seed=0)
    # (3*2, layers=8, dim=64)
    assert out.shape == (6, 8, 64)


def test_compute_som_directions_unit_norm(synthetic_states):
    benign, target = synthetic_states
    out = compute_som_directions(benign, target, grid_h=2, grid_w=2, n_iters=30, seed=0)
    norms = torch.linalg.vector_norm(out, dim=2)
    # Every per-layer direction must be unit-norm.
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_compute_som_directions_distinct():
    """Different SOM nodes should give distinct directions."""
    torch.manual_seed(0)
    benign = torch.randn(80, 4, 32)
    # Heterogeneous target — multiple clusters in residual space.
    cluster_centers = torch.randn(3, 4, 32) * 5.0
    pick = torch.randint(0, 3, (80,))
    target = benign + cluster_centers[pick]

    out = compute_som_directions(
        benign, target, grid_h=2, grid_w=2, n_iters=300, seed=0
    )
    # Per-layer pairwise cosine similarities should NOT all be ~1.0.
    layer0 = out[:, 0, :]
    cos_sims = layer0 @ layer0.T  # (4, 4)
    off_diag = cos_sims - torch.eye(4) * cos_sims.diag()
    # At least one pair should have cosine below 0.95.
    assert off_diag.abs().max().item() < 0.999
    assert (off_diag.abs() < 0.95).any().item()


def test_compute_som_directions_handles_too_few_samples():
    """If target has fewer rows than n_nodes, fall back to mean-diff."""
    benign = torch.randn(4, 3, 16)
    target = torch.randn(4, 3, 16) + 1.0
    out = compute_som_directions(benign, target, grid_h=3, grid_w=3, n_iters=20, seed=0)
    # All 9 directions equal the mean-diff direction.
    expected = F.normalize(target.mean(dim=0) - benign.mean(dim=0), p=2, dim=1)
    for i in range(9):
        assert torch.allclose(out[i].float(), expected.float(), atol=1e-4)


def test_compute_som_directions_projection(synthetic_states):
    """orthogonal_projection must zero the benign-mean component per direction."""
    benign, target = synthetic_states
    out = compute_som_directions(
        benign,
        target,
        grid_h=2,
        grid_w=2,
        n_iters=30,
        seed=0,
        orthogonal_projection=True,
    )
    benign_dir = F.normalize(benign.mean(dim=0), p=2, dim=1)
    for i in range(out.shape[0]):
        for layer_idx in range(out.shape[1]):
            dot = (
                torch.dot(out[i, layer_idx].float(), benign_dir[layer_idx].float())
                .abs()
                .item()
            )
            assert dot < 1e-4, f"dir {i} layer {layer_idx}: {dot}"


# ---------------------------------------------------------------------------
# Public entry point via compute_steering_vectors
# ---------------------------------------------------------------------------


def test_compute_steering_vectors_routes_to_som(synthetic_states):
    benign, target = synthetic_states
    via_method = compute_steering_vectors(
        benign,
        target,
        VectorMethod.SOM,
        False,
        som_grid_h=2,
        som_grid_w=2,
        som_n_iters=20,
        som_seed=0,
    )
    direct = compute_som_directions(
        benign, target, grid_h=2, grid_w=2, n_iters=20, seed=0
    )
    assert via_method.shape == direct.shape == (4, 8, 64)
    assert torch.allclose(via_method.float(), direct.float(), atol=1e-5)
