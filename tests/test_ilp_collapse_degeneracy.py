"""Characterization test: documents *why* the Stage-2 ILP collapses into mega-clusters
when ``max_layer_span`` is loose, and which knob actually controls granularity.

Root cause: the ILP objective is ``sum_{same cluster} (theta - cos_ij)``. At
``theta=0`` every pair with ``cos > 0`` carries a negative (merge-rewarding)
coefficient and nothing penalises cluster *size*, so the optimum merges everything
that span and epsilon constraints allow. ``lambda_causal`` is ignored; ``theta``
(the resolution threshold) and ``eps_causal`` are the real levers.
"""

import numpy as np
import pytest
import torch

import summarization.cluster as ic
from summarization.cluster import _cosine_similarity, cluster_graph_ilp
from summarization.prune import PruneGraph
from summarization.summarize import Node


def _make_graph(
    monkeypatch: pytest.MonkeyPatch,
    monkeypatch_phi: np.ndarray,
    adj: torch.Tensor,
) -> PruneGraph:
    n = monkeypatch_phi.shape[0]
    nodes = [
        Node(node_id=f"{i}", node_idx=i, feature=i, layer=str(i), ctx_idx=0, feature_type="feature")
        for i in range(n)
    ]
    # cluster_graph_ilp recomputes phi internally; pin it to our synthetic vectors.
    # (*_a/**_k so it tolerates the normalize_weights kwarg cluster_graph_ilp passes.)
    monkeypatch.setattr(
        ic,
        "compute_phi_vectors",
        lambda *_a, **_k: torch.tensor(monkeypatch_phi, dtype=torch.float32),
    )
    return PruneGraph(nodes=nodes, pruned_adj=adj, metadata={})


def _n_feature_clusters(pg: PruneGraph, theta: float, lam: float) -> int:
    clusters = cluster_graph_ilp(
        pg, theta=theta, lambda_causal=lam, max_sn=None, max_layer_span=1000, time_limit=30.0
    )
    return len([c for c in clusters if c and c[0][0].isdigit()])


def test_ilp_collapse_mechanism(monkeypatch: pytest.MonkeyPatch) -> None:
    # 5 features sharing one positive component => all pairwise cos = 0.5 (> 0),
    # with NO edges between them => the causal term is identically 0 for every pair.
    n = 5
    phi = np.zeros((n, n + 1))
    phi[:, 0] = 1.0  # shared positive component -> positive pairwise cosine
    for i in range(n):
        phi[i, 1 + i] = 1.0  # idiosyncratic component
    adj = torch.zeros(n, n, dtype=torch.float32)  # edge-free
    pg = _make_graph(monkeypatch, phi, adj)

    # theta=0: positive cosines reward merging, nothing penalises size -> full collapse.
    assert _n_feature_clusters(pg, theta=0.0, lam=0.0) == 1

    # lambda_causal is a deprecated compatibility argument and does not enter the
    # ILP objective.
    assert _n_feature_clusters(pg, theta=0.0, lam=1e9) == 1

    # theta is the real lever. Below the 0.5 cosine it still merges; above it, splits.
    assert _n_feature_clusters(pg, theta=0.45, lam=0.0) == 1
    assert _n_feature_clusters(pg, theta=0.55, lam=0.0) == n


def test_adaptive_theta_resolves_to_percentile(monkeypatch: pytest.MonkeyPatch) -> None:
    # Adaptive theta "p<q>" must equal using the q-th percentile of the allowed-pair
    # cosines as a fixed threshold -> identical partition.
    rng = np.random.default_rng(0)
    phi = rng.standard_normal((7, 10))
    phi[:, 0] += 1.5  # shared positive component -> a spread of positive cosines
    pg = _make_graph(monkeypatch, phi, torch.zeros(7, 7, dtype=torch.float32))

    cos = _cosine_similarity(phi)  # loose span -> allowed pairs are all off-diagonal pairs
    iu, ju = np.triu_indices(7, k=1)
    thr = float(np.percentile(cos[iu, ju], 65))

    def norm(clusters: list[list[str]]) -> list[list[str]]:
        return sorted(sorted(c) for c in clusters)

    adaptive = cluster_graph_ilp(
        pg, theta="p65", lambda_causal=0.0, max_sn=None, max_layer_span=1000
    )
    fixed = cluster_graph_ilp(pg, theta=thr, lambda_causal=0.0, max_sn=None, max_layer_span=1000)
    assert norm(adaptive) == norm(fixed)


def test_adaptive_theta_rejects_malformed(monkeypatch: pytest.MonkeyPatch) -> None:
    pg = _make_graph(monkeypatch, np.eye(3) + 0.1, torch.zeros(3, 3, dtype=torch.float32))
    with pytest.raises(ValueError):
        cluster_graph_ilp(pg, theta="65")  # missing 'p' prefix
    with pytest.raises(ValueError):
        cluster_graph_ilp(pg, theta="p150")  # percentile out of [0, 100]
