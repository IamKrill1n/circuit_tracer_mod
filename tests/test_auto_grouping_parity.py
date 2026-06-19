from __future__ import annotations

import torch

from eval.legacy_cluster_baselines import (
    compute_similarity,
    eigengap_analysis,
    find_best_k,
    find_best_k_for_clusterer,
)
from summarization.prune import PruneGraph
from summarization.utils import _node_from_json_dict


def _build_test_graph() -> PruneGraph:
    node_specs: list[tuple[str, dict]] = [
        ("E_0_0", {"feature_type": "embedding", "is_target_logit": False, "layer": "E"}),
        ("1_0_0", {"feature_type": "sae_feature", "is_target_logit": False, "layer": "1"}),
        ("1_1_0", {"feature_type": "sae_feature", "is_target_logit": False, "layer": "1"}),
        ("2_0_0", {"feature_type": "sae_feature", "is_target_logit": False, "layer": "2"}),
        ("2_1_0", {"feature_type": "sae_feature", "is_target_logit": False, "layer": "2"}),
        ("27_0_0", {"feature_type": "logit", "is_target_logit": True, "layer": "27"}),
    ]
    nodes = [_node_from_json_dict({"node_id": nid, **spec}) for nid, spec in node_specs]

    pruned_adj = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.4, 0.0, 0.1, 0.0, 0.0, 0.0],
            [0.3, 0.2, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.7, 0.6, 0.0, 0.2, 0.0],
            [0.0, 0.6, 0.7, 0.3, 0.0, 0.0],
            [0.0, 0.1, 0.2, 0.8, 0.9, 0.0],
        ],
        dtype=torch.float32,
    )

    edge_relevance = pruned_adj.clone() * 0.8
    edge_influence = pruned_adj.clone() * 1.2
    node_relevance = torch.tensor([0.0, 0.4, 0.45, 0.7, 0.75, 0.0], dtype=torch.float32)
    node_influence = torch.tensor([0.0, 0.3, 0.35, 0.8, 0.85, 0.0], dtype=torch.float32)
    return PruneGraph(
        nodes=nodes,
        pruned_adj=pruned_adj,
        metadata={},
        node_influence=node_influence,
        node_relevance=node_relevance,
        edge_influence=edge_influence,
        edge_relevance=edge_relevance,
    )


def test_eigengap_analysis_outputs_expected_keys() -> None:
    prune_graph = _build_test_graph()
    similarity = compute_similarity(prune_graph, mean_method="arith")
    result = eigengap_analysis(similarity, prune_graph, max_k=5)
    assert {"eigengap_k", "eigenvalues", "gaps", "search_range"} <= set(result.keys())
    assert result["search_range"][0] <= result["search_range"][1]


def test_find_best_k_returns_L_metrics_and_picks_argmin() -> None:
    prune_graph = _build_test_graph()
    best_k, results = find_best_k(
        prune_graph,
        max_layer_span=4,
        k_min_override=2,
        k_max_override=3,
        max_sn=None,
    )
    assert best_k in results
    # Each entry should carry the Stage-2 objective terms and the partition.
    for v in results.values():
        assert {"L", "L_atom", "L_atom_norm", "C_causal", "prune_loss"} <= set(v.keys())
        assert "final_supernodes" in v
        assert 0.0 <= float(v["L"]) <= 1.0
    # best_k must minimize L.
    best_L = float(results[best_k]["L"])
    assert best_L == min(float(v["L"]) for v in results.values())


def test_find_best_k_high_lambda_causal_prefers_more_clusters() -> None:
    prune_graph = _build_test_graph()
    # lambda_causal=0 minimizes only atomicity; a large lambda_causal also penalizes
    # intra-supernode edge mass, so it never prefers fewer clusters than lambda=0.
    best_lo, _ = find_best_k(
        prune_graph, k_min_override=2, k_max_override=3, max_sn=None, lambda_causal=0.0
    )
    best_hi, _ = find_best_k(
        prune_graph, k_min_override=2, k_max_override=3, max_sn=None, lambda_causal=100.0
    )
    assert best_hi >= best_lo


def test_find_best_k_for_clusterer_picks_argmin_L() -> None:
    prune_graph = _build_test_graph()
    middle_ids = ["1_0_0", "1_1_0", "2_0_0", "2_1_0"]

    def clusterer(k: int) -> list[list[str]]:
        # Deterministic: split middle ids into k contiguous chunks.
        if k <= 0:
            return []
        chunks: list[list[str]] = [[] for _ in range(k)]
        for i, nid in enumerate(middle_ids):
            chunks[i % k].append(nid)
        return [c for c in chunks if c]

    best_k, results = find_best_k_for_clusterer(
        prune_graph=prune_graph,
        clusterer=clusterer,
        k_min_override=2,
        k_max_override=3,
    )
    assert best_k in results
    for v in results.values():
        assert "L" in v
        assert "final_supernodes" in v
    assert float(results[best_k]["L"]) == min(float(v["L"]) for v in results.values())
