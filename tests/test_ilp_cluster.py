from __future__ import annotations

import pytest
import torch

from summarization.cluster import cluster
from summarization.ilp_cluster import cluster_graph_ilp
from summarization.prune import PruneGraph
from summarization.summarize import Supernode
from summarization.utils import _node_from_json_dict, layer_index_from_node, node_is_fixed


def _build_test_graph() -> PruneGraph:
    # 4 middle features at layers 1,1,2,2 + one embedding + one logit.
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
            [0.8, 0.0, 0.2, 0.0, 0.0, 0.0],
            [0.7, 0.1, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.9, 0.8, 0.0, 0.3, 0.0],
            [0.0, 0.8, 0.9, 0.2, 0.0, 0.0],
            [0.0, 0.2, 0.3, 0.8, 0.9, 0.0],
        ],
        dtype=torch.float32,
    )
    return PruneGraph(
        nodes=nodes,
        pruned_adj=pruned_adj,
        metadata={},
        node_influence=torch.tensor([0.0, 0.3, 0.35, 0.8, 0.85, 0.0], dtype=torch.float32),
        node_relevance=torch.tensor([0.0, 0.4, 0.45, 0.7, 0.75, 0.0], dtype=torch.float32),
        edge_influence=pruned_adj.clone() * 1.2,
        edge_relevance=pruned_adj.clone() * 0.8,
    )


def _cluster_set(clusters: list[list[str]]) -> frozenset[frozenset[str]]:
    return frozenset(frozenset(c) for c in clusters)


def _span(cluster_ids: list[str], nodes_by_id: dict) -> int:
    layers = [layer_index_from_node(nodes_by_id[nid]) for nid in cluster_ids]
    return max(layers) - min(layers)


@pytest.mark.parametrize("max_span", [0, 1, 4])
def test_ilp_respects_layer_span(max_span: int) -> None:
    pg = _build_test_graph()
    nodes_by_id = {n.node_id: n for n in pg.nodes}
    clusters = cluster_graph_ilp(pg, max_layer_span=max_span, gamma=0.1)
    for c in clusters:
        assert _span(c, nodes_by_id) <= max_span


def test_ilp_valid_partition() -> None:
    pg = _build_test_graph()
    clusters = cluster_graph_ilp(pg, max_layer_span=4)
    members = [nid for c in clusters for nid in c]
    assert set(members) == set(pg.node_ids)  # full cover
    assert len(members) == len(set(members))  # no overlap


def test_ilp_emb_logit_singletons() -> None:
    pg = _build_test_graph()
    clusters = cluster_graph_ilp(pg, max_layer_span=4)
    assert ["E_0_0"] in clusters
    assert ["27_0_0"] in clusters


def test_ilp_gamma_monotonic() -> None:
    # Larger opening cost => same-or-fewer middle supernodes.
    pg = _build_test_graph()
    nodes_by_id = {n.node_id: n for n in pg.nodes}

    def n_middle(clusters: list[list[str]]) -> int:
        return sum(1 for c in clusters if not node_is_fixed(nodes_by_id[c[0]]))

    few = n_middle(cluster_graph_ilp(pg, max_layer_span=4, gamma=10.0))
    many = n_middle(cluster_graph_ilp(pg, max_layer_span=4, gamma=0.001))
    assert few <= many
    assert few == 1  # one medoid can cover layers 1..2 under span=4
    assert many == 4  # tiny opening cost => every node is its own supernode


def test_ilp_deterministic() -> None:
    pg = _build_test_graph()
    a = cluster_graph_ilp(pg, max_layer_span=4, gamma=0.3)
    b = cluster_graph_ilp(pg, max_layer_span=4, gamma=0.3)
    assert _cluster_set(a) == _cluster_set(b)


def test_cluster_integration_ilp() -> None:
    pg = _build_test_graph()
    rows = cluster(pg, method="ilp", max_layer_span=4)
    assert all(isinstance(r, Supernode) for r in rows)
    members = [nid for r in rows for nid in r.member_node_ids()]
    assert set(members) == set(pg.node_ids)
    assert len(members) == len(set(members))
    assert sum(1 for r in rows if r.type == "features") >= 1


def test_ilp_infeasible_max_sn_raises() -> None:
    # span=0 forces same-layer clusters: layers {1, 2} need >= 2 medoids, but max_sn=1.
    pg = _build_test_graph()
    with pytest.raises(ValueError):
        cluster_graph_ilp(pg, max_layer_span=0, max_sn=1)


def test_ilp_too_large_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # Fast-fail (not a 30s timeout) when the MILP would exceed the variable cap.
    import summarization.ilp_cluster as ilp

    monkeypatch.setattr(ilp, "MAX_ILP_VARS", 1)
    pg = _build_test_graph()
    with pytest.raises(ValueError, match="too large"):
        cluster_graph_ilp(pg, max_layer_span=4)
