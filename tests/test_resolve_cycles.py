from __future__ import annotations

import networkx as nx
import torch

from summarization.cluster import _supernode_edges, resolve_cluster_cycles
from summarization.prune import PruneGraph
from summarization.utils import _node_from_json_dict


def _build_cycle_fixture() -> PruneGraph:
    """6 nodes: emb (0), m1@L1 (1), m2@L1 (2), m3@L2 (3), m4@L2 (4), logit (5).

    Adjacency is hand-crafted so the supernode graph over clusters
    A={m1,m4}, B={m2}, C={m3} forms the 3-cycle A->B->C->A under the
    dominant-direction tie-break in compute_sn_adj.
    """
    node_specs: list[tuple[str, dict]] = [
        ("E_0_0", {"feature_type": "embedding", "is_target_logit": False, "layer": "E"}),
        ("1_0_0", {"feature_type": "sae_feature", "is_target_logit": False, "layer": "1"}),
        ("1_1_0", {"feature_type": "sae_feature", "is_target_logit": False, "layer": "1"}),
        ("2_0_0", {"feature_type": "sae_feature", "is_target_logit": False, "layer": "2"}),
        ("2_1_0", {"feature_type": "sae_feature", "is_target_logit": False, "layer": "2"}),
        ("27_0_0", {"feature_type": "logit", "is_target_logit": True, "layer": "27"}),
    ]
    nodes = [_node_from_json_dict({"node_id": nid, **spec}) for nid, spec in node_specs]

    # adj[t, s] convention. Only three edges, each picking the desired direction
    # at the cluster level (the other direction is zero, so dominance trivially picks it).
    pruned_adj = torch.zeros(6, 6, dtype=torch.float32)
    pruned_adj[2, 1] = 1.0  # m1 -> m2  drives cluster A -> B
    pruned_adj[3, 2] = 1.0  # m2 -> m3  drives cluster B -> C
    pruned_adj[4, 3] = 1.0  # m3 -> m4  drives cluster C -> A

    zero = torch.zeros(6, dtype=torch.float32)
    return PruneGraph(
        nodes=nodes,
        pruned_adj=pruned_adj,
        metadata={},
        node_influence=zero,
        node_relevance=zero,
        edge_influence=pruned_adj.clone(),
        edge_relevance=pruned_adj.clone(),
    )


def _cluster_graph_is_dag(clusters: list[list[str]], prune_graph: PruneGraph) -> bool:
    id_to_idx = {nid: i for i, nid in enumerate(prune_graph.node_ids)}
    edges = _supernode_edges(clusters, prune_graph.pruned_adj, id_to_idx)
    g = nx.DiGraph()
    g.add_nodes_from(range(len(clusters)))
    g.add_edges_from(edges)
    return nx.is_directed_acyclic_graph(g)


def test_resolve_cluster_cycles_breaks_known_cycle() -> None:
    pg = _build_cycle_fixture()
    clusters = [["1_0_0", "2_1_0"], ["1_1_0"], ["2_0_0"]]  # A, B, C — A->B->C->A
    assert not _cluster_graph_is_dag(clusters, pg), "Fixture must start cyclic"

    resolved = resolve_cluster_cycles(clusters, pg)

    assert _cluster_graph_is_dag(resolved, pg)
    # Same node set, no losses or duplicates.
    flat_in = sorted(n for c in clusters for n in c)
    flat_out = sorted(n for c in resolved for n in c)
    assert flat_in == flat_out


def test_resolve_cluster_cycles_idempotent() -> None:
    pg = _build_cycle_fixture()
    clusters = [["1_0_0", "2_1_0"], ["1_1_0"], ["2_0_0"]]
    once = resolve_cluster_cycles(clusters, pg)
    twice = resolve_cluster_cycles(once, pg)
    assert once == twice


def test_resolve_cluster_cycles_dag_input_passthrough() -> None:
    pg = _build_cycle_fixture()
    # Singletons across the same fixture — graph is already acyclic.
    clusters = [["1_0_0"], ["1_1_0"], ["2_0_0"], ["2_1_0"]]
    assert _cluster_graph_is_dag(clusters, pg)
    assert resolve_cluster_cycles(clusters, pg) == clusters
