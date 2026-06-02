"""Tests for the π force-DAG operator (``get_adj``) behind SummaryGraph.adj_matrix."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
import torch

from summarization.summarize import (
    Node,
    SummaryGraph,
    Supernode,
    compute_sn_adj,
)


# --- Fixtures -----------------------------------------------------------------


def _feat_node(node_id: str, node_idx: int, layer: int, ctx_idx: int = 0) -> Node:
    return Node(
        node_id=node_id,
        node_idx=node_idx,
        feature=node_idx,
        layer=str(layer),
        ctx_idx=ctx_idx,
        feature_type="sae_feature",
    )


def _emb_node(node_id: str, node_idx: int, ctx_idx: int = 0) -> Node:
    return Node(
        node_id=node_id,
        node_idx=node_idx,
        feature=node_idx,
        layer="E",
        ctx_idx=ctx_idx,
        feature_type="embedding",
    )


def _logit_node(node_id: str, node_idx: int, ctx_idx: int = 0) -> Node:
    return Node(
        node_id=node_id,
        node_idx=node_idx,
        feature=node_idx,
        layer="27",
        ctx_idx=ctx_idx,
        feature_type="logit",
        is_target_logit=True,
    )


def _sng_from_blocks(
    sns: list[Supernode], adj_block: np.ndarray
) -> SummaryGraph:
    """Build a SummaryGraph whose pre-π block-sum is exactly ``adj_block``.

    Each Supernode is given a single member with a unique ``node_idx``, so the
    block-aggregation collapses to the supernode-level matrix we pass in.
    """
    n = len(sns)
    assert adj_block.shape == (n, n)
    # Ensure each supernode has exactly one member with a unique node_idx.
    for k, sn in enumerate(sns):
        assert len(sn.features) == 1, "test fixture: one member per supernode"
        assert sn.features[0].node_idx == k, "test fixture: node_idx must match sn idx"
    pruned_adj = torch.zeros((n, n), dtype=torch.float32)
    for t in range(n):
        for s in range(n):
            pruned_adj[t, s] = float(adj_block[t, s])
    return SummaryGraph(supernodes=sns, pruned_adj=pruned_adj)


def _is_dag(sn_adj: np.ndarray) -> bool:
    g = nx.DiGraph()
    n = sn_adj.shape[0]
    g.add_nodes_from(range(n))
    for t in range(n):
        for s in range(n):
            if sn_adj[t, s] != 0.0:
                g.add_edge(s, t)
    return nx.is_directed_acyclic_graph(g)


# --- Stage A: antiparallel collapse ------------------------------------------


def test_pi_collapses_2cycle() -> None:
    a = Supernode("A", [_feat_node("a", 0, layer=1)], "features", 1, 1)
    b = Supernode("B", [_feat_node("b", 1, layer=2)], "features", 2, 2)
    # M[B,A]=3 (mass A→B, forward); M[A,B]=1 (mass B→A, back).
    block = np.array([[0.0, 1.0], [3.0, 0.0]])
    sng = _sng_from_blocks([a, b], block)

    assert sng.adj_matrix[1, 0] == pytest.approx(2.0)  # forward survives with net magnitude
    assert sng.adj_matrix[0, 1] == 0.0


def test_pi_collapse_preserves_dominant_sign() -> None:
    a = Supernode("A", [_feat_node("a", 0, layer=1)], "features", 1, 1)
    b = Supernode("B", [_feat_node("b", 1, layer=2)], "features", 2, 2)
    # Dominant direction is negative-signed.
    block = np.array([[0.0, 2.0], [-5.0, 0.0]])
    sng = _sng_from_blocks([a, b], block)

    assert sng.adj_matrix[1, 0] == pytest.approx(-3.0)  # sign(-5) * (5-2)
    assert sng.adj_matrix[0, 1] == 0.0


def test_pi_collapse_zeros_on_tie() -> None:
    a = Supernode("A", [_feat_node("a", 0, layer=1)], "features", 1, 1)
    b = Supernode("B", [_feat_node("b", 1, layer=2)], "features", 2, 2)
    block = np.array([[0.0, 4.0], [4.0, 0.0]])
    sng = _sng_from_blocks([a, b], block)

    assert sng.adj_matrix[0, 1] == 0.0
    assert sng.adj_matrix[1, 0] == 0.0


# --- Stage B: back-edge removal ----------------------------------------------


def test_pi_breaks_3cycle_at_same_depth() -> None:
    a = Supernode("A", [_feat_node("a", 0, layer=5)], "features", 5, 5)
    b = Supernode("B", [_feat_node("b", 1, layer=5)], "features", 5, 5)
    c = Supernode("C", [_feat_node("c", 2, layer=5)], "features", 5, 5)
    # 3-cycle: A→B, B→C, C→A. Same depth → tiebreak by index → ordering [A,B,C].
    block = np.zeros((3, 3))
    block[1, 0] = 1.0  # A→B (forward)
    block[2, 1] = 1.0  # B→C (forward)
    block[0, 2] = 1.0  # C→A (back-edge)
    sng = _sng_from_blocks([a, b, c], block)

    assert sng.adj_matrix[0, 2] == 0.0  # back-edge removed
    assert sng.adj_matrix[1, 0] == 1.0
    assert sng.adj_matrix[2, 1] == 1.0
    assert _is_dag(sng.adj_matrix)


def test_pi_idempotent_on_acyclic_input() -> None:
    a = Supernode("A", [_feat_node("a", 0, layer=1)], "features", 1, 1)
    b = Supernode("B", [_feat_node("b", 1, layer=2)], "features", 2, 2)
    c = Supernode("C", [_feat_node("c", 2, layer=3)], "features", 3, 3)
    block = np.zeros((3, 3))
    block[1, 0] = 1.5  # A→B
    block[2, 1] = 2.5  # B→C
    block[2, 0] = 0.5  # A→C (forward)
    sng = _sng_from_blocks([a, b, c], block)

    # Acyclic input → π is the identity: adj_matrix equals the plain block-sum.
    raw = compute_sn_adj([[0], [1], [2]], sng.pruned_adj)
    np.testing.assert_array_equal(sng.adj_matrix, raw)


# --- Emb / logit invariants --------------------------------------------------


def test_pi_emb_is_source_only() -> None:
    """Edges incoming to an embedding supernode get removed by Stage B."""
    emb = Supernode("EMB", [_emb_node("E_0_0", 0)], "emb", -1, -1)
    a = Supernode("A", [_feat_node("a", 1, layer=2)], "features", 2, 2)
    block = np.zeros((2, 2))
    block[0, 1] = 0.7  # A → EMB (removed: emb is forced source, depth=-inf)
    block[1, 0] = 1.0  # EMB → A (forward)
    sng = _sng_from_blocks([emb, a], block)

    assert sng.adj_matrix[0, 1] == 0.0  # incoming to emb removed
    # Stage A collapses the antiparallel {EMB, A} pair before Stage B, so the
    # surviving EMB→A edge carries net magnitude 1.0 - 0.7 = 0.3.
    assert sng.adj_matrix[1, 0] == pytest.approx(0.3)  # outgoing from emb survives


def test_pi_logit_is_sink_only() -> None:
    """Edges outgoing from a logit supernode get removed by Stage B."""
    a = Supernode("A", [_feat_node("a", 0, layer=5)], "features", 5, 5)
    log = Supernode("LOG", [_logit_node("27_0_0", 1)], "logit", 27, 27)
    block = np.zeros((2, 2))
    block[1, 0] = 0.9  # A → LOG (forward)
    block[0, 1] = 0.4  # LOG → A (removed: logit is forced sink, depth=+inf)
    sng = _sng_from_blocks([a, log], block)

    assert sng.adj_matrix[0, 1] == 0.0  # outgoing from logit removed
    # Antiparallel collapse runs first: surviving A→LOG carries 0.9 - 0.4 = 0.5.
    assert sng.adj_matrix[1, 0] == pytest.approx(0.5)


# --- Aggregate properties ----------------------------------------------------


def test_pi_is_dag_on_complex_input() -> None:
    """Composite test: antiparallel + 3-cycle + back-edge all in one fixture."""
    a = Supernode("A", [_feat_node("a", 0, layer=1)], "features", 1, 1)
    b = Supernode("B", [_feat_node("b", 1, layer=2)], "features", 2, 2)
    c = Supernode("C", [_feat_node("c", 2, layer=3)], "features", 3, 3)
    block = np.array(
        [
            [0.0, 2.0, 0.7],   # to A: from B (2), from C (0.7 — back-edge)
            [5.0, 0.0, 1.5],   # to B: from A (5), from C (1.5 — back-edge)
            [0.5, 1.0, 0.0],   # to C: from A (0.5), from B (1)
        ]
    )
    sng = _sng_from_blocks([a, b, c], block)
    assert _is_dag(sng.adj_matrix)


# --- compute_L_causal integration -------------------------------------------


def test_compute_L_causal_is_internal_mass_fraction() -> None:
    # Eq. Lcausal: fraction of pruned edge mass absorbed *inside* a supernode.
    from summarization.scoring import compute_L_causal

    a = _feat_node("a", 0, layer=1)
    b = _feat_node("b", 1, layer=2)
    c = _feat_node("c", 2, layer=2)
    ab = Supernode("AB", [a, b], "features", 1, 2)  # a, b merged: a->b edge is internal
    c_sn = Supernode("C", [c], "features", 2, 2)
    pruned_adj = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # to a
            [2.0, 0.0, 0.0],  # to b: from a (2.0) — internal to AB
            [1.0, 3.0, 0.0],  # to c: from a (1.0), from b (3.0) — external
        ],
        dtype=torch.float32,
    )
    sng = SummaryGraph(supernodes=[ab, c_sn], pruned_adj=pruned_adj)

    internal = 2.0           # only a->b lives inside a supernode
    total = 2.0 + 1.0 + 3.0  # all pruned edge mass
    assert compute_L_causal(sng) == pytest.approx(internal / total)


def test_compute_L_causal_zero_for_all_singletons() -> None:
    from summarization.scoring import compute_L_causal

    a = Supernode("A", [_feat_node("a", 0, layer=1)], "features", 1, 1)
    b = Supernode("B", [_feat_node("b", 1, layer=2)], "features", 2, 2)
    block = np.array([[0.0, 1.0], [3.0, 0.0]])  # no two nodes share a supernode
    sng = _sng_from_blocks([a, b], block)
    assert compute_L_causal(sng) == pytest.approx(0.0)


# --- Plain block-sum sanity --------------------------------------------------


def test_compute_sn_adj_is_plain_block_sum() -> None:
    """Confirm compute_sn_adj keeps both antiparallel directions (no tie-break)."""
    pruned_adj = torch.tensor(
        [[0.0, 3.0], [1.0, 0.0]], dtype=torch.float32
    )  # [tgt, src]
    block = compute_sn_adj([[0], [1]], pruned_adj)
    assert block[0, 1] == 3.0
    assert block[1, 0] == 1.0
    assert block[0, 0] == 0.0
    assert block[1, 1] == 0.0
