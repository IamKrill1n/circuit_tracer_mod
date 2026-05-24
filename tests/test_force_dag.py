"""Tests for the π force-DAG operator built into SummarizationGraph."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
import torch

from summarization.supernode_graph import (
    Node,
    SummarizationGraph,
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
) -> SummarizationGraph:
    """Build a SummarizationGraph whose ``sn_adj_raw`` is exactly ``adj_block``.

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
    return SummarizationGraph(supernodes=sns, pruned_adj=pruned_adj)


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

    assert sng.sn_adj[1, 0] == pytest.approx(2.0)  # forward survives with net magnitude
    assert sng.sn_adj[0, 1] == 0.0
    assert sng.l_collapse == pytest.approx(2.0)    # 2 * min(1, 3)
    assert sng.l_back == 0.0


def test_pi_collapse_preserves_dominant_sign() -> None:
    a = Supernode("A", [_feat_node("a", 0, layer=1)], "features", 1, 1)
    b = Supernode("B", [_feat_node("b", 1, layer=2)], "features", 2, 2)
    # Dominant direction is negative-signed.
    block = np.array([[0.0, 2.0], [-5.0, 0.0]])
    sng = _sng_from_blocks([a, b], block)

    assert sng.sn_adj[1, 0] == pytest.approx(-3.0)  # sign(-5) * (5-2)
    assert sng.sn_adj[0, 1] == 0.0
    assert sng.l_collapse == pytest.approx(4.0)


def test_pi_collapse_zeros_on_tie() -> None:
    a = Supernode("A", [_feat_node("a", 0, layer=1)], "features", 1, 1)
    b = Supernode("B", [_feat_node("b", 1, layer=2)], "features", 2, 2)
    block = np.array([[0.0, 4.0], [4.0, 0.0]])
    sng = _sng_from_blocks([a, b], block)

    assert sng.sn_adj[0, 1] == 0.0
    assert sng.sn_adj[1, 0] == 0.0
    assert sng.l_collapse == pytest.approx(8.0)


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

    assert sng.l_collapse == 0.0
    assert sng.l_back == pytest.approx(1.0)
    assert sng.sn_adj[0, 2] == 0.0  # back-edge removed
    assert sng.sn_adj[1, 0] == 1.0
    assert sng.sn_adj[2, 1] == 1.0
    assert _is_dag(sng.sn_adj)


def test_pi_idempotent_on_acyclic_input() -> None:
    a = Supernode("A", [_feat_node("a", 0, layer=1)], "features", 1, 1)
    b = Supernode("B", [_feat_node("b", 1, layer=2)], "features", 2, 2)
    c = Supernode("C", [_feat_node("c", 2, layer=3)], "features", 3, 3)
    block = np.zeros((3, 3))
    block[1, 0] = 1.5  # A→B
    block[2, 1] = 2.5  # B→C
    block[2, 0] = 0.5  # A→C (forward)
    sng = _sng_from_blocks([a, b, c], block)

    assert sng.l_collapse == 0.0
    assert sng.l_back == 0.0
    np.testing.assert_array_equal(sng.sn_adj, sng.sn_adj_raw)


def test_pi_cache_returns_stable_values() -> None:
    a = Supernode("A", [_feat_node("a", 0, layer=1)], "features", 1, 1)
    b = Supernode("B", [_feat_node("b", 1, layer=2)], "features", 2, 2)
    block = np.array([[0.0, 1.0], [3.0, 0.0]])
    sng = _sng_from_blocks([a, b], block)

    first_adj = sng.sn_adj.copy()
    first_collapse = sng.l_collapse
    first_back = sng.l_back
    # Multiple accesses must return identical cached values.
    np.testing.assert_array_equal(sng.sn_adj, first_adj)
    assert sng.l_collapse == first_collapse
    assert sng.l_back == first_back


# --- Emb / logit invariants --------------------------------------------------


def test_pi_emb_is_source_only() -> None:
    """Edges incoming to an embedding supernode get removed by Stage B."""
    emb = Supernode("EMB", [_emb_node("E_0_0", 0)], "emb", -1, -1)
    a = Supernode("A", [_feat_node("a", 1, layer=2)], "features", 2, 2)
    block = np.zeros((2, 2))
    block[0, 1] = 0.7  # A → EMB (should be removed: emb is forced source, depth=-inf)
    block[1, 0] = 1.0  # EMB → A (forward)
    sng = _sng_from_blocks([emb, a], block)

    assert sng.sn_adj[0, 1] == 0.0  # incoming to emb removed
    assert sng.sn_adj[1, 0] == 1.0  # outgoing from emb preserved
    assert sng.l_back == pytest.approx(0.7)
    assert sng.depths[0] == -np.inf


def test_pi_logit_is_sink_only() -> None:
    """Edges outgoing from a logit supernode get removed by Stage B."""
    a = Supernode("A", [_feat_node("a", 0, layer=5)], "features", 5, 5)
    log = Supernode("LOG", [_logit_node("27_0_0", 1)], "logit", 27, 27)
    block = np.zeros((2, 2))
    block[1, 0] = 0.9  # A → LOG (forward)
    block[0, 1] = 0.4  # LOG → A (should be removed: logit is forced sink, depth=+inf)
    sng = _sng_from_blocks([a, log], block)

    assert sng.sn_adj[1, 0] == 0.9
    assert sng.sn_adj[0, 1] == 0.0
    assert sng.l_back == pytest.approx(0.4)
    assert sng.depths[1] == np.inf


# --- Aggregate properties ----------------------------------------------------


def test_pi_mass_conservation() -> None:
    """l_collapse + l_back + |sn_adj| equals |sn_adj_raw| in the unsigned case."""
    a = Supernode("A", [_feat_node("a", 0, layer=1)], "features", 1, 1)
    b = Supernode("B", [_feat_node("b", 1, layer=2)], "features", 2, 2)
    c = Supernode("C", [_feat_node("c", 2, layer=3)], "features", 3, 3)
    block = np.zeros((3, 3))
    # Forward edges and one antiparallel pair (A↔B).
    block[1, 0] = 5.0  # A→B
    block[0, 1] = 2.0  # B→A (antiparallel: cancels 4 mass, leaves 3 on A→B)
    block[2, 1] = 1.0  # B→C
    block[1, 2] = 0.0  # (no antiparallel here)
    block[0, 2] = 0.5  # C→A (back-edge under ordering A<B<C: removed)
    sng = _sng_from_blocks([a, b, c], block)

    raw_mass = float(np.abs(sng.sn_adj_raw).sum())
    surviving = float(np.abs(sng.sn_adj).sum())
    assert raw_mass == pytest.approx(surviving + sng.l_collapse + sng.l_back)


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
    assert _is_dag(sng.sn_adj)


# --- compute_D_agg integration ----------------------------------------------


def test_compute_D_agg_uses_post_pi_mass() -> None:
    from summarization.objective import compute_D_agg

    a = Supernode("A", [_feat_node("a", 0, layer=1)], "features", 1, 1)
    b = Supernode("B", [_feat_node("b", 1, layer=2)], "features", 2, 2)
    block = np.array([[0.0, 1.0], [3.0, 0.0]])  # antiparallel
    sng = _sng_from_blocks([a, b], block)

    total_pruned_mass = float(np.abs(sng.pruned_adj.numpy()).sum())  # 4.0
    surviving_sn_mass = float(np.abs(sng.sn_adj).sum())              # 2.0 (post-π)
    expected = 1.0 - surviving_sn_mass / total_pruned_mass

    assert compute_D_agg(sng) == pytest.approx(expected)


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
