"""Closed-form objective terms from paper/reformulation.tex.

Each term is in [0, 1] so L_total = L_coh + D_agg + L_cplx is in [0, 3]. Lower is better.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from summarization.prune import PruneGraph
from summarization.supernode_graph import Supernode
from summarization.utils import layer_index_from_node, node_is_embedding


def _block_sum_matrix(
    supernodes: list[Supernode], prune_graph: PruneGraph
) -> np.ndarray:
    """Signed block sum W^SN per paper Eq. 7 — no dominant-direction tie-breaker.

    Returns block[t, s] = sum over u in S_s, v in S_t of pruned_adj[v, u].
    """
    adj = prune_graph.pruned_adj.detach().cpu().numpy().astype(np.float64)  # [tgt, src]
    n_total = adj.shape[0]
    n_sn = len(supernodes)
    indicator = np.zeros((n_sn, n_total), dtype=np.float64)
    for sn_idx, sn in enumerate(supernodes):
        for node in sn.features:
            if 0 <= node.node_idx < n_total:
                indicator[sn_idx, node.node_idx] = 1.0
    return indicator @ adj @ indicator.T


def compute_L_coh(
    role_vectors_middle: np.ndarray, labels: np.ndarray
) -> float:
    """Mean (1 - cos(r(u), centroid_of_cluster(u))) over middle features.

    role_vectors_middle: [n_middle, d] feature role vectors.
    labels: [n_middle] integer cluster assignment per middle feature. Label -1 marks
    features with no assigned cluster; they are dropped from the mean.
    """
    if role_vectors_middle.size == 0:
        return 0.0
    valid_mask = labels >= 0
    if not np.any(valid_mask):
        return 0.0
    vecs = role_vectors_middle[valid_mask]
    lbls = labels[valid_mask]
    norms = np.linalg.norm(vecs, axis=1)

    distances: list[float] = []
    for cluster in np.unique(lbls):
        members = vecs[lbls == cluster]
        member_norms = norms[lbls == cluster]
        centroid = members.mean(axis=0)
        c_norm = float(np.linalg.norm(centroid))
        if c_norm < 1e-12:
            # Degenerate centroid (e.g., features cancel). Treat as fully unaligned.
            distances.extend([1.0] * len(members))
            continue
        safe_member = np.where(member_norms > 1e-12, member_norms, 1.0)
        cosines = (members @ centroid) / (safe_member * c_norm)
        cosines = np.where(member_norms > 1e-12, cosines, 0.0)
        distances.extend((1.0 - cosines).tolist())
    return float(np.mean(distances)) if distances else 0.0


def compute_D_agg(
    supernodes: list[Supernode], prune_graph: PruneGraph
) -> float:
    """Aggregation loss per paper Eq. 12.

    D_agg = 1 - (sum over S != T of |W^SN_ST|) / (sum over (u,v) in E' of |W_uv|).
    """
    adj = prune_graph.pruned_adj.detach().cpu().numpy().astype(np.float64)
    total_mag = float(np.abs(adj).sum())
    if total_mag <= 0.0:
        return 0.0
    block = _block_sum_matrix(supernodes, prune_graph)
    np.fill_diagonal(block, 0.0)
    retained_mag = float(np.abs(block).sum())
    return 1.0 - retained_mag / total_mag


def _num_model_layers(prune_graph: PruneGraph) -> int:
    """Largest finite non-embedding layer index, + 1. Used as the L_cplx denominator base."""
    finite = []
    for node in prune_graph.nodes:
        if node_is_embedding(node):
            continue
        layer = layer_index_from_node(node)
        if layer < 10_000:  # filter the fallback for unparseable ids
            finite.append(layer)
    return (max(finite) + 1) if finite else 1


def compute_L_cplx(
    supernodes: list[Supernode], prune_graph: PruneGraph
) -> float:
    """Average shortest path V_emb -> V_logit in G_SN (supernode hops), normalized.

    Denominator is (num_model_layers + 1) so dense end-to-end supernode chains land
    near 1.0 and tight summaries land near 2 / denom. Returns 0 if no reachable
    embedding-logit pair (degenerate disconnected summary).
    """
    block = _block_sum_matrix(supernodes, prune_graph)
    n_sn = block.shape[0]
    if n_sn == 0:
        return 0.0

    graph = nx.DiGraph()
    graph.add_nodes_from(range(n_sn))
    for t in range(n_sn):
        for s in range(n_sn):
            if t == s:
                continue
            if block[t, s] != 0.0:
                graph.add_edge(s, t)  # block[tgt, src] != 0 => edge src -> tgt

    emb_idx = [i for i, sn in enumerate(supernodes) if sn.type == "emb"]
    logit_idx = [i for i, sn in enumerate(supernodes) if sn.type == "logit"]
    if not emb_idx or not logit_idx:
        return 0.0

    lengths: list[int] = []
    for e in emb_idx:
        path_lengths = nx.single_source_shortest_path_length(graph, e)
        for ell in logit_idx:
            if ell in path_lengths:
                lengths.append(path_lengths[ell])
    if not lengths:
        return 0.0

    denom = _num_model_layers(prune_graph) + 1
    return float(np.mean(lengths) / denom)


def _supernode_labels_for_middle(
    supernodes: list[Supernode],
    middle_node_id_to_local: dict[str, int],
    n_middle: int,
) -> np.ndarray:
    """Per-middle-feature cluster label. -1 for middle features not in any feature-type supernode."""
    labels = np.full(n_middle, -1, dtype=np.int64)
    next_label = 0
    for sn in supernodes:
        if sn.type != "features" and sn.type != "feature":
            continue
        for node in sn.features:
            local = middle_node_id_to_local.get(node.node_id)
            if local is not None:
                labels[local] = next_label
        next_label += 1
    return labels


def compute_L(
    supernodes: list[Supernode],
    role_vectors_middle: np.ndarray,
    middle_node_id_to_local: dict[str, int],
    prune_graph: PruneGraph,
) -> dict[str, float | int]:
    """Bundle the three losses + L_total + simple counts for one partition."""
    n_middle = role_vectors_middle.shape[0]
    labels = _supernode_labels_for_middle(supernodes, middle_node_id_to_local, n_middle)
    L_coh = compute_L_coh(role_vectors_middle, labels)
    D_agg = compute_D_agg(supernodes, prune_graph)
    L_cplx = compute_L_cplx(supernodes, prune_graph)
    L_total = L_coh + D_agg + L_cplx
    n_middle_supernodes = sum(1 for sn in supernodes if sn.type in ("features", "feature"))
    return {
        "L_coh": float(L_coh),
        "D_agg": float(D_agg),
        "L_cplx": float(L_cplx),
        "L_total": float(L_total),
        "n_supernodes": int(len(supernodes)),
        "n_middle_supernodes": int(n_middle_supernodes),
    }
