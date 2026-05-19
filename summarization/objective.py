"""Closed-form objective terms from paper/reformulation.tex.

Each term is in [0, 1]. The scalar L = lambda_coh L_coh + lambda_cons L_cons +
lambda_cplx L_cplx is a simplex-weighted sum (default 1/3 each) in [0, 1], with
L_cons = 0.5 (prune_loss + D_agg). Lower is better.
"""

from __future__ import annotations

import numpy as np

from summarization.prune import PruneGraph
from summarization.supernode_graph import Supernode, compute_sn_adj
from summarization.utils import node_is_fixed


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
    index_lists = [[n.node_idx for n in sn.features] for sn in supernodes]
    block = compute_sn_adj(index_lists, prune_graph.pruned_adj)
    retained_mag = float(np.abs(block).sum())
    return 1.0 - retained_mag / total_mag


def compute_L_cplx(
    supernodes: list[Supernode], prune_graph: PruneGraph
) -> float:
    """Number of feature supernodes / |V'_mid|, per paper Eq. (Lcplx).

    Embedding and logit supernodes are forced singletons by (F2) and excluded
    from both numerator and denominator. Returns 0 on a degenerate pruned
    graph with no mid-graph features.
    """
    n_feature_supernodes = sum(
        1 for sn in supernodes if sn.type in ("features", "feature")
    )
    n_middle = sum(1 for node in prune_graph.nodes if not node_is_fixed(node))
    if n_middle == 0:
        return 0.0
    return float(n_feature_supernodes / n_middle)


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
    *,
    prune_loss: float = 0.0,
    lambdas: tuple[float, float, float] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
) -> dict[str, float | int]:
    """Bundle the per-axis losses + simplex-weighted scalar L for one partition.

    prune_loss is partition-invariant within a pruned graph; pass it in once per
    PruneGraph. lambdas = (lambda_coh, lambda_cons, lambda_cplx) must sit on the
    probability simplex (sum to 1, each in [0, 1]).
    """
    n_middle = role_vectors_middle.shape[0]
    labels = _supernode_labels_for_middle(supernodes, middle_node_id_to_local, n_middle)
    L_coh = compute_L_coh(role_vectors_middle, labels)
    D_agg = compute_D_agg(supernodes, prune_graph)
    L_cplx = compute_L_cplx(supernodes, prune_graph)
    L_cons = 0.5 * (float(prune_loss) + D_agg)
    lam_coh, lam_cons, lam_cplx = lambdas
    L = lam_coh * L_coh + lam_cons * L_cons + lam_cplx * L_cplx
    n_middle_supernodes = sum(1 for sn in supernodes if sn.type in ("features", "feature"))
    return {
        "L_coh": float(L_coh),
        "D_agg": float(D_agg),
        "prune_loss": float(prune_loss),
        "L_cons": float(L_cons),
        "L_cplx": float(L_cplx),
        "L": float(L),
        "L_total": float(L_coh + D_agg + L_cplx),  # back-compat
        "n_supernodes": int(len(supernodes)),
        "n_middle_supernodes": int(n_middle_supernodes),
    }
