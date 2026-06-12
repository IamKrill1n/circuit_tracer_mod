
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import silhouette_score

from summarization.prune import PruneGraph
from summarization.summarize import Supernode, SummaryGraph, compute_sn_adj
from summarization.utils import node_is_fixed, node_is_logit

def compute_L_atom(
    role_vectors_middle: np.ndarray,
    labels: np.ndarray,
    theta: float = 0.0,
) -> dict[str, float]:
    """Atomicity loss (paper Eq. Latom): signed-cosine correlation-clustering loss
    over the free feature nodes.

    With signed cosine ``s_ij`` of the role vectors and resolution ``theta``, the
    raw objective the Stage-2 ILP minimises is

        L_atom = sum_{i<j, same supernode} (theta - s_ij),

    lower (more negative) is better: a similar pair (``s > theta``) rewards merging,
    a dissimilar pair penalises it. ``L_atom_norm`` is the normalised *disagreement
    fraction* in ``[0, 1]`` — the weight on the "wrong" side (dissimilar-but-merged +
    similar-but-split) over the total weight ``sum |s_ij - theta|`` — so 0 is a
    perfect partition and 1 the worst.
    """
    n = role_vectors_middle.shape[0] if role_vectors_middle.size else 0
    if n < 2 or labels.size == 0:
        return {"L_atom": 0.0, "L_atom_norm": 0.0}

    norms = np.linalg.norm(role_vectors_middle, axis=1, keepdims=True)
    safe_norms = np.where(norms > 1e-12, norms, 1.0)
    unit = role_vectors_middle / safe_norms
    cos = np.clip(unit @ unit.T, -1.0, 1.0)

    iu, ju = np.triu_indices(n, k=1)
    w = cos[iu, ju] - theta  # signed merge weight per pair
    same = (labels[iu] == labels[ju]) & (labels[iu] >= 0)

    raw = float((-w[same]).sum())  # = sum_{same} (theta - cos)
    total = float(np.abs(w).sum())
    if total <= 1e-12:
        return {"L_atom": raw, "L_atom_norm": 0.0}
    # Disagreement: dissimilar pairs merged (w<0 & same) + similar pairs split (w>0 & ~same).
    disagreement = float(np.abs(w[(w < 0) & same]).sum() + np.abs(w[(w > 0) & ~same]).sum())
    return {"L_atom": raw, "L_atom_norm": disagreement / total}

def compute_L_causal(sng: SummaryGraph) -> float:
    """Causal-preservation loss (paper Eq. Lcausal): the fraction of pruned edge
    mass absorbed *inside* a supernode.

        L_causal = sum_{f(u)=f(v)} |W_vu| / sum_{(u,v) in E'} |W_vu|  in [0, 1].

    Edges between two members of the same supernode are collapsed into the node by
    Stage 3, so this is the causal information the summary necessarily loses. Only
    the intra-supernode (diagonal-block) mass counts — the dominant-direction and
    sign-cancellation losses of the π operator are *not* part of this loss.
    """
    adj = sng.pruned_adj.detach().cpu().numpy().astype(np.float64)  # [tgt, src]
    total_mag = float(np.abs(adj).sum())
    if total_mag <= 0.0:
        return 0.0
    internal_mag = 0.0
    for sn in sng.supernodes:
        idxs = [n.node_idx for n in sn.features if n.node_idx >= 0]  # rows/cols in pruned_adj
        if len(idxs) < 2:
            continue
        block = adj[np.ix_(idxs, idxs)]  # both directions among members; diagonal is 0 (DAG)
        internal_mag += float(np.abs(block).sum())
    return internal_mag / total_mag


def compute_edge_mass_metrics(sng: SummaryGraph) -> dict[str, float | int]:
    """Mass accounting for summary-graph construction.

    ``compute_sn_adj`` gives the raw external block-sum after clustering but before
    π force-DAG construction. ``sng.adj_matrix`` is the final DAG adjacency after
    antiparallel collapse and back-edge removal.
    """
    total_fine_edge_mass = float(np.abs(sng.pruned_adj.detach().cpu().numpy()).sum())
    index_lists = [[n.node_idx for n in sn.features if n.node_idx >= 0] for sn in sng.supernodes]
    raw_adj = compute_sn_adj(index_lists, sng.pruned_adj)
    raw_superedge_mass = float(np.abs(raw_adj).sum())
    final_superedge_mass = float(np.abs(sng.adj_matrix).sum())
    dag_removed_mass = max(raw_superedge_mass - final_superedge_mass, 0.0)
    dag_removed_mass_fraction = (
        dag_removed_mass / raw_superedge_mass if raw_superedge_mass > 1e-12 else 0.0
    )
    final_retained_mass_fraction = (
        final_superedge_mass / total_fine_edge_mass if total_fine_edge_mass > 1e-12 else 0.0
    )
    return {
        "total_fine_edge_mass": total_fine_edge_mass,
        "raw_superedge_mass": raw_superedge_mass,
        "final_superedge_mass": final_superedge_mass,
        "dag_removed_mass": dag_removed_mass,
        "dag_removed_mass_fraction": dag_removed_mass_fraction,
        "final_retained_mass_fraction": final_retained_mass_fraction,
        "n_superedges": int(np.count_nonzero(np.abs(sng.adj_matrix) > 0.0)),
    }

def _supernode_labels_for_middle(
    sng: SummaryGraph,
    middle_node_id_to_local: dict[str, int],
    n_middle: int,
) -> np.ndarray:
    """Per-middle-feature cluster label. -1 for middle features not in any feature-type supernode."""
    labels = np.full(n_middle, -1, dtype=np.int64)
    next_label = 0
    for sn in sng.supernodes:
        if sn.type != "features" and sn.type != "feature":
            continue
        for node in sn.features:
            local = middle_node_id_to_local.get(node.node_id)
            if local is not None:
                labels[local] = next_label
        next_label += 1
    return labels


def compute_L(
    sng: SummaryGraph,
    role_vectors_middle: np.ndarray,
    middle_node_id_to_local: dict[str, int],
    *,
    prune_loss: float = 0.0,
    lambda_causal: float = 1.0,
) -> dict[str, float | int]:
    """Stage-2 objective bundle for one partition (paper Eq. Lstage2).

    Reports the two partition-dependent losses and their trade-off scalar

        L = (L_atom_norm + lambda_causal * L_causal) / (1 + lambda_causal)  in [0, 1],

    a positive rescaling of the paper objective ``L_atom + lambda_causal * L_causal``
    (same argmin for fixed ``lambda_causal``) that uses the normalised atomicity term
    so L stays comparable across graphs. The raw ``L_atom`` (Eq. Latom) is reported
    too. ``prune_loss`` (Stage 1, D1) is reported alongside but not folded into L.
    """
    n_middle = role_vectors_middle.shape[0]
    labels = _supernode_labels_for_middle(sng, middle_node_id_to_local, n_middle)
    atom = compute_L_atom(role_vectors_middle, labels)
    causal = compute_L_causal(sng)
    edge_mass = compute_edge_mass_metrics(sng)
    n_feature_sn = sum(1 for sn in sng.supernodes if sn.type in ("features", "feature"))
    L = (atom["L_atom_norm"] + lambda_causal * causal) / (1.0 + lambda_causal)
    return {
        "L": float(L),
        "L_atom": float(atom["L_atom"]),
        "L_atom_norm": float(atom["L_atom_norm"]),
        "L_causal": float(causal),
        "internalized_mass_fraction": float(causal),
        **edge_mass,
        "prune_loss": float(prune_loss),
        "K": int(n_feature_sn),
        "n_supernodes": int(len(sng.supernodes)),
    }


# ---------------------------------------------------------------------------
# Cluster-quality metrics
# ---------------------------------------------------------------------------


def _middle_indices(prune_graph: PruneGraph) -> list[int]:
    return [i for i, n in enumerate(prune_graph.nodes) if not node_is_fixed(n)]


def _cosine_similarity(features: np.ndarray, nonnegative: bool = False) -> np.ndarray:
    """Pairwise cosine similarity of row-features. If nonnegative, map [-1, 1] -> [0, 1]."""
    safe = np.asarray(features, dtype=np.float64)
    if safe.size == 0:
        return np.zeros((0, 0), dtype=np.float64)
    norms = np.linalg.norm(safe, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    normalized = safe / norms
    similarity = np.nan_to_num(normalized @ normalized.T, nan=0.0, posinf=0.0, neginf=0.0)
    if nonnegative:
        similarity = np.clip((similarity + 1.0) / 2.0, 0.0, 1.0)
    return similarity


def _size_entropy_norm(rows: list[Supernode]) -> float:
    """H(size_dist) / log K over feature-type supernodes; 1 = uniform sizes, 0 = degenerate."""
    sizes = np.array(
        [len(r.member_node_ids()) for r in rows if r.type == "features"],
        dtype=np.float64,
    )
    k = len(sizes)
    if k < 2 or sizes.sum() == 0:
        return 1.0
    p = sizes / sizes.sum()
    h = -np.sum(p * np.log(p + 1e-12))
    return float(h / np.log(k))


def compute_ppr_vectors(
    prune_graph: PruneGraph,
    alpha: float = 0.15,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """Personalized PageRank vectors per node on the symmetrized |pruned_adj| graph.

    Row i is the PPR distribution of a random walker seeded at node i. Symmetrizing
    makes PPR meaningful for nodes at any layer (otherwise late-layer features have
    no outgoing mass).
    """
    adj = prune_graph.pruned_adj.detach().cpu().numpy().astype(np.float64)
    w = np.abs(adj)
    w = (w + w.T) / 2.0
    row_sums = w.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 1e-12, row_sums, 1.0)
    p_trans = w / row_sums  # row-stochastic transition matrix (isolated rows stay self-loop)
    isolated = (w.sum(axis=1) <= 1e-12)
    if isolated.any():
        # isolated nodes: keep walker at the seed by self-transition
        p_trans[isolated] = 0.0
        p_trans[isolated, np.where(isolated)[0]] = 1.0

    n = p_trans.shape[0]
    teleport = alpha * np.eye(n, dtype=np.float64)
    r = np.eye(n, dtype=np.float64)  # row i starts as one-hot at i
    for _ in range(max_iter):
        r_new = teleport + (1.0 - alpha) * (r @ p_trans)
        if np.max(np.abs(r_new - r)) < tol:
            r = r_new
            break
        r = r_new
    return r


def silhouette_from_features(
    features: np.ndarray,
    prune_graph: PruneGraph,
    rows: list[Supernode],
) -> tuple[float, float]:
    """Cosine-similarity silhouette over middle nodes for a fixed feature matrix."""
    sim = _cosine_similarity(features, nonnegative=True)
    return _silhouette_over_middle(sim, prune_graph, rows)


def _silhouette_over_middle(
    similarity: np.ndarray,
    prune_graph: PruneGraph,
    rows: list[Supernode],
) -> tuple[float, float]:
    """
    Mean silhouette score over middle nodes, plus its [0, 1]-normalized form.

    Returns (silhouette_raw, silhouette_norm) where silhouette_norm = (sil + 1) / 2.
    Returns (0.0, 0.5) when silhouette is undefined (single cluster, all singletons,
    or no middle nodes assigned).
    """
    ids = prune_graph.node_ids
    id_to_idx = {nid: i for i, nid in enumerate(ids)}

    nid_to_label: dict[str, int] = {}
    label_idx = 0
    for row in rows:
        if row.type != "features":
            continue
        assigned = False
        for nid in row.member_node_ids():
            if nid in id_to_idx:
                nid_to_label[nid] = label_idx
                assigned = True
        if assigned:
            label_idx += 1

    if not nid_to_label:
        return 0.0, 0.5

    node_indices = [id_to_idx[nid] for nid in nid_to_label]
    labels_arr = np.fromiter(
        (nid_to_label[ids[i]] for i in node_indices),
        dtype=np.int64,
        count=len(node_indices),
    )
    n_distinct = int(len(set(labels_arr.tolist())))
    if n_distinct < 2 or n_distinct >= len(labels_arr):
        return 0.0, 0.5

    s_block = similarity[np.ix_(node_indices, node_indices)]
    s_block = (s_block + s_block.T) / 2.0
    s_block = np.clip(s_block, 0.0, 1.0)
    distance = 1.0 - s_block
    np.fill_diagonal(distance, 0.0)
    sil = float(silhouette_score(distance, labels_arr, metric="precomputed"))
    return sil, float((sil + 1.0) / 2.0)


def _internal_independence_score(
    supernodes: list[Supernode],
    prune_adj: torch.Tensor,
    id_to_idx: dict[str, int],
) -> float:
    """
    Internal independence score
    S_ind = 1/|V| * sum(u in V) (Total External Edge Weight attached to u / Total Edge Weight attached to u)
    """
    n_nodes = prune_adj.shape[0]
    if n_nodes == 0:
        return 0.0

    adj_abs = prune_adj.abs()

    # Total edge weight attached to each node (incoming + outgoing)
    total_weights = adj_abs.sum(dim=0) + adj_abs.sum(dim=1)

    # Build mapping from node index to its supernode index
    node_to_sn: dict[int, int] = {}
    for sn_idx, sn in enumerate(supernodes):
        for nid in sn.member_node_ids():
            idx = id_to_idx.get(nid)
            if idx is not None:
                node_to_sn[idx] = sn_idx

    total_score = 0.0
    for u in range(n_nodes):
        if total_weights[u] == 0:
            continue

        u_sn = node_to_sn.get(u)

        if u_sn is not None:
            # sum of edge weights within the same supernode
            sn_nodes = [id_to_idx[nid] for nid in supernodes[u_sn].member_node_ids() if nid in id_to_idx]
            internal_weight = adj_abs[u, sn_nodes].sum() + adj_abs[sn_nodes, u].sum()
        else:
            internal_weight = 0.0

        external_weight = total_weights[u] - internal_weight
        total_score += float(external_weight / total_weights[u])

    return total_score / n_nodes


def _dag_interleave_edge_fraction(
    sn_adj: np.ndarray,
    sn_names: list[str],
    rows: list[Supernode],
) -> float:
    """Fraction of middle-supernode edge mass that flows backward in layer order. Lower is better."""
    # sn_adj[target, source] — same convention as pruned_adj.
    # Layer floor = supernode.layer_min, which uses Node.layer (correct for error nodes,
    # whose node_id prefix "0_..." would be misleading if parsed directly).
    row_by_name = {row.name: row for row in rows}
    layer_floor: dict[str, int] = {}
    for name in sn_names:
        row = row_by_name.get(name)
        if row is None or row.type != "features":
            continue
        layer_floor[name] = int(row.layer_min)

    total_middle_mass = 0.0
    backward_mass = 0.0
    for tgt_idx, tgt_name in enumerate(sn_names):
        tgt_layer = layer_floor.get(tgt_name)
        if tgt_layer is None:
            continue
        for src_idx, src_name in enumerate(sn_names):
            if tgt_idx == src_idx:
                continue
            src_layer = layer_floor.get(src_name)
            if src_layer is None:
                continue
            w = float(abs(sn_adj[tgt_idx, src_idx]))
            if w <= 0.0:
                continue
            total_middle_mass += w
            if tgt_layer < src_layer:  # source at higher layer than target → backward
                backward_mass += w

    if total_middle_mass <= 0.0:
        return 0.0
    return float(backward_mass / total_middle_mass)


def _mean_layer_span(rows: list[Supernode]) -> float:
    """Mean layer span over feature-type supernodes. Lower = thinner clusters (F2)."""
    spans = [
        int(r.layer_max) - int(r.layer_min)
        for r in rows
        if r.type == "features"
    ]
    if not spans:
        return 0.0
    return float(sum(spans) / len(spans))


def _cv_cluster_sizes(rows: list[Supernode]) -> float:
    """Coefficient of variation of cluster sizes (std/mean). Lower = more balanced."""
    sizes = [len(row.member_node_ids()) for row in rows if row.type == "features"]
    if len(sizes) < 2:
        return 0.0
    sizes_arr = np.array(sizes, dtype=np.float64)
    mean = sizes_arr.mean()
    if mean == 0.0:
        return 0.0
    return float(sizes_arr.std() / mean)


def _opposing_sign_fraction(
    rows: list[Supernode],
    prune_adj: torch.Tensor,
    nodes: list,
) -> float:
    """Fraction of intra-cluster pairs whose net output contributions have opposing signs."""
    logit_indices = [i for i, n in enumerate(nodes) if node_is_logit(n)]
    if not logit_indices:
        return 0.0

    id_to_idx = {n.node_id: i for i, n in enumerate(nodes)}
    # net signed contribution of each node to all logit nodes
    # prune_adj[target, source], so prune_adj[logit, u] = contribution of u toward logit
    net_out = prune_adj[logit_indices, :].sum(dim=0)  # [n_nodes]

    total_pairs = 0
    opposing_pairs = 0
    for row in rows:
        if row.type != "features":
            continue
        indices = [id_to_idx[nid] for nid in row.member_node_ids() if nid in id_to_idx]
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                total_pairs += 1
                if float(net_out[indices[i]]) * float(net_out[indices[j]]) < 0:
                    opposing_pairs += 1

    return opposing_pairs / total_pairs if total_pairs > 0 else 0.0


def _cluster_metrics_from_parts(
    rows: list[Supernode],
    prune_graph: PruneGraph,
    similarity: Any,
    sn_adj: np.ndarray,
    sn_names: list[str],
) -> dict[str, Any]:
    """Shared scoring core: same inputs as ``score_clusters`` after graph construction."""
    n_middle = sum(1 for r in rows if r.type == "features")

    if n_middle == 0:
        return {
            "score_arith": 0.0,
            "score_harm": 0.0,
            "score_geo": 0.0,
            "sil_raw": 0.0,
            "sil_norm": 0.0,
            "silhouette_phi_norm": 0.0,
            "internal_independence": 0.0,
            "dag_score": 0.0,
            "back_edge_fraction": 0.0,
            "size_entropy": 1.0,
            "cv_cluster_sizes": 0.0,
            "opposing_sign_frac": 0.0,
            "mean_layer_span": 0.0,
            "n_middle": 0,
        }

    s = np.asarray(
        similarity.detach().cpu().numpy() if hasattr(similarity, "detach") else similarity,
        dtype=np.float64,
    )
    sil_raw, sil_norm = _silhouette_over_middle(s, prune_graph, rows)
    id_to_idx = {nid: i for i, nid in enumerate(prune_graph.node_ids)}
    internal_independence = _internal_independence_score(rows, prune_graph.pruned_adj, id_to_idx)

    sn_adj_arr = np.asarray(sn_adj, dtype=np.float64)
    dag_score = _dag_interleave_edge_fraction(sn_adj_arr, sn_names, rows)
    cv = _cv_cluster_sizes(rows)
    opp = _opposing_sign_fraction(rows, prune_graph.pruned_adj, prune_graph.nodes)
    size_entropy = _size_entropy_norm(rows)
    mean_span = _mean_layer_span(rows)
    score_arith = (sil_norm + internal_independence) / 2.0
    score_harm = 2 / ((1 / (sil_norm + 1e-12)) + (1 / (internal_independence + 1e-12)))
    score_geo = np.sqrt(sil_norm * internal_independence)

    return {
        "score_arith": float(score_arith),
        "score_harm": float(score_harm),
        "score_geo": float(score_geo),
        "sil_raw": float(sil_raw),
        "sil_norm": float(sil_norm),
        "silhouette_phi_norm": float(sil_norm),
        "internal_independence": float(internal_independence),
        "dag_score": float(dag_score),
        "back_edge_fraction": float(dag_score),
        "size_entropy": float(size_entropy),
        "cv_cluster_sizes": float(cv),
        "opposing_sign_frac": float(opp),
        "mean_layer_span": float(mean_span),
        "n_middle": int(n_middle),
    }


def score_clusters(
    supernode_rows: list[Supernode],
    prune_graph: PruneGraph,
    similarity: Any,
) -> dict[str, Any]:
    """
    Score a clustering using two complementary metrics:

      total = silhouette_norm * internal_independence

    where:
      - silhouette_norm = (mean silhouette over middle nodes + 1) / 2, in [0, 1].
      - internal_independence = 1 - (internal edge mass ratio), in [0, 1].
    """
    sng = SummaryGraph(supernodes=supernode_rows, pruned_adj=prune_graph.pruned_adj)
    return _cluster_metrics_from_parts(
        supernode_rows,
        prune_graph,
        similarity,
        np.asarray(sng.adj_matrix, dtype=np.float64),
        list(sng.sn_names),
    )


def score_summarization_graph(
    sng: SummaryGraph,
    prune_graph: PruneGraph,
    similarity: Any,
) -> dict[str, Any]:
    """
    Score an existing ``SummaryGraph`` using the same metrics as ``score_clusters``.

    Use this when the supernode graph is already built (e.g. for evaluation pipelines).
    """
    return _cluster_metrics_from_parts(
        sng.supernodes,
        prune_graph,
        similarity,
        np.asarray(sng.adj_matrix, dtype=np.float64),
        list(sng.sn_names),
    )
