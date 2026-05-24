from __future__ import annotations

from typing import Literal

import logging
import numpy as np
import torch
from sklearn.cluster import SpectralClustering


from summarization.prune import PruneGraph
from summarization.supernode_graph import (
    Node,
    Supernode,
    cluster_kind_to_supernode_type,
    node_from_prune_graph,
)
from summarization.utils import (
    layer_index_from_node,
    layer_index_from_node_id,
    node_is_embedding,
    node_is_fixed,
    node_is_logit,
)

logger = logging.getLogger(__name__)


def _nodes_by_id(prune_graph: PruneGraph) -> dict[str, Node]:
    return {n.node_id: n for n in prune_graph.nodes}


def _fixed_singletons(
    kept_ids: list[str], nodes_by_id: dict[str, Node]
) -> tuple[list[list[str]], list[list[str]]]:
    emb = [[nid] for nid in kept_ids if node_is_embedding(nodes_by_id[nid])]
    logit = [[nid] for nid in kept_ids if node_is_logit(nodes_by_id[nid])]
    return emb, logit


def _classify_node(node_id: str, nodes_by_id: dict[str, Node]) -> str:
    n = nodes_by_id.get(node_id)
    if n is None:
        return "middle"
    if node_is_embedding(n):
        return "emb"
    if node_is_logit(n):
        return "logit"
    return "middle"


def _layer_numeric(node_id: str, nodes_by_id: dict[str, Node]) -> int:
    n = nodes_by_id.get(node_id)
    return layer_index_from_node(n) if n is not None else layer_index_from_node_id(node_id)


def _cosine_norm(matrix: torch.Tensor) -> torch.Tensor:
    diag = torch.sqrt(torch.diag(matrix).clamp(min=1e-8))
    return matrix / diag.unsqueeze(1) / diag.unsqueeze(0)


def _weighted_row_cosine(features: torch.Tensor) -> torch.Tensor:
    gram = features @ features.T
    return _cosine_norm(gram)


def _prepare_node_weights(
    scores: torch.Tensor | None, n_nodes: int, device: torch.device, normalize: bool = False
) -> torch.Tensor:
    # Missing tensors can happen for older serialized PruneGraph payloads.
    if scores is None:
        return torch.ones(n_nodes, dtype=torch.float32, device=device)

    values = scores.detach().float().to(device).reshape(-1)
    if values.numel() != n_nodes:
        return torch.ones(n_nodes, dtype=torch.float32, device=device)

    values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
    if normalize:
        v_max = float(values.max().item())
        v_min = float(values.min().item())
        if v_max - v_min > 1e-8:
            values = (values - v_min) / (v_max - v_min + 1e-8)
        else:
            values = torch.ones_like(values)
    return values

def compute_phi_vectors(
    prune_graph: PruneGraph,
    normalize_weights: bool = False,
) -> torch.Tensor:
    """Per-feature [v_out; v_in] vectors used by `compute_similarity` (no layer decay).

    Shape [N, 2N]. The two blocks are:
      v_out[i, k] = pruned_adj[k, i] * sqrt(Inf_k)   (i's outgoing edges, weighted by target influence)
      v_in[i, k]  = pruned_adj[i, k] * sqrt(Rel_k)   (i's incoming edges, weighted by source relevance)
    """
    adj = prune_graph.pruned_adj.float()  # [target, source] convention
    n_nodes = adj.shape[0]
    device = adj.device
    inf = _prepare_node_weights(prune_graph.node_influence, n_nodes, device, normalize=normalize_weights)
    rel = _prepare_node_weights(prune_graph.node_relevance, n_nodes, device, normalize=normalize_weights)
    sqrt_inf = inf.clamp(min=0.0).sqrt()
    sqrt_rel = rel.clamp(min=0.0).sqrt()
    v_out = adj.T * sqrt_inf.unsqueeze(0)  # row i = pruned_adj[:, i] (outgoing from i), scaled by sqrt(Inf)
    v_in = adj * sqrt_rel.unsqueeze(0)  # row i = pruned_adj[i, :] (incoming to i), scaled by sqrt(Rel)
    return torch.cat([v_out, v_in], dim=1)


def compute_similarity(
    prune_graph: PruneGraph,
    mean_method: Literal["geo", "harm", "arith"] = "arith",
    normalize_weights: bool = False,
    decay_rate: float | None = None,
    max_layer_span: int | None = None,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """
    Compute node similarity from weighted shared out/in structure.

    Output/input cosine similarities are always clamped to ``[0, 1]`` before
    being combined.
    """
    adj = prune_graph.pruned_adj.clone().float().T
    n_nodes = adj.shape[0]
    node_inf = _prepare_node_weights(prune_graph.node_influence, n_nodes, adj.device, normalize=normalize_weights)
    node_rel = _prepare_node_weights(prune_graph.node_relevance, n_nodes, adj.device, normalize=normalize_weights)
    s_out = adj @ torch.diag(node_inf) @ adj.T
    s_in = adj.T @ torch.diag(node_rel) @ adj

    s_out_cos = _cosine_norm(s_out).clamp(0.0, 1.0)
    s_in_cos = _cosine_norm(s_in).clamp(0.0, 1.0)

    if mean_method == "geo":
        s = (s_out_cos * s_in_cos).sqrt()
    elif mean_method == "harm":
        # Equivalent to 2 / (1/a + 1/b) but safe when a or b is exactly 0
        # (cosine values are clamped to [0, 1] above, so zeros are common).
        s = (2.0 * s_out_cos * s_in_cos) / (s_out_cos + s_in_cos + 1e-12)
    elif mean_method == "arith":
        s = (s_out_cos + s_in_cos) / 2.0
    else:
        raise ValueError(f"Unsupported mean_method={mean_method!r}.")

    if decay_rate is not None and decay_rate > 0.0:
        layer_indices = []
        for n in prune_graph.nodes:
            try:
                layer_indices.append(layer_index_from_node(n))
            except Exception:
                layer_indices.append(0)

        layers_t = torch.tensor(layer_indices, dtype=torch.float32, device=s.device)
        layer_diffs = torch.abs(layers_t.unsqueeze(1) - layers_t.unsqueeze(0))
        
        penalty = torch.exp(-decay_rate * layer_diffs)
        if max_layer_span is not None:
            penalty[layer_diffs > max_layer_span] = 0.0
            
        s = s * penalty

    s = s.clamp(epsilon, 1.0)
    
    return s


def _merge_to_budget(
    clusters: list[list[str]], nodes_by_id: dict[str, Node], max_sn: int
) -> list[list[str]]:
    """Greedily merge layer-adjacent clusters until budget is met."""
    while len(clusters) > max_sn:
        best_i = -1
        best_gap = float("inf")
        for i in range(len(clusters) - 1):
            hi_i = max(_layer_numeric(n, nodes_by_id) for n in clusters[i])
            lo_j = min(_layer_numeric(n, nodes_by_id) for n in clusters[i + 1])
            gap = abs(lo_j - hi_i)
            if gap < best_gap:
                best_gap = gap
                best_i = i

        if best_i < 0:
            break

        merged = clusters[best_i] + clusters[best_i + 1]
        clusters = clusters[:best_i] + [merged] + clusters[best_i + 2 :]

    return clusters


def _name_middle_supernodes(
    clusters: list[list[str]], nodes_by_id: dict[str, Node]
) -> dict[str, list[str]]:
    clusters = sorted(clusters, key=lambda c: min(_layer_numeric(n, nodes_by_id) for n in c))
    return {f"SN_{i}": members for i, members in enumerate(clusters)}


def _supernode_from_member_ids(
    prune_graph: PruneGraph,
    name: str,
    member_ids: list[str],
    kind: str,
    id_to_idx: dict[str, int] | None = None,
) -> Supernode:
    nodes = [node_from_prune_graph(prune_graph, node_id, id_to_idx=id_to_idx) for node_id in member_ids]
    nodes_map = _nodes_by_id(prune_graph)
    layers = [_layer_numeric(node_id, nodes_map) for node_id in member_ids]
    if kind == "emb":
        sn_type = cluster_kind_to_supernode_type("emb")
    elif kind == "logit":
        sn_type = cluster_kind_to_supernode_type("logit")
    else:
        sn_type = cluster_kind_to_supernode_type("middle")
    return Supernode(
        name=name,
        features=nodes,
        type=sn_type,
        layer_min=min(layers) if layers else 0,
        layer_max=max(layers) if layers else 0,
    )


def labels_to_supernodes(
    prune_graph: PruneGraph,
    middle_ids: list[str],
    labels: np.ndarray,
) -> list[list[str]]:
    grouped: dict[int, list[str]] = {}
    for node_id, label in zip(middle_ids, labels):
        grouped.setdefault(int(label), []).append(node_id)

    middle_clusters = [grouped[label] for label in sorted(grouped)]
    emb_singletons = [[n.node_id] for n in prune_graph.nodes if node_is_embedding(n)]
    logit_singletons = [[n.node_id] for n in prune_graph.nodes if node_is_logit(n)]
    return middle_clusters + emb_singletons + logit_singletons


def cluster_graph_spectral(
    prune_graph: PruneGraph,
    target_k: int = 7,
    max_layer_span: int = 4,
    max_sn: int | None = None,
    mean_method: Literal["geo", "harm", "arith"] = "arith",
    normalize_weights: bool = False,
    decay_rate: float | None = 1.0,
    enforce_dag: bool = False,
    random_state: int = 42,
    n_init: int = 20,
) -> list[list[str]]:
    """
    Cluster a pruned attribution graph into supernodes.

    Args:
        prune_graph: Output of `prune_graph_pipeline`.
        target_k: Target number of middle supernodes.
        max_layer_span: Maximum allowed layer span within a middle supernode.
        max_sn: Optional hard cap on number of middle supernodes.
        mean_method: Mean used to combine output/input cosine similarities.
        normalize_weights: If True, min-max normalize influence/relevance weights before computing similarity.
        random_state: Random seed for spectral clustering k-means init.
        n_init: Number of k-means runs for `SpectralClustering(assign_labels="kmeans")`.

    Returns:
        List of supernodes where each supernode is a list of node ids.
        Embedding/logit nodes are returned as singleton supernodes.
    """
    del enforce_dag  # legacy: π in SummarizationGraph is always on; partition is preserved
    logger.info("Starting cluster_graph_spectral (target_k=%d, max_layer_span=%s)", target_k, max_layer_span)
    kept_ids = prune_graph.node_ids
    nodes_by_id = _nodes_by_id(prune_graph)

    if not kept_ids:
        logger.info("No kept_ids found, returning empty clusters.")
        return []

    logger.info("Computing similarity matrix...")
    sim = compute_similarity(
        prune_graph,
        mean_method=mean_method,
        normalize_weights=normalize_weights,
        decay_rate=decay_rate,
        max_layer_span=max_layer_span,
    )

    middle_idx = [i for i, nid in enumerate(kept_ids) if not node_is_fixed(nodes_by_id[nid])]
    middle_ids = [kept_ids[i] for i in middle_idx]
    logger.info("Extracted %d middle nodes.", len(middle_ids))

    if not middle_ids:
        logger.info("No middle nodes found, returning singletons.")
        fixed_only = [[nid] for nid in kept_ids]
        return fixed_only

    mid_sim = sim[middle_idx][:, middle_idx].detach().cpu().numpy().clip(0.0, 1.0)
    mid_sim = ((mid_sim + mid_sim.T) / 2.0).clip(0.0, 1.0)
    target_k = max(1, min(target_k, len(middle_ids)))

    if target_k == 1:
        labels = np.zeros(len(middle_ids), dtype=np.int64)
    elif target_k == len(middle_ids):
        labels = np.arange(len(middle_ids), dtype=np.int64)
    else:
        logger.info("Running SpectralClustering with target_k=%d...", target_k)
        labels = SpectralClustering(
            n_clusters=target_k,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=int(random_state),
            n_init=int(n_init),
        ).fit_predict(mid_sim)
        logger.info("Finished SpectralClustering.")

    grouped: dict[int, list[str]] = {}
    for nid, lbl in zip(middle_ids, labels):
        grouped.setdefault(int(lbl), []).append(nid)
    middle_clusters = list(grouped.values())

    if max_sn is not None:
        logger.info("Merging to budget of %d supernodes...", max_sn)
        middle_clusters = _merge_to_budget(middle_clusters, nodes_by_id, max_sn=max_sn)

    # Keep deterministic naming order for middle SNs, but return member lists only.
    named_middle = _name_middle_supernodes(middle_clusters, nodes_by_id)

    emb_singletons, logit_singletons = _fixed_singletons(kept_ids, nodes_by_id)

    supernodes = list(named_middle.values()) + emb_singletons + logit_singletons
    logger.info("Returning %d total supernodes.", len(supernodes))
    return supernodes


def _merge_creates_cycle(ca: np.ndarray, a: int, b: int, active: set[int]) -> bool:
    """
    True iff merging clusters a and b would create a cycle in the cluster DAG.

    A direct edge a→b (or b→a) is fine — it just becomes internal to the merged
    cluster. A cycle arises only when an out-neighbor of the merged cluster can
    reach {a, b} via other clusters, i.e. there is an indirect path (length ≥ 2)
    between a and b.
    """
    n = ca.shape[0]
    # Out-neighbors of the merged cluster that are neither a nor b
    out_nbrs = [c for c in range(n) if c in active and c != a and c != b and (ca[a, c] or ca[b, c])]
    for c in out_nbrs:
        visited = {c}
        stack = [c]
        while stack:
            node = stack.pop()
            for nbr in range(n):
                if ca[node, nbr] and nbr in active and nbr not in visited:
                    if nbr == a or nbr == b:
                        return True
                    visited.add(nbr)
                    stack.append(nbr)
    return False


def cluster_graph_agglomerative(
    prune_graph: PruneGraph,
    target_k: int = 7,
    max_layer_span: int = 4,
    max_sn: int | None = None,
    mean_method: Literal["geo", "harm", "arith"] = "arith",
    normalize_weights: bool = False,
    decay_rate: float | None = 1.0,
) -> list[list[str]]:
    """
    Cycle-Constrained Agglomerative Clustering (Approach 1 from clusterv3.md).

    Merges the most-similar pair of clusters that (a) keeps layer span <= max_layer_span
    and (b) does not create a cycle in the supernode DAG, until target_k middle clusters
    remain. The DAG guarantee is native — no post-processing needed.
    """
    logger.info("Starting cluster_graph_agglomerative (target_k=%d)...", target_k)
    kept_ids = prune_graph.node_ids
    nodes_by_id = _nodes_by_id(prune_graph)

    if not kept_ids:
        logger.info("No kept_ids, returning empty list.")
        return []

    logger.info("Computing similarity matrix...")
    sim = compute_similarity(
        prune_graph,
        mean_method=mean_method,
        normalize_weights=normalize_weights,
        decay_rate=decay_rate,
        max_layer_span=max_layer_span,
    )

    middle_idx = [i for i, nid in enumerate(kept_ids) if not node_is_fixed(nodes_by_id[nid])]
    middle_ids = [kept_ids[i] for i in middle_idx]

    if not middle_ids:
        logger.info("No middle nodes, returning mapped fixed nodes as separate clusters.")
        return [[nid] for nid in kept_ids]

    m = len(middle_ids)
    target_k = max(1, min(target_k, m))
    logger.info("Extracted %d middle nodes.", m)

    # Symmetrized similarity between middle nodes
    mid_sim = sim[middle_idx][:, middle_idx].detach().cpu().numpy().clip(0.0, 1.0)
    mid_sim = ((mid_sim + mid_sim.T) / 2.0).clip(0.0, 1.0)

    # Working cluster-level similarity matrix (weighted average linkage)
    cs = mid_sim.copy()
    np.fill_diagonal(cs, -np.inf)

    sizes = np.ones(m, dtype=float)

    # members[k] = list of indices into middle_ids
    members: dict[int, list[int]] = {i: [i] for i in range(m)}

    logger.info("Initializing cluster-level DAG...")
    # Cluster-level DAG: ca[s, t] = True iff any edge from cluster s -> cluster t
    # pruned_adj[target, source]: pruned_adj[t, s] > 0 means edge s -> t
    full_adj = prune_graph.pruned_adj.detach().cpu().numpy()
    adj_mid = full_adj[np.ix_(middle_idx, middle_idx)]  # adj_mid[t, s] = edge s -> t

    ca = np.zeros((m, m), dtype=bool)
    tgts, srcs = np.where(adj_mid > 0)
    for s, t in zip(srcs, tgts):
        if s != t:
            ca[s, t] = True  # edge s -> t

    active: set[int] = set(range(m))

    # Precompute per-node layer values for span checks
    node_layers = [_layer_numeric(nid, nodes_by_id) for nid in middle_ids]

    logger.info("Starting agglomerative merge loop...")
    while len(active) > target_k:
        active_list = sorted(active)
        best_sim = -np.inf
        best_a = best_b = -1

        for ii in range(len(active_list)):
            a = active_list[ii]
            for jj in range(ii + 1, len(active_list)):
                b = active_list[jj]
                if cs[a, b] <= best_sim:
                    continue
                # Hard layer-span constraint
                merged_layers = [node_layers[i] for i in members[a]] + [node_layers[i] for i in members[b]]
                if max(merged_layers) - min(merged_layers) > max_layer_span:
                    continue
                # Cycle constraint: reject if merge would create a cycle
                if _merge_creates_cycle(ca, a, b, active):
                    continue
                best_sim = cs[a, b]
                best_a, best_b = a, b

        if best_a < 0:
            logger.info("No valid merge remaining. Terminating merge early.")
            break  # no valid merge remains

        logger.debug("Merging cluster index %d into %d. Remaining clusters: %d", best_b, best_a, len(active) - 1)

        # Merge best_b into best_a (weighted average linkage)
        sa, sb = sizes[best_a], sizes[best_b]
        for k in active:
            if k == best_a or k == best_b:
                continue
            merged_val = (sa * cs[best_a, k] + sb * cs[best_b, k]) / (sa + sb)
            cs[best_a, k] = merged_val
            cs[k, best_a] = merged_val

        sizes[best_a] = sa + sb
        members[best_a].extend(members.pop(best_b))

        # Update cluster DAG: best_a inherits best_b's edges
        ca[best_a, :] |= ca[best_b, :]
        ca[:, best_a] |= ca[:, best_b]
        ca[best_a, best_a] = False  # no self-loop
        ca[best_b, :] = False
        ca[:, best_b] = False

        active.discard(best_b)

    logger.info("Finished agglomeration loop. Formulating output clusters...")
    middle_clusters = [[middle_ids[i] for i in members[k]] for k in sorted(active)]

    if max_sn is not None and len(middle_clusters) > max_sn:
        logger.info("Merging up to maximum supernode budget of %d...", max_sn)
        middle_clusters = _merge_to_budget(middle_clusters, nodes_by_id, max_sn=max_sn)

    named_middle = _name_middle_supernodes(middle_clusters, nodes_by_id)
    emb_singletons, logit_singletons = _fixed_singletons(kept_ids, nodes_by_id)

    total_supernodes = len(named_middle) + len(emb_singletons) + len(logit_singletons)
    logger.info("Agglomerative clustering done. Returning %d total supernodes.", total_supernodes)
    return list(named_middle.values()) + emb_singletons + logit_singletons


def cluster_graph_with_labels(
    prune_graph: PruneGraph,
    method: Literal["spectral", "agglomerative"] = "spectral",
    **kwargs,
) -> list[list[str]]:
    """
    Convenience wrapper for Neuronpedia-style format:
    [[label, node_id, ...], ...]
    """
    if method == "spectral":
        raw = cluster_graph_spectral(prune_graph, **kwargs)
    elif method == "agglomerative":
        raw = cluster_graph_agglomerative(prune_graph, **kwargs)
    else:
        raise ValueError(f"Invalid method: {method}")
    out: list[list[str]] = []
    for i, members in enumerate(raw):
        if len(members) == 1:
            continue
        out.append([f"cluster_{i}", *members])
    return out


def clusters_to_supernodes(
    prune_graph: PruneGraph,
    supernodes: list[list[str]],
    middle_prefix: str = "SN",
    *,
    enforce_dag: bool = True,
) -> list[Supernode]:
    """Convert `cluster_graph_spectral` member lists into named `Supernode` rows (middle + emb + logit).

    The legacy ``enforce_dag`` parameter is ignored — π in ``SummarizationGraph`` is
    always on and operates at the edge level, so the clusterer's partition is
    preserved exactly as given.
    """
    del enforce_dag
    nodes_by_id = _nodes_by_id(prune_graph)
    middle: list[list[str]] = []
    emb: list[list[str]] = []
    logit: list[list[str]] = []

    for sn in supernodes:
        if not sn:
            continue
        first = sn[0]
        kind = _classify_node(first, nodes_by_id)
        if kind == "emb":
            emb.append(sn)
        elif kind == "logit":
            logit.append(sn)
        else:
            middle.append(sn)

    middle = sorted(middle, key=lambda m: min(_layer_numeric(n, nodes_by_id) for n in m))
    out: list[Supernode] = []
    id_to_idx = {n.node_id: i for i, n in enumerate(prune_graph.nodes)}
    for i, sn in enumerate(middle):
        k = _classify_node(sn[0], nodes_by_id)
        out.append(_supernode_from_member_ids(prune_graph, f"{middle_prefix}_{i}", list(sn), k, id_to_idx=id_to_idx))
    emb_logit: list[Supernode] = []
    for i, sn in enumerate(emb):
        k = _classify_node(sn[0], nodes_by_id)
        emb_logit.append(_supernode_from_member_ids(prune_graph, f"SN_EMB_{i}", list(sn), k, id_to_idx=id_to_idx))
    for i, sn in enumerate(logit):
        k = _classify_node(sn[0], nodes_by_id)
        emb_logit.append(_supernode_from_member_ids(prune_graph, f"SN_LOGIT_{i}", list(sn), k, id_to_idx=id_to_idx))
    return out + emb_logit


def supernodes_to_mapping(
    prune_graph: PruneGraph,
    supernodes: list[list[str]],
    middle_prefix: str = "SN",
) -> dict[str, list[str]]:
    """Convert `cluster_graph_spectral` output into a named supernode mapping (dict shim)."""
    rows = clusters_to_supernodes(prune_graph, supernodes, middle_prefix=middle_prefix)
    return {s.name: s.member_node_ids() for s in rows}


def mapping_dict_to_supernodes(prune_graph: PruneGraph, mapping: dict[str, list[str]]) -> list[Supernode]:
    """Preserve dict insertion order; one `Supernode` per key (features may be filtered later in graph build)."""
    nodes_by_id = _nodes_by_id(prune_graph)
    out: list[Supernode] = []
    id_to_idx = {n.node_id: i for i, n in enumerate(prune_graph.nodes)}
    for name, feats in mapping.items():
        if not feats:
            continue
        k = _classify_node(feats[0], nodes_by_id)
        out.append(_supernode_from_member_ids(prune_graph, name, list(feats), k, id_to_idx=id_to_idx))
    return out

