from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scipy.linalg import eigvalsh

from summarization.cluster import (
    cluster_graph_spectral,
    clusters_to_supernodes,
    compute_phi_vectors,
)
from summarization.cluster_scoring import (
    _cosine_similarity,
    _middle_indices,
    score_clusters,
)
from summarization.prune import PruneGraph


def eigengap_analysis(
    similarity: Any,
    prune_graph: PruneGraph,
    max_k: int = 20,
) -> dict[str, Any]:
    """Estimate a plausible k range via normalized-Laplacian eigengap."""
    s = np.asarray(similarity.detach().cpu().numpy() if hasattr(similarity, "detach") else similarity)
    mid = _middle_indices(prune_graph)
    m = len(mid)
    if m < 3:
        return {"eigengap_k": 2, "eigenvalues": np.array([0.0, 1.0]), "gaps": np.array([1.0]), "search_range": (2, 2)}

    s_mid = ((s[np.ix_(mid, mid)] + s[np.ix_(mid, mid)].T) / 2.0).clip(0.0, 1.0)
    deg = s_mid.sum(axis=1)
    deg_safe = np.where(deg > 1e-8, deg, 1e-8)
    d_inv = np.diag(1.0 / np.sqrt(deg_safe))
    l_norm = d_inv @ (np.diag(deg) - s_mid) @ d_inv

    n_eig = min(max_k + 1, m)
    evals = np.sort(eigvalsh(l_norm))[:n_eig]
    gaps = np.diff(evals)

    search_end = min(len(gaps), max_k)
    if search_end < 2:
        k_hat = 2
    else:
        k_hat = int(np.argmax(gaps[1:search_end])) + 2

    k_min = max(2, k_hat - 2)
    k_max = min(m - 1, k_hat + 2)
    if k_max - k_min < 2:
        k_max = min(m - 1, k_min + 4)

    return {"eigengap_k": k_hat, "eigenvalues": evals, "gaps": gaps, "search_range": (k_min, k_max)}


def _phi_similarity(prune_graph: PruneGraph) -> np.ndarray:
    """Cosine similarity of the influence/relevance-weighted phi vectors over all nodes.

    This is the shared method-neutral similarity used by auto-k for both eigengap
    and silhouette scoring.
    """
    phi = compute_phi_vectors(prune_graph).detach().cpu().numpy()
    return _cosine_similarity(phi, nonnegative=True)


def find_best_k(
    prune_graph: PruneGraph,
    similarity: Any | None = None,
    max_layer_span: int = 4,
    k_min_override: int | None = None,
    k_max_override: int | None = None,
    weights: dict[str, float] | None = None,
    max_sn: int | None = None,
    mean_method: Literal["geo", "harm", "arith"] = "arith",
    decay_rate: float | None = None,
    enforce_dag: bool = False,
    random_state: int = 42,
    n_init: int = 20,
) -> tuple[int, dict[int, dict[str, Any]]]:
    """
    Auto-select k for `cluster_graph_spectral` via composite-score sweep on phi-similarity.

    The eigengap range and silhouette are computed against sim_phi (the cosine of the
    influence/relevance-weighted feature vectors), so every method picks k on the same
    method-neutral yardstick.

    Returns `(best_k, results)` where each results[k] includes `final_supernodes`.
    """
    del similarity  # legacy arg; we always use sim_phi for scoring/eigengap now
    del weights  # legacy weight kwargs
    sim_phi = _phi_similarity(prune_graph)
    n_middle = len(_middle_indices(prune_graph))
    if n_middle < 3:
        return 2, {}

    eg = eigengap_analysis(sim_phi, prune_graph, max_k=min(20, n_middle - 1))
    k_min = k_min_override if k_min_override is not None else int(eg["search_range"][0])
    k_max = k_max_override if k_max_override is not None else int(eg["search_range"][1])
    k_min = max(2, k_min)
    k_max = min(n_middle - 1, k_max)
    if k_min > k_max:
        k_min = k_max

    results: dict[int, dict[str, Any]] = {}
    for k in range(k_min, k_max + 1):
        supernodes = cluster_graph_spectral(
            prune_graph,
            target_k=k,
            max_layer_span=max_layer_span,
            max_sn=max_sn,
            mean_method=mean_method,
            decay_rate=decay_rate,
            enforce_dag=enforce_dag,
            random_state=random_state,
            n_init=n_init,
        )
        rows = clusters_to_supernodes(prune_graph, supernodes)
        sc = score_clusters(
            rows,
            prune_graph,
            sim_phi,
            enforce_dag=enforce_dag,
        )
        sc["final_supernodes"] = {s.name: s.member_node_ids() for s in rows}
        results[k] = sc

    if not results:
        return int(eg["eigengap_k"]), {}
    best_k = max(results, key=lambda x: float(results[x]["sil_norm"]))
    return best_k, results


def find_best_k_for_clusterer(
    *,
    prune_graph: PruneGraph,
    similarity: Any | None = None,
    clusterer: Any,
    k_min_override: int | None = None,
    k_max_override: int | None = None,
    weights: dict[str, float] | None = None,
    enforce_dag: bool = False,
) -> tuple[int, dict[int, dict[str, Any]]]:
    """
    Auto-select k for an arbitrary clusterer via composite-score sweep on phi-similarity.
    """
    del similarity  # legacy arg; phi-similarity is computed internally
    del weights
    sim_phi = _phi_similarity(prune_graph)
    n_middle = len(_middle_indices(prune_graph))
    if n_middle < 3:
        fallback_k = max(0, n_middle)
        clusters = clusterer(fallback_k)
        rows = clusters_to_supernodes(prune_graph, clusters)
        result = score_clusters(
            rows,
            prune_graph,
            sim_phi,
            enforce_dag=enforce_dag,
        )
        result["final_supernodes"] = {s.name: s.member_node_ids() for s in rows}
        return fallback_k, {fallback_k: result}

    eigengap = eigengap_analysis(sim_phi, prune_graph, max_k=min(20, n_middle - 1))
    k_min = k_min_override if k_min_override is not None else int(eigengap["search_range"][0])
    k_max = k_max_override if k_max_override is not None else int(eigengap["search_range"][1])
    k_min = max(2, min(k_min, n_middle))
    k_max = max(k_min, min(k_max, n_middle))

    results: dict[int, dict[str, Any]] = {}
    for target_k in range(k_min, k_max + 1):
        clusters = clusterer(target_k)
        rows = clusters_to_supernodes(prune_graph, clusters)
        result = score_clusters(
            rows,
            prune_graph,
            sim_phi,
            enforce_dag=enforce_dag,
        )
        result["final_supernodes"] = {s.name: s.member_node_ids() for s in rows}
        results[target_k] = result

    best_k = max(results, key=lambda k: float(results[k]["sil_norm"]))
    return best_k, results
