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
)
from summarization.objective import compute_L
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


def _phi_and_similarity(prune_graph: PruneGraph) -> tuple[np.ndarray, np.ndarray]:
    """Phi role vectors over all nodes and their cosine similarity (nonnegative).

    The similarity is the method-neutral matrix used by auto-k for the eigengap range.
    """
    phi = compute_phi_vectors(prune_graph).detach().cpu().numpy()
    return phi, _cosine_similarity(phi, nonnegative=True)


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
    prune_loss: float = 0.0,
    lambdas: tuple[float, float, float] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
) -> tuple[int, dict[int, dict[str, Any]]]:
    """
    Auto-select k for `cluster_graph_spectral` by minimizing the closed-form L objective.

    The eigengap search range is computed against sim_phi (the cosine of the
    influence/relevance-weighted feature vectors). For each k in the range, we build
    the spectral partition and score it with `compute_L` (paper Eq. L). Best k = argmin(L).

    Returns `(best_k, results)` where each results[k] includes `final_supernodes`.
    """
    del similarity  # legacy arg; we always use sim_phi for eigengap now
    del weights  # legacy weight kwargs; objective uses `lambdas`
    phi, sim_phi = _phi_and_similarity(prune_graph)
    mid_idx = _middle_indices(prune_graph)
    n_middle = len(mid_idx)
    if n_middle < 3:
        return 2, {}

    role_vectors_middle = phi[mid_idx]
    middle_id_to_local = {prune_graph.nodes[i].node_id: local for local, i in enumerate(mid_idx)}

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
        sc: dict[str, Any] = dict(compute_L(
            rows,
            role_vectors_middle,
            middle_id_to_local,
            prune_graph,
            prune_loss=prune_loss,
            lambdas=lambdas,
        ))
        sc["final_supernodes"] = {s.name: s.member_node_ids() for s in rows}
        results[k] = sc

    if not results:
        return int(eg["eigengap_k"]), {}
    best_k = min(results, key=lambda x: float(results[x]["L"]))
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
    prune_loss: float = 0.0,
    lambdas: tuple[float, float, float] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
) -> tuple[int, dict[int, dict[str, Any]]]:
    """
    Auto-select k for an arbitrary clusterer by minimizing the closed-form L objective.
    """
    del similarity  # legacy arg; phi is computed internally
    del weights
    del enforce_dag  # legacy arg; objective is partition-only
    phi, sim_phi = _phi_and_similarity(prune_graph)
    mid_idx = _middle_indices(prune_graph)
    n_middle = len(mid_idx)
    role_vectors_middle = phi[mid_idx] if n_middle else phi[:0]
    middle_id_to_local = {prune_graph.nodes[i].node_id: local for local, i in enumerate(mid_idx)}

    if n_middle < 3:
        fallback_k = max(0, n_middle)
        clusters = clusterer(fallback_k)
        rows = clusters_to_supernodes(prune_graph, clusters)
        result: dict[str, Any] = dict(compute_L(
            rows,
            role_vectors_middle,
            middle_id_to_local,
            prune_graph,
            prune_loss=prune_loss,
            lambdas=lambdas,
        ))
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
        result = dict(compute_L(
            rows,
            role_vectors_middle,
            middle_id_to_local,
            prune_graph,
            prune_loss=prune_loss,
            lambdas=lambdas,
        ))
        result["final_supernodes"] = {s.name: s.member_node_ids() for s in rows}
        results[target_k] = result

    best_k = min(results, key=lambda k: float(results[k]["L"]))
    return best_k, results
