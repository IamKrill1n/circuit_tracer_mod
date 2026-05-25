from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix

from summarization.cluster import _fixed_singletons, _nodes_by_id, compute_phi_vectors
from summarization.prune import PruneGraph
from summarization.scoring import _cosine_similarity
from summarization.utils import layer_index_from_node, node_is_fixed

logger = logging.getLogger(__name__)

# Hard cap on MILP assignment variables; beyond this, exact HiGHS solves are intractable.
MAX_ILP_VARS = 200_000


def cluster_graph_ilp(
    prune_graph: PruneGraph,
    max_layer_span: int = 4,
    gamma: float | None = None,
    max_sn: int | None = None,
    time_limit: float = 30.0,
) -> list[list[str]]:
    """Cluster a pruned graph by an uncapacitated facility-location MILP (k-medoids with opening cost).

    Minimizes within-cluster role dissimilarity + gamma * (#supernodes), so the number of
    supernodes K is chosen endogenously by the single knob gamma (dissimilarity units, [0, 1]).
    Layer span is hard-bounded by an exact directional encoding (see below). Returns the same
    ``list[list[str]]`` contract as ``cluster_graph_spectral``: middle clusters + emb/logit singletons.

    gamma=None uses an adaptive default = median of allowed off-diagonal distances.
    """
    kept_ids = prune_graph.node_ids
    nodes_by_id = _nodes_by_id(prune_graph)
    if not kept_ids:
        return []

    mid_idx = [i for i, nid in enumerate(kept_ids) if not node_is_fixed(nodes_by_id[nid])]
    middle_ids = [kept_ids[i] for i in mid_idx]
    emb_singletons, logit_singletons = _fixed_singletons(kept_ids, nodes_by_id)

    if not middle_ids:
        return emb_singletons + logit_singletons
    if len(middle_ids) == 1:
        return [[middle_ids[0]]] + emb_singletons + logit_singletons

    n = len(middle_ids)
    phi = compute_phi_vectors(prune_graph).detach().cpu().numpy()  # [N, 2N]
    sim = _cosine_similarity(phi[mid_idx])  # [n, n] cosine in [-1, 1]
    d = 1.0 - np.clip(sim, 0.0, 1.0)  # rectified-cosine distance, matches L_coh's cos_+
    layers = np.array([layer_index_from_node(nodes_by_id[nid]) for nid in middle_ids])

    # Allowed assignments (u -> medoid m): directional upward window 0 <= layer(u) - layer(m) <= L.
    # This forces the medoid to be a lowest-layer cluster member, so every member sits in
    # [layer(m), layer(m)+L] and the cluster span is <= L exactly. Diagonal (u, u) is always allowed.
    pairs: list[tuple[int, int]] = [
        (u, m)
        for u in range(n)
        for m in range(n)
        if 0 <= layers[u] - layers[m] <= max_layer_span
    ]
    col = {(u, m): k for k, (u, m) in enumerate(pairs)}
    n_var = len(pairs)

    # Exact MILP has ~O(n * features-within-L-layers) binaries; it is only tractable for modest
    # graphs. Real pruned graphs can have thousands of middle features (millions of vars), which
    # HiGHS cannot solve — fail fast with guidance instead of timing out.
    if n_var > MAX_ILP_VARS:
        raise ValueError(
            f"ILP too large: {n_var:,} assignment variables for n_middle={n} "
            f"(max_layer_span={max_layer_span}). Exact MILP is intractable at this size; "
            "prune more aggressively or use method='spectral'/'agglomerative'."
        )

    if gamma is None:
        off_diag = [d[u, m] for (u, m) in pairs if u != m]
        gamma = float(np.median(off_diag)) if off_diag else 0.5

    # z[u, m] is the only variable; a node is an open medoid iff z[m, m] == 1 (self-assigned).
    c = np.empty(n_var, dtype=np.float64)
    for k, (u, m) in enumerate(pairs):
        c[k] = d[u, m] + (gamma if u == m else 0.0)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    b_l: list[float] = []
    b_u: list[float] = []

    def add(row: int, c_idx: int, val: float) -> None:
        rows.append(row)
        cols.append(c_idx)
        data.append(val)

    r = 0
    # (C1) each u assigned to exactly one medoid: sum_m z[u, m] = 1.
    for u in range(n):
        for m in range(n):
            if (u, m) in col:
                add(r, col[(u, m)], 1.0)
        b_l.append(1.0)
        b_u.append(1.0)
        r += 1

    # (C2) a node may only join an open medoid: z[u, m] - z[m, m] <= 0 (u != m).
    for (u, m) in pairs:
        if u == m:
            continue
        add(r, col[(u, m)], 1.0)
        add(r, col[(m, m)], -1.0)
        b_l.append(-np.inf)
        b_u.append(0.0)
        r += 1

    # (C3) budget cap: sum_m z[m, m] <= max_sn.
    if max_sn is not None:
        for m in range(n):
            add(r, col[(m, m)], 1.0)
        b_l.append(-np.inf)
        b_u.append(float(max_sn))
        r += 1

    A = csr_matrix((data, (rows, cols)), shape=(r, n_var))
    constraints = LinearConstraint(A, np.array(b_l), np.array(b_u))
    res = milp(
        c=c,
        constraints=constraints,
        integrality=np.ones(n_var),
        bounds=Bounds(0, 1),
        options={"time_limit": time_limit},
    )

    if res.x is None:
        if res.status == 2:  # 2 = infeasible
            raise ValueError(
                f"ILP infeasible (n_middle={n}, max_layer_span={max_layer_span}, max_sn={max_sn}). "
                "Increase max_sn or max_layer_span."
            )
        raise ValueError(
            f"ILP found no solution within time_limit={time_limit}s (status={res.status}, "
            f"n_middle={n}). Raise time_limit or reduce graph size."
        )
    if res.status == 1:  # 1 = time/iteration limit, but an incumbent exists
        logger.warning("ILP hit the %.1fs time limit; using the best incumbent (may be suboptimal).", time_limit)

    z = res.x > 0.5
    # Group members by their assigned medoid; each u has exactly one assignment by (C1).
    grouped: dict[int, list[str]] = {}
    for (u, m) in pairs:
        if z[col[(u, m)]]:
            grouped.setdefault(m, []).append(middle_ids[u])
    middle_clusters = list(grouped.values())

    return middle_clusters + emb_singletons + logit_singletons
