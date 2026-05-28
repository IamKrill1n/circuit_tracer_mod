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

# Hard cap on total MILP variables (z + y); beyond this, exact HiGHS solves are intractable.
MAX_ILP_VARS = 200_000


def cluster_graph_ilp(
    prune_graph: PruneGraph,
    lambdas: tuple[float, float, float] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    max_layer_span: int = 4,
    max_sn: int | None = None,
    time_limit: float = 30.0,
) -> list[list[str]]:
    """Cluster a pruned graph by minimising a linearised form of the paper's L.

    Objective tracks ``L = lam_cplx * L_cplx + lam_atom * L_atom + lam_causal * L_causal``
    from paper Eq. (Lsum):
      - L_cplx (exact): K / |V'|, with K = (opened medoids) + (constant) fixed
        singletons. Only the K term varies with z; the fixed-singleton offset is a
        constant ignored by the MILP and recovered post-hoc by ``compute_L_cplx``.
      - L_atom (medoid proxy): mean rectified-cosine distance from each middle
        feature to its assigned medoid, normalised by n_middle so the term lies in
        [0, 1]. The paper's silhouette is non-linear (min / max over neighbours)
        and cannot enter an MILP; medoid distance is the standard k-medoids
        surrogate and agrees with the silhouette in spirit (penalises within-
        cluster dispersion).
      - L_causal (intra-edge proxy): fraction of pruned edge weight that falls
        inside supernodes, via same-cluster indicators y[i, j]. Captures
        mechanism (i) from paper Eq. (Lcausal); the dominant-direction drop (ii)
        and sign-cancellation (iii) depend on the post-projection summary
        adjacency and cannot be linearised cheaply, so they are not in the MILP
        objective — they are still scored post-hoc by ``compute_L_causal``.

    ``lambdas = (lam_cplx, lam_atom, lam_causal)`` must lie on the probability
    simplex. ``max_layer_span`` hard-bounds cluster layer width (the paper has
    no such constraint; this is a tractability prior — set to a large value to
    relax). ``max_sn`` is an optional hard cap on K.

    Returns the same ``list[list[str]]`` contract as ``cluster_graph_spectral``:
    middle clusters + embedding/error/logit singletons.
    """
    lam_cplx, lam_atom, lam_causal = lambdas
    if min(lam_cplx, lam_atom, lam_causal) < 0:
        raise ValueError(f"lambdas must be non-negative, got {lambdas}")
    if abs((lam_cplx + lam_atom + lam_causal) - 1.0) > 1e-6:
        raise ValueError(f"lambdas must sum to 1, got {lam_cplx + lam_atom + lam_causal}")

    kept_ids = prune_graph.node_ids
    nodes_by_id = _nodes_by_id(prune_graph)
    if not kept_ids:
        return []

    n_pruned = len(kept_ids)
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
    d = 1.0 - np.clip(sim, 0.0, 1.0)  # rectified-cosine distance, paper's cos_+
    layers = np.array([layer_index_from_node(nodes_by_id[nid]) for nid in middle_ids])

    # Allowed assignments u -> medoid m: directional upward window
    # 0 <= layer(u) - layer(m) <= max_layer_span. Forces the medoid to sit at the
    # lowest layer of its cluster, so each cluster spans <= L exactly.
    pairs: list[tuple[int, int]] = [
        (u, m)
        for u in range(n)
        for m in range(n)
        if 0 <= layers[u] - layers[m] <= max_layer_span
    ]
    col_z = {(u, m): k for k, (u, m) in enumerate(pairs)}
    n_var_z = len(pairs)

    # Same-cluster indicators for the L_causal proxy. y[(i, j)] = 1 iff middle
    # features i, j land in the same cluster; linearised via constraint (C3).
    prune_adj = prune_graph.pruned_adj.detach().cpu().numpy()
    W_total = float(np.abs(prune_adj).sum())
    global_to_local: dict[int, int] = {gi: li for li, gi in enumerate(mid_idx)}
    edge_weight: dict[tuple[int, int], float] = {}
    if lam_causal > 0.0 and W_total > 0.0:
        nz_t, nz_s = np.nonzero(prune_adj)
        for tgt_g, src_g in zip(nz_t.tolist(), nz_s.tolist()):
            if tgt_g == src_g:
                continue
            u_l = global_to_local.get(src_g)
            v_l = global_to_local.get(tgt_g)
            if u_l is None or v_l is None:
                continue
            key = (min(u_l, v_l), max(u_l, v_l))
            edge_weight[key] = edge_weight.get(key, 0.0) + float(abs(prune_adj[tgt_g, src_g]))

    y_keys = sorted(edge_weight.keys())
    col_y = {key: n_var_z + k for k, key in enumerate(y_keys)}
    n_var_y = len(y_keys)
    n_var = n_var_z + n_var_y

    if n_var > MAX_ILP_VARS:
        raise ValueError(
            f"ILP too large: {n_var:,} variables for n_middle={n} "
            f"({n_var_z} assignment + {n_var_y} same-cluster), "
            f"max_layer_span={max_layer_span}. Reduce graph size, set "
            "lam_causal=0 to skip same-cluster variables, or use "
            "method='spectral'/'agglomerative'."
        )

    # Objective coefficients (each loss term is already in [0, 1]).
    c = np.zeros(n_var, dtype=np.float64)
    atom_scale = lam_atom / n
    cplx_scale = lam_cplx / n_pruned
    for k, (u, m) in enumerate(pairs):
        c[k] = atom_scale * d[u, m]
        if u == m:
            c[k] += cplx_scale  # opening cost per supernode
    if W_total > 0.0:
        causal_scale = lam_causal / W_total
        for key, w in edge_weight.items():
            c[col_y[key]] = causal_scale * w

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
            if (u, m) in col_z:
                add(r, col_z[(u, m)], 1.0)
        b_l.append(1.0)
        b_u.append(1.0)
        r += 1

    # (C2) a node may only join an open medoid: z[u, m] - z[m, m] <= 0 (u != m).
    for (u, m) in pairs:
        if u == m:
            continue
        add(r, col_z[(u, m)], 1.0)
        add(r, col_z[(m, m)], -1.0)
        b_l.append(-np.inf)
        b_u.append(0.0)
        r += 1

    # (C3) same-cluster: z[i, m] + z[j, m] - y[i, j] <= 1 for each m where both
    # (i, m) and (j, m) are allowed. With z binary this forces y[i, j] = 1 iff
    # i, j share a medoid; y is left continuous (LP-tight at integer z).
    for (i, j) in y_keys:
        for m in range(n):
            if (i, m) in col_z and (j, m) in col_z:
                add(r, col_z[(i, m)], 1.0)
                add(r, col_z[(j, m)], 1.0)
                add(r, col_y[(i, j)], -1.0)
                b_l.append(-np.inf)
                b_u.append(1.0)
                r += 1

    # (C4) optional budget cap on opened medoids: sum_m z[m, m] <= max_sn.
    if max_sn is not None:
        for m in range(n):
            add(r, col_z[(m, m)], 1.0)
        b_l.append(-np.inf)
        b_u.append(float(max_sn))
        r += 1

    A = csr_matrix((data, (rows, cols)), shape=(r, n_var))
    constraints = LinearConstraint(A, np.array(b_l), np.array(b_u))
    integrality = np.zeros(n_var, dtype=np.int64)
    integrality[:n_var_z] = 1  # z binary; y continuous (LP-tight at integer z).
    res = milp(
        c=c,
        constraints=constraints,
        integrality=integrality,
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

    z = res.x[:n_var_z] > 0.5
    grouped: dict[int, list[str]] = {}
    for (u, m) in pairs:
        if z[col_z[(u, m)]]:
            grouped.setdefault(m, []).append(middle_ids[u])
    middle_clusters = list(grouped.values())

    return middle_clusters + emb_singletons + logit_singletons
