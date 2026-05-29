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

# Hard caps beyond which exact HiGHS solves become intractable.
MAX_ILP_VARS = 200_000
MAX_ILP_CONSTRAINTS = 2_000_000


def cluster_graph_ilp(
    prune_graph: PruneGraph,
    *,
    theta: float = 0.0,
    eps_causal: float | None = None,
    max_sn: int | None = None,
    max_layer_span: int = 4,
    time_limit: float = 30.0,
) -> list[list[str]]:
    """Cluster a pruned graph by exact correlation clustering of its feature nodes.

    This is the Stage-2 solver of the methodology. It minimises the
    correlation-clustering atomicity objective on the *signed* cosine of the role
    vectors, subject to the complexity budget (D2) and the causal preservation
    budget (D4) as hard constraints:

        min_phi  sum_{i<j} x_ij (theta - cos(r_i, r_j))
        s.t.     K <= max_sn                                (D2, complexity)
                 sum_{feature edges} x_ij |W_ij| <= eps_causal * W_total   (D4, causal)

    with ``x_ij = 1`` iff middle features ``i, j`` share a supernode. The
    objective is the symmetric over/under-merge penalty: a pair with
    ``cos > theta`` (similar) carries a negative coefficient (merging is
    rewarded), a pair with ``cos < theta`` (dissimilar / antagonistic) a positive
    one (merging is penalised). Signed cosine is used directly — the negative
    region marks antagonistic features that should repel, which also keeps this
    objective consistent with ``L_causal``'s sign-cancellation term.

    Parameters
    ----------
    theta:
        Resolution threshold in ``[-1, 1]`` on the signed cosine. ``theta = 0``
        (the default) makes the cosine sign itself the merge boundary, so no
        threshold needs to be invented; raise it to merge more conservatively.
    eps_causal:
        Causal budget (D4). When set, the fraction of pruned edge weight absorbed
        *inside* supernodes (the linear surrogate of mechanism (i) from the
        causal-preservation loss) is constrained to ``<= eps_causal``. ``None``
        leaves causal preservation unconstrained. Sweeping ``eps_causal`` traces
        the 2-D Pareto front (atomicity vs. causal).
    max_sn:
        Complexity budget (D2): a hard cap ``K <= max_sn`` on the number of
        feature supernodes. ``None`` leaves K endogenous (driven by ``theta``).
    max_layer_span:
        Tractability prior: forbid merging two features more than this many
        layers apart. The methodology imposes no such constraint; raise it to
        relax. Combined with transitivity this bounds every cluster's layer span.

    Returns the same ``list[list[str]]`` contract as ``cluster_graph_spectral``:
    feature clusters + embedding/error/logit singletons.
    """
    if eps_causal is not None and eps_causal < 0:
        raise ValueError(f"eps_causal must be non-negative, got {eps_causal}")
    if max_sn is not None and max_sn < 1:
        raise ValueError(f"max_sn must be >= 1, got {max_sn}")

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
    phi = compute_phi_vectors(prune_graph).detach().cpu().numpy()  # (N, 2N)
    cos = _cosine_similarity(phi[mid_idx])  # (n, n) signed cosine in [-1, 1]
    layers = np.array([layer_index_from_node(nodes_by_id[nid]) for nid in middle_ids])

    # Allowed same-cluster pairs: features within max_layer_span layers of each other.
    # A disallowed pair is structurally x_ij = 0 (cannot share a supernode).
    pairs: list[tuple[int, int]] = [
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if abs(int(layers[i]) - int(layers[j])) <= max_layer_span
    ]
    col_x = {pair: k for k, pair in enumerate(pairs)}
    n_var_x = len(pairs)

    # Representative variables r_i (i is the lowest-index member of its cluster) are
    # only needed to count K for the complexity budget; skip them when max_sn is None.
    use_reps = max_sn is not None
    col_r = {i: n_var_x + i for i in range(n)} if use_reps else {}
    n_var_r = n if use_reps else 0
    n_var = n_var_x + n_var_r

    if n_var > MAX_ILP_VARS:
        raise ValueError(
            f"ILP too large: {n_var:,} variables for n_middle={n} "
            f"({n_var_x} same-cluster + {n_var_r} representative), "
            f"max_layer_span={max_layer_span}. Reduce graph size, lower "
            "max_layer_span, or use method='spectral'/'agglomerative'."
        )

    def pkey(a: int, b: int) -> tuple[int, int]:
        return (a, b) if a < b else (b, a)

    # Objective: minimise sum_pairs x_ij (theta - cos_ij). r has zero cost.
    c = np.zeros(n_var, dtype=np.float64)
    for pair, k in col_x.items():
        i, j = pair
        c[k] = theta - float(cos[i, j])

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    b_l: list[float] = []
    b_u: list[float] = []
    r = 0

    def add(row: int, c_idx: int, val: float) -> None:
        rows.append(row)
        cols.append(c_idx)
        data.append(val)

    # Allowed partners of each node, used for both transitivity and representatives.
    partners: list[list[int]] = [[] for _ in range(n)]
    for (i, j) in pairs:
        partners[i].append(j)
        partners[j].append(i)
    for lst in partners:
        lst.sort()

    # (T) Transitivity. Each inequality has a unique apex node a shared by its two
    # positive pairs; enumerate over (apex, unordered partner pair). The third pair
    # (p, q) may be disallowed, in which case its term is 0 and the constraint
    # x_ap + x_aq <= 1 forbids the layer-span-violating transitive merge.
    for a in range(n):
        pa = partners[a]
        for ii in range(len(pa)):
            p = pa[ii]
            x_ap = col_x[pkey(a, p)]
            for jj in range(ii + 1, len(pa)):
                q = pa[jj]
                add(r, x_ap, 1.0)
                add(r, col_x[pkey(a, q)], 1.0)
                pq = col_x.get(pkey(p, q))
                if pq is not None:
                    add(r, pq, -1.0)
                b_l.append(-np.inf)
                b_u.append(1.0)
                r += 1

    if use_reps:
        # (R) r_i <= 1 - x_ji for each earlier allowed partner j < i, and
        #     r_i >= 1 - sum_{j<i} x_ji. Together: r_i = 1 iff i has no earlier
        #     same-cluster partner, i.e. i is the lowest-index member of its cluster.
        for i in range(n):
            earlier = [j for j in partners[i] if j < i]
            for j in earlier:
                add(r, col_r[i], 1.0)
                add(r, col_x[pkey(j, i)], 1.0)
                b_l.append(-np.inf)
                b_u.append(1.0)
                r += 1
            add(r, col_r[i], 1.0)
            for j in earlier:
                add(r, col_x[pkey(j, i)], 1.0)
            b_l.append(1.0)
            b_u.append(np.inf)
            r += 1

        # (D2) complexity budget: sum_i r_i <= max_sn.
        for i in range(n):
            add(r, col_r[i], 1.0)
        b_l.append(-np.inf)
        b_u.append(float(max_sn))
        r += 1

    # (D4) causal epsilon-constraint: absorbed feature-edge mass <= eps_causal * W_total.
    if eps_causal is not None:
        prune_adj = prune_graph.pruned_adj.detach().cpu().numpy()
        W_total = float(np.abs(prune_adj).sum())
        global_to_local = {gi: li for li, gi in enumerate(mid_idx)}
        edge_weight: dict[tuple[int, int], float] = {}
        nz_t, nz_s = np.nonzero(prune_adj)
        for tgt_g, src_g in zip(nz_t.tolist(), nz_s.tolist()):
            if tgt_g == src_g:
                continue
            u_l = global_to_local.get(src_g)
            v_l = global_to_local.get(tgt_g)
            if u_l is None or v_l is None:
                continue
            key = pkey(u_l, v_l)
            if key in col_x:  # span-disallowed pairs cannot merge -> absorb nothing
                edge_weight[key] = edge_weight.get(key, 0.0) + float(abs(prune_adj[tgt_g, src_g]))
        if W_total > 0.0 and edge_weight:
            for key, w in edge_weight.items():
                add(r, col_x[key], w)
            b_l.append(-np.inf)
            b_u.append(eps_causal * W_total)
            r += 1

    if r > MAX_ILP_CONSTRAINTS:
        raise ValueError(
            f"ILP too large: {r:,} constraints for n_middle={n}, "
            f"max_layer_span={max_layer_span}. Lower max_layer_span or graph size."
        )

    A = csr_matrix((data, (rows, cols)), shape=(r, n_var))
    constraints = LinearConstraint(A, np.array(b_l), np.array(b_u))
    integrality = np.ones(n_var, dtype=np.int64)  # x and r are binary
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
                f"ILP infeasible (n_middle={n}, max_sn={max_sn}, eps_causal={eps_causal}, "
                f"max_layer_span={max_layer_span}). Raise max_sn/eps_causal/max_layer_span."
            )
        raise ValueError(
            f"ILP found no solution within time_limit={time_limit}s (status={res.status}, "
            f"n_middle={n}). Raise time_limit or reduce graph size."
        )
    if res.status == 1:  # 1 = time/iteration limit, but an incumbent exists
        logger.warning("ILP hit the %.1fs time limit; using the best incumbent (may be suboptimal).", time_limit)

    # Recover clusters by union-find over the same-cluster pairs set to 1.
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    x_sol = res.x[:n_var_x] > 0.5
    for (i, j), k in col_x.items():
        if x_sol[k]:
            union(i, j)

    grouped: dict[int, list[str]] = {}
    for i in range(n):
        grouped.setdefault(find(i), []).append(middle_ids[i])
    middle_clusters = list(grouped.values())

    return middle_clusters + emb_singletons + logit_singletons
