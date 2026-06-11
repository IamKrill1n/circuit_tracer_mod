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


def _validate_args(lambda_causal: float, eps_causal: float | None, max_sn: int | None) -> None:
    # lambda_causal is kept only so older call sites fail less abruptly; it no longer
    # contributes to the ILP objective.
    if lambda_causal < 0:
        raise ValueError(f"lambda_causal must be non-negative, got {lambda_causal}")
    if eps_causal is not None and not 0.0 <= eps_causal <= 1.0:
        raise ValueError(f"eps_causal must be in [0, 1], got {eps_causal}")
    if max_sn is not None and max_sn < 1:
        raise ValueError(f"max_sn must be >= 1, got {max_sn}")


def _pair_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _allowed_pairs(layers: np.ndarray, max_layer_span: int) -> list[tuple[int, int]]:
    n = len(layers)
    return [
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if abs(int(layers[i]) - int(layers[j])) <= max_layer_span
    ]


def _resolve_theta(theta: float | str, cos: np.ndarray, pairs: list[tuple[int, int]]) -> float:
    if not isinstance(theta, str):
        return float(theta)

    if not (theta.startswith("p") and theta[1:].replace(".", "", 1).isdigit()):
        raise ValueError(f"theta string must be 'p<percentile>' (e.g. 'p65'), got {theta!r}")
    q = float(theta[1:])
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"theta percentile must be in [0, 100], got {q}")
    allowed_cos = np.array([cos[i, j] for i, j in pairs], dtype=np.float64)
    return float(np.percentile(allowed_cos, q)) if allowed_cos.size else 0.0


def _partners_by_node(n: int, pairs: list[tuple[int, int]]) -> list[list[int]]:
    partners: list[list[int]] = [[] for _ in range(n)]
    for i, j in pairs:
        partners[i].append(j)
        partners[j].append(i)
    for lst in partners:
        lst.sort()
    return partners


def _check_problem_size(n: int, n_var_x: int, n_var_r: int, max_layer_span: int) -> None:
    n_var = n_var_x + n_var_r
    if n_var <= MAX_ILP_VARS:
        return
    raise ValueError(
        f"ILP too large: {n_var:,} variables for n_middle={n} "
        f"({n_var_x} same-cluster + {n_var_r} representative), "
        f"max_layer_span={max_layer_span}. Reduce graph size, lower "
        "max_layer_span, or use method='spectral'/'agglomerative'."
    )


def _causal_coefficients(
    prune_graph: PruneGraph,
    mid_idx: list[int],
    col_x: dict[tuple[int, int], int],
) -> tuple[np.ndarray, float]:
    prune_adj = prune_graph.pruned_adj.detach().cpu().numpy()  # adj[tgt, src]
    w_total = float(np.abs(prune_adj).sum())
    coeff = np.zeros(len(col_x), dtype=np.float64)
    if w_total <= 0.0:
        return coeff, w_total

    for (i, j), k in col_x.items():
        gi, gj = mid_idx[i], mid_idx[j]  # local -> global pruned_adj indices
        pair_mass = abs(float(prune_adj[gi, gj])) + abs(float(prune_adj[gj, gi]))
        coeff[k] = pair_mass / w_total
    return coeff, w_total


def _objective(
    *,
    n_var: int,
    col_x: dict[tuple[int, int], int],
    cos: np.ndarray,
    theta_val: float,
) -> np.ndarray:
    c = np.zeros(n_var, dtype=np.float64)
    for (i, j), k in col_x.items():
        c[k] = theta_val - float(cos[i, j])
    return c


def _build_constraints(
    *,
    n: int,
    n_var: int,
    pairs: list[tuple[int, int]],
    col_x: dict[tuple[int, int], int],
    col_r: dict[int, int],
    max_sn: int | None,
    eps_causal: float | None,
    causal_coeff: np.ndarray,
    has_causal_mass: bool,
    max_layer_span: int,
) -> LinearConstraint:
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    b_l: list[float] = []
    b_u: list[float] = []
    row = 0

    def add(col: int, val: float) -> None:
        rows.append(row)
        cols.append(col)
        data.append(val)

    partners = _partners_by_node(n, pairs)

    # Transitivity over same-cluster indicators.
    for apex in range(n):
        apex_partners = partners[apex]
        for ii, p in enumerate(apex_partners):
            x_ap = col_x[_pair_key(apex, p)]
            for q in apex_partners[ii + 1 :]:
                add(x_ap, 1.0)
                add(col_x[_pair_key(apex, q)], 1.0)
                pq = col_x.get(_pair_key(p, q))
                if pq is not None:
                    add(pq, -1.0)
                b_l.append(-np.inf)
                b_u.append(1.0)
                row += 1

    if max_sn is not None:
        for i in range(n):
            earlier = [j for j in partners[i] if j < i]
            for j in earlier:
                add(col_r[i], 1.0)
                add(col_x[_pair_key(j, i)], 1.0)
                b_l.append(-np.inf)
                b_u.append(1.0)
                row += 1

            add(col_r[i], 1.0)
            for j in earlier:
                add(col_x[_pair_key(j, i)], 1.0)
            b_l.append(1.0)
            b_u.append(np.inf)
            row += 1

        for i in range(n):
            add(col_r[i], 1.0)
        b_l.append(-np.inf)
        b_u.append(float(max_sn))
        row += 1

    if eps_causal is not None and has_causal_mass:
        for k, coeff in enumerate(causal_coeff):
            if coeff > 0.0:
                add(k, float(coeff))
        b_l.append(-np.inf)
        b_u.append(float(eps_causal))
        row += 1

    if row > MAX_ILP_CONSTRAINTS:
        raise ValueError(
            f"ILP too large: {row:,} constraints for n_middle={n}, "
            f"max_layer_span={max_layer_span}. Lower max_layer_span or graph size."
        )

    matrix = csr_matrix((data, (rows, cols)), shape=(row, n_var))
    return LinearConstraint(matrix, np.array(b_l), np.array(b_u))


def _solve_ilp(
    *,
    c: np.ndarray,
    constraints: LinearConstraint,
    time_limit: float,
    n_middle: int,
    max_sn: int | None,
    max_layer_span: int,
    eps_causal: float | None,
) -> np.ndarray:
    res = milp(
        c=c,
        constraints=constraints,
        integrality=np.ones(len(c), dtype=np.int64),
        bounds=Bounds(0, 1),
        options={"time_limit": time_limit},
    )

    if res.x is None:
        if res.status == 2:  # 2 = infeasible
            raise ValueError(
                f"ILP infeasible (n_middle={n_middle}, max_sn={max_sn}, "
                f"max_layer_span={max_layer_span}, eps_causal={eps_causal}). "
                "Raise max_sn, max_layer_span, or eps_causal."
            )
        raise ValueError(
            f"ILP found no solution within time_limit={time_limit}s (status={res.status}, "
            f"n_middle={n_middle}). Raise time_limit or reduce graph size."
        )
    if res.status == 1:  # 1 = time/iteration limit, but an incumbent exists
        logger.warning(
            "ILP hit the %.1fs time limit; using the best incumbent (may be suboptimal).",
            time_limit,
        )
    return res.x


def _recover_clusters(
    middle_ids: list[str],
    col_x: dict[tuple[int, int], int],
    x_values: np.ndarray,
) -> list[list[str]]:
    parent = list(range(len(middle_ids)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for (i, j), k in col_x.items():
        if x_values[k] > 0.5:
            union(i, j)

    grouped: dict[int, list[str]] = {}
    for i, node_id in enumerate(middle_ids):
        grouped.setdefault(find(i), []).append(node_id)
    return list(grouped.values())


def cluster_graph_ilp(
    prune_graph: PruneGraph,
    *,
    theta: float | str = 0.0,
    lambda_causal: float = 1.0,
    eps_causal: float | None = None,
    max_sn: int | None = None,
    max_layer_span: int = 1000,
    normalize_weights: bool = False,
    time_limit: float = 30.0,
) -> list[list[str]]:
    """Cluster a pruned graph by exactly minimising the Stage-2 objective.

    This is the Stage-2 solver of the methodology. It minimises atomicity
    (signed-cosine correlation clustering, Eq. Latom) and treats causal
    preservation as an optional epsilon constraint:

        min_f  L_atom(f)
             = sum_{i<j} x_ij (theta - cos(r_i, r_j))
        s.t.   L_causal(f) <= eps_causal                         (D3, optional)
               K <= max_sn                                       (C2, optional)

    with ``x_ij = 1`` iff middle features ``i, j`` share a supernode, ``|W_ij|`` the
    pruned edge mass between them (both directions), and ``W_total`` the total pruned
    edge mass (the constant Eq. Lcausal denominator). The atomicity term is the
    symmetric over/under-merge penalty: a similar pair (``cos > theta``)
    carries a negative coefficient (merging rewarded), a dissimilar/antagonistic pair
    a positive one. Acyclicity (C1) is deferred to Stage 3.

    Parameters
    ----------
    theta:
        Resolution threshold in ``[-1, 1]`` on the signed cosine. ``theta = 0``
        (the default) makes the cosine sign itself the merge boundary, so no
        threshold needs to be invented; raise it to merge more conservatively.
        May also be an *adaptive* percentile spec ``"p<q>"`` (e.g. ``"p65"``):
        theta is then the q-th percentile of the allowed-pair cosine distribution
        of *this* graph, so the boundary tracks each graph's similarity scale
        instead of a fixed global value (the cosine scale varies widely per graph).
    lambda_causal:
        Deprecated compatibility argument. The ILP objective is always ``L_atom``;
        use ``eps_causal`` to constrain causal preservation.
    eps_causal:
        Optional hard budget on ``L_causal`` in ``[0, 1]``. When set, the ILP
        minimises atomicity subject to hiding at most this fraction of retained edge
        mass inside feature supernodes.
    max_sn:
        Complexity budget (C2): a hard cap ``K <= max_sn`` on the number of
        feature supernodes. ``None`` leaves K endogenous (driven by ``theta``).
    max_layer_span:
        Tractability prior: forbid merging two features more than this many
        layers apart. The methodology imposes no such constraint; raise it to
        relax. Combined with transitivity this bounds every cluster's layer span.
    normalize_weights:
        If True, min-max normalise the per-node influence/relevance weights before
        they scale the role vectors (forwarded to ``compute_phi_vectors``). Default
        False (raw weights).

    Returns the same ``list[list[str]]`` contract as ``cluster_graph_spectral``:
    feature clusters + embedding/error/logit singletons.
    """
    _validate_args(lambda_causal, eps_causal, max_sn)

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
    phi = (
        compute_phi_vectors(prune_graph, normalize_weights=normalize_weights).detach().cpu().numpy()
    )  # (N, 2N)
    cos = _cosine_similarity(phi[mid_idx])  # (n, n) signed cosine in [-1, 1]
    layers = np.array([layer_index_from_node(nodes_by_id[nid]) for nid in middle_ids])

    # Disallowed pairs are structurally x_ij = 0 (cannot share a supernode).
    pairs = _allowed_pairs(layers, max_layer_span)
    col_x = {pair: k for k, pair in enumerate(pairs)}
    n_var_x = len(pairs)

    if n_var_x == 0 and max_sn is None:
        return [[node_id] for node_id in middle_ids] + emb_singletons + logit_singletons

    theta_val = _resolve_theta(theta, cos, pairs)

    # Representative variables r_i (i is the lowest-index member of its cluster) are
    # only needed to count K for the complexity budget; skip them when max_sn is None.
    use_reps = max_sn is not None
    col_r = {i: n_var_x + i for i in range(n)} if use_reps else {}
    n_var_r = n if use_reps else 0
    n_var = n_var_x + n_var_r

    _check_problem_size(n, n_var_x, n_var_r, max_layer_span)

    causal_coeff, w_total = _causal_coefficients(prune_graph, mid_idx, col_x)
    c = _objective(n_var=n_var, col_x=col_x, cos=cos, theta_val=theta_val)
    constraints = _build_constraints(
        n=n,
        n_var=n_var,
        pairs=pairs,
        col_x=col_x,
        col_r=col_r,
        max_sn=max_sn,
        eps_causal=eps_causal,
        causal_coeff=causal_coeff,
        has_causal_mass=w_total > 0.0,
        max_layer_span=max_layer_span,
    )
    solution = _solve_ilp(
        c=c,
        constraints=constraints,
        time_limit=time_limit,
        n_middle=n,
        max_sn=max_sn,
        max_layer_span=max_layer_span,
        eps_causal=eps_causal,
    )
    middle_clusters = _recover_clusters(middle_ids, col_x, solution[:n_var_x])

    return middle_clusters + emb_singletons + logit_singletons
