from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import torch
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix

from summarization.prune import PruneGraph
from summarization.summarize import (
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

# Hard caps beyond which exact HiGHS solves become intractable.
MAX_ILP_VARS = 200_000
MAX_ILP_CONSTRAINTS = 2_000_000
DEFAULT_THETA: str = "p65"
DEFAULT_EPS_CAUSAL: float = 0.05
DEFAULT_MAX_SN: int = 20
DEFAULT_MAX_LAYER_SPAN: int = 7
DEFAULT_NORMALIZE_WEIGHTS: bool = False
DEFAULT_TIME_LIMIT: float = 30.0


def _nodes_by_id(prune_graph: PruneGraph) -> dict[str, Node]:
    return {n.node_id: n for n in prune_graph.nodes}


def _fixed_singletons(
    kept_ids: list[str],
    nodes_by_id: dict[str, Node],
) -> tuple[list[list[str]], list[list[str]]]:
    emb = [[nid] for nid in kept_ids if node_is_embedding(nodes_by_id[nid])]
    logit = [[nid] for nid in kept_ids if node_is_logit(nodes_by_id[nid])]
    return emb, logit


def _classify_node(node_id: str, nodes_by_id: dict[str, Node]) -> str:
    node = nodes_by_id.get(node_id)
    if node is None:
        return "middle"
    if node_is_embedding(node):
        return "emb"
    if node_is_logit(node):
        return "logit"
    return "middle"


def _layer_numeric(node_id: str, nodes_by_id: dict[str, Node]) -> int:
    node = nodes_by_id.get(node_id)
    return layer_index_from_node(node) if node is not None else layer_index_from_node_id(node_id)


def _prepare_node_weights(
    scores: torch.Tensor | None,
    n_nodes: int,
    device: torch.device,
    normalize: bool = False,
) -> torch.Tensor:
    # Older serialized PruneGraph payloads may not carry influence/relevance tensors.
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
    """Role vectors rᵢ = [v_out; v_in] used by ILP clustering.

    Shape: (N, 2N). ``pruned_adj[target, source]`` is the adjacency convention.
    """
    adj = prune_graph.pruned_adj.float()  # (N, N)
    n_nodes = adj.shape[0]
    device = adj.device
    influence = _prepare_node_weights(
        prune_graph.node_influence,
        n_nodes,
        device,
        normalize=normalize_weights,
    )
    relevance = _prepare_node_weights(
        prune_graph.node_relevance,
        n_nodes,
        device,
        normalize=normalize_weights,
    )
    sqrt_influence = influence.clamp(min=0.0).sqrt()
    sqrt_relevance = relevance.clamp(min=0.0).sqrt()
    v_out = adj.T * sqrt_influence.unsqueeze(0)
    v_in = adj * sqrt_relevance.unsqueeze(0)
    return torch.cat([v_out, v_in], dim=1)


def _cosine_similarity(features: np.ndarray, nonnegative: bool = False) -> np.ndarray:
    """Pairwise cosine similarity of row features."""
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


def _sort_clusters_by_layer(
    clusters: list[list[str]],
    nodes_by_id: dict[str, Node],
) -> list[list[str]]:
    return sorted(clusters, key=lambda c: min(_layer_numeric(n, nodes_by_id) for n in c))


def _groups_from_labels(node_ids: list[str], labels: np.ndarray) -> list[list[str]]:
    grouped: dict[int, list[str]] = {}
    for node_id, label in zip(node_ids, labels, strict=True):
        grouped.setdefault(int(label), []).append(node_id)
    return [grouped[label] for label in sorted(grouped)]


def labels_to_supernodes(
    prune_graph: PruneGraph,
    middle_ids: list[str],
    labels: np.ndarray,
) -> list[list[str]]:
    middle_clusters = _groups_from_labels(middle_ids, labels)
    nodes_by_id = _nodes_by_id(prune_graph)
    emb_singletons, logit_singletons = _fixed_singletons(prune_graph.node_ids, nodes_by_id)
    return middle_clusters + emb_singletons + logit_singletons


def _supernode_from_member_ids(
    prune_graph: PruneGraph,
    name: str,
    member_ids: list[str],
    kind: str,
    id_to_idx: dict[str, int] | None = None,
) -> Supernode:
    nodes = [
        node_from_prune_graph(prune_graph, node_id, id_to_idx=id_to_idx) for node_id in member_ids
    ]
    nodes_map = _nodes_by_id(prune_graph)
    layers = [_layer_numeric(node_id, nodes_map) for node_id in member_ids]
    if kind == "emb":
        supernode_type = cluster_kind_to_supernode_type("emb")
    elif kind == "logit":
        supernode_type = cluster_kind_to_supernode_type("logit")
    else:
        supernode_type = cluster_kind_to_supernode_type("middle")
    return Supernode(
        name=name,
        features=nodes,
        type=supernode_type,
        layer_min=min(layers) if layers else 0,
        layer_max=max(layers) if layers else 0,
    )


def clusters_to_supernodes(
    prune_graph: PruneGraph,
    supernodes: list[list[str]],
    middle_prefix: str = "SN",
    *,
    enforce_dag: bool = True,
) -> list[Supernode]:
    """Convert member-id clusters into typed ``Supernode`` rows."""
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

    middle = _sort_clusters_by_layer(middle, nodes_by_id)
    out: list[Supernode] = []
    id_to_idx = {n.node_id: i for i, n in enumerate(prune_graph.nodes)}
    for i, sn in enumerate(middle):
        kind = _classify_node(sn[0], nodes_by_id)
        out.append(
            _supernode_from_member_ids(
                prune_graph,
                f"{middle_prefix}_{i}",
                list(sn),
                kind,
                id_to_idx=id_to_idx,
            )
        )
    emb_logit: list[Supernode] = []
    for i, sn in enumerate(emb):
        kind = _classify_node(sn[0], nodes_by_id)
        emb_logit.append(
            _supernode_from_member_ids(
                prune_graph,
                f"SN_EMB_{i}",
                list(sn),
                kind,
                id_to_idx=id_to_idx,
            )
        )
    for i, sn in enumerate(logit):
        kind = _classify_node(sn[0], nodes_by_id)
        emb_logit.append(
            _supernode_from_member_ids(
                prune_graph,
                f"SN_LOGIT_{i}",
                list(sn),
                kind,
                id_to_idx=id_to_idx,
            )
        )
    return out + emb_logit


def supernodes_to_mapping(
    prune_graph: PruneGraph,
    supernodes: list[list[str]],
    middle_prefix: str = "SN",
) -> dict[str, list[str]]:
    rows = clusters_to_supernodes(prune_graph, supernodes, middle_prefix=middle_prefix)
    return {s.name: s.member_node_ids() for s in rows}


def mapping_dict_to_supernodes(
    prune_graph: PruneGraph, mapping: dict[str, list[str]]
) -> list[Supernode]:
    nodes_by_id = _nodes_by_id(prune_graph)
    out: list[Supernode] = []
    id_to_idx = {n.node_id: i for i, n in enumerate(prune_graph.nodes)}
    for name, feats in mapping.items():
        if not feats:
            continue
        kind = _classify_node(feats[0], nodes_by_id)
        out.append(
            _supernode_from_member_ids(prune_graph, name, list(feats), kind, id_to_idx=id_to_idx)
        )
    return out


def _validate_args(eps_causal: float | None, max_sn: int | None) -> None:
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
        "max_layer_span, or use an eval-owned legacy baseline."
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
    max_sn: int | None = DEFAULT_MAX_SN,
    eps_causal: float | None = DEFAULT_EPS_CAUSAL,
    causal_coeff: np.ndarray,
    has_causal_mass: bool,
    max_layer_span: int = DEFAULT_MAX_LAYER_SPAN,
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
    theta: float | str = DEFAULT_THETA,
    eps_causal: float | None = DEFAULT_EPS_CAUSAL,
    max_sn: int | None = DEFAULT_MAX_SN,
    max_layer_span: int = DEFAULT_MAX_LAYER_SPAN,
    normalize_weights: bool = DEFAULT_NORMALIZE_WEIGHTS,
    time_limit: float = DEFAULT_TIME_LIMIT,
) -> list[list[str]]:
    """Cluster a pruned graph by exactly minimising the Stage-2 objective.

    This is the Stage-2 solver of the methodology. It minimises atomicity
    (signed-cosine correlation clustering, Eq. Latom) and treats causal
    preservation as an optional epsilon constraint:

        min_f  L_atom(f)
             = sum_{i<j} x_ij (theta - cos(r_i, r_j))
        s.t.   C_causal(f) <= eps_causal                         (D3, optional)
               K <= max_sn                                       (C2, optional)

    with ``x_ij = 1`` iff middle features ``i, j`` share a supernode, ``|W_ij|`` the
    pruned edge mass between them (both directions), and ``W_total`` the total pruned
    edge mass (the constant causal denominator). The atomicity term is the
    symmetric over/under-merge penalty: a similar pair (``cos > theta``)
    carries a negative coefficient (merging rewarded), a dissimilar/antagonistic pair
    a positive one. Acyclicity (C1) is deferred to Stage 3.

    Parameters
    ----------
    theta:
        Resolution threshold in ``[-1, 1]`` on the signed cosine. ``theta = 0``
        makes the cosine sign itself the merge boundary. The default is the
        adaptive percentile spec ``"p65"``: theta is the 65th percentile of the
        allowed-pair cosine distribution of *this* graph, so the boundary tracks
        each graph's similarity scale instead of a fixed global value.
    eps_causal:
        Optional hard budget on ``C_causal`` in ``[0, 1]``. When set, the ILP
        minimises atomicity subject to hiding at most this fraction of retained edge
        mass inside feature supernodes. Default ``0.05``.
    max_sn:
        Complexity budget (C2): a hard cap ``K <= max_sn`` on the number of
        feature supernodes. Default ``20``. ``None`` leaves K endogenous (driven by
        ``theta``).
    max_layer_span:
        Tractability prior: forbid merging two features more than this many
        layers apart. The methodology imposes no such constraint; raise it to
        relax. Combined with transitivity this bounds every cluster's layer span.
        Default ``7``.
    normalize_weights:
        If True, min-max normalise the per-node influence/relevance weights before
        they scale the role vectors (forwarded to ``compute_phi_vectors``). Default
        False (raw weights).

    Returns raw member-id clusters: feature clusters plus embedding/logit singletons.
    """
    _validate_args(eps_causal, max_sn)

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


def cluster(
    prune_graph: PruneGraph,
    *,
    method: Literal["ilp"] = "ilp",
    theta: float | str = DEFAULT_THETA,
    eps_causal: float | None = DEFAULT_EPS_CAUSAL,
    max_sn: int | None = DEFAULT_MAX_SN,
    max_layer_span: int = DEFAULT_MAX_LAYER_SPAN,
    normalize_weights: bool = DEFAULT_NORMALIZE_WEIGHTS,
    time_limit: float = DEFAULT_TIME_LIMIT,
    ilp_time_limit: float | None = None,
) -> list[Supernode]:
    """Stage 2: cluster a ``PruneGraph`` into typed ``Supernode`` rows.

    ``summarization.cluster`` is the canonical ILP clustering stage. Legacy
    spectral/agglomerative baselines live in ``eval.legacy_cluster_baselines``.
    """
    if method != "ilp":
        raise ValueError(
            "summarization.cluster only supports method='ilp'. "
            "Use eval.legacy_cluster_baselines for spectral/agglomerative baselines."
        )
    clusters = cluster_graph_ilp(
        prune_graph,
        theta=theta,
        eps_causal=eps_causal,
        max_sn=max_sn,
        max_layer_span=max_layer_span,
        normalize_weights=normalize_weights,
        time_limit=ilp_time_limit if ilp_time_limit is not None else time_limit,
    )
    return clusters_to_supernodes(prune_graph, clusters)
