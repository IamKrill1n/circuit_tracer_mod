"""Clustering evaluation: signed causal-role coherence + summary-graph mass loss.

For every pruned graph we run each clustering method (ILP + matched-K baselines),
compute the per-graph metrics below, then report the unweighted mean across graphs
(method_means.csv). Pairs are never pooled across graphs.

Per-graph metrics (cosine uses cos(x, y) = xᵀy / (|x| |y| + ε), ε = 1e-12):

  1. Role gap ↑        mean cos(rᵢ, rⱼ) over same-cluster pairs − over cross-cluster pairs
  2. Signed up gap ↑   same, on the upstream role vᵢⁿ
  3. Signed down gap ↑ same, on the downstream role vᵒᵘᵗ
  4. C_causal ↓      fraction of retained edge mass hidden inside a feature supernode
  5. DAG loss ↓        backward-edge mass fraction Rσ over the aggregated superedges A

rᵢ = [vᵢᵒᵘᵗ ; vᵢⁱⁿ] is the full signed role vector from ``compute_phi_vectors``: the
first half is the downstream (sender) role, the second the upstream (receiver) role.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from sklearn.cluster import KMeans, SpectralClustering

from summarization.cluster import (
    DEFAULT_EPS_CAUSAL,
    DEFAULT_MAX_SN,
    DEFAULT_THETA,
    DEFAULT_TIME_LIMIT,
    _allowed_pairs,
    _cosine_similarity,
    _resolve_theta,
    cluster_graph_ilp,
    clusters_to_supernodes,
    compute_phi_vectors,
    labels_to_supernodes,
)
from summarization.prune import PruneGraph, load_prune_graph
from summarization.summarize import (
    SummaryGraph,
    _compute_depths,
    _stable_argsort,
    compute_sn_adj,
)
from summarization.utils import layer_index_from_node, node_is_fixed

logger = logging.getLogger(__name__)

# Small numerical constant for the cosine denominator and mass-fraction guards.
EPS = 1e-12

# Cross-cluster pair set Q_diff is exactly averaged unless it exceeds this many pairs,
# in which case |Q_same| of them are sampled uniformly with a fixed seed (plan §Inputs).
DIFF_PAIR_SAMPLE_CAP = 50_000

_NORM_TOKENS = {"softmax", "entmax", "sparsemax", "entmax15"}

METHODS = [
    "ours-ilp",
    "baseline-spectral-cosine",
    "baseline-kmeans",
    "baseline-spectral-adj",
    "baseline-random-same-size",
]

SUMMARY_COLUMNS = [
    "graph_name",
    "dataset",
    "graph_path",
    "token_normalization",
    "num_features",
    "method",
    "matched_k",
    "theta",
    "n_supernodes",
    "n_superedges",
    "role_intra",
    "role_inter",
    "role_gap",
    "signed_up_intra",
    "signed_up_inter",
    "signed_up_gap",
    "signed_down_intra",
    "signed_down_inter",
    "signed_down_gap",
    "C_causal",
    "dag_loss",
    "L_atom",
    "raw_superedge_mass",
    "final_superedge_mass",
    "dag_removed_mass_fraction",
    "final_retained_mass_fraction",
    "n_same_pairs",
    "n_diff_pairs",
    "supernode_map_path",
    "result_path",
]

# Recommended final-table columns (means across graphs); the headline 5 metrics + L_atom.
MEAN_COLUMNS = [
    "method",
    "n_graphs",
    "role_gap",
    "signed_up_gap",
    "signed_down_gap",
    "C_causal",
    "dag_loss",
    "L_atom",
]


# --- Cosine + pair gaps -------------------------------------------------------


def _pairwise_cosine(vectors: np.ndarray) -> np.ndarray:
    """Signed pairwise cosine cos(x, y) = xᵀy / (|x| |y| + ε), in [-1, 1]."""
    v = np.asarray(vectors, dtype=np.float64)  # (m, d)
    gram = v @ v.T
    norms = np.sqrt(np.clip(np.diag(gram), 0.0, None))
    denom = np.outer(norms, norms) + EPS
    return np.clip(gram / denom, -1.0, 1.0)


def compute_role_gaps(
    role_vectors_middle: np.ndarray,
    labels: np.ndarray,
    *,
    random_state: int,
    diff_cap: int = DIFF_PAIR_SAMPLE_CAP,
) -> dict[str, float]:
    """Signed within − cross cluster cosine gaps for the full / upstream / downstream roles.

    Only feature nodes assigned to a feature supernode (label >= 0) participate.
    role_vectors_middle is r = [v_out ; v_in], shape (n_middle, 2N).
    """
    nan = float("nan")
    keep = labels >= 0
    r = role_vectors_middle[keep]  # (m, 2N)
    lab = labels[keep]
    m = r.shape[0]
    n_total = r.shape[1] // 2
    v_out = r[:, :n_total]
    v_in = r[:, n_total:]

    empty = {
        "role_intra": nan,
        "role_inter": nan,
        "role_gap": nan,
        "signed_up_intra": nan,
        "signed_up_inter": nan,
        "signed_up_gap": nan,
        "signed_down_intra": nan,
        "signed_down_inter": nan,
        "signed_down_gap": nan,
        "n_same_pairs": 0,
        "n_diff_pairs": 0,
    }
    if m < 2:
        return empty

    iu, ju = np.triu_indices(m, k=1)
    same = lab[iu] == lab[ju]
    same_i, same_j = iu[same], ju[same]
    diff_i, diff_j = iu[~same], ju[~same]
    n_same = int(same_i.size)
    n_diff = int(diff_i.size)
    if n_same == 0 or n_diff == 0:
        return empty

    # Cross-cluster set is huge → sample |Q_same| pairs uniformly with a fixed seed.
    if n_diff > diff_cap:
        rng = np.random.default_rng(random_state)
        sel = rng.choice(n_diff, size=n_same, replace=False)
        diff_i, diff_j = diff_i[sel], diff_j[sel]

    def gap(vectors: np.ndarray) -> tuple[float, float, float]:
        cos = _pairwise_cosine(vectors)
        intra = float(cos[same_i, same_j].mean())
        inter = float(cos[diff_i, diff_j].mean())
        return intra, inter, intra - inter

    role_intra, role_inter, role_gap = gap(r)
    up_intra, up_inter, up_gap = gap(v_in)
    down_intra, down_inter, down_gap = gap(v_out)
    return {
        "role_intra": role_intra,
        "role_inter": role_inter,
        "role_gap": role_gap,
        "signed_up_intra": up_intra,
        "signed_up_inter": up_inter,
        "signed_up_gap": up_gap,
        "signed_down_intra": down_intra,
        "signed_down_inter": down_inter,
        "signed_down_gap": down_gap,
        "n_same_pairs": n_same,
        "n_diff_pairs": int(diff_i.size),
    }


# --- Mass metrics -------------------------------------------------------------


def _feature_member_index_lists(sng: SummaryGraph) -> list[list[int]]:
    return [
        [n.node_idx for n in sn.features if n.node_idx >= 0]
        for sn in sng.supernodes
        if sn.type in ("features", "feature")
    ]


def compute_C_causal(sng: SummaryGraph) -> float:
    """Causal budget (plan §4): same-cluster feature edge mass / total pruned edge mass.

    The numerator is restricted to edges between features hidden inside a feature
    supernode. The denominator matches ``summarization.cluster``'s ILP constraint:
    all retained pruned edge mass, including embedding→feature and feature→logit edges.
    """
    adj = sng.pruned_adj.detach().cpu().numpy().astype(np.float64)  # adj[target, source]
    member_lists = _feature_member_index_lists(sng)
    total_mag = float(np.abs(adj).sum())
    if total_mag <= EPS:
        return 0.0
    internal_mag = 0.0
    for idxs in member_lists:
        if len(idxs) < 2:
            continue
        block = adj[np.ix_(idxs, idxs)]  # both directions among members; diagonal is 0
        internal_mag += float(np.abs(block).sum())
    return internal_mag / total_mag


def compute_dag_loss(sng: SummaryGraph) -> float:
    """DAG loss (plan §5): backward-edge mass fraction Rσ over aggregated superedges A.

    A_{a→b} = block-sum of pruned edge mass from cluster a to cluster b (raw, pre-π).
    σ is the topological order SummaryGraph uses (anchor depth, emb source / logit sink).
    R is the set of backward edges A_{a→b} with σ(a) >= σ(b); we report
    Σ_R |A| / (Σ_{a≠b} |A| + ε). Computed on the raw A, so antiparallel collapse (a
    separate π step) is not folded into this number.
    """
    index_lists = [[n.node_idx for n in sn.features if n.node_idx >= 0] for sn in sng.supernodes]
    A = compute_sn_adj(index_lists, sng.pruned_adj)  # A[t, s] = mass s → t, diagonal 0
    total_mag = float(np.abs(A).sum())
    if total_mag <= EPS:
        return 0.0
    depths = _compute_depths(sng.supernodes)
    ordering = _stable_argsort(sng.supernodes, depths)
    rank = np.empty(len(sng.supernodes), dtype=np.int64)
    rank[ordering] = np.arange(len(sng.supernodes))
    backward_mag = 0.0
    n = A.shape[0]
    for t in range(n):
        for s in range(n):
            if A[t, s] != 0.0 and rank[s] > rank[t]:  # source ranks later → backward edge
                backward_mag += abs(float(A[t, s]))
    return backward_mag / (total_mag + EPS)


def compute_edge_mass_metrics(sng: SummaryGraph) -> dict[str, float | int]:
    """Mass accounting for the post-π summary graph (raw vs final superedge mass)."""
    total_fine_edge_mass = float(np.abs(sng.pruned_adj.detach().cpu().numpy()).sum())
    index_lists = [[n.node_idx for n in sn.features if n.node_idx >= 0] for sn in sng.supernodes]
    raw_adj = compute_sn_adj(index_lists, sng.pruned_adj)  # raw external block-sum, pre force-DAG
    raw_superedge_mass = float(np.abs(raw_adj).sum())
    final_superedge_mass = float(np.abs(sng.adj_matrix).sum())  # after antiparallel + back-edge cut
    dag_removed_mass = max(raw_superedge_mass - final_superedge_mass, 0.0)
    dag_removed_mass_fraction = (
        dag_removed_mass / raw_superedge_mass if raw_superedge_mass > EPS else 0.0
    )
    final_retained_mass_fraction = (
        final_superedge_mass / total_fine_edge_mass if total_fine_edge_mass > EPS else 0.0
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


def compute_atomicity_loss(cos_middle: np.ndarray, labels: np.ndarray, theta: float) -> float:
    """Atomicity diagnostic L_atom = Σ_{same-cluster i<j} (θ − cos(rᵢ, rⱼ)) (plan diagnostics).

    Lower (more negative) means more high-similarity pairs (cos > θ) were merged.
    cos_middle is the full signed role cosine over middle features (same matrix the ILP uses).
    """
    n = labels.size
    if n < 2:
        return 0.0
    iu, ju = np.triu_indices(n, k=1)
    same = (labels[iu] == labels[ju]) & (labels[iu] >= 0)
    s = cos_middle[iu, ju][same]
    return float((theta - s).sum())


# --- Cluster label bookkeeping ------------------------------------------------


def _middle_indices(prune_graph: PruneGraph) -> list[int]:
    return [i for i, n in enumerate(prune_graph.nodes) if not node_is_fixed(n)]


def _middle_labels_from_clusters(
    clusters: list[list[str]],
    middle_id_to_local: dict[str, int],
    n_middle: int,
) -> np.ndarray:
    """One label per feature cluster; -1 for middle features in no feature cluster."""
    labels = np.full(n_middle, -1, dtype=np.int64)
    next_label = 0
    for cluster in clusters:
        locals_ = [middle_id_to_local[m] for m in cluster if m in middle_id_to_local]
        if not locals_:
            continue  # emb / logit singleton
        for local in locals_:
            labels[local] = next_label
        next_label += 1
    return labels


# --- Baseline label producers -------------------------------------------------


def _adjacency_affinity(prune_graph: PruneGraph) -> np.ndarray:
    adj_sender = prune_graph.pruned_adj.clone().float().T.detach().cpu().numpy()
    affinity = np.abs(adj_sender)
    affinity = (affinity + affinity.T) / 2.0
    max_val = float(affinity.max()) if affinity.size else 0.0
    if max_val > 0.0:
        affinity = affinity / max_val
    np.fill_diagonal(affinity, 1.0)
    return affinity


def _kmeans_middle_labels(
    features: np.ndarray, target_k: int, random_state: int, n_init: int
) -> np.ndarray:
    n = features.shape[0]
    if n == 0:
        return np.array([], dtype=np.int64)
    k = max(1, min(target_k, n))
    if k == 1:
        return np.zeros(n, dtype=np.int64)
    if k == n:
        return np.arange(n, dtype=np.int64)
    # Spherical K-means: unit-normalise r(u) so Euclidean distance tracks cosine distance.
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features_normed = features / np.where(norms > 1e-12, norms, 1.0)
    return (
        KMeans(n_clusters=k, random_state=random_state, n_init=n_init)  # type: ignore[arg-type]
        .fit_predict(features_normed)
        .astype(np.int64)
    )


def _spectral_cosine_middle_labels(
    features: np.ndarray, target_k: int, random_state: int, n_init: int
) -> np.ndarray:
    n = features.shape[0]
    if n == 0:
        return np.array([], dtype=np.int64)
    k = max(1, min(target_k, n))
    if k == 1:
        return np.zeros(n, dtype=np.int64)
    if k == n:
        return np.arange(n, dtype=np.int64)
    affinity = _cosine_similarity(features, nonnegative=True)  # rectified cosine in [0, 1]
    return (
        SpectralClustering(
            n_clusters=k,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=random_state,
            n_init=n_init,  # type: ignore[arg-type]
        )
        .fit_predict(affinity)
        .astype(np.int64)
    )


def _spectral_affinity_middle_labels(
    affinity: np.ndarray, target_k: int, random_state: int, n_init: int
) -> np.ndarray:
    n = affinity.shape[0]
    if n == 0:
        return np.array([], dtype=np.int64)
    k = max(1, min(target_k, n))
    if k == 1:
        return np.zeros(n, dtype=np.int64)
    if k == n:
        return np.arange(n, dtype=np.int64)
    safe_affinity = np.asarray(affinity, dtype=np.float64)
    safe_affinity = np.clip((safe_affinity + safe_affinity.T) / 2.0, 0.0, None)
    np.fill_diagonal(safe_affinity, 1.0)
    return (
        SpectralClustering(
            n_clusters=k,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=random_state,
            n_init=n_init,  # type: ignore[arg-type]
        )
        .fit_predict(safe_affinity)
        .astype(np.int64)
    )


def _random_same_size_middle_labels(
    cluster_sizes: list[int],
    target_k: int,
    n_middle: int,
    random_state: int,
) -> np.ndarray:
    if n_middle == 0:
        return np.array([], dtype=np.int64)
    k = max(1, min(target_k, n_middle))
    if k == 1:
        return np.zeros(n_middle, dtype=np.int64)
    rng = np.random.default_rng(random_state)
    order = rng.permutation(n_middle)
    sizes = [size for size in cluster_sizes if size > 0]
    if sum(sizes) != n_middle or len(sizes) != k:
        sizes = [n_middle // k] * k
        for i in range(n_middle % k):
            sizes[i] += 1
    labels = np.empty(n_middle, dtype=np.int64)
    start = 0
    for label, size in enumerate(sizes):
        labels[order[start : start + size]] = label
        start += size
    return labels


# --- IO helpers ---------------------------------------------------------------


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(key): _to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(value) for value in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_jsonable(payload), indent=2), encoding="utf-8")


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _safe_slug(text: str) -> str:
    cleaned = [ch if (ch.isalnum() or ch in ("-", "_")) else "-" for ch in text]
    return "".join(cleaned).strip("-") or "graph"


# --- Graph discovery + identity ----------------------------------------------


_RE_NODE_INF_REL = re.compile(
    r"^node_inf_(?P<inf>\d+(?:\.\d+)?)_rel_(?P<rel>\d+(?:\.\d+)?)$",
    re.IGNORECASE,
)
_RE_NODE_SIMPLE = re.compile(r"^node_(?P<v>\d+(?:\.\d+)?)$", re.IGNORECASE)


def _path_matches_node_threshold(graph_path: Path, threshold: float, *, eps: float = 1e-5) -> bool:
    """True if a parent folder encodes this threshold (SHAP or simple layouts)."""

    def close(x: float) -> bool:
        return abs(x - threshold) <= eps

    for part in graph_path.parts:
        m = _RE_NODE_INF_REL.match(part)
        if m:
            if close(float(m.group("inf"))) and close(float(m.group("rel"))):
                return True
            continue
        m = _RE_NODE_SIMPLE.match(part)
        if m and close(float(m.group("v"))):
            return True
    return False


def _discover_prune_graphs(
    input_paths: Sequence[str],
    *,
    node_threshold: float | None,
) -> list[Path]:
    discovered: list[Path] = []
    for raw_path in input_paths:
        path = Path(raw_path).expanduser().resolve()
        if path.is_file():
            if path.suffix == ".pt" and path.name.endswith("_prune_graph.pt"):
                discovered.append(path)
            continue
        if path.is_dir():
            discovered.extend(
                sorted(p.resolve() for p in path.rglob("*_prune_graph.pt") if p.is_file())
            )
    unique = sorted(dict.fromkeys(discovered))
    if not unique:
        raise FileNotFoundError(
            "No prune-graph .pt files were found in the provided --input-path locations."
        )
    if node_threshold is None:
        return unique
    filtered = [p for p in unique if _path_matches_node_threshold(p, node_threshold)]
    if not filtered:
        raise FileNotFoundError(
            f"No prune-graph .pt files matched --node-threshold {node_threshold:g}. "
            "Expected a path directory like 'node_0.7' or 'node_inf_0.7_rel_0.7' under --input-path."
        )
    return filtered


def _default_input_paths() -> list[str]:
    candidates = ["eval_outputs/prune/subgraph/clt-hp"]
    existing = [path for path in candidates if Path(path).exists()]
    return existing or ["demos/eval_shap_prune"]


def _graph_identity(graph_path: Path, input_paths: Sequence[str]) -> tuple[str, str]:
    resolved_inputs = [
        Path(raw).expanduser().resolve() for raw in input_paths if Path(raw).expanduser().exists()
    ]
    for root in resolved_inputs:
        if root.is_file() and root == graph_path:
            return graph_path.stem, root.parent.name or "graphs"
        if root.is_dir():
            try:
                rel = graph_path.relative_to(root)
            except ValueError:
                continue
            dataset = root.name or "graphs"
            stem = _safe_slug(str(rel.with_suffix("")).replace("/", "__"))
            return stem, dataset
    return _safe_slug(graph_path.stem), graph_path.parent.name or "graphs"


def _token_normalization_from_path(graph_path: Path) -> str:
    for part in graph_path.parts:
        if part in _NORM_TOKENS:
            return part
    return "unknown"


# --- Per-partition evaluation -------------------------------------------------


def _evaluate_partition(
    *,
    prune_graph: PruneGraph,
    clusters: list[list[str]],
    method: str,
    theta_resolved: float,
    role_vectors_middle: np.ndarray,
    cos_middle: np.ndarray,
    middle_id_to_local: dict[str, int],
    random_state: int,
    graph_name: str,
    dataset: str,
    graph_path: Path,
    token_normalization: str,
    num_features: int,
    output_dir: Path,
) -> dict[str, Any]:
    rows = clusters_to_supernodes(prune_graph, clusters)
    sng = SummaryGraph(supernodes=rows, pruned_adj=prune_graph.pruned_adj)
    labels = _middle_labels_from_clusters(
        clusters, middle_id_to_local, role_vectors_middle.shape[0]
    )
    matched_k = int(labels.max() + 1) if labels.size and labels.max() >= 0 else 0

    gaps = compute_role_gaps(role_vectors_middle, labels, random_state=random_state)
    c_causal = compute_C_causal(sng)
    dag_loss = compute_dag_loss(sng)
    edge_mass = compute_edge_mass_metrics(sng)
    l_atom = compute_atomicity_loss(cos_middle, labels, theta_resolved)

    metrics: dict[str, Any] = {
        **gaps,
        "C_causal": float(c_causal),
        "dag_loss": float(dag_loss),
        "L_atom": float(l_atom),
        "matched_k": matched_k,
        "n_supernodes": int(len(rows)),
        "n_superedges": int(edge_mass["n_superedges"]),
        "raw_superedge_mass": float(edge_mass["raw_superedge_mass"]),
        "final_superedge_mass": float(edge_mass["final_superedge_mass"]),
        "dag_removed_mass_fraction": float(edge_mass["dag_removed_mass_fraction"]),
        "final_retained_mass_fraction": float(edge_mass["final_retained_mass_fraction"]),
        "theta": float(theta_resolved),
    }

    run_dir = output_dir / "runs" / graph_name / method
    supernode_map_path = run_dir / "supernode_map.json"
    result_path = run_dir / "result.json"
    supernode_map = {sn.name: sn.member_node_ids() for sn in rows}
    _write_json(supernode_map_path, supernode_map)
    _write_json(
        result_path,
        {
            "method": method,
            "graph_name": graph_name,
            "graph_path": str(graph_path),
            "metrics": metrics,
            "supernode_map_path": str(supernode_map_path),
        },
    )

    return {
        "graph_name": graph_name,
        "dataset": dataset,
        "graph_path": str(graph_path),
        "token_normalization": token_normalization,
        "num_features": num_features,
        "method": method,
        "supernode_map_path": str(supernode_map_path),
        "result_path": str(result_path),
        **metrics,
    }


def evaluate_prune_graph(
    *,
    graph_path: Path,
    input_paths: Sequence[str],
    output_dir: Path,
    map_location: str,
    max_layer_span: int,
    ilp_theta: float | str,
    ilp_eps_causal: float | None,
    ilp_max_sn: int | None,
    ilp_time_limit: float,
    random_state: int,
    n_init: int,
    theta_sweep: list[str],
) -> list[dict[str, Any]]:
    """Evaluate the ILP method and matched-K baselines on a single prune graph."""
    prune_graph = load_prune_graph(str(graph_path), map_location=map_location)
    graph_name, dataset = _graph_identity(graph_path, input_paths)
    token_normalization = _token_normalization_from_path(graph_path)

    mid_idx = _middle_indices(prune_graph)
    middle_ids = [prune_graph.nodes[i].node_id for i in mid_idx]
    middle_id_to_local = {nid: i for i, nid in enumerate(middle_ids)}
    num_features = len(mid_idx)

    phi = compute_phi_vectors(prune_graph).detach().cpu().numpy()  # (N, 2N)
    role_vectors_middle = phi[mid_idx]
    cos_middle = _pairwise_cosine(role_vectors_middle)  # signed role cosine over middle features
    adjacency_mid = _adjacency_affinity(prune_graph)[np.ix_(mid_idx, mid_idx)]

    # Resolved θ for L_atom diagnostics (graph-specific percentile of allowed-pair cosines).
    layers = np.array([layer_index_from_node(prune_graph.nodes[i]) for i in mid_idx])
    allowed_pairs = _allowed_pairs(layers, max_layer_span)
    theta_main = _resolve_theta(ilp_theta, cos_middle, allowed_pairs)

    logger.info(
        "  %s | dataset=%s norm=%s n_features=%d theta=%.4f",
        graph_name,
        dataset,
        token_normalization,
        num_features,
        theta_main,
    )

    ilp_clusters = cluster_graph_ilp(
        prune_graph,
        theta=ilp_theta,
        eps_causal=ilp_eps_causal,
        max_sn=ilp_max_sn,
        max_layer_span=max_layer_span,
        time_limit=ilp_time_limit,
    )
    ilp_feature_clusters = [c for c in ilp_clusters if c and c[0] in middle_id_to_local]
    matched_k = len(ilp_feature_clusters)
    ilp_cluster_sizes = [len(c) for c in ilp_feature_clusters]
    logger.info("  matched_k from ILP: %d", matched_k)

    def baseline(labels: np.ndarray) -> list[list[str]]:
        return labels_to_supernodes(prune_graph, middle_ids, labels)

    builders: list[tuple[str, float, Callable[[], list[list[str]]]]] = [
        ("ours-ilp", theta_main, lambda: ilp_clusters),
        (
            "baseline-spectral-cosine",
            theta_main,
            lambda: baseline(
                _spectral_cosine_middle_labels(role_vectors_middle, matched_k, random_state, n_init)
            ),
        ),
        (
            "baseline-kmeans",
            theta_main,
            lambda: baseline(
                _kmeans_middle_labels(role_vectors_middle, matched_k, random_state, n_init)
            ),
        ),
        (
            "baseline-spectral-adj",
            theta_main,
            lambda: baseline(
                _spectral_affinity_middle_labels(adjacency_mid, matched_k, random_state, n_init)
            ),
        ),
        (
            "baseline-random-same-size",
            theta_main,
            lambda: baseline(
                _random_same_size_middle_labels(
                    ilp_cluster_sizes, matched_k, len(middle_ids), random_state
                )
            ),
        ),
    ]

    # Optional θ-sensitivity rows: rerun ILP at each requested percentile (Ours only).
    for theta_str in theta_sweep:
        theta_val = _resolve_theta(theta_str, cos_middle, allowed_pairs)
        builders.append(
            (
                f"ours-ilp-{theta_str}",
                theta_val,
                (
                    lambda t=theta_str: cluster_graph_ilp(
                        prune_graph,
                        theta=t,
                        eps_causal=ilp_eps_causal,
                        max_sn=ilp_max_sn,
                        max_layer_span=max_layer_span,
                        time_limit=ilp_time_limit,
                    )
                ),
            )
        )

    rows: list[dict[str, Any]] = []
    for method, theta_resolved, build in builders:
        logger.info("  evaluating %s ...", method)
        rows.append(
            _evaluate_partition(
                prune_graph=prune_graph,
                clusters=build(),
                method=method,
                theta_resolved=theta_resolved,
                role_vectors_middle=role_vectors_middle,
                cos_middle=cos_middle,
                middle_id_to_local=middle_id_to_local,
                random_state=random_state,
                graph_name=graph_name,
                dataset=dataset,
                graph_path=graph_path,
                token_normalization=token_normalization,
                num_features=num_features,
                output_dir=output_dir,
            )
        )
    return rows


# --- Mean across graphs -------------------------------------------------------


def _mean_table(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Unweighted mean of each metric across graphs, grouped by method (plan final table)."""
    metric_keys = [
        "role_gap",
        "signed_up_gap",
        "signed_down_gap",
        "C_causal",
        "dag_loss",
        "L_atom",
    ]
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in summary_rows:
        by_method.setdefault(row["method"], []).append(row)

    table: list[dict[str, Any]] = []
    for method, rows in by_method.items():
        entry: dict[str, Any] = {"method": method, "n_graphs": len(rows)}
        for key in metric_keys:
            values = [float(r[key]) for r in rows if key in r and not _is_nan(r[key])]
            entry[key] = float(np.mean(values)) if values else float("nan")
        table.append(entry)
    table.sort(key=lambda e: (0 if e["method"].startswith("ours") else 1, e["method"]))
    return table


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and value != value


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    ilp_theta = _parse_ilp_theta(str(args.ilp_theta))
    theta_sweep = _parse_theta_sweep(getattr(args, "theta_sweep", None))
    graph_paths = _discover_prune_graphs(args.input_path, node_threshold=args.node_threshold)

    logger.info("Discovered %d prune graph(s); writing under %s", len(graph_paths), output_dir)
    if args.node_threshold is not None:
        logger.info("Node-threshold filter: %g", args.node_threshold)
    if theta_sweep:
        logger.info("Theta sweep (ILP-only extra rows): %s", ", ".join(theta_sweep))

    summary_rows: list[dict[str, Any]] = []
    n_graphs = len(graph_paths)
    for i, graph_path in enumerate(graph_paths, start=1):
        logger.info("Graph %d/%d: %s", i, n_graphs, graph_path)
        before = len(summary_rows)
        summary_rows.extend(
            evaluate_prune_graph(
                graph_path=graph_path,
                input_paths=args.input_path,
                output_dir=output_dir,
                map_location=args.map_location,
                max_layer_span=args.max_layer_span,
                ilp_theta=ilp_theta,
                ilp_eps_causal=args.ilp_eps_causal,
                ilp_max_sn=args.ilp_max_sn,
                ilp_time_limit=args.ilp_time_limit,
                random_state=args.random_state,
                n_init=args.n_init,
                theta_sweep=theta_sweep,
            )
        )
        logger.info(
            "Graph %d/%d finished: +%d rows (total %d)",
            i,
            n_graphs,
            len(summary_rows) - before,
            len(summary_rows),
        )

    mean_rows = _mean_table(summary_rows)

    logger.info("Writing summary.csv, method_means.csv, results.json, manifest.json")
    summary_path = output_dir / "summary.csv"
    means_path = output_dir / "method_means.csv"
    results_path = output_dir / "results.json"
    manifest_path = output_dir / "manifest.json"
    _write_csv(summary_path, SUMMARY_COLUMNS, summary_rows)
    _write_csv(means_path, MEAN_COLUMNS, mean_rows)
    _write_json(results_path, summary_rows)
    _write_json(
        manifest_path,
        {
            "input_paths": list(args.input_path),
            "node_threshold": args.node_threshold,
            "graph_paths": [str(p) for p in graph_paths],
            "output_dir": str(output_dir),
            "metrics": {
                "role_gap": "mean signed cos(r_i, r_j) within − across clusters (higher better)",
                "signed_up_gap": "same on upstream role v_in (higher better)",
                "signed_down_gap": "same on downstream role v_out (higher better)",
                "C_causal": (
                    "same-cluster feature edge mass / total retained pruned edge mass "
                    "(lower better)"
                ),
                "dag_loss": "backward-edge mass fraction Rσ over aggregated superedges (lower better)",
                "L_atom": "atomicity diagnostic Σ_same (θ − cos(r_i,r_j))",
            },
            "protocol": (
                "ILP runs once per graph; baselines run at the ILP's number of feature "
                "supernodes (matched-K). Per-graph metrics are averaged unweighted across "
                "graphs in method_means.csv; pairs are never pooled across graphs."
            ),
            "methods": METHODS,
            "theta_sweep": theta_sweep,
            "summary_csv": str(summary_path),
            "method_means_csv": str(means_path),
            "results_json": str(results_path),
            "n_graphs": n_graphs,
            "n_runs": len(summary_rows),
            "config": {
                "node_threshold": args.node_threshold,
                "max_layer_span": args.max_layer_span,
                "ilp_theta": ilp_theta,
                "ilp_eps_causal": args.ilp_eps_causal,
                "ilp_max_sn": args.ilp_max_sn,
                "ilp_time_limit": args.ilp_time_limit,
                "map_location": args.map_location,
                "random_state": args.random_state,
                "n_init": args.n_init,
                "diff_pair_sample_cap": DIFF_PAIR_SAMPLE_CAP,
                "eps": EPS,
            },
        },
    )
    logger.info("Done: %d graphs, %d runs total", n_graphs, len(summary_rows))
    return {
        "output_dir": str(output_dir),
        "summary_csv": str(summary_path),
        "method_means_csv": str(means_path),
        "results_json": str(results_path),
        "manifest_json": str(manifest_path),
        "n_graphs": n_graphs,
        "n_runs": len(summary_rows),
    }


def _parse_ilp_theta(raw: str) -> float | str:
    return raw if raw.startswith("p") else float(raw)


def _parse_theta_sweep(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [token.strip() for token in str(raw).split(",") if token.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate clustering methods on saved prune-graph .pt files: signed causal-role "
            "coherence gaps, C_causal, and DAG-loss, reported as the mean "
            "across graphs."
        )
    )
    parser.add_argument(
        "--input-path",
        action="append",
        default=[],
        help="File or directory with prune-graph .pt files. Repeatable; dirs searched recursively.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_outputs/clustering",
        help="Directory for summary.csv, method_means.csv, results.json, and per-run artifacts.",
    )
    parser.add_argument("--map-location", type=str, default="cuda")
    parser.add_argument(
        "--node-threshold",
        type=float,
        default=None,
        metavar="T",
        help="Only evaluate prune graphs whose path includes a threshold folder (e.g. 'node_0.7').",
    )
    parser.add_argument("--max-layer-span", type=int, default=4)
    parser.add_argument(
        "--ilp-theta",
        type=str,
        default=DEFAULT_THETA,
        help="ILP cosine resolution threshold. Float or percentile string like p65.",
    )
    parser.add_argument(
        "--theta-sweep",
        type=str,
        default=None,
        help="Optional comma list of θ for ILP-only sensitivity rows, e.g. 'p50,p65,p80'.",
    )
    parser.add_argument("--ilp-eps-causal", type=float, default=DEFAULT_EPS_CAUSAL)
    parser.add_argument("--ilp-max-sn", type=int, default=DEFAULT_MAX_SN)
    parser.add_argument("--ilp-time-limit", type=float, default=DEFAULT_TIME_LIMIT)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-init", type=int, default=20)
    return parser


def main() -> None:
    if not logging.root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
        )
    parser = build_parser()
    args = parser.parse_args()
    if not args.input_path:
        args.input_path = _default_input_paths()
    logger.info("=== Clustering evaluation (signed role gaps; C_causal; DAG loss) ===")
    result = run_evaluation(args)
    for key in ("output_dir", "summary_csv", "method_means_csv", "results_json", "manifest_json"):
        logger.info("%s: %s", key, result[key])
    logger.info("n_graphs: %s | n_runs: %s", result["n_graphs"], result["n_runs"])


if __name__ == "__main__":
    main()
