from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, Literal, Sequence, cast

import networkx as nx
import numpy as np
import torch
from sklearn.cluster import KMeans, SpectralClustering

from summarization.cluster_scoring import (
    _cv_cluster_sizes,
    _dag_interleave_edge_fraction,
    _internal_independence_score,
    _opposing_sign_fraction,
)
from summarization.cluster import (
    cluster_graph_spectral,
    cluster_graph_agglomerative,
    labels_to_supernodes,
    mapping_dict_to_supernodes,
    supernodes_to_mapping,
)
from summarization.prune import PruneGraph, load_prune_graph
from summarization.supernode_graph import SummarizationGraph
from summarization.utils import node_is_fixed, node_is_logit

logger = logging.getLogger(__name__)

METHOD_GRID: list[dict[str, str | float]] = [
    {
        "mean_method": mean_method,
        "decay_rate": decay_rate,
    }
    for mean_method in ("arith", "harm", "geo")
    for decay_rate in (i / 10.0 for i in range(11))
]

SUMMARY_COLUMNS = [
    "graph_name",
    "dataset",
    "graph_path",
    "num_nodes",
    "k_selection",
    "method",
    "method_family",
    "mean_method",
    "decay_rate",
    "best_k",
    "auto_k_candidates",
    "n_supernodes",
    "back_edge_fraction",
    "internal_independence",
    "cv_cluster_sizes",
    "size_entropy",
    "opposing_sign_frac",
    "out_dir_cosine_intra",
    "out_dir_cosine_inter",
    "out_dir_cosine_gap",
    "n_middle",
    "result_path",
    "supernode_map_path",
    "auto_k_sweep_path",
]


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(key): _to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(value) for value in obj]
    if isinstance(obj, tuple):
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


def _size_entropy_norm(rows: list) -> float:
    # H(size_dist) / log K, 1 = uniform sizes, 0 = degenerate. Features-type only.
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


def _output_direction_cosine(
    rows: list,
    prune_graph: PruneGraph,
    n_inter_samples: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Mean intra-cluster vs cross-cluster cosine of feature output vectors.

    Output vector for feature f = pruned_adj[logit_indices, f] (signed contribution to each
    logit node). This uses only the output-to-logit slice of the adjacency, so it is
    independent of the bidirectional edge profile that drives clustering.
    """
    nodes = prune_graph.nodes
    logit_indices = [i for i, n in enumerate(nodes) if node_is_logit(n)]
    if not logit_indices:
        return 0.0, 0.0

    out_vecs = (
        prune_graph.pruned_adj[logit_indices, :]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
        .T
    )  # [n_nodes, n_logit] — row f is feature f's output vector to each logit
    norms = np.linalg.norm(out_vecs, axis=1, keepdims=True)
    safe = np.where(norms > 1e-12, norms, 1.0)
    normed = out_vecs / safe
    has_output = (norms.squeeze(-1) > 1e-12)

    id_to_idx = {nid: i for i, nid in enumerate(prune_graph.node_ids)}
    intra_cosines: list[float] = []
    cluster_indices: list[list[int]] = []
    for row in rows:
        if row.type != "features":
            continue
        indices = [
            id_to_idx[nid]
            for nid in row.member_node_ids()
            if nid in id_to_idx and has_output[id_to_idx[nid]]
        ]
        cluster_indices.append(indices)
        if len(indices) < 2:
            continue
        sub = normed[indices]
        sim = sub @ sub.T
        mask = np.triu(np.ones_like(sim, dtype=bool), k=1)
        intra_cosines.extend(sim[mask].tolist())

    intra_mean = float(np.mean(intra_cosines)) if intra_cosines else 0.0

    valid = [c for c in cluster_indices if c]
    if len(valid) < 2:
        return intra_mean, 0.0
    rng = np.random.default_rng(seed)
    inter_cosines: list[float] = []
    for _ in range(n_inter_samples):
        i, j = rng.choice(len(valid), size=2, replace=False)
        a = valid[i][int(rng.integers(len(valid[i])))]
        b = valid[j][int(rng.integers(len(valid[j])))]
        inter_cosines.append(float(normed[a] @ normed[b]))
    inter_mean = float(np.mean(inter_cosines)) if inter_cosines else 0.0
    return intra_mean, inter_mean


def _score_independent(
    final_supernodes: dict[str, list[str]],
    prune_graph: PruneGraph,
) -> dict[str, Any]:
    """Compute similarity-free supernode metrics: back-edge fraction, internal independence,
    cluster-size CV, size entropy, opposing-sign fraction, output-direction cosine."""
    rows = mapping_dict_to_supernodes(prune_graph, final_supernodes)
    n_middle = sum(1 for r in rows if r.type == "features")
    if n_middle == 0:
        return {
            "back_edge_fraction": 0.0,
            "internal_independence": 0.0,
            "cv_cluster_sizes": 0.0,
            "size_entropy": 1.0,
            "opposing_sign_frac": 0.0,
            "out_dir_cosine_intra": 0.0,
            "out_dir_cosine_inter": 0.0,
            "out_dir_cosine_gap": 0.0,
            "n_middle": 0,
        }

    sng = SummarizationGraph(supernodes=rows, pruned_adj=prune_graph.pruned_adj)
    sn_adj = np.asarray(sng.sn_adj, dtype=np.float64)
    id_to_idx = {nid: i for i, nid in enumerate(prune_graph.node_ids)}
    intra_cos, inter_cos = _output_direction_cosine(rows, prune_graph)
    return {
        "back_edge_fraction": float(
            _dag_interleave_edge_fraction(sn_adj, list(sng.sn_names), rows)
        ),
        "internal_independence": float(
            _internal_independence_score(rows, prune_graph.pruned_adj, id_to_idx)
        ),
        "cv_cluster_sizes": float(_cv_cluster_sizes(rows)),
        "size_entropy": _size_entropy_norm(rows),
        "opposing_sign_frac": float(
            _opposing_sign_fraction(rows, prune_graph.pruned_adj, prune_graph.nodes)
        ),
        "out_dir_cosine_intra": float(intra_cos),
        "out_dir_cosine_inter": float(inter_cos),
        "out_dir_cosine_gap": float(intra_cos - inter_cos),
        "n_middle": int(n_middle),
    }


def _safe_slug(text: str) -> str:
    cleaned = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_"):
            cleaned.append(ch)
        else:
            cleaned.append("-")
    slug = "".join(cleaned).strip("-")
    return slug or "graph"


_RE_NODE_INF_REL = re.compile(
    r"^node_inf_(?P<inf>\d+(?:\.\d+)?)_rel_(?P<rel>\d+(?:\.\d+)?)$",
    re.IGNORECASE,
)
_RE_NODE_SIMPLE = re.compile(
    r"^node_(?P<v>\d+(?:\.\d+)?)$",
    re.IGNORECASE,
)


def _path_matches_node_threshold(
    graph_path: Path, threshold: float, *, eps: float = 1e-5
) -> bool:
    """True if a parent folder encodes this threshold (SHAP or simple layouts)."""

    def close(x: float) -> bool:
        return abs(x - threshold) <= eps

    for part in graph_path.parts:
        m = _RE_NODE_INF_REL.match(part)
        if m:
            inf = float(m.group("inf"))
            rel = float(m.group("rel"))
            if close(inf) and close(rel):
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
                sorted(
                    p.resolve()
                    for p in path.rglob("*_prune_graph.pt")
                    if p.is_file()
                )
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
            "Expected a path directory like 'node_0.7' or 'node_inf_0.7_rel_0.7' "
            "under --input-path."
        )
    return filtered


def _default_input_paths() -> list[str]:
    candidates = [
        "eval_outputs/prune/subgraph/clt-hp"
    ]
    existing = [path for path in candidates if Path(path).exists()]
    return existing or ["demos/eval_shap_prune"]


def _graph_identity(graph_path: Path, input_paths: Sequence[str]) -> tuple[str, str]:
    resolved_inputs = []
    for raw_path in input_paths:
        path = Path(raw_path).expanduser()
        if path.exists():
            resolved_inputs.append(path.resolve())

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


def _middle_indices(prune_graph: PruneGraph) -> list[int]:
    return [i for i, n in enumerate(prune_graph.nodes) if not node_is_fixed(n)]


def _fixed_k_from_num_nodes(n_middle: int, divisor: int) -> int:
    """Integer k from floor(num_nodes / divisor), clamped to [1, n_middle]."""
    if n_middle <= 0:
        return 1
    return max(1, min(n_middle // divisor, n_middle))


def _fixed_k_schedule(n_middle: int) -> list[tuple[str, int]]:
    """Two policies: k ≈ n/2 and k ≈ n/3 (middle-node count)."""
    return [
        ("n_over_2", _fixed_k_from_num_nodes(n_middle, 2)),
        ("n_over_3", _fixed_k_from_num_nodes(n_middle, 3)),
    ]


def _cosine_similarity(features: np.ndarray, nonnegative: bool = False) -> np.ndarray:
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


def _adjacency_affinity(prune_graph: PruneGraph) -> np.ndarray:
    adj_sender = prune_graph.pruned_adj.clone().float().T.detach().cpu().numpy()
    affinity = np.abs(adj_sender)
    affinity = (affinity + affinity.T) / 2.0
    max_val = float(affinity.max()) if affinity.size else 0.0
    if max_val > 0.0:
        affinity = affinity / max_val
    np.fill_diagonal(affinity, 1.0)
    return affinity


def _node_features_bidir(prune_graph: PruneGraph, mid_idx: list[int]) -> np.ndarray:
    # pruned_adj[target, source] — concat(pruned_adj[i,:], pruned_adj[:,i]) per middle index i
    adj = prune_graph.pruned_adj.clone().float().detach().cpu().numpy()
    n_tot = adj.shape[1]
    if not mid_idx:
        return np.zeros((0, 2 * n_tot), dtype=np.float64)
    idx = np.array(mid_idx, dtype=np.intp)
    incoming = adj[idx, :]
    outgoing = adj[:, idx].T
    return np.concatenate([incoming, outgoing], axis=1)


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
    return KMeans(n_clusters=k, random_state=random_state, n_init=n_init).fit_predict(features).astype(np.int64)


def _spectral_rbf_middle_labels(
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
    return SpectralClustering(
        n_clusters=k,
        affinity="rbf",
        assign_labels="kmeans",
        random_state=random_state,
        n_init=n_init,  # type: ignore[arg-type]
    ).fit_predict(features).astype(np.int64)


def _modularity_middle_labels(
    adjacency_mid: np.ndarray, target_k: int
) -> np.ndarray:
    """K-matched modularity baseline via networkx.greedy_modularity_communities(best_n=K)."""
    n = adjacency_mid.shape[0]
    if n == 0:
        return np.array([], dtype=np.int64)
    k = max(1, min(target_k, n))
    if k == 1:
        return np.zeros(n, dtype=np.int64)
    if k == n:
        return np.arange(n, dtype=np.int64)
    graph = nx.from_numpy_array(adjacency_mid)
    communities = nx.community.greedy_modularity_communities(
        graph, weight="weight", best_n=k
    )
    labels = np.zeros(n, dtype=np.int64)
    for label, community in enumerate(communities):
        for idx in community:
            labels[int(idx)] = int(label)
    return labels


def _flatten_metrics(
    *,
    graph_name: str,
    dataset: str,
    graph_path: Path,
    num_nodes: int,
    k_selection: str,
    method: str,
    method_family: str,
    mean_method: str | None,
    decay_rate: float | None,
    best_k: int,
    auto_k_candidates: int,
    final_supernodes: dict[str, list[str]],
    base_score: dict[str, Any],
    result_path: Path,
    supernode_map_path: Path,
    auto_k_sweep_path: Path | None,
) -> dict[str, Any]:
    row = {
        "graph_name": graph_name,
        "dataset": dataset,
        "graph_path": str(graph_path),
        "num_nodes": num_nodes,
        "k_selection": k_selection,
        "method": method,
        "method_family": method_family,
        "mean_method": mean_method,
        "decay_rate": decay_rate,
        "best_k": best_k,
        "auto_k_candidates": auto_k_candidates,
        "n_supernodes": len(final_supernodes),
        "back_edge_fraction": base_score.get("back_edge_fraction"),
        "internal_independence": base_score.get("internal_independence"),
        "cv_cluster_sizes": base_score.get("cv_cluster_sizes"),
        "size_entropy": base_score.get("size_entropy"),
        "opposing_sign_frac": base_score.get("opposing_sign_frac"),
        "out_dir_cosine_intra": base_score.get("out_dir_cosine_intra"),
        "out_dir_cosine_inter": base_score.get("out_dir_cosine_inter"),
        "out_dir_cosine_gap": base_score.get("out_dir_cosine_gap"),
        "n_middle": base_score.get("n_middle"),
        "result_path": str(result_path),
        "supernode_map_path": str(supernode_map_path),
        "auto_k_sweep_path": str(auto_k_sweep_path) if auto_k_sweep_path else "",
    }
    return row


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in SUMMARY_COLUMNS})


def _evaluate_ours_fixed_k(
    *,
    prune_graph: PruneGraph,
    graph_path: Path,
    graph_name: str,
    dataset: str,
    output_dir: Path,
    method_config: dict[str, str | float],
    target_k: int,
    k_selection: str,
    num_nodes: int,
    max_layer_span: int,
    random_state: int,
    n_init: int,
    enforce_dag: bool,
) -> dict[str, Any]:
    mean_method = cast(Literal["geo", "harm", "arith"], method_config["mean_method"])
    decay_rate = float(cast(float, method_config["decay_rate"]))
    clusters = cluster_graph_spectral(
        prune_graph,
        target_k=target_k,
        max_layer_span=max_layer_span,
        mean_method=mean_method,
        decay_rate=decay_rate,
        enforce_dag=enforce_dag,
        random_state=random_state,
        n_init=n_init,
    )
    final_supernodes = supernodes_to_mapping(prune_graph, clusters)
    base_score = _score_independent(final_supernodes, prune_graph)

    method_slug = f"ours-{method_config['mean_method']}-decay-{decay_rate:.1f}"
    run_dir = output_dir / "runs" / graph_name / method_slug / k_selection
    supernode_map_path = run_dir / "supernode_map.json"
    auto_k_sweep_path = run_dir / "fixed_k_metrics.json"
    result_path = run_dir / "result.json"

    _write_json(supernode_map_path, final_supernodes)
    _write_json(
        auto_k_sweep_path,
        {
            "k_selection": k_selection,
            "target_k": target_k,
            "num_nodes": num_nodes,
            "metrics": {key: value for key, value in base_score.items()},
        },
    )

    summary_row = _flatten_metrics(
        graph_name=graph_name,
        dataset=dataset,
        graph_path=graph_path,
        num_nodes=num_nodes,
        k_selection=k_selection,
        method=method_slug,
        method_family="ours",
        mean_method=str(method_config["mean_method"]),
        decay_rate=decay_rate,
        best_k=target_k,
        auto_k_candidates=0,
        final_supernodes=final_supernodes,
        base_score=base_score,
        result_path=result_path,
        supernode_map_path=supernode_map_path,
        auto_k_sweep_path=auto_k_sweep_path,
    )
    result_payload = {
        **summary_row,
        "final_supernodes": final_supernodes,
        "score_details": base_score.get("details", {}),
    }
    _write_json(result_path, result_payload)
    return result_payload


def _evaluate_ours_agglomerative_fixed_k(
    *,
    prune_graph: PruneGraph,
    graph_path: Path,
    graph_name: str,
    dataset: str,
    output_dir: Path,
    method_config: dict[str, str | float],
    target_k: int,
    k_selection: str,
    num_nodes: int,
    max_layer_span: int,
    enforce_dag: bool,
) -> dict[str, Any]:
    mean_method = cast(Literal["geo", "harm", "arith"], method_config["mean_method"])
    decay_rate = float(cast(float, method_config["decay_rate"]))
    clusters = cluster_graph_agglomerative(
        prune_graph,
        target_k=target_k,
        max_layer_span=max_layer_span,
        mean_method=mean_method,
        decay_rate=decay_rate,
    )
    final_supernodes = supernodes_to_mapping(prune_graph, clusters)
    base_score = _score_independent(final_supernodes, prune_graph)

    method_slug = f"ours-agglomerative-{method_config['mean_method']}-decay-{decay_rate:.1f}"
    run_dir = output_dir / "runs" / graph_name / method_slug / k_selection
    supernode_map_path = run_dir / "supernode_map.json"
    auto_k_sweep_path = run_dir / "fixed_k_metrics.json"
    result_path = run_dir / "result.json"

    _write_json(supernode_map_path, final_supernodes)
    _write_json(
        auto_k_sweep_path,
        {
            "k_selection": k_selection,
            "target_k": target_k,
            "num_nodes": num_nodes,
            "metrics": {key: value for key, value in base_score.items()},
        },
    )

    summary_row = _flatten_metrics(
        graph_name=graph_name,
        dataset=dataset,
        graph_path=graph_path,
        num_nodes=num_nodes,
        k_selection=k_selection,
        method=method_slug,
        method_family="ours-agglomerative",
        mean_method=str(method_config["mean_method"]),
        decay_rate=decay_rate,
        best_k=target_k,
        auto_k_candidates=0,
        final_supernodes=final_supernodes,
        base_score=base_score,
        result_path=result_path,
        supernode_map_path=supernode_map_path,
        auto_k_sweep_path=auto_k_sweep_path,
    )
    result_payload = {
        **summary_row,
        "final_supernodes": final_supernodes,
        "score_details": base_score.get("details", {}),
    }
    _write_json(result_path, result_payload)
    return result_payload


def _evaluate_baseline_fixed_k(
    *,
    prune_graph: PruneGraph,
    graph_path: Path,
    graph_name: str,
    dataset: str,
    output_dir: Path,
    method: str,
    clusterer: Callable[[int], list[list[str]]],
    target_k: int,
    k_selection: str,
    num_nodes: int,
    enforce_dag: bool,
) -> dict[str, Any]:
    clusters = clusterer(target_k)
    final_supernodes = supernodes_to_mapping(prune_graph, clusters)
    base_score = _score_independent(final_supernodes, prune_graph)

    run_dir = output_dir / "runs" / graph_name / method / k_selection
    supernode_map_path = run_dir / "supernode_map.json"
    auto_k_sweep_path = run_dir / "fixed_k_metrics.json"
    result_path = run_dir / "result.json"

    _write_json(supernode_map_path, final_supernodes)
    _write_json(
        auto_k_sweep_path,
        {
            "k_selection": k_selection,
            "target_k": target_k,
            "num_nodes": num_nodes,
            "metrics": {key: value for key, value in base_score.items()},
        },
    )

    summary_row = _flatten_metrics(
        graph_name=graph_name,
        dataset=dataset,
        graph_path=graph_path,
        num_nodes=num_nodes,
        k_selection=k_selection,
        method=method,
        method_family="baseline",
        mean_method=None,
        decay_rate=None,
        best_k=target_k,
        auto_k_candidates=0,
        final_supernodes=final_supernodes,
        base_score=base_score,
        result_path=result_path,
        supernode_map_path=supernode_map_path,
        auto_k_sweep_path=auto_k_sweep_path,
    )
    result_payload = {
        **summary_row,
        "final_supernodes": final_supernodes,
        "score_details": base_score.get("details", {}),
    }
    _write_json(result_path, result_payload)
    return result_payload


def evaluate_prune_graph(
    *,
    graph_path: Path,
    input_paths: Sequence[str],
    output_dir: Path,
    map_location: str,
    max_layer_span: int,
    random_state: int,
    n_init: int,
    enforce_dag: bool,
) -> list[dict[str, Any]]:
    prune_graph = load_prune_graph(str(graph_path), map_location=map_location)
    graph_name, dataset = _graph_identity(graph_path, input_paths)
    rows: list[dict[str, Any]] = []

    num_nodes = len(_middle_indices(prune_graph))
    k_schedule = _fixed_k_schedule(num_nodes)
    n_ours_grid = len(METHOD_GRID) * len(k_schedule)
    logger.info(
        "  %s | dataset=%s n_middle=%d k_schedule=%s",
        graph_name,
        dataset,
        num_nodes,
        [(sel, k) for sel, k in k_schedule],
    )

    logger.info("  ours spectral: %d runs (method grid × k schedule)", n_ours_grid)
    for method_config in METHOD_GRID:
        for k_selection, target_k in k_schedule:
            rows.append(
                _evaluate_ours_fixed_k(
                    prune_graph=prune_graph,
                    graph_path=graph_path,
                    graph_name=graph_name,
                    dataset=dataset,
                    output_dir=output_dir,
                    method_config=method_config,
                    target_k=target_k,
                    k_selection=k_selection,
                    num_nodes=num_nodes,
                    max_layer_span=max_layer_span,
                    random_state=random_state,
                    n_init=n_init,
                    enforce_dag=enforce_dag,
                )
            )

    logger.info("  ours agglomerative: %d runs", n_ours_grid)
    for method_config in METHOD_GRID:
        for k_selection, target_k in k_schedule:
            rows.append(
                _evaluate_ours_agglomerative_fixed_k(
                    prune_graph=prune_graph,
                    graph_path=graph_path,
                    graph_name=graph_name,
                    dataset=dataset,
                    output_dir=output_dir,
                    method_config=method_config,
                    target_k=target_k,
                    k_selection=k_selection,
                    num_nodes=num_nodes,
                    max_layer_span=max_layer_span,
                    enforce_dag=enforce_dag,
                )
            )

    logger.info(
        "  baselines: modularity + spectral-rbf + kmeans (%d k each)",
        len(k_schedule),
    )
    mid_idx = _middle_indices(prune_graph)
    middle_ids = [prune_graph.nodes[i].node_id for i in mid_idx]
    adjacency_mid = _adjacency_affinity(prune_graph)[np.ix_(mid_idx, mid_idx)]
    features_bidir = _node_features_bidir(prune_graph, mid_idx)

    for k_selection, target_k in k_schedule:
        rows.append(
            _evaluate_baseline_fixed_k(
                prune_graph=prune_graph,
                graph_path=graph_path,
                graph_name=graph_name,
                dataset=dataset,
                output_dir=output_dir,
                method="baseline-modularity",
                clusterer=lambda k=target_k: labels_to_supernodes(
                    prune_graph,
                    middle_ids,
                    _modularity_middle_labels(adjacency_mid, k),
                ),
                target_k=target_k,
                k_selection=k_selection,
                num_nodes=num_nodes,
                enforce_dag=enforce_dag,
            )
        )
        rows.append(
            _evaluate_baseline_fixed_k(
                prune_graph=prune_graph,
                graph_path=graph_path,
                graph_name=graph_name,
                dataset=dataset,
                output_dir=output_dir,
                method="baseline-spectral-rbf",
                clusterer=lambda k=target_k: labels_to_supernodes(
                    prune_graph,
                    middle_ids,
                    _spectral_rbf_middle_labels(features_bidir, k, random_state, n_init),
                ),
                target_k=target_k,
                k_selection=k_selection,
                num_nodes=num_nodes,
                enforce_dag=enforce_dag,
            )
        )
        rows.append(
            _evaluate_baseline_fixed_k(
                prune_graph=prune_graph,
                graph_path=graph_path,
                graph_name=graph_name,
                dataset=dataset,
                output_dir=output_dir,
                method="baseline-kmeans",
                clusterer=lambda k=target_k: labels_to_supernodes(
                    prune_graph,
                    middle_ids,
                    _kmeans_middle_labels(features_bidir, k, random_state, n_init),
                ),
                target_k=target_k,
                k_selection=k_selection,
                num_nodes=num_nodes,
                enforce_dag=enforce_dag,
            )
        )
    return rows


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    graph_paths = _discover_prune_graphs(
        args.input_path,
        node_threshold=args.node_threshold,
    )
    logger.info(
        "Discovered %d prune graph(s); writing under %s",
        len(graph_paths),
        output_dir,
    )
    if args.node_threshold is not None:
        logger.info("Node-threshold filter: %g", args.node_threshold)

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
                random_state=args.random_state,
                n_init=args.n_init,
                enforce_dag=args.enforce_dag,
            )
        )
        logger.info(
            "Graph %d/%d finished: +%d rows (total %d)",
            i,
            n_graphs,
            len(summary_rows) - before,
            len(summary_rows),
        )

    logger.info("Writing summary.csv, results.json, manifest.json")
    summary_path = output_dir / "summary.csv"
    results_path = output_dir / "results.json"
    manifest_path = output_dir / "manifest.json"
    _write_summary_csv(summary_path, summary_rows)
    _write_json(results_path, summary_rows)
    _write_json(
        manifest_path,
        {
            "input_paths": list(args.input_path),
            "node_threshold": args.node_threshold,
            "graph_paths": [str(path) for path in graph_paths],
            "output_dir": str(output_dir),
            "method_grid": METHOD_GRID,
            "baselines": [
                "baseline-modularity",
                "baseline-spectral-rbf",
                "baseline-kmeans",
            ],
            "cluster_k_policy": (
                "Middle-node count n = |non-fixed nodes|. Ours and all baselines run at "
                "k = round(n/2) and k = round(n/3), clamped to [1, n]. Modularity uses "
                "networkx.greedy_modularity_communities(best_n=k) for K-matched comparison."
            ),
            "summary_csv": str(summary_path),
            "results_json": str(results_path),
            "n_graphs": len(graph_paths),
            "n_runs": len(summary_rows),
            "config": {
                "node_threshold": args.node_threshold,
                "max_layer_span": args.max_layer_span,
                "enforce_dag": args.enforce_dag,
                "map_location": args.map_location,
                "random_state": args.random_state,
                "n_init": args.n_init,
            },
        },
    )
    logger.info("Done: %d graphs, %d runs total", len(graph_paths), len(summary_rows))
    return {
        "output_dir": str(output_dir),
        "summary_csv": str(summary_path),
        "results_json": str(results_path),
        "manifest_json": str(manifest_path),
        "n_graphs": len(graph_paths),
        "n_runs": len(summary_rows),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate clustering variants and simple baselines on saved prune-graph .pt files."
        )
    )
    parser.add_argument(
        "--input-path",
        action="append",
        default=[],
        help=(
            "File or directory containing prune-graph .pt files. "
            "May be repeated; directories are searched recursively."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_outputs/clustering",
        help="Directory where summary CSV/JSON and per-run artifacts will be saved.",
    )
    parser.add_argument("--map-location", type=str, default="cpu")
    parser.add_argument(
        "--node-threshold",
        type=float,
        default=None,
        metavar="T",
        help=(
            "Only evaluate prune graphs whose path includes a threshold folder, e.g. "
            "'node_0.7' (…\\\\entmax15\\\\node_0.7\\\\…) or SHAP-style "
            "'node_inf_0.7_rel_0.7'. Omit to use every discovered *_prune_graph.pt."
        ),
    )
    parser.add_argument("--max-layer-span", type=int, default=4)
    parser.add_argument(
        "--enforce-dag",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Apply DAG constraints in our spectral clustering. "
            "Default on (matches prior hardcoded eval). Use --no-enforce-dag to disable."
        ),
    )
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
    logger.info(
        "=== Clustering evaluation (ours grid + modularity / spectral-rbf / kmeans) ==="
    )
    result = run_evaluation(args)
    logger.info("output_dir: %s", result["output_dir"])
    logger.info("summary_csv: %s", result["summary_csv"])
    logger.info("results_json: %s", result["results_json"])
    logger.info("manifest_json: %s", result["manifest_json"])
    logger.info("n_graphs: %s", result["n_graphs"])
    logger.info("n_runs: %s", result["n_runs"])


if __name__ == "__main__":
    main()
