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
import networkx as nx
from sklearn.cluster import KMeans, SpectralClustering

from eval.eval_prune import compute_prune_loss
from summarization.attr_graph import AttrGraph
from summarization.cluster import (
    clusters_to_supernodes,
    compute_phi_vectors,
    labels_to_supernodes,
)
from summarization.ilp_cluster import (
    DEFAULT_EPS_CAUSAL,
    DEFAULT_MAX_SN,
    DEFAULT_THETA,
    DEFAULT_TIME_LIMIT,
    cluster_graph_ilp,
)
from summarization.scoring import _cosine_similarity, compute_L, silhouette_from_features
from summarization.prune import PruneGraph, load_prune_graph
from summarization.summarize import SummaryGraph
from summarization.utils import node_is_fixed

logger = logging.getLogger(__name__)

_NORM_TOKENS = {"softmax", "entmax", "sparsemax", "entmax15"}

SUMMARY_COLUMNS = [
    "graph_name",
    "dataset",
    "graph_path",
    "token_normalization",
    "num_nodes",
    "solver",
    "matched_k",
    "n_supernodes",
    "n_superedges",
    "L",
    "L_atom",
    "L_atom_norm",
    "sil_raw",
    "sil_norm",
    "L_causal",
    "internalized_mass_fraction",
    "dag_removed_mass_fraction",
    "final_retained_mass_fraction",
    "raw_superedge_mass",
    "final_superedge_mass",
    "total_fine_edge_mass",
    "prune_loss",
    "supernode_map_path",
    "result_path",
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
    candidates = ["eval_outputs/prune/subgraph/clt-hp"]
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


def _token_normalization_from_path(graph_path: Path) -> str:
    """Pull softmax/entmax/... from the prune-graphs output layout."""
    for part in graph_path.parts:
        if part in _NORM_TOKENS:
            return part
    return "unknown"


def _find_prune_graphs_manifest(graph_path: Path) -> Path | None:
    """Walk up from graph_path looking for the prune_graphs manifest.json."""
    for parent in graph_path.parents:
        candidate = parent / "manifest.json"
        if candidate.is_file():
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                continue
            # prune_graphs manifest carries these two keys; reject other manifests.
            if isinstance(payload, dict) and "results_json" in payload and "summary_csv" in payload:
                return candidate
    return None


def _lookup_prune_manifest_row(
    manifest_path: Path, graph_path: Path
) -> dict[str, Any] | None:
    """Find the results.json row whose prune_graph_path matches graph_path."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results_json = Path(manifest.get("results_json", ""))
    if not results_json.is_file():
        results_json = manifest_path.parent / "results.json"
    if not results_json.is_file():
        return None
    rows = json.loads(results_json.read_text(encoding="utf-8"))
    target = graph_path.resolve()
    target_name = target.name
    for row in rows:
        row_path = row.get("prune_graph_path")
        if not row_path:
            continue
        candidate = Path(row_path)
        # Resolve relative paths relative to repo root (parent of eval_outputs).
        if not candidate.is_absolute():
            candidate = (manifest_path.parents[-1] / candidate).resolve()
            # If that doesn't exist, also try resolving against the manifest dir.
            if not candidate.exists():
                candidate = (manifest_path.parent / candidate).resolve()
        else:
            candidate = candidate.resolve()
        if candidate == target or (candidate.name == target_name and str(candidate).endswith(str(target).split("eval_outputs", 1)[-1])):
            return row
    return None


def _compute_prune_loss_for_graph(
    *,
    graph_path: Path,
    prune_graph: PruneGraph,
    map_location: str,
) -> float:
    """Look up the original AttrGraph + token_weights via the prune_graphs
    manifest, then compute 1 - token_attribution_faithfulness. Returns 0.0
    if the manifest can't be found or the wiring fails (with a warning)."""
    manifest = _find_prune_graphs_manifest(graph_path)
    if manifest is None:
        logger.warning(
            "Could not find prune_graphs manifest for %s; prune_loss=0.0", graph_path
        )
        return 0.0
    row = _lookup_prune_manifest_row(manifest, graph_path)
    if row is None:
        logger.warning(
            "Manifest %s has no row for %s; prune_loss=0.0", manifest, graph_path
        )
        return 0.0
    orig_graph_path = row.get("graph_path")
    token_weights = row.get("token_weights")
    if not orig_graph_path or not token_weights:
        logger.warning(
            "Manifest row for %s missing graph_path/token_weights; prune_loss=0.0",
            graph_path,
        )
        return 0.0
    try:
        attr_graph = AttrGraph.from_graph(str(orig_graph_path))
    except Exception as exc:
        logger.warning("AttrGraph.from_graph failed for %s: %s", orig_graph_path, exc)
        return 0.0
    device = "cpu" if map_location == "cpu" else "cuda"
    try:
        return compute_prune_loss(
            attr_graph,
            prune_graph,
            [float(w) for w in token_weights],
            device=device,
        )
    except Exception as exc:
        logger.warning("compute_prune_loss failed for %s: %s", graph_path, exc)
        return 0.0


def _middle_indices(prune_graph: PruneGraph) -> list[int]:
    return [i for i, n in enumerate(prune_graph.nodes) if not node_is_fixed(n)]


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
    # Spherical K-means: unit-normalise r(u) so Euclidean distance = cosine distance,
    # making this a direct surrogate of L_atom.
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features_normed = features / np.where(norms > 1e-12, norms, 1.0)
    return (
        KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
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
    # Rectified cosine affinity matches the cos+ in L_atom exactly.
    affinity = _cosine_similarity(features, nonnegative=True)
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


def _modularity_middle_labels(affinity: np.ndarray, target_k: int) -> np.ndarray:
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
    np.fill_diagonal(safe_affinity, 0.0)

    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    rows, cols = np.triu_indices(n, k=1)
    for i, j in zip(rows, cols, strict=True):
        weight = float(safe_affinity[i, j])
        if weight > 0.0:
            graph.add_edge(int(i), int(j), weight=weight)

    if graph.number_of_edges() == 0:
        return np.arange(n, dtype=np.int64) % k

    communities = nx.community.greedy_modularity_communities(
        graph,
        weight="weight",
        cutoff=k,
        best_n=k,
    )
    labels = np.empty(n, dtype=np.int64)
    for label, community in enumerate(communities):
        labels[list(community)] = label
    return labels


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


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in SUMMARY_COLUMNS})


def _evaluate_partition(
    *,
    prune_graph: PruneGraph,
    graph_path: Path,
    graph_name: str,
    dataset: str,
    token_normalization: str,
    output_dir: Path,
    solver: str,
    clusters: list[list[str]],
    matched_k: int,
    role_vectors_all: np.ndarray,
    role_vectors_middle: np.ndarray,
    middle_id_to_local: dict[str, int],
    num_nodes: int,
    prune_loss: float,
    lambda_causal: float,
) -> dict[str, Any]:
    rows = clusters_to_supernodes(prune_graph, clusters)
    sng = SummaryGraph(supernodes=rows, pruned_adj=prune_graph.pruned_adj)
    metrics = compute_L(
        sng,
        role_vectors_middle,
        middle_id_to_local,
        prune_loss=prune_loss,
        lambda_causal=lambda_causal,
    )
    sil_raw, sil_norm = silhouette_from_features(role_vectors_all, prune_graph, rows)
    metrics = {
        **metrics,
        "sil_raw": float(sil_raw),
        "sil_norm": float(sil_norm),
    }
    supernode_map = {sn.name: sn.member_node_ids() for sn in rows}
    run_dir = output_dir / "runs" / graph_name / solver
    supernode_map_path = run_dir / "supernode_map.json"
    result_path = run_dir / "result.json"

    _write_json(supernode_map_path, supernode_map)
    _write_json(
        result_path,
        {
            "solver": solver,
            "graph_name": graph_name,
            "graph_path": str(graph_path),
            "matched_k": int(matched_k),
            "metrics": metrics,
            "supernode_map_path": str(supernode_map_path),
        },
    )

    summary: dict[str, Any] = {
        "graph_name": graph_name,
        "dataset": dataset,
        "graph_path": str(graph_path),
        "token_normalization": token_normalization,
        "num_nodes": num_nodes,
        "solver": solver,
        "matched_k": int(matched_k),
        "supernode_map_path": str(supernode_map_path),
        "result_path": str(result_path),
        **metrics,
    }
    return summary


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
    lambda_causal: float,
) -> list[dict[str, Any]]:
    """Evaluate ILP and K-matched baselines on a single prune graph."""
    prune_graph = load_prune_graph(str(graph_path), map_location=map_location)
    graph_name, dataset = _graph_identity(graph_path, input_paths)
    token_normalization = _token_normalization_from_path(graph_path)

    mid_idx = _middle_indices(prune_graph)
    middle_ids = [prune_graph.nodes[i].node_id for i in mid_idx]
    middle_id_to_local = {nid: i for i, nid in enumerate(middle_ids)}
    num_nodes = len(mid_idx)

    phi = compute_phi_vectors(prune_graph).detach().cpu().numpy()
    role_vectors_middle = phi[mid_idx]
    phi_mid = role_vectors_middle  # same array, distinct name for baseline clustering input
    adjacency_mid = _adjacency_affinity(prune_graph)[np.ix_(mid_idx, mid_idx)]

    prune_loss = _compute_prune_loss_for_graph(
        graph_path=graph_path,
        prune_graph=prune_graph,
        map_location=map_location,
    )

    logger.info(
        "  %s | dataset=%s norm=%s n_middle=%d prune_loss=%.4f",
        graph_name,
        dataset,
        token_normalization,
        num_nodes,
        prune_loss,
    )

    ilp_clusters = cluster_graph_ilp(
        prune_graph,
        theta=ilp_theta,
        eps_causal=ilp_eps_causal,
        max_sn=ilp_max_sn,
        max_layer_span=max_layer_span,
        time_limit=ilp_time_limit,
    )
    ilp_feature_clusters = [
        cluster
        for cluster in ilp_clusters
        if cluster and cluster[0] in middle_id_to_local
    ]
    matched_k = len(ilp_feature_clusters)
    ilp_cluster_sizes = [len(cluster) for cluster in ilp_feature_clusters]

    logger.info("  matched_k from ILP: %d", matched_k)

    def baseline_from_labels(labels: np.ndarray) -> list[list[str]]:
        return labels_to_supernodes(prune_graph, middle_ids, labels)

    baseline_builders: list[tuple[str, Callable[[], list[list[str]]]]] = [
        ("ours-ilp", lambda: ilp_clusters),
        (
            "baseline-spectral-cosine",
            lambda: baseline_from_labels(
                _spectral_cosine_middle_labels(phi_mid, matched_k, random_state, n_init)
            ),
        ),
        (
            "baseline-kmeans",
            lambda: baseline_from_labels(
                _kmeans_middle_labels(phi_mid, matched_k, random_state, n_init)
            ),
        ),
        (
            "baseline-spectral-adj",
            lambda: baseline_from_labels(
                _spectral_affinity_middle_labels(adjacency_mid, matched_k, random_state, n_init)
            ),
        ),
        (
            "baseline-random-same-size",
            lambda: baseline_from_labels(
                _random_same_size_middle_labels(
                    ilp_cluster_sizes,
                    matched_k,
                    len(middle_ids),
                    random_state,
                )
            ),
        ),
    ]

    rows: list[dict[str, Any]] = []
    for solver_name, build_clusters in baseline_builders:
        logger.info("  solving %s ...", solver_name)
        clusters = build_clusters()
        rows.append(
            _evaluate_partition(
                prune_graph=prune_graph,
                graph_path=graph_path,
                graph_name=graph_name,
                dataset=dataset,
                token_normalization=token_normalization,
                output_dir=output_dir,
                solver=solver_name,
                clusters=clusters,
                matched_k=matched_k,
                role_vectors_all=phi,
                role_vectors_middle=role_vectors_middle,
                middle_id_to_local=middle_id_to_local,
                num_nodes=num_nodes,
                prune_loss=prune_loss,
                lambda_causal=lambda_causal,
            )
        )
    return rows


def _validate_lambda_causal(lambda_causal: float) -> float:
    if lambda_causal < 0.0:
        raise ValueError(f"--lambda-causal must be non-negative, got {lambda_causal}")
    return lambda_causal


def _parse_ilp_theta(raw: str) -> float | str:
    if raw.startswith("p"):
        return raw
    return float(raw)


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    lambda_causal = _validate_lambda_causal(args.lambda_causal)
    args.ilp_theta = _parse_ilp_theta(str(args.ilp_theta))
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
    logger.info("lambda_causal: %.4f", lambda_causal)

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
                ilp_theta=args.ilp_theta,
                ilp_eps_causal=args.ilp_eps_causal,
                ilp_max_sn=args.ilp_max_sn,
                ilp_time_limit=args.ilp_time_limit,
                random_state=args.random_state,
                n_init=args.n_init,
                lambda_causal=lambda_causal,
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
            "objective": (
                "Matched-K ILP cluster evaluation. L = (L_atom_norm + lambda_causal "
                "* L_causal) / (1 + lambda_causal), where L_causal is the fraction "
                "of fine edge mass internalized inside feature supernodes. Additional "
                "mass metrics report raw external superedge mass, final DAG-retained "
                "mass, and mass removed by π DAG construction."
            ),
            "selection_protocol": (
                "ILP runs once per graph. Baselines run at the ILP's actual number "
                "of feature supernodes, using the same fixed-node singleton policy "
                "and the same SummaryGraph π DAG construction."
            ),
            "lambdas": {
                "lambda_causal": lambda_causal,
            },
            "solvers": [
                "ours-ilp",
                "baseline-spectral-cosine",
                "baseline-kmeans",
                "baseline-spectral-adj",
                "baseline-random-same-size",
            ],
            "summary_csv": str(summary_path),
            "results_json": str(results_path),
            "n_graphs": len(graph_paths),
            "n_runs": len(summary_rows),
            "config": {
                "node_threshold": args.node_threshold,
                "max_layer_span": args.max_layer_span,
                "ilp_theta": args.ilp_theta,
                "ilp_eps_causal": args.ilp_eps_causal,
                "ilp_max_sn": args.ilp_max_sn,
                "ilp_time_limit": args.ilp_time_limit,
                "map_location": args.map_location,
                "random_state": args.random_state,
                "n_init": args.n_init,
                "lambda_causal": lambda_causal,
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
            "Evaluate clustering solvers on saved prune-graph .pt files under the "
            "matched-K ILP protocol, reporting role-vector silhouette, internalized "
            "edge mass, and DAG-construction mass loss."
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
    parser.add_argument("--map-location", type=str, default="cuda")
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
        "--ilp-theta",
        type=str,
        default=DEFAULT_THETA,
        help="ILP cosine resolution threshold. Use a float or percentile string like p65.",
    )
    parser.add_argument(
        "--ilp-eps-causal",
        type=float,
        default=DEFAULT_EPS_CAUSAL,
        help="ILP hard budget on internalized causal mass fraction. Use default p65 setup.",
    )
    parser.add_argument(
        "--ilp-max-sn",
        type=int,
        default=DEFAULT_MAX_SN,
        help="Maximum number of ILP feature supernodes.",
    )
    parser.add_argument(
        "--ilp-time-limit",
        type=float,
        default=DEFAULT_TIME_LIMIT,
        help="HiGHS ILP time limit in seconds per graph.",
    )
    parser.add_argument(
        "--enforce-dag",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Deprecated no-op. SummaryGraph always applies π DAG construction uniformly "
            "for every solver."
        ),
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-init", type=int, default=20)
    parser.add_argument(
        "--lambda-causal",
        type=float,
        default=1.0,
        help="Trade-off weight (>= 0) on L_causal in L = L_atom + lambda_causal * L_causal "
        "(default 1.0). 0 = pure atomicity.",
    )
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
        "=== Clustering evaluation (ILP matched-K baselines; edge-mass metrics) ==="
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
