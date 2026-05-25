from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import networkx as nx
import numpy as np
import torch
from sklearn.cluster import KMeans, SpectralClustering

from eval.eval_prune import compute_prune_loss
from summarization.attr_graph import AttrGraph
from summarization.cluster import (
    cluster_graph_spectral,
    clusters_to_supernodes,
    compute_phi_vectors,
    eigengap_analysis,
    labels_to_supernodes,
)
from summarization.scoring import _cosine_similarity, compute_L
from summarization.prune import PruneGraph, load_prune_graph
from summarization.summarize import SummarizationGraph
from summarization.utils import node_is_fixed

logger = logging.getLogger(__name__)

METHOD_GRID_DECAY: list[dict[str, float]] = [
    {"decay_rate": i / 10.0} for i in range(11)
]

_NORM_TOKENS = {"softmax", "entmax", "sparsemax", "entmax15"}

SUMMARY_COLUMNS = [
    "graph_name",
    "dataset",
    "graph_path",
    "token_normalization",
    "num_nodes",
    "solver",
    "decay_rate",
    "best_k",
    "k_candidates",
    "n_supernodes",
    "n_middle_supernodes",
    "L",
    "L_coh",
    "L_cons",
    "L_cplx",
    "prune_loss",
    "D_agg",
    "sweep_path",
    "best_dir",
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
    # making this a direct surrogate of L_coh.
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
    # Rectified cosine affinity matches the cos+ in L_coh exactly.
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


def _modularity_middle_labels(adjacency_mid: np.ndarray, target_k: int) -> np.ndarray:
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


def _build_k_candidates(prune_graph: PruneGraph, sim_phi: np.ndarray) -> list[int]:
    """Eigengap-bounded K range on the symmetrized phi similarity. Shared by all solvers."""
    n_middle = len(_middle_indices(prune_graph))
    if n_middle < 3:
        return [max(1, n_middle)]
    eg = eigengap_analysis(sim_phi, prune_graph, max_k=min(20, n_middle - 1))
    k_min_raw, k_max_raw = eg["search_range"]
    k_min = max(2, int(k_min_raw))
    k_max = min(n_middle, int(k_max_raw))
    if k_min > k_max:
        k_min = k_max
    return list(range(k_min, k_max + 1))


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in SUMMARY_COLUMNS})


def _evaluate_solver(
    *,
    prune_graph: PruneGraph,
    graph_path: Path,
    graph_name: str,
    dataset: str,
    token_normalization: str,
    output_dir: Path,
    solver: str,
    clusterer_factory: Callable[..., Callable[[int], list[list[str]]]],
    hyperparam_grid: list[dict[str, Any]],
    k_candidates: list[int],
    role_vectors_middle: np.ndarray,
    middle_id_to_local: dict[str, int],
    num_nodes: int,
    prune_loss: float,
    lambdas: tuple[float, float, float],
    enforce_dag: bool,
) -> dict[str, Any]:
    """Full hyperparameter × K sweep; pick argmin(L) and write artifacts."""
    sweep: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_supernode_map: dict[str, list[str]] | None = None

    grid = hyperparam_grid or [{}]
    for hp in grid:
        try:
            clusterer = clusterer_factory(**hp)
        except Exception as exc:
            logger.warning("Factory %s failed for %s: %s", solver, hp, exc)
            continue
        for target_k in k_candidates:
            try:
                clusters = clusterer(target_k)
                rows = clusters_to_supernodes(
                    prune_graph, clusters, enforce_dag=enforce_dag
                )
                sng = SummarizationGraph(supernodes=rows, pruned_adj=prune_graph.pruned_adj)
                metrics = compute_L(
                    sng,
                    role_vectors_middle,
                    middle_id_to_local,
                    prune_graph,
                    prune_loss=prune_loss,
                    lambdas=lambdas,
                )
            except Exception as exc:
                logger.warning(
                    "%s failed at K=%d hp=%s: %s", solver, target_k, hp, exc
                )
                continue
            record = {**hp, "K": int(target_k), **metrics}
            sweep.append(record)
            if best is None or record["L"] < best["L"]:
                best = record
                best_supernode_map = {sn.name: sn.member_node_ids() for sn in rows}

    run_dir = output_dir / "runs" / graph_name / solver
    sweep_path = run_dir / "sweep.json"
    best_dir = run_dir / "best"
    supernode_map_path = best_dir / "supernode_map.json"
    result_path = best_dir / "result.json"

    _write_json(
        sweep_path,
        {
            "solver": solver,
            "graph_name": graph_name,
            "k_candidates": list(map(int, k_candidates)),
            "hyperparam_grid": hyperparam_grid,
            "best": best,
            "sweep": sweep,
        },
    )

    if best is not None and best_supernode_map is not None:
        _write_json(supernode_map_path, best_supernode_map)
        _write_json(
            result_path,
            {
                "solver": solver,
                "graph_name": graph_name,
                "graph_path": str(graph_path),
                "best": best,
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
        "decay_rate": best.get("decay_rate") if best else "",
        "best_k": best.get("K") if best else "",
        "k_candidates": json.dumps(list(map(int, k_candidates))),
        "n_supernodes": best.get("n_supernodes") if best else "",
        "n_middle_supernodes": best.get("n_middle_supernodes") if best else "",
        "L": best.get("L") if best else math.nan,
        "L_coh": best.get("L_coh") if best else math.nan,
        "L_cons": best.get("L_cons") if best else math.nan,
        "L_cplx": best.get("L_cplx") if best else math.nan,
        "prune_loss": best.get("prune_loss") if best else math.nan,
        "D_agg": best.get("D_agg") if best else math.nan,
        "sweep_path": str(sweep_path),
        "best_dir": str(best_dir),
        "supernode_map_path": str(supernode_map_path) if best is not None else "",
        "result_path": str(result_path) if best is not None else "",
    }
    return summary


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
    lambdas: tuple[float, float, float],
) -> list[dict[str, Any]]:
    """Evaluate all 6 solvers on a single prune graph; return one summary row per solver."""
    prune_graph = load_prune_graph(str(graph_path), map_location=map_location)
    graph_name, dataset = _graph_identity(graph_path, input_paths)
    token_normalization = _token_normalization_from_path(graph_path)

    mid_idx = _middle_indices(prune_graph)
    middle_ids = [prune_graph.nodes[i].node_id for i in mid_idx]
    middle_id_to_local = {nid: i for i, nid in enumerate(middle_ids)}
    num_nodes = len(mid_idx)

    phi = compute_phi_vectors(prune_graph).detach().cpu().numpy()
    sim_phi = _cosine_similarity(phi, nonnegative=True)
    role_vectors_middle = phi[mid_idx]
    phi_mid = role_vectors_middle  # same array, distinct name for baseline clustering input
    adjacency_mid = _adjacency_affinity(prune_graph)[np.ix_(mid_idx, mid_idx)]

    k_candidates = _build_k_candidates(prune_graph, sim_phi)

    prune_loss = _compute_prune_loss_for_graph(
        graph_path=graph_path,
        prune_graph=prune_graph,
        map_location=map_location,
    )

    logger.info(
        "  %s | dataset=%s norm=%s n_middle=%d k_candidates=%s prune_loss=%.4f",
        graph_name,
        dataset,
        token_normalization,
        num_nodes,
        k_candidates,
        prune_loss,
    )

    def make_ours_spectral_factory(mean_method: Literal["geo", "harm", "arith"]):
        def factory(*, decay_rate: float) -> Callable[[int], list[list[str]]]:
            dr = float(decay_rate)

            def clusterer(target_k: int) -> list[list[str]]:
                # DAG enforcement happens uniformly at clusters_to_supernodes for all solvers.
                return cluster_graph_spectral(
                    prune_graph,
                    target_k=target_k,
                    max_layer_span=max_layer_span,
                    mean_method=mean_method,
                    decay_rate=dr,
                    enforce_dag=False,
                    random_state=random_state,
                    n_init=n_init,
                )

            return clusterer

        return factory

    def make_modularity() -> Callable[[int], list[list[str]]]:
        def clusterer(target_k: int) -> list[list[str]]:
            return labels_to_supernodes(
                prune_graph,
                middle_ids,
                _modularity_middle_labels(adjacency_mid, target_k),
            )

        return clusterer

    def make_spectral_cosine() -> Callable[[int], list[list[str]]]:
        def clusterer(target_k: int) -> list[list[str]]:
            return labels_to_supernodes(
                prune_graph,
                middle_ids,
                _spectral_cosine_middle_labels(phi_mid, target_k, random_state, n_init),
            )

        return clusterer

    def make_kmeans() -> Callable[[int], list[list[str]]]:
        def clusterer(target_k: int) -> list[list[str]]:
            return labels_to_supernodes(
                prune_graph,
                middle_ids,
                _kmeans_middle_labels(phi_mid, target_k, random_state, n_init),
            )

        return clusterer

    rows: list[dict[str, Any]] = []
    solver_specs: list[tuple[str, Callable[..., Any], list[dict[str, Any]]]] = [
        ("ours-spectral-arith", make_ours_spectral_factory("arith"), METHOD_GRID_DECAY),
        ("ours-spectral-harm",  make_ours_spectral_factory("harm"),  METHOD_GRID_DECAY),
        ("ours-spectral-geo",   make_ours_spectral_factory("geo"),   METHOD_GRID_DECAY),
        ("baseline-modularity", make_modularity, []),
        ("baseline-spectral-cosine", make_spectral_cosine, []),
        ("baseline-kmeans", make_kmeans, []),
    ]
    for solver_name, factory, grid in solver_specs:
        logger.info("  solving %s ...", solver_name)
        rows.append(
            _evaluate_solver(
                prune_graph=prune_graph,
                graph_path=graph_path,
                graph_name=graph_name,
                dataset=dataset,
                token_normalization=token_normalization,
                output_dir=output_dir,
                solver=solver_name,
                clusterer_factory=factory,
                hyperparam_grid=grid,
                k_candidates=k_candidates,
                role_vectors_middle=role_vectors_middle,
                middle_id_to_local=middle_id_to_local,
                num_nodes=num_nodes,
                prune_loss=prune_loss,
                lambdas=lambdas,
                enforce_dag=enforce_dag,
            )
        )
    return rows


def _validate_lambdas(lambda_coh: float, lambda_cons: float) -> tuple[float, float, float]:
    if not (0.0 <= lambda_coh <= 1.0 and 0.0 <= lambda_cons <= 1.0):
        raise ValueError("--lambda-coh and --lambda-cons must each be in [0, 1].")
    lambda_cplx = 1.0 - lambda_coh - lambda_cons
    if lambda_cplx < -1e-9:
        raise ValueError(
            f"--lambda-coh + --lambda-cons must be <= 1; got {lambda_coh + lambda_cons:.4f}"
        )
    return lambda_coh, lambda_cons, max(0.0, lambda_cplx)


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    lambdas = _validate_lambdas(args.lambda_coh, args.lambda_cons)
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
    logger.info(
        "lambdas: (coh=%.4f, cons=%.4f, cplx=%.4f)", lambdas[0], lambdas[1], lambdas[2]
    )

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
                lambdas=lambdas,
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
                "L = lambda_coh L_coh + lambda_cons L_cons + lambda_cplx L_cplx, "
                "simplex weights (default 1/3 each) per paper/reformulation.tex Section 3.4. "
                "L_cons = 0.5 (prune_loss + D_agg) where prune_loss = "
                "1 - token_attribution_faithfulness from eval/eval_prune.py. "
                "Each per-axis loss is in [0, 1], so L is in [0, 1]."
            ),
            "selection_protocol": (
                "Each solver sweeps its hyperparameter grid x the shared "
                "eigengap-bounded K candidate set; the (config, K) minimizing "
                "L is reported as that solver's row. K candidates come from "
                "auto_grouping.eigengap_analysis on the symmetrized phi cosine."
            ),
            "method_grid": METHOD_GRID_DECAY,
            "lambdas": {
                "lambda_coh": lambdas[0],
                "lambda_cons": lambdas[1],
                "lambda_cplx": lambdas[2],
            },
            "solvers": [
                "ours-spectral-arith",
                "ours-spectral-harm",
                "ours-spectral-geo",
                "baseline-modularity",
                "baseline-spectral-cosine",
                "baseline-kmeans",
            ],
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
                "lambda_coh": lambdas[0],
                "lambda_cons": lambdas[1],
                "lambda_cplx": lambdas[2],
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
            "simplex-weighted L = lambda_coh L_coh + lambda_cons L_cons + "
            "lambda_cplx L_cplx objective from paper/reformulation.tex."
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
    parser.add_argument(
        "--lambda-coh",
        type=float,
        default=1.0 / 3.0,
        help="Simplex weight on L_coh (default 1/3).",
    )
    parser.add_argument(
        "--lambda-cons",
        type=float,
        default=1.0 / 3.0,
        help="Simplex weight on L_cons (default 1/3). lambda_cplx = 1 - lambda_coh - lambda_cons.",
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
        "=== Clustering evaluation (simplex-weighted L = lambda_coh L_coh + "
        "lambda_cons L_cons + lambda_cplx L_cplx; per-solver best) ==="
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
