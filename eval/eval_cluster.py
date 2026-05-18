from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Callable, Literal, Sequence, cast

import networkx as nx
import numpy as np
import torch
from sklearn.cluster import KMeans, SpectralClustering

from summarization.auto_grouping import eigengap_analysis
from summarization.cluster import (
    cluster_graph_agglomerative,
    cluster_graph_spectral,
    clusters_to_supernodes,
    compute_phi_vectors,
    labels_to_supernodes,
)
from summarization.cluster_scoring import _cosine_similarity
from summarization.objective import compute_L
from summarization.prune import PruneGraph, load_prune_graph
from summarization.utils import node_is_fixed

logger = logging.getLogger(__name__)

METHOD_GRID: list[dict[str, str | float]] = [
    {"mean_method": mean_method, "decay_rate": decay_rate}
    for mean_method in ("arith", "harm", "geo")
    for decay_rate in (i / 10.0 for i in range(11))
]

SUMMARY_COLUMNS = [
    "graph_name",
    "dataset",
    "graph_path",
    "num_nodes",
    "solver",
    "mean_method",
    "decay_rate",
    "best_k",
    "k_candidates",
    "n_supernodes",
    "n_middle_supernodes",
    "L_total",
    "L_coh",
    "D_agg",
    "L_cplx",
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
    return (
        KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        .fit_predict(features)
        .astype(np.int64)
    )


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
    return (
        SpectralClustering(
            n_clusters=k,
            affinity="rbf",
            assign_labels="kmeans",
            random_state=random_state,
            n_init=n_init,  # type: ignore[arg-type]
        )
        .fit_predict(features)
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
    output_dir: Path,
    solver: str,
    clusterer_factory: Callable[..., Callable[[int], list[list[str]]]],
    hyperparam_grid: list[dict[str, Any]],
    k_candidates: list[int],
    role_vectors_middle: np.ndarray,
    middle_id_to_local: dict[str, int],
    num_nodes: int,
) -> dict[str, Any]:
    """Full hyperparameter × K sweep; pick argmin(L_total) and write artifacts."""
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
                rows = clusters_to_supernodes(prune_graph, clusters)
                metrics = compute_L(
                    rows, role_vectors_middle, middle_id_to_local, prune_graph
                )
            except Exception as exc:
                logger.warning(
                    "%s failed at K=%d hp=%s: %s", solver, target_k, hp, exc
                )
                continue
            record = {**hp, "K": int(target_k), **metrics}
            sweep.append(record)
            if best is None or record["L_total"] < best["L_total"]:
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
        "num_nodes": num_nodes,
        "solver": solver,
        "mean_method": best.get("mean_method") if best else "",
        "decay_rate": best.get("decay_rate") if best else "",
        "best_k": best.get("K") if best else "",
        "k_candidates": json.dumps(list(map(int, k_candidates))),
        "n_supernodes": best.get("n_supernodes") if best else "",
        "n_middle_supernodes": best.get("n_middle_supernodes") if best else "",
        "L_total": best.get("L_total") if best else math.nan,
        "L_coh": best.get("L_coh") if best else math.nan,
        "D_agg": best.get("D_agg") if best else math.nan,
        "L_cplx": best.get("L_cplx") if best else math.nan,
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
) -> list[dict[str, Any]]:
    """Evaluate all 5 solvers on a single prune graph; return one summary row per solver."""
    prune_graph = load_prune_graph(str(graph_path), map_location=map_location)
    graph_name, dataset = _graph_identity(graph_path, input_paths)

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

    logger.info(
        "  %s | dataset=%s n_middle=%d k_candidates=%s",
        graph_name,
        dataset,
        num_nodes,
        k_candidates,
    )

    def make_ours_spectral(
        *,
        mean_method: str,
        decay_rate: float,
    ) -> Callable[[int], list[list[str]]]:
        mm = cast(Literal["geo", "harm", "arith"], mean_method)
        dr = float(decay_rate)

        def clusterer(target_k: int) -> list[list[str]]:
            return cluster_graph_spectral(
                prune_graph,
                target_k=target_k,
                max_layer_span=max_layer_span,
                mean_method=mm,
                decay_rate=dr,
                enforce_dag=enforce_dag,
                random_state=random_state,
                n_init=n_init,
            )

        return clusterer

    def make_ours_agglomerative(
        *,
        mean_method: str,
        decay_rate: float,
    ) -> Callable[[int], list[list[str]]]:
        mm = cast(Literal["geo", "harm", "arith"], mean_method)
        dr = float(decay_rate)

        def clusterer(target_k: int) -> list[list[str]]:
            return cluster_graph_agglomerative(
                prune_graph,
                target_k=target_k,
                max_layer_span=max_layer_span,
                mean_method=mm,
                decay_rate=dr,
            )

        return clusterer

    def make_modularity() -> Callable[[int], list[list[str]]]:
        def clusterer(target_k: int) -> list[list[str]]:
            return labels_to_supernodes(
                prune_graph,
                middle_ids,
                _modularity_middle_labels(adjacency_mid, target_k),
            )

        return clusterer

    def make_spectral_rbf() -> Callable[[int], list[list[str]]]:
        def clusterer(target_k: int) -> list[list[str]]:
            return labels_to_supernodes(
                prune_graph,
                middle_ids,
                _spectral_rbf_middle_labels(phi_mid, target_k, random_state, n_init),
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
        ("ours-spectral", make_ours_spectral, METHOD_GRID),
        ("ours-agglomerative", make_ours_agglomerative, METHOD_GRID),
        ("baseline-modularity", make_modularity, []),
        ("baseline-spectral-rbf", make_spectral_rbf, []),
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
                output_dir=output_dir,
                solver=solver_name,
                clusterer_factory=factory,
                hyperparam_grid=grid,
                k_candidates=k_candidates,
                role_vectors_middle=role_vectors_middle,
                middle_id_to_local=middle_id_to_local,
                num_nodes=num_nodes,
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
            "objective": (
                "L = L_coh + D_agg + L_cplx per paper/reformulation.tex Section 2-3, "
                "with all three terms redefined to lie in [0, 1]. L_coh is mean "
                "(1 - cos(r(u), centroid)) over middle features; D_agg is the "
                "magnitude-retention loss of Eq. 12 computed from the signed "
                "block sum (no dominant-direction tie-breaker); L_cplx is the "
                "average emb->logit shortest-path length in G_SN divided by "
                "(num_model_layers + 1)."
            ),
            "selection_protocol": (
                "Each solver sweeps its full hyperparameter grid x the shared "
                "eigengap-bounded K candidate set; the (config, K) minimizing "
                "L_total is reported as that solver's row. K candidates come from "
                "auto_grouping.eigengap_analysis on the symmetrized phi cosine."
            ),
            "method_grid": METHOD_GRID,
            "solvers": [
                "ours-spectral",
                "ours-agglomerative",
                "baseline-modularity",
                "baseline-spectral-rbf",
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
            "L_coh + D_agg + L_cplx objective from paper/reformulation.tex."
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
        "=== Clustering evaluation (L_coh + D_agg + L_cplx; per-solver best) ==="
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
