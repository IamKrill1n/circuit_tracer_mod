"""Materialize K-means summary graphs at K matched to saved ILP summary graphs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from eval.eval_cluster import _kmeans_middle_labels, _middle_indices
from eval.eval_intervention import (
    _baseline_k_from_summary_graph,
    _prune_graph_stem,
    _sng_from_clusters,
)
from summarization.cluster import compute_phi_vectors, labels_to_supernodes
from summarization.prune import load_prune_graph
from summarization.summarize import SummaryGraph

logger = logging.getLogger(__name__)


def export_kmeans_kmatched(
    *,
    prune_graphs_dir: Path,
    k_source_summary_graphs_dir: Path,
    output_dir: Path,
    random_state: int,
    n_init: int,
    limit: int | None,
    skip_existing: bool,
) -> None:
    prune_paths = sorted(prune_graphs_dir.glob("*_prune_graph.pt"))
    if not prune_paths:
        raise FileNotFoundError(f"No *_prune_graph.pt files in {prune_graphs_dir}")
    if limit is not None:
        prune_paths = prune_paths[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    for path in prune_paths:
        stem = _prune_graph_stem(path)
        out_path = output_dir / f"{stem}_summary_graph.pt"
        if skip_existing and out_path.is_file():
            logger.info("skip existing %s", out_path.name)
            continue

        matched_k = _baseline_k_from_summary_graph(path, k_source_summary_graphs_dir)
        prune_graph = load_prune_graph(str(path))
        mid_idx = _middle_indices(prune_graph)
        middle_ids = [prune_graph.nodes[i].node_id for i in mid_idx]
        phi_mid = compute_phi_vectors(prune_graph).detach().cpu().numpy()[mid_idx]
        labels = _kmeans_middle_labels(phi_mid, matched_k, random_state, n_init)
        clusters = labels_to_supernodes(prune_graph, middle_ids, labels)
        sng = _sng_from_clusters(prune_graph, clusters)
        sng.metadata = dict(prune_graph.metadata)
        sng.metadata["clustering_method"] = "baseline-kmeans"
        sng.metadata["matched_k"] = matched_k
        sng.metadata["matched_k_source"] = str(k_source_summary_graphs_dir)
        sng.save(str(out_path))
        logger.info(
            "wrote %s  matched_k=%d  n_supernodes=%d",
            out_path.name,
            matched_k,
            len(sng.supernodes),
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export K-means summary graphs at K from saved ILP summary graphs"
    )
    parser.add_argument(
        "--prune-graphs-dir",
        type=Path,
        default=Path("pruned_graphs/entmax/alpha_0.50/node_0.02"),
    )
    parser.add_argument(
        "--k-source-summary-graphs-dir",
        type=Path,
        default=Path("summary_graphs/entmax/alpha_0.50/node_0.02"),
        help="Directory of ILP (or reference) summary graphs used only for per-graph K.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("summary_graphs/entmax/alpha_0.50/node_0.02_kmeans_kmatched"),
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-init", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip graphs whose output *_summary_graph.pt already exists.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    export_kmeans_kmatched(
        prune_graphs_dir=args.prune_graphs_dir,
        k_source_summary_graphs_dir=args.k_source_summary_graphs_dir,
        output_dir=args.output_dir,
        random_state=args.random_state,
        n_init=args.n_init,
        limit=args.limit,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
