"""Register stored dataset graphs and summaries with the visualization app.

Examples:
    conda run -n circuit python import_dataset.py \
      --dataset analogies \
      --graphs-root dataset/analogies \
      --summary-dir labeled_summary/entmax/alpha_0.50/node_0.02

    conda run -n circuit python import_dataset.py \
      --dataset multihop \
      --graphs-root dataset/multihop \
      --summary-dir labeled_summary/entmax/alpha_0.50/node_0.02
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from visualization_app import services


DEFAULT_SUMMARY_DIR = Path("labeled_summary/entmax/alpha_0.50/node_0.02")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import dataset graphs into visualization_app.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=services.KNOWN_DATASETS,
        help="Dataset name (controls graph_files/{dataset}/ and summary/{dataset}/).",
    )
    parser.add_argument(
        "--graphs-root",
        type=Path,
        default=None,
        help="Directory with raw attribution .pt files (default: dataset/{dataset}).",
    )
    parser.add_argument(
        "--source-set",
        default="",
        help="Optional source-set namespace, for example mntss/clt-gemma-2-2b-426k.",
    )
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--graph-root", type=Path, default=services.GRAPH_ROOT)
    parser.add_argument("--summary-root", type=Path, default=services.SUMMARY_ROOT)
    parser.add_argument("--node-threshold", type=float, default=0.8)
    parser.add_argument("--edge-threshold", type=float, default=0.98)
    parser.add_argument("--copy", action="store_true", help="Copy summaries instead of symlinking.")
    parser.add_argument("--allow-missing-summary", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--rebuild-storage",
        action="store_true",
        help="Rebuild summary/supernode_storage.json after import.",
    )
    return parser.parse_args()


def _replace_with_summary_link(source: Path, destination: Path, *, copy: bool) -> str:
    if not source.exists():
        raise FileNotFoundError(f"Summary does not exist: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == destination.resolve():
        return "already-present"
    if destination.exists() or destination.is_symlink():
        destination.unlink()

    if copy:
        shutil.copy2(source, destination)
        return "copied"

    relative_source = os.path.relpath(source.resolve(), destination.parent.resolve())
    destination.symlink_to(relative_source)
    return "symlinked"


def main() -> None:
    args = _parse_args()
    source_set = services.validate_source_set(args.source_set)
    graphs_root = args.graphs_root or (
        services.dataset_pt_path("placeholder", args.dataset, source_set=source_set).parent
        if source_set
        else services.DATASET_ROOT / args.dataset
    )
    graph_paths = sorted(graphs_root.glob("*.pt"))
    if args.limit is not None:
        graph_paths = graph_paths[: args.limit]
    if not graph_paths:
        raise SystemExit(f"No .pt graphs found in {graphs_root}")

    imported = linked = missing = 0
    for i, graph_path in enumerate(graph_paths, start=1):
        stem = graph_path.stem
        summary_candidates = [
            args.summary_dir / f"{stem}.pt",
            args.summary_dir / f"{stem}_labeled_summary_graph.pt",
            args.summary_dir / f"{stem}_summary_graph.pt",
        ]
        stored_summary = next((path for path in summary_candidates if path.exists()), summary_candidates[0])
        app_summary = services.app_summary_path(
            stem,
            args.dataset,
            args.summary_root,
            source_set=source_set,
        )
        graph_name = "/".join(part for part in (args.dataset, source_set, stem) if part)
        print(f"[{i}/{len(graph_paths)}] {graph_name}", flush=True)

        services.convert_pt_to_viewer(
            graph_path,
            slug=stem,
            dataset=args.dataset,
            source_set=source_set,
            root=args.graph_root,
            summary_root=args.summary_root,
            node_threshold=args.node_threshold,
            edge_threshold=args.edge_threshold,
        )
        imported += 1

        if not stored_summary.exists() and args.allow_missing_summary:
            print(f"  missing summary: {stored_summary}", flush=True)
            missing += 1
            continue

        action = _replace_with_summary_link(stored_summary, app_summary, copy=args.copy)
        print(f"  {action}: {app_summary} -> {stored_summary}", flush=True)
        linked += 1

    if args.rebuild_storage:
        services.rebuild_supernode_storage(args.graph_root, args.summary_root)

    print(
        f"Done. imported={imported} linked={linked} missing_summaries={missing}",
        flush=True,
    )


if __name__ == "__main__":
    main()
