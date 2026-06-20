"""Register stored analogies graphs and summaries with the visualization app.

Example:
    conda run -n circuit python -m visualization_app.import_analogies \
      --graphs-root dataset/analogies \
      --summary-dir labeled_summary/entmax/alpha_0.50/node_0.02
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from visualization_app import services


DEFAULT_GRAPHS_ROOT = Path("dataset/analogies")
DEFAULT_SUMMARY_DIR = Path("labeled_summary/entmax/alpha_0.50/node_0.02")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import analogies into visualization_app.")
    parser.add_argument("--graphs-root", type=Path, default=DEFAULT_GRAPHS_ROOT)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--graph-root", type=Path, default=services.GRAPH_ROOT)
    parser.add_argument("--pt-root", type=Path, default=services.GENERATED_GRAPH_ROOT)
    parser.add_argument("--node-threshold", type=float, default=0.8)
    parser.add_argument("--edge-threshold", type=float, default=0.98)
    parser.add_argument("--copy", action="store_true", help="Copy summaries instead of symlinking.")
    parser.add_argument("--allow-missing-summary", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def _replace_with_summary_link(source: Path, destination: Path, *, copy: bool) -> str:
    if not source.exists():
        raise FileNotFoundError(f"Summary does not exist: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
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
    graph_paths = sorted(args.graphs_root.glob("*.pt"))
    if args.limit is not None:
        graph_paths = graph_paths[: args.limit]
    if not graph_paths:
        raise SystemExit(f"No .pt graphs found in {args.graphs_root}")

    imported = linked = missing = 0
    for i, graph_path in enumerate(graph_paths, start=1):
        stem = graph_path.stem
        stored_summary = args.summary_dir / f"{stem}_labeled_summary_graph.pt"
        app_summary = services.summary_path(stem, args.pt_root)
        print(f"[{i}/{len(graph_paths)}] {stem}", flush=True)

        services.convert_pt_to_viewer(
            graph_path,
            slug=stem,
            root=args.graph_root,
            pt_root=args.pt_root,
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

    print(
        f"Done. imported={imported} linked={linked} missing_summaries={missing}",
        flush=True,
    )


if __name__ == "__main__":
    main()
