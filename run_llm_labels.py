"""Batch-label supernodes for all prune graphs in a directory using an LLM.

Usage:
    python run_llm_labels.py                          # uses default path
    python run_llm_labels.py --graphs-dir eval_outputs/analogies/clt-hp/entmax/alpha_0.50/node_0.02
    python run_llm_labels.py --output labels.json --target-k 7
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch LLM supernode labelling")
    p.add_argument(
        "--graphs-dir",
        default="eval_outputs/analogies/clt-hp/entmax/alpha_0.50/node_0.02",
        help="Directory containing *_prune_graph.pt files",
    )
    p.add_argument(
        "--output",
        default="llm_labels.json",
        help="Path to write JSON results (default: llm_labels.json)",
    )
    p.add_argument("--target-k", type=int, default=7)
    p.add_argument("--mean-method", default="geo", choices=["geo", "harm", "arith"])
    p.add_argument("--normalize-weights", action="store_true", default=True)
    p.add_argument("--model-name", default="gpt-4o-mini")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip graphs already present in the output file",
    )
    p.add_argument("--limit", type=int, default=None, help="Process at most N graphs (for quick tests)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Lazy imports so arg parsing is fast
    from summarization.prune import load_prune_graph
    from summarization.cluster import cluster_graph_spectral, clusters_to_supernodes
    from summarization.supernode_graph import SummarizationGraph
    from summarization.group_llm import label_summarization_graph

    graphs_dir = Path(args.graphs_dir)
    pt_files = sorted(graphs_dir.glob("*_prune_graph.pt"))
    if not pt_files:
        print(f"No *_prune_graph.pt files found in {graphs_dir}", file=sys.stderr)
        sys.exit(1)

    if args.limit:
        pt_files = pt_files[: args.limit]

    output_path = Path(args.output)

    # Load existing results for resume — only skip graphs that actually succeeded
    results: dict[str, list[str]] = {}
    if args.resume and output_path.exists():
        with output_path.open() as f:
            results = json.load(f)
        done = sum(1 for v in results.values() if v)
        print(f"Resuming — {done} graphs already labelled ({len(results) - done} failed/empty, will retry).")

    # tqdm is optional
    try:
        from tqdm import tqdm
        iterator = tqdm(pt_files, desc="Labelling graphs", unit="graph")
    except ImportError:
        iterator = pt_files  # type: ignore[assignment]

    ok = err = skipped = 0
    for pt_path in iterator:
        key = pt_path.stem  # e.g. "000_prune_graph"
        if args.resume and key in results and results[key]:
            skipped += 1
            continue

        try:
            prune_graph = load_prune_graph(str(pt_path))
            clusters = cluster_graph_spectral(
                prune_graph,
                target_k=args.target_k,
                mean_method=args.mean_method,
                normalize_weights=args.normalize_weights,
            )
            supernodes = clusters_to_supernodes(prune_graph, clusters)
            sng = SummarizationGraph(supernodes=supernodes, pruned_adj=prune_graph.pruned_adj)

            sng = label_summarization_graph(
                sng,
                prune_graph.metadata,
                model_name=args.model_name,
                temperature=args.temperature,
            )

            results[key] = sng.sn_names
            ok += 1

        except Exception:
            print(f"\n[ERROR] {pt_path.name}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            results[key] = []
            err += 1

        # Save after every graph so progress is not lost on interrupt
        with output_path.open("w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nDone. ok={ok}  errors={err}  skipped={skipped}")
    print(f"Results written to {output_path.resolve()}")


if __name__ == "__main__":
    main()
