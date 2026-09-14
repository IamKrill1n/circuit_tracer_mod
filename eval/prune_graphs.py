"""Sweep semantic, uniform, shuffled, and output-only pruning on saved graphs."""
from __future__ import annotations

import argparse
import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from summarization.attr_graph import AttrGraph
from summarization.prune import prune_attr_graph, save_prune_graph
from summarization.semantic_seeds import embedding_weights, select_semantic_seeds


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def seed_conditions(ag: AttrGraph, semantic: list[float], seed: int) -> dict[str, list[float] | None]:
    """Shuffle only ordinary-token weights, preserving the special-token exclusion."""
    special = set(ag.metadata["special_token_indices"])
    positions = [int(n.ctx_idx) for n in ag.nodes if n.feature_type == "embedding"]
    candidates = [i for i, pos in enumerate(positions) if pos not in special]
    values = [semantic[i] for i in candidates]
    random.Random(seed).shuffle(values)
    shuffled = list(semantic)
    for i, value in zip(candidates, values):
        shuffled[i] = value
    uniform = [0.0 if i in special else 1.0 for i in range(len(positions))]
    total = sum(uniform)
    if total == 0:
        raise ValueError("No ordinary prompt tokens for uniform baseline.")
    return {"semantic": semantic, "uniform": embedding_weights(ag, [w / total for w in uniform]),
            "shuffled": shuffled, "output_only": embedding_weights(ag, [w / total for w in uniform])}


def run_prune_sweep(args: argparse.Namespace) -> None:
    if not args.claim.strip() or not args.selector_model.strip():
        raise ValueError("Supply a nonempty claim and selector model.")
    if not args.node_thresholds or any(not 0 < t <= 1 for t in args.node_thresholds):
        raise ValueError("Node thresholds must be in (0, 1].")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    root = Path(args.output_root) if args.output_root else Path("experiments/runs") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    root.mkdir(parents=True, exist_ok=False)
    paths = sorted(Path(args.graphs_root).glob("*.pt"))
    if args.limit is not None:
        paths = paths[:args.limit]
    if not paths:
        raise ValueError("No .pt attribution graphs found.")
    records: list[dict] = []
    manifest = {"parameters": vars(args), "git_commit": git_commit(), "records": records}
    for path in paths:
        ag = AttrGraph.from_graph(str(path))
        ag.adj = ag.adj.to(args.device)
        semantic = select_semantic_seeds(ag, args.claim, args.selector_model)
        (root / f"{path.stem}_selection.json").write_text(json.dumps(ag.metadata, indent=2))
        for condition, weights in seed_conditions(ag, semantic, args.seed).items():
            for threshold in args.node_thresholds:
                # Arithmetic alpha=1 is explicitly output-only; no relevance gating.
                pg = prune_attr_graph(ag, token_weights=weights, node_threshold=threshold,
                    edge_threshold=args.edge_threshold, combine_method="arithmetic" if condition == "output_only" else "geometric",
                    alpha=1.0 if condition == "output_only" else args.alpha,
                    keep_all_tokens_and_logits=True)
                out = root / condition / f"node_{threshold}" / f"{path.stem}_prune_graph.pt"
                out.parent.mkdir(parents=True, exist_ok=True)
                pg.metadata = {**pg.metadata, "seed_condition": condition, "seed": args.seed,
                               "git_commit": manifest["git_commit"]}
                save_prune_graph(pg, str(out))
                records.append({"graph_path": str(path.resolve()), "graph_stem": path.stem,
                    "prune_graph_path": str(out.resolve()), "source_set": "semantic_pilot",
                    "seed_condition": condition, "normalize_method": condition,
                    "node_threshold": threshold, "edge_threshold": args.edge_threshold,
                    "alpha": 1.0 if condition == "output_only" else args.alpha,
                    "combine_method": "arithmetic" if condition == "output_only" else "geometric",
                    "score_normalization": "rank", "logit_weights": "target",
                    "keep_all_tokens_and_logits": True,
                    "token_weights": weights, "num_nodes": pg.num_nodes, "num_edges": pg.num_edges,
                    "num_features": sum(n.feature_type == "cross layer transcoder" for n in pg.nodes)})
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(root / "manifest.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs-root", required=True)
    parser.add_argument("--claim", required=True)
    parser.add_argument("--selector-model", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--node-thresholds", nargs="+", type=float, default=[0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
    parser.add_argument("--edge-threshold", type=float, default=0.95)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    run_prune_sweep(build_parser().parse_args())


if __name__ == "__main__":
    main()
