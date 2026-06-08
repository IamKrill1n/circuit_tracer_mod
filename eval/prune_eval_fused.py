"""Fused prune+eval sweep: prune in memory, compute metrics, never save graphs.

The two-stage path (prune_graphs.py saves PruneGraph .pt files, eval_prune.py reads
them back) is infeasible at scale here: each saved PruneGraph stores dense pruned_adj
+ edge_influence + edge_relevance (~140 MB each), so a 9000-cell sweep needs ~0.5 TB.

This script sweeps node_threshold x alpha x SHAP-normalization over a graph set,
pruning each cell in memory and computing only the three reported metrics
(completeness_score, replacement_score via compute_clt_graph_scores, and
token_attribution_faithfulness). It writes one results.json + summary.csv (a few MB).

A fresh GraphCache is built per graph: caching every graph's dense n x n adjacency on
the GPU at once would OOM, but one graph's artifacts (adj, row-normalized transpose,
per-embedding full relevance) fit easily and are reused across that graph's cells.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

from summarization.prune import prune_attr_graph
from eval.prune_graphs import (
    _build_shap_lookup,
    _discover_graph_files,
    _load_shap_values_json,
    _match_shap_row,
    _token_weights_for_embeddings,
    normalize_shap_values_for_prune,
)
from eval.eval_prune import (
    GraphCache,
    compute_clt_graph_scores,
    plot_pareto_frontier,
    token_attribution_faithfulness,
)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def run(args: argparse.Namespace) -> None:
    shap = _load_shap_values_json(Path(args.shap_values_json))
    by_prompt, by_index = _build_shap_lookup(shap)
    json_keep = shap.get("masker_keep_prefix")
    if args.masker_keep_prefix is not None:
        eff_keep: int | None = int(args.masker_keep_prefix)
    elif isinstance(json_keep, (int, float)) and int(json_keep) > 0:
        eff_keep = int(json_keep)
    else:
        eff_keep = None

    norms = tuple(args.eval_normalizations)
    node_thrs = [float(t) for t in args.node_thresholds]
    alphas = [float(a) for a in args.alpha_sweep]
    discovered = _discover_graph_files(Path(args.graphs_root), tuple(args.source_sets))

    rows: list[dict[str, Any]] = []
    failures: list[str] = []

    all_paths = [(ss, gp) for ss, paths in discovered.items() for gp in paths]
    if args.limit is not None:
        all_paths = all_paths[: args.limit]
    ntot = len(all_paths)

    for gi, (source_set, gp_path) in enumerate(all_paths):
        gp = str(gp_path)
        stem = gp_path.stem
        # Fresh per-graph cache so GPU memory stays bounded to a single graph.
        cache = GraphCache(device=args.device)
        try:
            ag = cache.attr_graph(gp)
            full_idx = cache.idx_sets(gp)
            A_full_norm = cache.a_full_norm(gp)
            id_to_idx = cache.id_to_idx(gp)
            md = ag.metadata
            node_ids = [n.node_id for n in ag.nodes]
            emb_idx = full_idx["embedding"]
            prompt_tokens = [str(t) for t in (md.get("prompt_tokens") or [])]
            if not prompt_tokens:
                raise ValueError("metadata.prompt_tokens missing or empty")

            row = _match_shap_row(stem, md, by_prompt, by_index)
            if row is None:
                raise ValueError("no matching SHAP row (prompt / pNN index)")
            raw_shap = row.get("raw_shap")
            if not isinstance(raw_shap, list) or not raw_shap:
                raise ValueError("matched SHAP row has no raw_shap list")

            n_cells = 0
            for norm in norms:
                normalized = normalize_shap_values_for_prune(
                    prompt_tokens,
                    [float(x) for x in raw_shap],
                    norm,  # type: ignore[arg-type]
                    masker_keep_prefix=eff_keep,
                    entmax_alpha=args.entmax_alpha,
                )
                token_weights = _token_weights_for_embeddings(normalized, node_ids, emb_idx)

                for alpha in alphas:
                    for thr in node_thrs:
                        pg = prune_attr_graph(
                            ag,
                            logit_weights=args.logit_weights,
                            token_weights=token_weights,
                            node_threshold=thr,
                            edge_threshold=args.edge_threshold,
                            combine_method=args.combine_method,  # type: ignore[arg-type]
                            normalization=args.normalization,  # type: ignore[arg-type]
                            alpha=alpha,
                            keep_all_tokens_and_logits=args.keep_all_tokens_and_logits,
                        )
                        rep, comp = compute_clt_graph_scores(ag, pg)
                        taf = token_attribution_faithfulness(
                            ag,
                            pg,
                            token_weights,
                            full_idx=full_idx,
                            A_full_norm=A_full_norm,
                            id_to_idx=id_to_idx,
                            full_target_rel_fn=lambda emb_i: cache.full_target_rel(gp, emb_i),
                        )
                        rows.append({
                            "source_set": source_set,
                            "graph_stem": stem,
                            "normalize_method": norm,
                            "alpha": alpha,
                            "node_threshold": thr,
                            "edge_threshold": args.edge_threshold,
                            "n_nodes": pg.num_nodes,
                            "n_edges": pg.num_edges,
                            "completeness_score": comp,
                            "replacement_score": rep,
                            "token_attribution_faithfulness": taf,
                        })
                        n_cells += 1
            print(f"[{gi + 1}/{ntot}] {stem}: {n_cells} cells (total rows={len(rows)})", flush=True)
        except Exception as exc:
            failures.append(f"{source_set}/{stem}: {exc}")
            print(f"[failed] {source_set}/{stem}: {exc}", flush=True)
        finally:
            del cache
            if str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    results_path = output_root / "results.json"
    summary_path = output_root / "summary.csv"
    manifest_path = output_root / "manifest.json"

    _write_json(results_path, rows)
    if rows:
        with summary_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    _write_json(manifest_path, {
        "graphs_root": str(args.graphs_root),
        "shap_values_json": str(args.shap_values_json),
        "source_sets": list(args.source_sets),
        "eval_normalizations": list(norms),
        "node_thresholds": node_thrs,
        "alphas": alphas,
        "edge_threshold": args.edge_threshold,
        "combine_method": args.combine_method,
        "score_normalization": args.normalization,
        "logit_weights": args.logit_weights,
        "keep_all_tokens_and_logits": bool(args.keep_all_tokens_and_logits),
        "masker_keep_prefix": eff_keep,
        "entmax_alpha": args.entmax_alpha,
        "n_graphs": ntot,
        "n_rows": len(rows),
        "n_failures": len(failures),
        "failures": failures,
    })

    print("\n=== Fused prune+eval summary ===")
    print(f"graphs: {ntot}  rows: {len(rows)}  failures: {len(failures)}")
    print(f"wrote {results_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {manifest_path}")
    if failures:
        print("\nFailures (first 20):")
        for item in failures[:20]:
            print(f"- {item}")

    if args.plot and rows:
        metrics = [
            ("completeness_score", "graph completeness score"),
            ("replacement_score", "replacement score"),
            ("token_attribution_faithfulness", "token attribution faithfulness"),
        ]
        for norm in norms:
            sub = [r for r in rows if r["normalize_method"] == norm]
            if not sub:
                continue
            out = plot_pareto_frontier(sub, output_root, metrics=metrics, filename=f"pareto_{norm}.png")
            if out is not None:
                print(f"wrote {out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fused in-memory prune+eval sweep (no graph .pt files).")
    p.add_argument("--graphs-root", required=True)
    p.add_argument("--source-sets", nargs="+", default=["analogies"])
    p.add_argument("--shap-values-json", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--eval-normalizations", nargs="+",
                   choices=["softmax", "entmax", "entmax15", "sparsemax"], default=["softmax", "entmax"])
    p.add_argument("--node-thresholds", nargs="+", type=float, required=True)
    p.add_argument("--alpha-sweep", nargs="+", type=float, required=True)
    p.add_argument("--edge-threshold", type=float, default=0.95)
    p.add_argument("--combine-method", choices=["geometric", "arithmetic", "harmonic"], default="geometric")
    p.add_argument("--normalization", choices=["rank", "min_max"], default="rank")
    p.add_argument("--logit-weights", choices=["target", "probs"], default="target")
    p.add_argument("--keep-all-tokens-and-logits", action="store_true")
    p.add_argument("--masker-keep-prefix", type=int, default=None)
    p.add_argument("--entmax-alpha", type=float, default=1.25)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--plot", action="store_true")
    return p


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
