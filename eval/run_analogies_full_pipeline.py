"""Run the confirmed analogies prune -> ILP cluster -> LLM label pipeline.

Outputs are intentionally split by stage:

- pruned_graphs/<normalization>/alpha_0.50/node_0.02/*_prune_graph.pt
- summary_graphs/<normalization>/alpha_0.50/node_0.02/*_summary_graph.pt
- labeled_summary/<normalization>/alpha_0.50/node_0.02/*_labeled_summary_graph.pt

The script is resumable and writes manifests after every graph.
"""

from __future__ import annotations

import argparse
import csv
import json
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from eval.prune_graphs import (
    _build_shap_lookup,
    _load_shap_values_json,
    _match_shap_row,
    _token_weights_for_embeddings,
    normalize_shap_values_for_prune,
)
from summarization.attr_graph import AttrGraph
from summarization.classify import filter_act_density
from summarization.cluster import clusters_to_supernodes
from summarization.group_llm import LabelScheme, ModelSettings, label_supernodes
from summarization.ilp_cluster import (
    DEFAULT_EPS_CAUSAL,
    DEFAULT_MAX_LAYER_SPAN,
    DEFAULT_MAX_SN,
    DEFAULT_NORMALIZE_WEIGHTS,
    DEFAULT_THETA,
    DEFAULT_TIME_LIMIT,
    cluster_graph_ilp,
)
from summarization.prune import PruneGraph, load_prune_graph, prune_attr_graph, save_prune_graph
from summarization.summarize import SummaryGraph
from summarization.utils import _build_index_sets


NORMALIZATIONS = ("softmax", "entmax")
ALPHA = 0.5
NODE_THRESHOLD = 0.02
ENTMAX_ALPHA = 1.25
EDGE_THRESHOLD = 0.95
COMBINE_METHOD = "geometric"
SCORE_NORMALIZATION = "rank"
LOGIT_WEIGHTS = "target"
KEEP_ALL_TOKENS_AND_LOGITS = False
ACT_DENSITY_LB = 2e-5
ACT_DENSITY_UB = 0.1
LABEL_MODEL = "gemma-4-31b-it"


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_to_jsonable(v) for v in obj]
    if hasattr(obj, "detach"):
        return obj.detach().cpu().tolist()
    return obj


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _save_stage_rows(root: Path, stage: str, rows: list[dict[str, Any]]) -> None:
    _write_json(root / f"{stage}_manifest.json", rows)
    _write_csv(root / f"{stage}_summary.csv", rows)


def _graph_index(stem: str) -> int | None:
    return int(stem) if stem.isdigit() else None


def _should_label(stem: str) -> bool:
    idx = _graph_index(stem)
    return idx is not None and idx % 10 == 1


def _prune_graph_to_cpu(prune_graph: PruneGraph) -> PruneGraph:
    def vec(t: torch.Tensor | None) -> torch.Tensor | None:
        return t.detach().cpu() if t is not None else None

    return PruneGraph(
        nodes=[replace(n) for n in prune_graph.nodes],
        pruned_adj=prune_graph.pruned_adj.detach().cpu(),
        metadata=prune_graph.metadata,
        node_influence=vec(prune_graph.node_influence),
        node_relevance=vec(prune_graph.node_relevance),
        edge_influence=vec(prune_graph.edge_influence),
        edge_relevance=vec(prune_graph.edge_relevance),
    )


def _stage_dir(root: Path, normalization: str) -> Path:
    return root / normalization / f"alpha_{ALPHA:.2f}" / f"node_{NODE_THRESHOLD:.2f}"


def _load_graphs(graphs_root: Path, limit: int | None) -> list[Path]:
    paths = sorted(graphs_root.glob("*.pt"))
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise ValueError(f"No .pt graphs found in {graphs_root}")
    return paths


def _prune_one(
    *,
    graph_path: Path,
    normalization: str,
    shap_row: dict[str, Any],
    output_path: Path,
    device: str,
) -> tuple[PruneGraph, dict[str, Any]]:
    if output_path.exists():
        prune_graph = load_prune_graph(str(output_path))
        return prune_graph, {"status": "skipped_existing"}

    attr_graph = AttrGraph.from_graph(str(graph_path))
    if device != "cpu":
        attr_graph.adj = attr_graph.adj.to(device)
    attr_graph.metadata.setdefault("info", {})["neuronpedia_source_set"] = "analogies"

    node_ids = [n.node_id for n in attr_graph.nodes]
    idx = _build_index_sets(attr_graph.nodes)
    emb_idx = idx["embedding"]
    prompt_tokens = [str(t) for t in (attr_graph.metadata.get("prompt_tokens") or [])]
    raw_shap = shap_row.get("raw_shap")
    if not isinstance(raw_shap, list) or not raw_shap:
        raise ValueError("matched SHAP row has no raw_shap list")

    normalized = normalize_shap_values_for_prune(
        prompt_tokens,
        [float(x) for x in raw_shap],
        normalization,  # type: ignore[arg-type]
        masker_keep_prefix=None,
        entmax_alpha=ENTMAX_ALPHA,
    )
    token_weights = _token_weights_for_embeddings(normalized, node_ids, emb_idx)

    prune_graph = prune_attr_graph(
        attr_graph,
        logit_weights=LOGIT_WEIGHTS,
        token_weights=token_weights,
        node_threshold=NODE_THRESHOLD,
        edge_threshold=EDGE_THRESHOLD,
        combine_method=COMBINE_METHOD,
        normalization=SCORE_NORMALIZATION,
        alpha=ALPHA,
        keep_all_tokens_and_logits=KEEP_ALL_TOKENS_AND_LOGITS,
    )
    prune_graph = filter_act_density(
        prune_graph,
        act_density_lb=ACT_DENSITY_LB,
        act_density_ub=ACT_DENSITY_UB,
    )
    prune_graph = _prune_graph_to_cpu(prune_graph)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_prune_graph(prune_graph, str(output_path))
    return prune_graph, {"status": "ok", "token_weights": [float(w) for w in token_weights]}


def _cluster_one(
    *,
    prune_graph: PruneGraph,
    output_path: Path,
) -> tuple[SummaryGraph, dict[str, Any]]:
    if output_path.exists():
        return SummaryGraph.load(str(output_path)), {"status": "skipped_existing"}

    clusters = cluster_graph_ilp(prune_graph)
    rows = clusters_to_supernodes(prune_graph, clusters)
    sng = SummaryGraph(
        supernodes=rows,
        pruned_adj=prune_graph.pruned_adj,
        metadata=prune_graph.metadata,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sng.save(str(output_path))
    feature_supernodes = sum(1 for sn in sng.supernodes if sn.type == "features")
    return sng, {"status": "ok", "feature_supernodes": feature_supernodes}


def _label_one(
    *,
    summary_graph: SummaryGraph,
    output_path: Path,
) -> tuple[SummaryGraph, dict[str, Any]]:
    if output_path.exists():
        return SummaryGraph.load(str(output_path)), {"status": "skipped_existing"}

    labelled = label_supernodes(
        summary_graph,
        LABEL_MODEL,
        settings=ModelSettings(temperature=0.2, thinking_effort=None),
        scheme=LabelScheme(scheme="one_pass"),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labelled.save(str(output_path))
    labelled_count = sum(1 for sn in labelled.supernodes if sn.role or sn.description)
    return labelled, {"status": "ok", "labelled_supernodes": labelled_count}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run analogies full pipeline.")
    parser.add_argument("--graphs-root", default="dataset/analogies")
    parser.add_argument("--shap-values-json", default="dataset/analogies/shap_values.json")
    parser.add_argument("--pruned-root", default="pruned_graphs")
    parser.add_argument("--summary-root", default="summary_graphs")
    parser.add_argument("--labeled-root", default="labeled_summary")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    graphs_root = Path(args.graphs_root)
    shap_path = Path(args.shap_values_json)
    pruned_root = Path(args.pruned_root)
    summary_root = Path(args.summary_root)
    labeled_root = Path(args.labeled_root)

    payload = _load_shap_values_json(shap_path)
    by_prompt, by_index = _build_shap_lookup(payload)
    graph_paths = _load_graphs(graphs_root, args.limit)

    settings = {
        "graphs_root": str(graphs_root),
        "shap_values_json": str(shap_path),
        "normalizations": list(NORMALIZATIONS),
        "alpha": ALPHA,
        "node_threshold": NODE_THRESHOLD,
        "entmax_alpha": ENTMAX_ALPHA,
        "edge_threshold": EDGE_THRESHOLD,
        "combine_method": COMBINE_METHOD,
        "score_normalization": SCORE_NORMALIZATION,
        "logit_weights": LOGIT_WEIGHTS,
        "keep_all_tokens_and_logits": KEEP_ALL_TOKENS_AND_LOGITS,
        "filter_act_density": True,
        "act_density_lb": ACT_DENSITY_LB,
        "act_density_ub": ACT_DENSITY_UB,
        "ilp": {
            "theta": DEFAULT_THETA,
            "eps_causal": DEFAULT_EPS_CAUSAL,
            "max_sn": DEFAULT_MAX_SN,
            "max_layer_span": DEFAULT_MAX_LAYER_SPAN,
            "normalize_weights": DEFAULT_NORMALIZE_WEIGHTS,
            "time_limit": DEFAULT_TIME_LIMIT,
        },
        "label_model": LABEL_MODEL,
        "label_temperature": 0.2,
        "label_thinking_effort": None,
        "label_scheme": "one_pass",
        "label_idx_mod": "idx % 10 == 1",
        "device": args.device,
        "limit": args.limit,
    }
    for root in (pruned_root, summary_root, labeled_root):
        _write_json(root / "settings.json", settings)

    prune_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []

    for normalization in NORMALIZATIONS:
        print(f"\n=== normalization={normalization} ===", flush=True)
        prune_dir = _stage_dir(pruned_root, normalization)
        summary_dir = _stage_dir(summary_root, normalization)
        label_dir = _stage_dir(labeled_root, normalization)

        for graph_path in graph_paths:
            stem = graph_path.stem
            print(f"[{normalization}] {stem}", flush=True)
            prune_path = prune_dir / f"{stem}_prune_graph.pt"
            summary_path = summary_dir / f"{stem}_summary_graph.pt"
            label_path = label_dir / f"{stem}_labeled_summary_graph.pt"

            base = {
                "normalization": normalization,
                "graph_file": graph_path.name,
                "graph_stem": stem,
                "graph_path": str(graph_path),
            }
            stage = "prune"
            try:
                attr_graph_for_match = AttrGraph.from_graph(str(graph_path))
                shap_row = _match_shap_row(stem, attr_graph_for_match.metadata, by_prompt, by_index)
                if shap_row is None:
                    raise ValueError("no matching SHAP row (prompt / pNN index)")

                prune_graph, prune_info = _prune_one(
                    graph_path=graph_path,
                    normalization=normalization,
                    shap_row=shap_row,
                    output_path=prune_path,
                    device=args.device,
                )
                prune_rows.append(
                    {
                        **base,
                        **prune_info,
                        "shap_row_index": shap_row.get("index"),
                        "num_nodes": prune_graph.num_nodes,
                        "num_edges": prune_graph.num_edges,
                        "prune_graph_path": str(prune_path),
                    }
                )
                _save_stage_rows(pruned_root, "pruned", prune_rows)

                stage = "cluster"
                sng, cluster_info = _cluster_one(prune_graph=prune_graph, output_path=summary_path)
                cluster_rows.append(
                    {
                        **base,
                        **cluster_info,
                        "num_supernodes": len(sng.supernodes),
                        "summary_graph_path": str(summary_path),
                    }
                )
                _save_stage_rows(summary_root, "summary", cluster_rows)

                if _should_label(stem):
                    stage = "label"
                    labelled, label_info = _label_one(summary_graph=sng, output_path=label_path)
                    label_rows.append(
                        {
                            **base,
                            **label_info,
                            "num_supernodes": len(labelled.supernodes),
                            "labeled_summary_graph_path": str(label_path),
                        }
                    )
                    _save_stage_rows(labeled_root, "labeled", label_rows)

            except Exception as exc:
                tb = traceback.format_exc()
                failure = {
                    **base,
                    "status": "error",
                    "stage": stage,
                    "error": repr(exc),
                    "traceback": tb,
                }
                if stage == "prune":
                    prune_rows.append(failure)
                    _save_stage_rows(pruned_root, "pruned", prune_rows)
                elif stage == "cluster":
                    cluster_rows.append(failure)
                    _save_stage_rows(summary_root, "summary", cluster_rows)
                else:
                    label_rows.append(failure)
                    _save_stage_rows(labeled_root, "labeled", label_rows)
                print(f"[ERROR] {normalization}/{stem}: {exc!r}", flush=True)

    _save_stage_rows(pruned_root, "pruned", prune_rows)
    _save_stage_rows(summary_root, "summary", cluster_rows)
    _save_stage_rows(labeled_root, "labeled", label_rows)
    print("\n=== done ===", flush=True)
    print(f"prune rows: {len(prune_rows)}", flush=True)
    print(f"summary rows: {len(cluster_rows)}", flush=True)
    print(f"label rows: {len(label_rows)}", flush=True)


if __name__ == "__main__":
    main()
