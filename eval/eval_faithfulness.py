"""Causal faithfulness eval for pruning.

For each (AttrGraph, PruneGraph) record in the prune sweep manifest, zero all
CLT features that are in the AttrGraph but missing from the PruneGraph, run a
forward pass via ``ReplacementModel.feature_intervention``, and measure
``P(target | pruned) / P(target | baseline)``. Error nodes, embeddings, and
logits are left untouched.

Baseline P(target) is read from the AttrGraph's target logit node
(``token_prob``), which the upstream attribution code records from a clean
forward pass — no extra baseline run is needed.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from circuit_tracer import ReplacementModel
from eval.eval_prune import plot_pareto_frontier
from summarization.attr_graph import AttrGraph
from summarization.prune import load_prune_graph

logger = logging.getLogger(__name__)

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class AttrGraphCache:
    """Cache AttrGraphs by path — many prune records share the same source graph."""

    def __init__(self) -> None:
        self._cache: dict[str, AttrGraph] = {}

    def get(self, path: str) -> AttrGraph:
        if path not in self._cache:
            self._cache[path] = (
                AttrGraph.from_graph_file(path) if path.endswith(".json") else AttrGraph.from_graph(path)
            )
        return self._cache[path]


def _evaluate_record(
    rec: dict[str, Any],
    cache: AttrGraphCache,
    model: ReplacementModel,
) -> dict[str, Any]:
    attr_graph = cache.get(rec["graph_path"])
    prune_graph = load_prune_graph(rec["prune_graph_path"], map_location="cpu")
    prompt = prune_graph.metadata["prompt"]

    # Target token + baseline P(target) from clean forward pass (stored upstream).
    target_node = next((n for n in attr_graph.nodes if n.is_target_logit), None)
    if target_node is None:
        raise RuntimeError("no target logit node in AttrGraph")
    target_token_id = int(target_node.feature)
    baseline_p = float(target_node.token_prob)

    # Ablate CLT features in AttrGraph but not in PruneGraph.
    kept_ids = {n.node_id for n in prune_graph.nodes}
    interventions: list[tuple[int, int, int, float]] = []
    for n in attr_graph.nodes:
        if n.feature_type != "cross layer transcoder" or n.node_id in kept_ids:
            continue
        # node_id is "{layer}_{feat_idx}_{pos}" — same parsing as eval_intervention.py
        layer_s, feat_s, _ = n.node_id.split("_")
        interventions.append((int(layer_s), n.ctx_idx, int(feat_s), 0.0))

    if not interventions:
        pruned_p = baseline_p
    else:
        logits, _ = model.feature_intervention(
            prompt, interventions, return_activations=False
        )
        last_logits = logits.squeeze(0)[-1] if logits.ndim == 3 else logits[-1]
        probs = F.softmax(last_logits.float(), dim=-1)
        pruned_p = float(probs[target_token_id].item())

    faithfulness = pruned_p / baseline_p if baseline_p > 0 else 0.0

    return {
        "n_nodes": prune_graph.num_nodes,
        "n_edges": prune_graph.num_edges,
        "n_ablated": len(interventions),
        "target_token_id": target_token_id,
        "baseline_p": baseline_p,
        "pruned_p": pruned_p,
        "faithfulness": faithfulness,
    }


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def run_eval(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest)
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    records: list[dict[str, Any]] = list(manifest.get("records") or [])
    if not records and manifest.get("results_json"):
        results_path = Path(manifest["results_json"])
        if results_path.exists():
            with results_path.open("r", encoding="utf-8") as f:
                records = json.load(f)
    if not records:
        raise RuntimeError(f"no records in {manifest_path}")

    if args.prompt_stride is not None and args.prompt_stride > 1:
        # Keep prompts whose numeric stem is divisible by the stride.
        records = [r for r in records if int(r.get("graph_stem", "0")) % args.prompt_stride == 0]
        logger.info("filtered to %d records via prompt-stride=%d", len(records), args.prompt_stride)

    if args.limit is not None:
        records = records[: int(args.limit)]

    output_root = (
        Path(args.output_root) if args.output_root else manifest_path.parent / "eval_faithfulness"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    logger.info("loading model %s / transcoder %s ...", args.model_name, args.transcoder_set)
    model = ReplacementModel.from_pretrained(
        args.model_name,
        args.transcoder_set,
        lazy_encoder=True,
        dtype=DTYPE_MAP[args.dtype],
        device=args.device,
    )

    cache = AttrGraphCache()
    rows: list[dict[str, Any]] = []
    failures: list[str] = []

    n = len(records)
    for i, rec in enumerate(records):
        stem = rec.get("graph_stem", rec.get("graph_file", "?"))
        nt = rec.get("node_threshold")
        alpha = rec.get("alpha")
        try:
            metrics = _evaluate_record(rec, cache, model)
            rows.append({**rec, **metrics})
            if (i + 1) % max(1, args.log_every) == 0:
                logger.info(
                    "[%d/%d] %s a=%s nt=%s n_abl=%d base=%.3f pruned=%.3f faith=%.3f",
                    i + 1, n, stem, alpha, nt,
                    metrics["n_ablated"], metrics["baseline_p"],
                    metrics["pruned_p"], metrics["faithfulness"],
                )
        except Exception as exc:
            msg = f"{stem} a={alpha} nt={nt}: {exc}"
            failures.append(msg)
            logger.warning("[failed] %s", msg)

    results_path = output_root / "results.json"
    summary_path = output_root / "summary.csv"
    eval_manifest_path = output_root / "eval_manifest.json"

    _write_json(results_path, rows)

    if rows:
        csv_fields = [k for k in rows[0].keys() if k != "token_weights"]
        with summary_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    _write_json(eval_manifest_path, {
        "source_manifest": str(manifest_path),
        "output_root": str(output_root),
        "model_name": args.model_name,
        "transcoder_set": args.transcoder_set,
        "dtype": args.dtype,
        "device": args.device,
        "limit": args.limit,
        "n_records_processed": len(rows),
        "n_failures": len(failures),
        "results_json": str(results_path),
        "summary_csv": str(summary_path),
        "failures": failures,
    })

    logger.info("rows=%d failures=%d", len(rows), len(failures))
    logger.info("wrote %s", results_path)
    logger.info("wrote %s", summary_path)
    logger.info("wrote %s", eval_manifest_path)

    if args.plot and rows:
        plot_path = plot_pareto_frontier(
            rows, output_root,
            metrics=[("faithfulness", "P(target | pruned) / P(target | baseline)")],
            filename="faithfulness_frontier.png",
        )
        if plot_path is not None:
            logger.info("wrote %s", plot_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Causal faithfulness eval: ablate CLT features outside the pruned "
        "subgraph, measure P(target token) retained vs baseline."
    )
    parser.add_argument("--manifest", required=True, help="manifest.json from eval/prune_graphs.py")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--model-name", default="google/gemma-2-2b")
    parser.add_argument("--transcoder-set", default="mntss/clt-gemma-2-2b-2.5M")
    parser.add_argument("--dtype", default="bfloat16", choices=list(DTYPE_MAP))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--prompt-stride",
        type=int,
        default=None,
        help="Keep prompts whose int(graph_stem) is divisible by stride (e.g. 5 → every 5th prompt).",
    )
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_eval(args)


if __name__ == "__main__":
    main()
