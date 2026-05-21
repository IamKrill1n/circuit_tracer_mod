"""Causal validation of supernode clusters via feature interventions.

For each supernode produced by spectral and agglomerative clustering:
  Exp B: ablate upstream SN -> measure fractional activation drop in downstream SN
  Exp D: ablate each node individually -> measure intra-cluster cosine similarity
         of logit-delta vectors (vs. inter-cluster baseline)

All methods are compared against the same baselines used in eval_cluster.py:
modularity (K-matched), spectral-rbf, and kmeans, all at our spectral's auto-k.
"""
from __future__ import annotations

import argparse
import csv
import logging
from functools import partial
from itertools import groupby
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from circuit_tracer import ReplacementModel
from eval.eval_cluster import (
    _adjacency_affinity,
    _kmeans_middle_labels,
    _middle_indices,
    _modularity_middle_labels,
    _spectral_cosine_middle_labels,
)
from summarization.auto_grouping import find_best_k, find_best_k_for_clusterer
from summarization.cluster import (
    cluster_graph_agglomerative,
    cluster_graph_spectral,
    clusters_to_supernodes,
    compute_phi_vectors,
    compute_similarity,
    labels_to_supernodes,
)
from summarization.prune import PruneGraph, load_prune_graph
from summarization.supernode_graph import Supernode, SummarizationGraph

logger = logging.getLogger(__name__)

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _clt_interventions(sn: Supernode, value: float = 0.0) -> list[tuple]:
    """Build (layer, pos, feature_idx, value) tuples for all CLT nodes in a supernode."""
    out = []
    for n in sn.features:
        if n.feature_type != "cross layer transcoder":
            continue
        parts = n.node_id.split("_")
        out.append((int(parts[0]), n.ctx_idx, int(parts[1]), value))
    return out


def _mean_activation(sn: Supernode, activations: torch.Tensor) -> float:
    """Mean activation of CLT nodes in a supernode. activations: [n_layers, n_pos, d_tc]"""
    vals = []
    for n in sn.features:
        if n.feature_type != "cross layer transcoder":
            continue
        parts = n.node_id.split("_")
        layer, pos, feat = int(parts[0]), n.ctx_idx, int(parts[1])
        if layer < activations.shape[0] and pos < activations.shape[1] and feat < activations.shape[2]:
            vals.append(activations[layer, pos, feat].item())
    return float(np.mean(vals)) if vals else 0.0


def _last_logits(logits: torch.Tensor) -> torch.Tensor:
    """Extract last-position logit vector from [1, seq, vocab] or [seq, vocab]."""
    return logits.squeeze(0)[-1] if logits.ndim == 3 else logits[-1]


def _intra_cosine(deltas: torch.Tensor) -> float:
    """Mean pairwise cosine similarity among rows of deltas. deltas: [n, vocab]"""
    if deltas.shape[0] < 2:
        return float("nan")
    normed = F.normalize(deltas.float(), dim=-1)  # [n, vocab]
    sim = normed @ normed.T  # [n, n]
    mask = torch.triu(torch.ones(sim.shape[0], sim.shape[0], dtype=torch.bool, device=sim.device), diagonal=1)
    return sim[mask].mean().item()


def _inter_cosine(sn_deltas: dict[str, torch.Tensor], n_samples: int = 1000) -> float:
    """Mean cosine similarity across randomly sampled cross-supernode pairs."""
    names = list(sn_deltas.keys())
    if len(names) < 2:
        return float("nan")
    rng = np.random.default_rng(42)
    pairs_a, pairs_b = [], []
    for _ in range(n_samples):
        i, j = rng.choice(len(names), size=2, replace=False)
        di, dj = sn_deltas[names[i]], sn_deltas[names[j]]
        pairs_a.append(di[rng.integers(len(di))])
        pairs_b.append(dj[rng.integers(len(dj))])
    a = torch.stack(pairs_a).float()
    b = torch.stack(pairs_b).float()
    return F.cosine_similarity(a, b).mean().item()


def _sng_from_clusters(
    prune_graph: PruneGraph, clusters: list[list[str]]
) -> SummarizationGraph:
    return SummarizationGraph(
        supernodes=clusters_to_supernodes(prune_graph, clusters),
        pruned_adj=prune_graph.pruned_adj,
    )


def _build_sngs(
    prune_graph: PruneGraph,
    random_state: int = 42,
    n_init: int = 20,
) -> dict[str, SummarizationGraph]:
    sim = compute_similarity(prune_graph)

    best_k_s, _ = find_best_k(prune_graph, similarity=sim)
    spectral_sng = _sng_from_clusters(
        prune_graph, cluster_graph_spectral(prune_graph, target_k=best_k_s)
    )

    agg_clusterer = partial(cluster_graph_agglomerative, prune_graph)
    best_k_a, _ = find_best_k_for_clusterer(
        prune_graph=prune_graph, similarity=sim, clusterer=agg_clusterer
    )
    agg_sng = _sng_from_clusters(
        prune_graph, cluster_graph_agglomerative(prune_graph, target_k=best_k_a)
    )

    # Baselines (match eval_cluster.py). All three baselines run at K = best_k_s
    # (our spectral's auto-k) so the per-method comparison is at the same supernode count.
    mid_idx = _middle_indices(prune_graph)
    middle_ids = [prune_graph.nodes[i].node_id for i in mid_idx]
    adjacency_mid = _adjacency_affinity(prune_graph)[np.ix_(mid_idx, mid_idx)]
    phi_mid = compute_phi_vectors(prune_graph).detach().cpu().numpy()[mid_idx]

    modularity_labels = _modularity_middle_labels(adjacency_mid, best_k_s)
    modularity_sng = _sng_from_clusters(
        prune_graph, labels_to_supernodes(prune_graph, middle_ids, modularity_labels)
    )

    spectral_cos_labels = _spectral_cosine_middle_labels(
        phi_mid, best_k_s, random_state, n_init
    )
    spectral_cos_sng = _sng_from_clusters(
        prune_graph, labels_to_supernodes(prune_graph, middle_ids, spectral_cos_labels)
    )

    kmeans_labels = _kmeans_middle_labels(phi_mid, best_k_s, random_state, n_init)
    kmeans_sng = _sng_from_clusters(
        prune_graph, labels_to_supernodes(prune_graph, middle_ids, kmeans_labels)
    )

    return {
        "spectral": spectral_sng,
        "agglomerative": agg_sng,
        "baseline-modularity": modularity_sng,
        "baseline-spectral-cosine": spectral_cos_sng,
        "baseline-kmeans": kmeans_sng,
    }


def _evaluate_sng(
    model: ReplacementModel,
    sng: SummarizationGraph,
    prompt: str,
    orig_logits: torch.Tensor,
    orig_activations: torch.Tensor,
    graph_name: str,
    method: str,
) -> tuple[list[dict], list[dict]]:
    edge_rows: list[dict] = []
    sn_rows: list[dict] = []
    orig_last = _last_logits(orig_logits)

    ablation_acts: dict[str, torch.Tensor] = {}
    sn_deltas: dict[str, torch.Tensor] = {}  # sn_name -> [n_nodes, vocab]

    for sn in sng.supernodes:
        if sn.type in ("emb", "logit"):
            continue
        interventions = _clt_interventions(sn)
        if not interventions:
            continue

        # Exp B: whole-supernode ablation for downstream activation measurement
        _, new_acts = model.feature_intervention(
            prompt, interventions, return_activations=True
        )
        ablation_acts[sn.name] = new_acts  # type: ignore[assignment]

        # Exp D: per-node ablation to measure logit-push direction of each feature
        deltas = []
        for layer, pos, feat, val in interventions:
            new_logits, _ = model.feature_intervention(
                prompt, [(layer, pos, feat, val)], return_activations=False
            )
            deltas.append(_last_logits(new_logits) - orig_last)
        sn_deltas[sn.name] = torch.stack(deltas)  # [n_nodes, vocab]

        intra = _intra_cosine(sn_deltas[sn.name])
        sn_rows.append(
            {
                "graph": graph_name,
                "supernode_id": sn.name,
                "method": method,
                "n_clt_features": len(interventions),
                "intra_cosine": intra,
            }
        )

    # Exp B: edge knockout — for each S_A → S_B edge, measure activation drop in S_B
    # after ablating S_A. sn_adj[target, source] same convention as pruned_adj.
    sn_adj = sng.sn_adj
    for i, target_sn in enumerate(sng.supernodes):
        for j, source_sn in enumerate(sng.supernodes):
            if i == j:
                continue
            edge_weight = float(sn_adj[i, j])
            if edge_weight <= 0 or source_sn.name not in ablation_acts:
                continue
            orig_mean = _mean_activation(target_sn, orig_activations)
            new_mean = _mean_activation(target_sn, ablation_acts[source_sn.name])
            edge_rows.append(
                {
                    "graph": graph_name,
                    "src_sn": source_sn.name,
                    "tgt_sn": target_sn.name,
                    "method": method,
                    "edge_weight": edge_weight,
                    "activation_drop_ratio": new_mean / (orig_mean + 1e-9),
                }
            )

    # Exp D: inter-cluster cosine (graph-level) — attach to each sn_row for convenience
    inter = _inter_cosine(sn_deltas)
    for row in sn_rows:
        row["inter_cosine"] = inter
        row["cosine_gap"] = row["intra_cosine"] - inter

    return edge_rows, sn_rows


def evaluate_graph(
    model: ReplacementModel,
    prune_graph: PruneGraph,
    graph_name: str,
) -> tuple[list[dict], list[dict]]:
    prompt: str = prune_graph.metadata["prompt"]
    orig_logits, orig_activations = model.get_activations(prompt)

    sngs = _build_sngs(prune_graph)
    all_edge: list[dict] = []
    all_sn: list[dict] = []
    for method, sng in sngs.items():
        logger.info("  method=%s  n_supernodes=%d", method, len(sng.supernodes))
        edge_rows, sn_rows = _evaluate_sng(
            model, sng, prompt, orig_logits, orig_activations, graph_name, method
        )
        all_edge.extend(edge_rows)
        all_sn.extend(sn_rows)
    return all_edge, all_sn


def _compute_summary(edge_rows: list[dict], sn_rows: list[dict]) -> list[dict]:
    summary = []
    key = lambda r: (r["graph"], r["method"])
    for (graph, method), group in groupby(sorted(edge_rows, key=key), key=key):
        items = list(group)
        rho_edge = float("nan")
        if len(items) >= 3:
            weights = [r["edge_weight"] for r in items]
            drops = [1.0 - r["activation_drop_ratio"] for r in items]
            rho_edge = spearmanr(weights, drops).statistic

        sn_items = [r for r in sn_rows if r["graph"] == graph and r["method"] == method]
        mean_intra = float(np.mean([r["intra_cosine"] for r in sn_items if not np.isnan(r["intra_cosine"])])) if sn_items else float("nan")
        mean_inter = float(np.mean([r["inter_cosine"] for r in sn_items if not np.isnan(r["inter_cosine"])])) if sn_items else float("nan")
        mean_gap = float(np.mean([r["cosine_gap"] for r in sn_items if not np.isnan(r["cosine_gap"])])) if sn_items else float("nan")

        summary.append(
            {
                "graph": graph,
                "method": method,
                "n_edges": len(items),
                "spearman_edge_drop": rho_edge,
                "mean_intra_cosine": mean_intra,
                "mean_inter_cosine": mean_inter,
                "mean_cosine_gap": mean_gap,
            }
        )
    return summary


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Causal validation of supernode clusters via feature interventions"
    )
    parser.add_argument("--prune-graphs-dir", required=True, type=Path)
    parser.add_argument("--model-name", default="google/gemma-2-2b")
    parser.add_argument("--transcoder-set", default="mntss/clt-gemma-2-2b-2.5M")
    parser.add_argument("--dtype", default="bfloat16", choices=list(DTYPE_MAP))
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None, help="Max graphs to process")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    graph_paths = sorted(args.prune_graphs_dir.glob("*.pt"))
    if args.limit:
        graph_paths = graph_paths[: args.limit]
    if not graph_paths:
        logger.error("No .pt files found in %s", args.prune_graphs_dir)
        return

    logger.info("Loading model %s / transcoder %s ...", args.model_name, args.transcoder_set)
    model = ReplacementModel.from_pretrained(
        args.model_name,
        args.transcoder_set,
        lazy_encoder=True,
        dtype=DTYPE_MAP[args.dtype],
        device=args.device,
    )

    all_edge: list[dict] = []
    all_sn: list[dict] = []
    for path in graph_paths:
        graph_name = path.stem
        logger.info("Processing %s ...", graph_name)
        prune_graph = load_prune_graph(str(path))
        edge_rows, sn_rows = evaluate_graph(model, prune_graph, graph_name)
        all_edge.extend(edge_rows)
        all_sn.extend(sn_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "edge_results.csv", all_edge)
    _write_csv(args.output_dir / "supernode_results.csv", all_sn)

    summary = _compute_summary(all_edge, all_sn)
    _write_csv(args.output_dir / "summary.csv", summary)

    header = f"{'graph':30s} {'method':15s} {'n_edges':>8s} {'ρ(edge,drop)':>12s} {'intra_cos':>10s} {'inter_cos':>10s} {'gap':>8s}"
    print(header)
    print("-" * len(header))
    for row in summary:
        print(
            f"{row['graph']:30s} {row['method']:15s} {row['n_edges']:>8d} "
            f"{row['spearman_edge_drop']:>12.3f} {row['mean_intra_cosine']:>10.3f} "
            f"{row['mean_inter_cosine']:>10.3f} {row['mean_cosine_gap']:>8.3f}"
        )
    logger.info("Done. Output in %s", args.output_dir)


if __name__ == "__main__":
    main()
