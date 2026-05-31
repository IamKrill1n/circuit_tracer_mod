"""Causal validation of supernode clusters via feature interventions.

For each supernode produced by spectral and agglomerative clustering:
  Exp B: steer the whole SN (value = factor * orig_activation; factor=-1 negates,
         matching the paper) -> measure the effect on every other SN (downstream
         feature SNs: activation ratio; logit SNs: Δ token probability) and on the
         target token (Δ probability). factor=0 reproduces pure ablation/knockout.
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
from summarization.cluster import (
    cluster_graph_agglomerative,
    cluster_graph_spectral,
    clusters_to_supernodes,
    compute_phi_vectors,
    find_best_k,
    find_best_k_for_clusterer,
    labels_to_supernodes,
)
from summarization.prune import PruneGraph, load_prune_graph
from summarization.summarize import Supernode, SummaryGraph

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


def _steer_interventions(
    sn: Supernode, orig_activations: torch.Tensor, factor: float
) -> list[tuple]:
    """Steering interventions for all CLT nodes in a supernode.

    value = factor * orig_activation per node. factor=-1 negates (paper's steering);
    factor=0 reproduces ablation. orig_activations: [n_layers, n_pos, d_tc].
    """
    out = []
    for n in sn.features:
        if n.feature_type != "cross layer transcoder":
            continue
        parts = n.node_id.split("_")
        layer, pos, feat = int(parts[0]), n.ctx_idx, int(parts[1])
        if layer < orig_activations.shape[0] and pos < orig_activations.shape[1] and feat < orig_activations.shape[2]:
            orig_val = orig_activations[layer, pos, feat].item()
            out.append((layer, pos, feat, factor * orig_val))
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


def _last_probs(logits: torch.Tensor) -> torch.Tensor:
    """Softmax over the last-position logit vector. Returns [vocab] on cpu/float."""
    return _last_logits(logits).float().softmax(-1)


def _logit_supernode_prob_delta(
    sn: Supernode, new_probs: torch.Tensor, orig_probs: torch.Tensor
) -> float:
    """Mean Δ probability over a logit supernode's member tokens (node.feature = vocab id)."""
    deltas = [
        new_probs[n.feature].item() - orig_probs[n.feature].item()
        for n in sn.features
        if n.feature_type == "logit"
    ]
    return float(np.mean(deltas)) if deltas else 0.0


def _target_token(sng: SummaryGraph) -> tuple[int, float] | None:
    """(vocab_id, orig_prob) of the target logit node (is_target_logit), if present."""
    for sn in sng.supernodes:
        if sn.type != "logit":
            continue
        for n in sn.features:
            if n.is_target_logit:
                return int(n.feature), float(n.token_prob)
    return None


def steer_source_effects(
    model: ReplacementModel,
    sng: SummaryGraph,
    source_sn: Supernode,
    prompt: str,
    orig_logits: torch.Tensor,
    orig_activations: torch.Tensor,
    factor: float,
) -> tuple[list[dict], list[dict]]:
    """Steer one supernode -> effect on every other SN + the target token.

    Runs a single steering intervention and reports, for ``source_sn``:
      - feature target SNs: activation ratio (new/orig) and signed delta,
      - logit target SNs: mean Δ token probability,
      - the global target token: Δ probability.
    Returns (effect_rows, target_token_rows). Rows omit graph/method, which the
    eval caller attaches; app.py uses them directly. ``sn_adj[t, s]`` = source s → target t.
    """
    interventions = _steer_interventions(source_sn, orig_activations, factor)
    new_logits, new_acts = model.feature_intervention(
        prompt, interventions, return_activations=True
    )
    orig_probs = _last_probs(orig_logits)
    new_probs = _last_probs(new_logits)

    sn_adj = sng.adj_matrix  # [tgt, src]
    names = [s.name for s in sng.supernodes]
    src_idx = names.index(source_sn.name)

    effect_rows: list[dict] = []
    for t_idx, tgt in enumerate(sng.supernodes):
        if tgt.name == source_sn.name or tgt.type == "emb":
            continue
        edge_weight = float(sn_adj[t_idx, src_idx])
        row = {
            "src_sn": source_sn.name,
            "tgt_sn": tgt.name,
            "tgt_type": "logit" if tgt.type == "logit" else "feature",
            "edge_weight": edge_weight,
            "factor": factor,
            "activation_ratio": float("nan"),
            "activation_delta": float("nan"),
            "prob_delta": float("nan"),
        }
        if tgt.type == "logit":
            row["prob_delta"] = _logit_supernode_prob_delta(tgt, new_probs, orig_probs)
        else:
            orig_mean = _mean_activation(tgt, orig_activations)
            new_mean = _mean_activation(tgt, new_acts)
            row["activation_ratio"] = new_mean / (orig_mean + 1e-9)
            row["activation_delta"] = new_mean - orig_mean
        effect_rows.append(row)

    target_rows: list[dict] = []
    tgt_tok = _target_token(sng)
    if tgt_tok is not None:
        vid, orig_p = tgt_tok
        new_p = float(new_probs[vid].item())
        target_rows.append(
            {
                "src_sn": source_sn.name,
                "factor": factor,
                "target_token": model.tokenizer.decode([vid]).strip(),
                "target_vid": vid,
                "orig_prob": orig_p,
                "new_prob": new_p,
                "prob_delta": new_p - orig_p,
            }
        )
    return effect_rows, target_rows


def _sng_from_clusters(
    prune_graph: PruneGraph, clusters: list[list[str]]
) -> SummaryGraph:
    return SummaryGraph(
        supernodes=clusters_to_supernodes(prune_graph, clusters),
        pruned_adj=prune_graph.pruned_adj,
    )


def _build_sngs(
    prune_graph: PruneGraph,
    random_state: int = 42,
    n_init: int = 20,
) -> dict[str, SummaryGraph]:
    best_k_s, _ = find_best_k(prune_graph)
    spectral_sng = _sng_from_clusters(
        prune_graph, cluster_graph_spectral(prune_graph, target_k=best_k_s)
    )

    agg_clusterer = partial(cluster_graph_agglomerative, prune_graph)
    best_k_a, _ = find_best_k_for_clusterer(
        prune_graph=prune_graph, clusterer=agg_clusterer
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
    sng: SummaryGraph,
    prompt: str,
    orig_logits: torch.Tensor,
    orig_activations: torch.Tensor,
    graph_name: str,
    method: str,
    factor: float,
) -> tuple[list[dict], list[dict], list[dict]]:
    edge_rows: list[dict] = []
    sn_rows: list[dict] = []
    logit_rows: list[dict] = []
    orig_last = _last_logits(orig_logits)

    sn_deltas: dict[str, torch.Tensor] = {}  # sn_name -> [n_nodes, vocab]

    for sn in sng.supernodes:
        if sn.type in ("emb", "logit"):
            continue
        interventions = _clt_interventions(sn)
        if not interventions:
            continue

        # Exp D: per-node ablation to measure logit-push direction of each feature
        deltas = []
        for layer, pos, feat, val in interventions:
            new_logits, _ = model.feature_intervention(
                prompt, [(layer, pos, feat, val)], return_activations=False
            )
            deltas.append(_last_logits(new_logits) - orig_last)
        sn_deltas[sn.name] = torch.stack(deltas)  # [n_nodes, vocab]
        sn_rows.append(
            {
                "graph": graph_name,
                "supernode_id": sn.name,
                "method": method,
                "n_clt_features": len(interventions),
                "intra_cosine": _intra_cosine(sn_deltas[sn.name]),
            }
        )

        # Exp B: steer the whole supernode -> effect on every other SN + the target token
        eff_rows, tgt_rows = steer_source_effects(
            model, sng, sn, prompt, orig_logits, orig_activations, factor
        )
        for r in eff_rows + tgt_rows:
            r["graph"] = graph_name
            r["method"] = method
        edge_rows.extend(eff_rows)
        logit_rows.extend(tgt_rows)

    # Exp D: inter-cluster cosine (graph-level) — attach to each sn_row for convenience
    inter = _inter_cosine(sn_deltas)
    for row in sn_rows:
        row["inter_cosine"] = inter
        row["cosine_gap"] = row["intra_cosine"] - inter

    return edge_rows, sn_rows, logit_rows


def evaluate_graph(
    model: ReplacementModel,
    prune_graph: PruneGraph,
    graph_name: str,
    factor: float,
) -> tuple[list[dict], list[dict], list[dict]]:
    prompt: str = prune_graph.metadata["prompt"]
    orig_logits, orig_activations = model.get_activations(prompt)

    sngs = _build_sngs(prune_graph)
    all_edge: list[dict] = []
    all_sn: list[dict] = []
    all_logit: list[dict] = []
    for method, sng in sngs.items():
        logger.info("  method=%s  n_supernodes=%d", method, len(sng.supernodes))
        edge_rows, sn_rows, logit_rows = _evaluate_sng(
            model, sng, prompt, orig_logits, orig_activations, graph_name, method, factor
        )
        all_edge.extend(edge_rows)
        all_sn.extend(sn_rows)
        all_logit.extend(logit_rows)
    return all_edge, all_sn, all_logit


def _compute_summary(
    edge_rows: list[dict], sn_rows: list[dict], logit_rows: list[dict]
) -> list[dict]:
    summary = []
    keys = sorted({(r["graph"], r["method"]) for r in edge_rows + sn_rows + logit_rows})
    for graph, method in keys:
        items = [r for r in edge_rows if r["graph"] == graph and r["method"] == method]

        # Feature-target edges drive the edge-effect correlation. Under steering
        # (factor != 0) the response is not a clean knockout, so correlate edge weight
        # against the *magnitude* of the activation change |1 - ratio| (factor-agnostic).
        feat_items = [r for r in items if r["tgt_type"] == "feature" and r["edge_weight"] > 0]
        rho_edge = float("nan")
        if len(feat_items) >= 3:
            weights = [r["edge_weight"] for r in feat_items]
            effects = [abs(1.0 - r["activation_ratio"]) for r in feat_items]
            rho_edge = spearmanr(weights, effects).statistic

        logit_items = [r for r in items if r["tgt_type"] == "logit"]
        mean_logit_sn = float(np.mean([abs(r["prob_delta"]) for r in logit_items])) if logit_items else float("nan")

        tgt_items = [r for r in logit_rows if r["graph"] == graph and r["method"] == method]
        mean_target = float(np.mean([abs(r["prob_delta"]) for r in tgt_items])) if tgt_items else float("nan")

        sn_items = [r for r in sn_rows if r["graph"] == graph and r["method"] == method]
        mean_intra = float(np.mean([r["intra_cosine"] for r in sn_items if not np.isnan(r["intra_cosine"])])) if sn_items else float("nan")
        mean_inter = float(np.mean([r["inter_cosine"] for r in sn_items if not np.isnan(r["inter_cosine"])])) if sn_items else float("nan")
        mean_gap = float(np.mean([r["cosine_gap"] for r in sn_items if not np.isnan(r["cosine_gap"])])) if sn_items else float("nan")

        summary.append(
            {
                "graph": graph,
                "method": method,
                "n_edges": len(feat_items),
                "spearman_edge_effect": rho_edge,
                "mean_target_abs_prob_delta": mean_target,
                "mean_logit_sn_abs_prob_delta": mean_logit_sn,
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
    parser.add_argument(
        "--steer-factor",
        type=float,
        default=-1.0,
        help="Multiplicative steering factor (value = factor * orig_activation); "
        "-1 negates (paper), 0 reproduces ablation/knockout.",
    )
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
    all_logit: list[dict] = []
    for path in graph_paths:
        graph_name = path.stem
        logger.info("Processing %s ...", graph_name)
        prune_graph = load_prune_graph(str(path))
        edge_rows, sn_rows, logit_rows = evaluate_graph(
            model, prune_graph, graph_name, args.steer_factor
        )
        all_edge.extend(edge_rows)
        all_sn.extend(sn_rows)
        all_logit.extend(logit_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "edge_results.csv", all_edge)
    _write_csv(args.output_dir / "supernode_results.csv", all_sn)
    _write_csv(args.output_dir / "logit_results.csv", all_logit)

    summary = _compute_summary(all_edge, all_sn, all_logit)
    _write_csv(args.output_dir / "summary.csv", summary)

    header = f"{'graph':30s} {'method':15s} {'n_edges':>8s} {'ρ(edge,eff)':>12s} {'tgtΔp':>8s} {'logitΔp':>8s} {'intra':>7s} {'inter':>7s} {'gap':>7s}"
    print(header)
    print("-" * len(header))
    for row in summary:
        print(
            f"{row['graph']:30s} {row['method']:15s} {row['n_edges']:>8d} "
            f"{row['spearman_edge_effect']:>12.3f} {row['mean_target_abs_prob_delta']:>8.3f} "
            f"{row['mean_logit_sn_abs_prob_delta']:>8.3f} {row['mean_intra_cosine']:>7.3f} "
            f"{row['mean_inter_cosine']:>7.3f} {row['mean_cosine_gap']:>7.3f}"
        )
    logger.info("Done. Output in %s", args.output_dir)


if __name__ == "__main__":
    main()
