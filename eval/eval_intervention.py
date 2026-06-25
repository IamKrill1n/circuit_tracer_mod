"""Causal validation of supernode clusters via feature interventions.

For each supernode in either a saved summary graph or a clustering rebuilt from a prune graph:
  Exp B: constrained-steer the whole SN (value = factor * orig_activation; factor=-1
         negates, matching the paper; 0 ablates) over the direct-effect window
         [l-below, l+above] (default [l, l+1], paper Fig. 9) -> measure the effect on every other SN
         (downstream feature SNs: activation ratio; logit SNs: Δ token probability) and
         on the target token (Δ probability). Edge faithfulness: fraction of summary edges
         whose downstream sign matches sign(edge_weight)·sign(Δa_source), |weight|-weighted.
  Exp D: ablate each node individually -> measure intra-cluster cosine similarity
         of logit-delta vectors (vs. inter-cluster baseline)

When prune graphs are used as input, methods are compared against the same baselines used in
eval_cluster.py: modularity (K-matched), spectral-rbf, and kmeans, all at our spectral's auto-k.
When summary graphs are used as input, the saved summary graph is evaluated as-is.
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
from eval.legacy_cluster_baselines import (
    cluster_graph_agglomerative,
    cluster_graph_spectral,
    find_best_k,
    find_best_k_for_clusterer,
)
from summarization.cluster import (
    clusters_to_supernodes,
    compute_phi_vectors,
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

# Ordering used for output; ILP is opt-in (only tractable on small graphs).
ALL_METHODS = (
    "spectral",
    "agglomerative",
    "ilp",
    "baseline-modularity",
    "baseline-spectral-cosine",
    "baseline-kmeans",
)
DEFAULT_METHODS = tuple(m for m in ALL_METHODS if m != "ilp")


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
        if (
            layer < orig_activations.shape[0]
            and pos < orig_activations.shape[1]
            and feat < orig_activations.shape[2]
        ):
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
        if (
            layer < activations.shape[0]
            and pos < activations.shape[1]
            and feat < activations.shape[2]
        ):
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
    mask = torch.triu(
        torch.ones(sim.shape[0], sim.shape[0], dtype=torch.bool, device=sim.device), diagonal=1
    )
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
    inputs: str | torch.Tensor,
    orig_logits: torch.Tensor,
    orig_activations: torch.Tensor,
    factor: float,
    layers_below: int = 0,
    layers_above: int = 1,
) -> tuple[list[dict], list[dict]]:
    """Constrained-steer one supernode -> effect on every other SN + the target token.

    Runs one *constrained* (direct-effect) steering pass over the window
    [layer_min-layers_below, layer_max+layers_above], so each source feature contributes its
    decode only within that window (cross-layer writes above it are dropped) — CLTs decode only
    into layers >= l, so any slot below the source layer is empty. The default (below=0,
    above=1) spans [layer_min, layer_max+1]; for a single-layer supernode this is the paper's
    Fig. 9 window [l, l+1] (own-layer decode + first cross-layer write). Reports, for
    ``source_sn``:
      - feature target SNs: activation ratio (new/orig) and signed delta,
      - logit target SNs: mean Δ token probability,
      - the global target token: Δ probability,
      - per-edge sign consistency: does sign(edge_weight)·sign(Δa_source) match the measured
        effect's sign? The source's own change is known by construction (value=factor·orig ⇒
        Δa_source=(factor-1)·orig); constrained patching alters the decode, not the encoder read,
        so it is not recoverable from the cache.
    Returns (effect_rows, target_token_rows). ``sn_adj[t, s]`` = source s → target t.
    """
    interventions = _steer_interventions(source_sn, orig_activations, factor)
    n_layers = orig_activations.shape[0]
    window = range(
        max(0, source_sn.layer_min - layers_below),
        min(n_layers, source_sn.layer_max + layers_above + 1),
    )
    new_logits, new_acts = model.feature_intervention(
        inputs,
        interventions,
        constrained_layers=window,
        freeze_attention=True,
        return_activations=True,
    )
    orig_probs = _last_probs(orig_logits)
    new_probs = _last_probs(new_logits)

    sn_adj = sng.adj_matrix  # [tgt, src]
    names = [s.name for s in sng.supernodes]
    src_idx = names.index(source_sn.name)

    src_orig = _mean_activation(source_sn, orig_activations)
    src_delta = (factor - 1.0) * src_orig  # the change the steering imposes on the source

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
            tgt_delta = row["prob_delta"]
        else:
            orig_mean = _mean_activation(tgt, orig_activations)
            new_mean = _mean_activation(tgt, new_acts)
            row["activation_ratio"] = new_mean / (orig_mean + 1e-9)
            row["activation_delta"] = new_mean - orig_mean
            tgt_delta = row["activation_delta"]
        # Edge-faithfulness sign test: direct effect is Δtgt ≈ edge_weight · Δa_source.
        predicted = float(np.sign(edge_weight) * np.sign(src_delta))
        measured = float(np.sign(tgt_delta))
        row["predicted_sign"] = predicted
        row["measured_sign"] = measured
        row["sign_consistent"] = bool(
            predicted != 0.0 and measured != 0.0 and predicted == measured
        )
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


def _sng_from_clusters(prune_graph: PruneGraph, clusters: list[list[str]]) -> SummaryGraph:
    return SummaryGraph(
        supernodes=clusters_to_supernodes(prune_graph, clusters),
        pruned_adj=prune_graph.pruned_adj,
    )


def _feature_supernode_count(sng: SummaryGraph) -> int:
    return sum(1 for s in sng.supernodes if s.type not in ("emb", "logit"))


def _prune_graph_stem(path: Path) -> str:
    stem = path.with_suffix("").name
    if stem.endswith("_prune_graph"):
        stem = stem.removesuffix("_prune_graph")
    return stem


def _summary_graph_path_for_prune_graph(prune_graph_path: Path, summary_graphs_dir: Path) -> Path:
    return summary_graphs_dir / f"{_prune_graph_stem(prune_graph_path)}_summary_graph.pt"


def _baseline_k_from_summary_graph(prune_graph_path: Path, summary_graphs_dir: Path) -> int:
    summary_path = _summary_graph_path_for_prune_graph(prune_graph_path, summary_graphs_dir)
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"No summary graph for {prune_graph_path.name}: expected {summary_path}"
        )
    return _feature_supernode_count(SummaryGraph.load(str(summary_path)))


def _build_sngs(
    prune_graph: PruneGraph,
    methods: list[str],
    random_state: int = 42,
    n_init: int = 20,
    ilp_max_sn: int | None = None,
    ilp_max_layer_span: int = 4,
    ilp_time_limit: float = 30.0,
    match_baseline_k_to_ilp: bool = False,
    baseline_k_override: int | None = None,
) -> dict[str, SummaryGraph]:
    """Build a SummaryGraph per requested clustering method. Only the requested methods are
    computed, so selecting a single method skips the others' clustering work.

    ``match_baseline_k_to_ilp`` runs the baselines at ILP's feature-supernode count (solving
    ILP for K even when it is not itself a requested method), so a baseline can be compared to
    ILP at equal granularity rather than at spectral's auto-k.

    ``baseline_k_override`` pins baselines to a caller-supplied K (e.g. from a saved summary
    graph) without re-solving ILP."""
    wanted = set(methods)
    sngs: dict[str, SummaryGraph] = {}

    # ILP is solved if requested as a method, or if baselines must be K-matched to it.
    ilp_k: int | None = None
    if "ilp" in wanted or match_baseline_k_to_ilp:
        from summarization.cluster import cluster_graph_ilp

        clusters = cluster_graph_ilp(
            prune_graph,
            theta=0.0,
            max_sn=ilp_max_sn,
            max_layer_span=ilp_max_layer_span,
            time_limit=ilp_time_limit,
        )
        ilp_sng = _sng_from_clusters(prune_graph, clusters)
        ilp_k = sum(1 for s in ilp_sng.supernodes if s.type not in ("emb", "logit"))
        if "ilp" in wanted:
            sngs["ilp"] = ilp_sng

    if "spectral" in wanted:
        best_k_s, _ = find_best_k(prune_graph)
        sngs["spectral"] = _sng_from_clusters(
            prune_graph, cluster_graph_spectral(prune_graph, target_k=best_k_s)
        )

    if "agglomerative" in wanted:
        agg_clusterer = partial(cluster_graph_agglomerative, prune_graph)
        best_k_a, _ = find_best_k_for_clusterer(prune_graph=prune_graph, clusterer=agg_clusterer)
        sngs["agglomerative"] = _sng_from_clusters(
            prune_graph, cluster_graph_agglomerative(prune_graph, target_k=best_k_a)
        )

    # Baselines (match eval_cluster.py). They run at K = best_k_s (spectral's auto-k) so the
    # per-method comparison is at the same supernode count — unless K-matched to ILP instead.
    baseline_methods = {"baseline-modularity", "baseline-spectral-cosine", "baseline-kmeans"}
    if wanted & baseline_methods:
        if baseline_k_override is not None:
            baseline_k = baseline_k_override
        elif match_baseline_k_to_ilp and ilp_k is not None:
            baseline_k = ilp_k
        else:
            baseline_k = find_best_k(prune_graph)[0]
        mid_idx = _middle_indices(prune_graph)
        middle_ids = [prune_graph.nodes[i].node_id for i in mid_idx]
        adjacency_mid = _adjacency_affinity(prune_graph)[np.ix_(mid_idx, mid_idx)]
        phi_mid = compute_phi_vectors(prune_graph).detach().cpu().numpy()[mid_idx]
        baseline_labels = {
            "baseline-modularity": _modularity_middle_labels(adjacency_mid, baseline_k),
            "baseline-spectral-cosine": _spectral_cosine_middle_labels(
                phi_mid, baseline_k, random_state, n_init
            ),
            "baseline-kmeans": _kmeans_middle_labels(phi_mid, baseline_k, random_state, n_init),
        }
        for name, labels in baseline_labels.items():
            if name in wanted:
                sngs[name] = _sng_from_clusters(
                    prune_graph, labels_to_supernodes(prune_graph, middle_ids, labels)
                )

    return {m: sngs[m] for m in ALL_METHODS if m in sngs}


def _evaluate_sng(
    model: ReplacementModel,
    sng: SummaryGraph,
    inputs: str | torch.Tensor,
    orig_logits: torch.Tensor,
    orig_activations: torch.Tensor,
    graph_name: str,
    method: str,
    factor: float,
    layers_below: int,
    layers_above: int,
    run_exp_d: bool = True,
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

        if run_exp_d:
            # Exp D: per-node ablation to measure logit-push direction of each feature. Left
            # unconstrained on purpose — this is a cluster-cohesion signature, not an edge test.
            deltas = []
            for layer, pos, feat, val in interventions:
                new_logits, _ = model.feature_intervention(
                    inputs, [(layer, pos, feat, val)], return_activations=False
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

        # Exp B: constrained-steer the supernode -> effect on every other SN + the target token
        eff_rows, tgt_rows = steer_source_effects(
            model,
            sng,
            sn,
            inputs,
            orig_logits,
            orig_activations,
            factor,
            layers_below=layers_below,
            layers_above=layers_above,
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
    layers_below: int,
    layers_above: int,
    methods: list[str],
    run_exp_d: bool = True,
    ilp_kwargs: dict | None = None,
    match_baseline_k_to_ilp: bool = False,
    baseline_k_override: int | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    prompt: str = prune_graph.metadata["prompt"]
    # The stored prompt begins with a literal "<bos>"; re-feeding it as a raw string would let
    # the tokenizer prepend a *second* BOS (gemma default), shifting every position by one so each
    # node's ctx_idx reads the wrong activation and steering silently no-ops. ensure_tokenized is
    # idempotent on an existing BOS — tokenize once and reuse the tensor for every pass.
    inputs = model.ensure_tokenized(prompt)
    orig_logits, orig_activations = model.get_activations(inputs)

    sngs = _build_sngs(
        prune_graph,
        methods,
        match_baseline_k_to_ilp=match_baseline_k_to_ilp,
        baseline_k_override=baseline_k_override,
        **(ilp_kwargs or {}),
    )
    all_edge: list[dict] = []
    all_sn: list[dict] = []
    all_logit: list[dict] = []
    for method, sng in sngs.items():
        logger.info("  method=%s  n_supernodes=%d", method, len(sng.supernodes))
        edge_rows, sn_rows, logit_rows = _evaluate_sng(
            model,
            sng,
            inputs,
            orig_logits,
            orig_activations,
            graph_name,
            method,
            factor,
            layers_below,
            layers_above,
            run_exp_d=run_exp_d,
        )
        all_edge.extend(edge_rows)
        all_sn.extend(sn_rows)
        all_logit.extend(logit_rows)
    return all_edge, all_sn, all_logit


def evaluate_summary_graph(
    model: ReplacementModel,
    sng: SummaryGraph,
    graph_name: str,
    method_name: str,
    factor: float,
    layers_below: int,
    layers_above: int,
    run_exp_d: bool = True,
) -> tuple[list[dict], list[dict], list[dict]]:
    prompt = sng.metadata.get("prompt")
    if not prompt:
        raise ValueError(f"{graph_name} summary graph is missing metadata['prompt']")

    # The stored prompt begins with a literal "<bos>"; re-feeding it as a raw string would let
    # the tokenizer prepend a *second* BOS (gemma default), shifting every position by one so each
    # node's ctx_idx reads the wrong activation and steering silently no-ops. ensure_tokenized is
    # idempotent on an existing BOS — tokenize once and reuse the tensor for every pass.
    inputs = model.ensure_tokenized(str(prompt))
    orig_logits, orig_activations = model.get_activations(inputs)
    logger.info("  method=%s  n_supernodes=%d", method_name, len(sng.supernodes))
    return _evaluate_sng(
        model,
        sng,
        inputs,
        orig_logits,
        orig_activations,
        graph_name,
        method_name,
        factor,
        layers_below,
        layers_above,
        run_exp_d=run_exp_d,
    )


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
        mean_logit_sn = (
            float(np.mean([abs(r["prob_delta"]) for r in logit_items]))
            if logit_items
            else float("nan")
        )

        # Edge faithfulness: over all real (nonzero) edges with a measurable source & target
        # effect, fraction whose downstream sign matches sign(edge_weight)·sign(Δa_source),
        # |edge_weight|-weighted so the structural edges dominate.
        sign_items = [
            r
            for r in items
            if r["edge_weight"] != 0.0 and r["predicted_sign"] != 0.0 and r["measured_sign"] != 0.0
        ]
        total_w = sum(abs(r["edge_weight"]) for r in sign_items)
        sign_consistency = (
            sum(abs(r["edge_weight"]) for r in sign_items if r["sign_consistent"]) / total_w
            if total_w
            else float("nan")
        )

        tgt_items = [r for r in logit_rows if r["graph"] == graph and r["method"] == method]
        mean_target = (
            float(np.mean([abs(r["prob_delta"]) for r in tgt_items])) if tgt_items else float("nan")
        )

        sn_items = [r for r in sn_rows if r["graph"] == graph and r["method"] == method]
        mean_intra = (
            float(np.mean([r["intra_cosine"] for r in sn_items if not np.isnan(r["intra_cosine"])]))
            if sn_items
            else float("nan")
        )
        mean_inter = (
            float(np.mean([r["inter_cosine"] for r in sn_items if not np.isnan(r["inter_cosine"])]))
            if sn_items
            else float("nan")
        )
        mean_gap = (
            float(np.mean([r["cosine_gap"] for r in sn_items if not np.isnan(r["cosine_gap"])]))
            if sn_items
            else float("nan")
        )

        summary.append(
            {
                "graph": graph,
                "method": method,
                "n_edges": len(feat_items),
                "n_sign_edges": len(sign_items),
                "sign_consistency": sign_consistency,
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


def _summary_graph_paths(summary_graph: Path | None, summary_graphs_dir: Path | None) -> list[Path]:
    if summary_graph is not None:
        if not summary_graph.is_file():
            raise FileNotFoundError(f"summary graph file not found: {summary_graph}")
        return [summary_graph]
    if summary_graphs_dir is None:
        return []
    paths = sorted(p for p in summary_graphs_dir.rglob("*_summary_graph.pt") if p.is_file())
    if not paths:
        raise FileNotFoundError(f"No *_summary_graph.pt files found under {summary_graphs_dir}")
    return paths


def _prune_graph_paths(prune_graphs_dir: Path | None) -> list[Path]:
    if prune_graphs_dir is None:
        return []
    paths = sorted(prune_graphs_dir.glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No .pt files found in {prune_graphs_dir}")
    return paths


def _summary_graph_name(path: Path, root: Path | None) -> str:
    stem_path = path.with_suffix("")
    if stem_path.name.endswith("_summary_graph"):
        stem_path = stem_path.with_name(stem_path.name.removesuffix("_summary_graph"))
    if root is None:
        return stem_path.name
    try:
        rel = stem_path.relative_to(root)
    except ValueError:
        return stem_path.name
    return "__".join(rel.parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Causal validation of supernode clusters via feature interventions"
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--prune-graphs-dir",
        type=Path,
        help="Directory containing PruneGraph .pt files; clustering methods are rebuilt.",
    )
    inputs.add_argument(
        "--summary-graphs-dir",
        type=Path,
        help="Directory tree containing *_summary_graph.pt files; saved summary graphs are evaluated as-is.",
    )
    inputs.add_argument(
        "--summary-graph",
        type=Path,
        help="Single saved SummaryGraph .pt file to evaluate as-is.",
    )
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
    parser.add_argument(
        "--layers-below",
        type=int,
        default=0,
        help="Constrained-patching window low end: [l-below, l+above] per source. "
        "Default 0 (with --layers-above 1 ⇒ the paper's Fig. 9 [l, l+1] direct-effect window).",
    )
    parser.add_argument(
        "--layers-above",
        type=int,
        default=1,
        help="Constrained-patching window high end (see --layers-below).",
    )
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help=f"Comma-separated clustering methods to evaluate (from {', '.join(ALL_METHODS)}). "
        "Default: all non-ILP methods. Only used with --prune-graphs-dir.",
    )
    parser.add_argument(
        "--summary-method-name",
        default="summary",
        help="Method label used in CSV output when evaluating saved summary graphs.",
    )
    parser.add_argument(
        "--skip-exp-d",
        action="store_true",
        help="Skip Exp D (per-node ablation cohesion); run only Exp B (edge/target steering).",
    )
    parser.add_argument(
        "--ilp-max-sn", type=int, default=None, help="ILP complexity budget K<=max_sn."
    )
    parser.add_argument(
        "--ilp-max-layer-span",
        type=int,
        default=4,
        help="ILP: forbid merging features more than this many layers apart (tractability).",
    )
    parser.add_argument(
        "--ilp-time-limit", type=float, default=30.0, help="ILP HiGHS time limit per graph (s)."
    )
    k_match = parser.add_mutually_exclusive_group()
    k_match.add_argument(
        "--match-baseline-k-to-ilp",
        action="store_true",
        help="Run baselines at ILP's feature-supernode count per graph (solving ILP for K even "
        "if 'ilp' is not in --methods), for an equal-granularity comparison against ILP.",
    )
    k_match.add_argument(
        "--match-baseline-k-from-summary-graphs-dir",
        type=Path,
        help="Run baselines at the feature-supernode count of each saved summary graph in this "
        "directory (paired by stem, e.g. 000_prune_graph.pt ↔ 000_summary_graph.pt).",
    )
    args = parser.parse_args()

    use_summary_graphs = args.summary_graph is not None or args.summary_graphs_dir is not None
    if args.match_baseline_k_from_summary_graphs_dir is not None and use_summary_graphs:
        parser.error(
            "--match-baseline-k-from-summary-graphs-dir requires --prune-graphs-dir, not "
            "saved summary graph inputs"
        )
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if not use_summary_graphs:
        unknown = set(methods) - set(ALL_METHODS)
        if unknown:
            parser.error(f"unknown --methods {sorted(unknown)}; choose from {list(ALL_METHODS)}")
    ilp_kwargs = {
        "ilp_max_sn": args.ilp_max_sn,
        "ilp_max_layer_span": args.ilp_max_layer_span,
        "ilp_time_limit": args.ilp_time_limit,
    }

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    graph_paths = (
        _summary_graph_paths(args.summary_graph, args.summary_graphs_dir)
        if use_summary_graphs
        else _prune_graph_paths(args.prune_graphs_dir)
    )
    if args.limit:
        graph_paths = graph_paths[: args.limit]

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
        graph_name = (
            _summary_graph_name(path, args.summary_graphs_dir) if use_summary_graphs else path.stem
        )
        logger.info("Processing %s ...", graph_name)
        if use_summary_graphs:
            sng = SummaryGraph.load(str(path))
            edge_rows, sn_rows, logit_rows = evaluate_summary_graph(
                model,
                sng,
                graph_name,
                args.summary_method_name,
                args.steer_factor,
                args.layers_below,
                args.layers_above,
                run_exp_d=not args.skip_exp_d,
            )
        else:
            prune_graph = load_prune_graph(str(path))
            baseline_k_override = None
            if args.match_baseline_k_from_summary_graphs_dir is not None:
                baseline_k_override = _baseline_k_from_summary_graph(
                    path, args.match_baseline_k_from_summary_graphs_dir
                )
                logger.info("  baseline K=%d from saved summary graph", baseline_k_override)
            edge_rows, sn_rows, logit_rows = evaluate_graph(
                model,
                prune_graph,
                graph_name,
                args.steer_factor,
                args.layers_below,
                args.layers_above,
                methods,
                run_exp_d=not args.skip_exp_d,
                ilp_kwargs=ilp_kwargs,
                match_baseline_k_to_ilp=args.match_baseline_k_to_ilp,
                baseline_k_override=baseline_k_override,
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

    header = f"{'graph':30s} {'method':15s} {'n_edges':>8s} {'signOK':>7s} {'ρ(edge,eff)':>12s} {'tgtΔp':>8s} {'logitΔp':>8s} {'intra':>7s} {'inter':>7s} {'gap':>7s}"
    print(header)
    print("-" * len(header))
    for row in summary:
        print(
            f"{row['graph']:30s} {row['method']:15s} {row['n_edges']:>8d} "
            f"{row['sign_consistency']:>7.3f} "
            f"{row['spearman_edge_effect']:>12.3f} {row['mean_target_abs_prob_delta']:>8.3f} "
            f"{row['mean_logit_sn_abs_prob_delta']:>8.3f} {row['mean_intra_cosine']:>7.3f} "
            f"{row['mean_inter_cosine']:>7.3f} {row['mean_cosine_gap']:>7.3f}"
        )
    logger.info("Done. Output in %s", args.output_dir)


if __name__ == "__main__":
    main()
