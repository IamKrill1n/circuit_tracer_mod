"""Evaluate PruneGraphs produced by ``eval/prune_graphs.py``.
Old code, might not use

Reads a manifest.json describing the prune sweep and computes a battery of
metrics per prune cell. In addition to the existing combined completeness
scores and mean influence/relevance stats, this implements four new metrics
that target the token-weighted relevance dimension that distinguishes our
pruning algorithm from Anthropic's:

1. relevance_conservation_rate: fraction of token-weighted forward relevance
   over feature nodes that survives pruning.
2. token_attribution_faithfulness: token-weight-weighted fraction of each
   input token's flow to the target logit that survives pruning, computed on
   the fixed full-graph row-normalized adjacency (mask but do not re-normalize).
3. asymmetric_pruning_divergence: Jaccard distance between kept node sets when
   pruning with the user's token weights vs uniform token weights at the same
   thresholds. High = token weights are actually steering pruning.
5. influence_relevance_agreement: Pearson correlation between node_influence
   and node_relevance over kept feature nodes. A diagnostic of whether the
   two flow signals carry independent information on the surviving features.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Callable

import torch

from summarization.attr_graph import AttrGraph
from summarization.prune import (
    PruneGraph,
    compute_influence,
    compute_relevance,
    load_prune_graph,
    normalize_matrix,
    prune_attr_graph,
)
from summarization.utils import _build_index_sets


# --- Caches ---------------------------------------------------------------


class GraphCache:
    """Caches per-graph artifacts shared across many prune cells."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._attr_graph: dict[str, AttrGraph] = {}
        self._idx_sets: dict[str, dict[str, list[int]]] = {}
        self._a_full_norm: dict[str, torch.Tensor] = {}
        self._full_target_rel: dict[str, dict[int, float]] = {}
        self._id_to_idx: dict[str, dict[str, int]] = {}
        self._uniform_prune: dict[tuple, PruneGraph] = {}

    def attr_graph(self, path: str) -> AttrGraph:
        if path not in self._attr_graph:
            ag = (
                AttrGraph.from_graph_file(path)
                if path.endswith(".json")
                else AttrGraph.from_graph(path)
            )
            ag.adj = ag.adj.to(self.device)
            self._attr_graph[path] = ag
            self._idx_sets[path] = _build_index_sets(ag.nodes)
            self._id_to_idx[path] = {nd.node_id: i for i, nd in enumerate(ag.nodes)}
        return self._attr_graph[path]

    def idx_sets(self, path: str) -> dict[str, list[int]]:
        self.attr_graph(path)
        return self._idx_sets[path]

    def id_to_idx(self, path: str) -> dict[str, int]:
        self.attr_graph(path)
        return self._id_to_idx[path]

    def a_full_norm(self, path: str) -> torch.Tensor:
        if path not in self._a_full_norm:
            ag = self.attr_graph(path)
            # row-normalized transpose; same as compute_node_relevance internals
            self._a_full_norm[path] = normalize_matrix(ag.adj.T)
        return self._a_full_norm[path]

    def full_target_rel(self, path: str, emb_idx: int) -> float:
        """Full-graph relevance reaching the target logit when seeded one-hot at ``emb_idx``."""
        per = self._full_target_rel.setdefault(path, {})
        if emb_idx in per:
            return per[emb_idx]
        ag = self.attr_graph(path)
        target_idx = self.idx_sets(path)["target_logit"]
        if not target_idx:
            per[emb_idx] = 0.0
            return 0.0
        n = ag.adj.shape[0]
        seed = torch.zeros(n, dtype=torch.float32, device=self.device)
        seed[emb_idx] = 1.0
        full_rel = compute_relevance(self.a_full_norm(path), seed)
        per[emb_idx] = float(full_rel[target_idx].sum().item())
        return per[emb_idx]

    def uniform_prune(
        self,
        path: str,
        node_threshold: float,
        edge_threshold: float,
        combine_method: str,
        normalization: str,
        alpha: float,
        logit_weights: str,
        keep_all_tokens_and_logits: bool,
    ) -> PruneGraph:
        key = (
            path,
            float(node_threshold),
            float(edge_threshold),
            str(combine_method),
            str(normalization),
            float(alpha),
            str(logit_weights),
            bool(keep_all_tokens_and_logits),
        )
        if key not in self._uniform_prune:
            ag = self.attr_graph(path)
            self._uniform_prune[key] = prune_attr_graph(
                ag,
                logit_weights=logit_weights,  # type: ignore[arg-type]
                token_weights=None,
                node_threshold=node_threshold,
                edge_threshold=edge_threshold,
                combine_method=combine_method,  # type: ignore[arg-type]
                normalization=normalization,  # type: ignore[arg-type]
                alpha=alpha,
                keep_all_tokens_and_logits=keep_all_tokens_and_logits,
            )
        return self._uniform_prune[key]


# --- Metric implementations ----------------------------------------------


def _build_masked_A(
    attr_graph: AttrGraph,
    prune_graph: PruneGraph,
    A_full_norm: torch.Tensor,
    id_to_idx: dict[str, int],
) -> torch.Tensor:
    """Mask the full row-normalized adjacency to only edges/nodes kept after pruning.

    Pruning preserves edges and nodes; we do not re-normalize. ``A_full_norm`` is
    indexed (source, target). Edge mask in adj convention is ``(target, source)``.
    """
    kept_full_idx = [id_to_idx[nd.node_id] for nd in prune_graph.nodes]
    pruned_full = torch.zeros_like(attr_graph.adj)
    kept_t = torch.tensor(kept_full_idx, dtype=torch.long, device=pruned_full.device)
    pruned_full[kept_t.unsqueeze(1), kept_t.unsqueeze(0)] = prune_graph.pruned_adj.to(
        device=pruned_full.device, dtype=pruned_full.dtype
    )
    edge_mask = pruned_full != 0  # (target, source)
    A_masked = A_full_norm.clone()
    A_masked[~edge_mask.T] = 0.0
    return A_masked


def relevance_conservation_rate(
    attr_graph: AttrGraph,
    prune_graph: PruneGraph,
    token_weights: list[float],
    *,
    full_idx: dict[str, list[int]],
    A_full_norm: torch.Tensor,
    id_to_idx: dict[str, int],
) -> float | None:
    all_feat_idx = full_idx["feature"]
    if not all_feat_idx:
        return None

    n = attr_graph.adj.shape[0]
    emb_seed = torch.zeros(n, dtype=torch.float32, device=A_full_norm.device)
    emb_idx_full = full_idx["embedding"]
    if len(token_weights) != len(emb_idx_full):
        raise ValueError(
            f"token_weights length {len(token_weights)} != embedding count {len(emb_idx_full)}"
        )
    for k, emb_i in enumerate(emb_idx_full):
        emb_seed[emb_i] = float(token_weights[k])

    full_rel = compute_relevance(A_full_norm, emb_seed)

    kept_feat_full = [
        id_to_idx[nd.node_id]
        for nd in prune_graph.nodes
        if nd.feature_type == "cross layer transcoder"
    ]
    kept_sum = float(full_rel[kept_feat_full].sum().item()) if kept_feat_full else 0.0
    total_sum = float(full_rel[all_feat_idx].sum().item())
    return kept_sum / total_sum if total_sum > 0 else 0.0


def token_attribution_faithfulness(
    attr_graph: AttrGraph,
    prune_graph: PruneGraph,
    token_weights: list[float],
    *,
    full_idx: dict[str, list[int]],
    A_full_norm: torch.Tensor,
    id_to_idx: dict[str, int],
    full_target_rel_fn: Callable[[int], float],
) -> float | None:
    target_idx = full_idx["target_logit"]
    emb_idx_full = full_idx["embedding"]
    if not target_idx or not emb_idx_full:
        return None
    if len(token_weights) != len(emb_idx_full):
        raise ValueError(
            f"token_weights length {len(token_weights)} != embedding count {len(emb_idx_full)}"
        )

    A_masked = _build_masked_A(attr_graph, prune_graph, A_full_norm, id_to_idx)
    n = attr_graph.adj.shape[0]

    weighted_sum = 0.0
    weight_total = 0.0
    for k, emb_i in enumerate(emb_idx_full):
        w_t = float(token_weights[k])
        if w_t <= 0:
            continue
        full_target = full_target_rel_fn(emb_i)
        seed = torch.zeros(n, dtype=torch.float32, device=A_masked.device)
        seed[emb_i] = 1.0
        pruned_rel = compute_relevance(A_masked, seed)
        pruned_target = float(pruned_rel[target_idx].sum().item())
        faith_t = (pruned_target / full_target) if full_target > 0 else 0.0
        weighted_sum += w_t * faith_t
        weight_total += w_t
    return weighted_sum / weight_total if weight_total > 0 else 0.0


def compute_prune_loss(
    attr_graph: AttrGraph,
    prune_graph: PruneGraph,
    token_weights: list[float],
    *,
    device: str = "cpu",
) -> float:
    """1 - token_attribution_faithfulness; one number per (full, pruned) pair.

    Reported separately alongside L per paper/formulation.tex ("Pruning is
    scored separately"); partition-invariant within a pruned graph.
    Constructs the full-graph cache lazily via GraphCache so callers don't
    have to wire A_full_norm/full_target_rel by hand.
    """
    cache = GraphCache(device=device)
    key = f"_inline_{id(attr_graph)}"
    attr_graph.adj = attr_graph.adj.to(device)
    cache._attr_graph[key] = attr_graph
    cache._idx_sets[key] = _build_index_sets(attr_graph.nodes)
    cache._id_to_idx[key] = {nd.node_id: i for i, nd in enumerate(attr_graph.nodes)}
    full_idx = cache._idx_sets[key]
    id_to_idx = cache._id_to_idx[key]
    A_full_norm = normalize_matrix(attr_graph.adj.T)
    cache._a_full_norm[key] = A_full_norm

    faith = token_attribution_faithfulness(
        attr_graph,
        prune_graph,
        token_weights,
        full_idx=full_idx,
        A_full_norm=A_full_norm,
        id_to_idx=id_to_idx,
        full_target_rel_fn=lambda emb_i: cache.full_target_rel(key, emb_i),
    )
    return 1.0 - float(faith if faith is not None else 0.0)


def asymmetric_pruning_divergence(
    prune_graph: PruneGraph,
    uniform_prune_graph: PruneGraph,
) -> float:
    A = {nd.node_id for nd in prune_graph.nodes}
    B = {nd.node_id for nd in uniform_prune_graph.nodes}
    union = A | B
    if not union:
        return 0.0
    return 1.0 - len(A & B) / len(union)


def compute_clt_graph_scores(attr_graph: AttrGraph, prune_graph: PruneGraph) -> tuple[float, float]:
    """CLT-style (replacement, completeness) for a pruned subgraph.

    Pruned features are merged into their corresponding error node at the same
    (layer, ctx_idx) in the full adjacency before computing influence, so the
    error signal reflects both the original reconstruction error and any features
    not captured by the subgraph.
    """
    full_nodes = attr_graph.nodes
    n_full = len(full_nodes)

    # error node lookup: (layer, ctx_idx) -> full-graph index
    error_lookup: dict[tuple[str, int], int] = {
        (node.layer, node.ctx_idx): i
        for i, node in enumerate(full_nodes)
        if node.feature_type == "mlp reconstruction error"
    }

    kept_ids = {nd.node_id for nd in prune_graph.nodes}

    # adj[target, source]; merge pruned features into error nodes in-place
    adj = attr_graph.adj.to(dtype=torch.float32).clone()
    for i, node in enumerate(full_nodes):
        if node.feature_type != "cross layer transcoder" or node.node_id in kept_ids:
            continue
        err_i = error_lookup.get((node.layer, node.ctx_idx))
        if err_i is None:
            continue
        # incoming edges to pruned feature → merge into error node row
        adj[err_i, :] += adj[i, :]
        adj[i, :] = 0.0
        # outgoing edges from pruned feature → merge into error node column
        adj[:, err_i] += adj[:, i]
        adj[:, i] = 0.0

    full_idx = _build_index_sets(full_nodes)
    error_idx = full_idx["error"]
    emb_idx = full_idx["embedding"]
    logit_idx = full_idx["logit"]
    if not logit_idx or not emb_idx:
        return float("nan"), float("nan")

    logit_w = torch.zeros(n_full, dtype=torch.float32, device=adj.device)
    for i in logit_idx:
        logit_w[i] = float(full_nodes[i].token_prob or 0.0)

    A_norm = normalize_matrix(adj)
    node_inf = compute_influence(A_norm, logit_w)

    token_inf = float(node_inf[emb_idx].sum())
    err_inf = float(node_inf[error_idx].sum()) if error_idx else 0.0
    denom_rep = token_inf + err_inf
    replacement = token_inf / denom_rep if denom_rep > 0 else float("nan")

    if error_idx:
        non_error = 1.0 - A_norm[:, error_idx].sum(dim=-1)
    else:
        non_error = torch.ones(n_full, dtype=A_norm.dtype, device=A_norm.device)
    out_inf = node_inf + logit_w
    denom_comp = float(out_inf.sum())
    completeness = (
        float((non_error * out_inf).sum()) / denom_comp if denom_comp > 0 else float("nan")
    )

    return replacement, completeness


def compute_clt_graph_completeness_relevance(
    attr_graph: AttrGraph,
    prune_graph: PruneGraph,
) -> float:
    """Forward-direction analog of completeness in `compute_clt_graph_scores`.

    Pruned features are merged into their error node (same merge as the backward
    version). Then we run forward relevance from uniformly-seeded embeddings and
    measure, weighted by each node's forward relevance, the fraction of its
    outgoing flow going to non-error targets.
    """
    full_nodes = attr_graph.nodes
    n_full = len(full_nodes)

    error_lookup: dict[tuple[str, int], int] = {
        (node.layer, node.ctx_idx): i
        for i, node in enumerate(full_nodes)
        if node.feature_type == "mlp reconstruction error"
    }

    kept_ids = {nd.node_id for nd in prune_graph.nodes}

    # adj[target, source]; merge pruned features into error nodes in-place
    adj = attr_graph.adj.to(dtype=torch.float32).clone()
    for i, node in enumerate(full_nodes):
        if node.feature_type != "cross layer transcoder" or node.node_id in kept_ids:
            continue
        err_i = error_lookup.get((node.layer, node.ctx_idx))
        if err_i is None:
            continue
        adj[err_i, :] += adj[i, :]
        adj[i, :] = 0.0
        adj[:, err_i] += adj[:, i]
        adj[:, i] = 0.0

    full_idx = _build_index_sets(full_nodes)
    error_idx = full_idx["error"]
    emb_idx = full_idx["embedding"]
    if not emb_idx:
        return float("nan")

    emb_w = torch.zeros(n_full, dtype=torch.float32, device=adj.device)
    for i in emb_idx:
        emb_w[i] = 1.0

    # A_fwd[s, t] = fraction of source s's outgoing weight to target t (row-sum = 1)
    A_fwd = normalize_matrix(adj.T)
    node_rel = compute_relevance(A_fwd, emb_w)

    if error_idx:
        non_error = 1.0 - A_fwd[:, error_idx].sum(dim=-1)
    else:
        non_error = torch.ones(n_full, dtype=A_fwd.dtype, device=A_fwd.device)
    denom = float(node_rel.sum())
    return float((non_error * node_rel).sum()) / denom if denom > 0 else float("nan")


def compute_clt_graph_completeness_combined(
    attr_graph: AttrGraph,
    prune_graph: PruneGraph,
) -> float:
    """Geometric mean of backward (influence) and forward (relevance) completeness.

    Penalizes asymmetry: a high score requires error nodes to be unimportant in
    *both* the upstream and downstream directions.
    """
    _, c_back = compute_clt_graph_scores(attr_graph, prune_graph)
    c_fwd = compute_clt_graph_completeness_relevance(attr_graph, prune_graph)
    # NaN propagation: either direction undefined → combined undefined
    if c_back != c_back or c_fwd != c_fwd:
        return float("nan")
    if c_back <= 0 or c_fwd <= 0:
        return 0.0
    return float((c_back * c_fwd) ** 0.5)


def influence_relevance_agreement(prune_graph: PruneGraph) -> float | None:
    if prune_graph.node_influence is None or prune_graph.node_relevance is None:
        return None
    feat_idx = [
        i for i, nd in enumerate(prune_graph.nodes) if nd.feature_type == "cross layer transcoder"
    ]
    if len(feat_idx) < 2:
        return None
    inf_t = prune_graph.node_influence[feat_idx].to(dtype=torch.float32)
    rel_t = prune_graph.node_relevance[feat_idx].to(dtype=torch.float32)
    inf_std = float(inf_t.std(unbiased=False).item())
    rel_std = float(rel_t.std(unbiased=False).item())
    if inf_std <= 1e-12 or rel_std <= 1e-12:
        return None
    inf_z = (inf_t - inf_t.mean()) / inf_std
    rel_z = (rel_t - rel_t.mean()) / rel_std
    return float((inf_z * rel_z).mean().item())


# --- Driver ---------------------------------------------------------------


def _evaluate_record(
    rec: dict[str, Any],
    cache: GraphCache,
    *,
    skip_relevance_conservation: bool,
    skip_token_faithfulness: bool,
    skip_pruning_divergence: bool,
    skip_influence_relevance_agreement: bool,
    skip_clt_scores: bool,
) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "relevance_conservation_rate": None,
        "token_attribution_faithfulness": None,
        "asymmetric_pruning_divergence": None,
        "influence_relevance_agreement": None,
        "replacement_score": None,
        "completeness_score": None,
        "n_nodes": None,
        "n_edges": None,
    }

    prune_graph = load_prune_graph(rec["prune_graph_path"], map_location=cache.device)
    metrics["n_nodes"] = prune_graph.num_nodes
    metrics["n_edges"] = prune_graph.num_edges

    needs_graph = not (
        skip_relevance_conservation
        and skip_token_faithfulness
        and skip_pruning_divergence
        and skip_clt_scores
    )
    needs_token_weights = not (skip_relevance_conservation and skip_token_faithfulness)

    if needs_graph:
        graph_path = rec["graph_path"]
        attr_graph = cache.attr_graph(graph_path)
        full_idx = cache.idx_sets(graph_path)
        A_full_norm = cache.a_full_norm(graph_path)
        id_to_idx = cache.id_to_idx(graph_path)
    if needs_token_weights:
        token_weights = [float(w) for w in rec["token_weights"]]

    if not skip_clt_scores:
        rep, comp = compute_clt_graph_scores(attr_graph, prune_graph)
        metrics["replacement_score"] = rep
        metrics["completeness_score"] = comp

    if not skip_relevance_conservation:
        metrics["relevance_conservation_rate"] = relevance_conservation_rate(
            attr_graph,
            prune_graph,
            token_weights,
            full_idx=full_idx,
            A_full_norm=A_full_norm,
            id_to_idx=id_to_idx,
        )

    if not skip_token_faithfulness:
        metrics["token_attribution_faithfulness"] = token_attribution_faithfulness(
            attr_graph,
            prune_graph,
            token_weights,
            full_idx=full_idx,
            A_full_norm=A_full_norm,
            id_to_idx=id_to_idx,
            full_target_rel_fn=lambda emb_i: cache.full_target_rel(graph_path, emb_i),
        )

    if not skip_pruning_divergence:
        uniform_pg = cache.uniform_prune(
            graph_path,  # set above when needs_graph
            float(rec["node_threshold"]),
            float(rec["edge_threshold"]),
            str(rec.get("combine_method", "geometric")),
            str(rec.get("score_normalization", "rank")),
            float(rec.get("alpha", 0.5)),
            str(rec["logit_weights"]),
            bool(rec.get("keep_all_tokens_and_logits", False)),
        )
        metrics["asymmetric_pruning_divergence"] = asymmetric_pruning_divergence(
            prune_graph, uniform_pg
        )

    if not skip_influence_relevance_agreement:
        metrics["influence_relevance_agreement"] = influence_relevance_agreement(prune_graph)

    return metrics


def plot_pareto_frontier(
    rows: list[dict[str, Any]],
    output_dir: Path,
    metrics: list[tuple[str, str]] | None = None,
    filename: str = "pareto_frontier.png",
) -> Path | None:
    """Plot metrics vs node-count and edges-per-node, one line per alpha.

    metrics: list of (column_name, y_label). Defaults to CLT-style scores.
    """
    import matplotlib.pyplot as plt
    from collections import defaultdict

    if metrics is None:
        metrics = [
            ("completeness_score", "graph completeness score"),
            ("replacement_score", "replacement score"),
        ]
    m_keys = [m[0] for m in metrics]

    # alpha -> (node_threshold, edge_threshold) -> list of (n_nodes, n_edges, *metric_values)
    grouped: dict[float, dict[tuple, list[tuple[float, ...]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in rows:
        alpha = r.get("alpha")
        n_nodes = r.get("n_nodes")
        n_edges = r.get("n_edges")
        vals = [r.get(k) for k in m_keys]
        if alpha is None or n_nodes is None or n_edges is None or any(v is None for v in vals):
            continue
        if any(isinstance(v, float) and v != v for v in vals):  # NaN check
            continue
        key = (float(r.get("node_threshold", 0.0)), float(r.get("edge_threshold", 0.0)))
        grouped[float(alpha)][key].append(
            (float(n_nodes), float(n_edges), *[float(v) for v in vals])
        )

    if not grouped:
        print("[plot] no usable rows (missing alpha / scores / counts).")
        return None

    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 2, figsize=(12, 5 * n_metrics), squeeze=False)
    for alpha in sorted(grouped):
        per_cell = grouped[alpha]
        points: list[tuple[float, float, list[float]]] = []
        for cell, vals_list in per_cell.items():
            arr = torch.tensor(vals_list, dtype=torch.float32).mean(dim=0).tolist()
            n_nodes, n_edges, *m_vals = arr
            epn = n_edges / n_nodes if n_nodes > 0 else 0.0
            points.append((n_nodes, epn, m_vals))
        points.sort(key=lambda p: p[0])
        xs_n = [p[0] for p in points]
        xs_e = [p[1] for p in points]
        label = f"α={alpha:g}"
        for mi in range(n_metrics):
            ys = [p[2][mi] for p in points]
            axes[mi, 0].plot(xs_n, ys, marker="o", label=label)
            axes[mi, 1].plot(xs_e, ys, marker="o", label=label)

    for mi, (_, y_label) in enumerate(metrics):
        axes[mi, 0].set_xlabel("average node count")
        axes[mi, 0].set_ylabel(y_label)
        axes[mi, 1].set_xlabel("average edges per node")
        axes[mi, 1].set_ylabel(y_label)
    for ax in axes.flat:
        ax.set_xscale("log")
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend(fontsize=8)
    fig.suptitle("Average Graph Metrics with Pruning (per α)")
    fig.tight_layout()

    out_path = output_dir / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


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
        raise RuntimeError(f"No records found in manifest {manifest_path}")

    if args.limit is not None:
        records = records[: int(args.limit)]

    output_root = Path(args.output_root) if args.output_root else manifest_path.parent / "eval"
    output_root.mkdir(parents=True, exist_ok=True)

    cache = GraphCache(device=args.device)
    rows_out: list[dict[str, Any]] = []
    failures: list[str] = []

    n = len(records)
    for i, rec in enumerate(records):
        stem = rec.get("graph_stem", rec.get("graph_file", "?"))
        norm = rec.get("normalize_method", "?")
        nt = rec.get("node_threshold")
        try:
            metrics = _evaluate_record(
                rec,
                cache,
                skip_relevance_conservation=args.skip_relevance_conservation,
                skip_token_faithfulness=args.skip_token_faithfulness,
                skip_pruning_divergence=args.skip_pruning_divergence,
                skip_influence_relevance_agreement=args.skip_influence_relevance_agreement,
                skip_clt_scores=args.skip_clt_scores,
            )
            row = {**rec, **metrics}
            rows_out.append(row)
            if (i + 1) % max(1, args.log_every) == 0:
                print(
                    f"[{i + 1}/{n}] {stem} norm={norm} node={nt} "
                    f"rcr={metrics['relevance_conservation_rate']} "
                    f"taf={metrics['token_attribution_faithfulness']} "
                    f"apd={metrics['asymmetric_pruning_divergence']} "
                    f"ira={metrics['influence_relevance_agreement']} "
                    f"rep={metrics['replacement_score']} "
                    f"comp={metrics['completeness_score']} "
                    f"n={metrics['n_nodes']} e={metrics['n_edges']}"
                )
        except Exception as exc:
            msg = f"{stem} norm={norm} node={nt}: {exc}"
            failures.append(msg)
            print(f"[failed] {msg}")

    eval_results_path = output_root / "eval_results.json"
    eval_summary_path = output_root / "eval_summary.csv"
    eval_manifest_path = output_root / "eval_manifest.json"

    _write_json(eval_results_path, rows_out)

    if rows_out:
        csv_fields = [k for k in rows_out[0].keys() if k != "token_weights"]
        with eval_summary_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows_out)

    eval_manifest = {
        "source_manifest": str(manifest_path),
        "output_root": str(output_root),
        "limit": args.limit,
        "skip_relevance_conservation": args.skip_relevance_conservation,
        "skip_token_faithfulness": args.skip_token_faithfulness,
        "skip_pruning_divergence": args.skip_pruning_divergence,
        "skip_influence_relevance_agreement": args.skip_influence_relevance_agreement,
        "skip_clt_scores": args.skip_clt_scores,
        "n_records_processed": len(rows_out),
        "n_failures": len(failures),
        "results_json": str(eval_results_path),
        "summary_csv": str(eval_summary_path),
        "failures": failures,
    }
    _write_json(eval_manifest_path, eval_manifest)

    print("\n=== Eval summary ===")
    print(f"manifest: {manifest_path}")
    print(f"output: {output_root}")
    print(f"rows: {len(rows_out)} (failures: {len(failures)})")
    print(f"wrote {eval_results_path}")
    print(f"wrote {eval_summary_path}")
    print(f"wrote {eval_manifest_path}")
    if failures:
        print("\nFailures (first 20):")
        for item in failures[:20]:
            print(f"- {item}")

    if args.plot and rows_out:
        plot_path = plot_pareto_frontier(rows_out, output_root)
        if plot_path is not None:
            print(f"wrote {plot_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute evaluation metrics on PruneGraphs produced by "
            "eval/prune_graphs.py. Reads the prune sweep manifest.json and "
            "emits eval_results.json + eval_summary.csv."
        )
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to manifest.json produced by eval/prune_graphs.py.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Output directory (defaults to <manifest_dir>/eval).",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--device", default="cpu", help="Torch device for adj/prune_graph tensors (e.g. 'cuda')."
    )
    parser.add_argument(
        "--skip-relevance-conservation",
        action="store_true",
        help="Skip metric 1 (relevance_conservation_rate).",
    )
    parser.add_argument(
        "--skip-token-faithfulness",
        action="store_true",
        help="Skip metric 2 (token_attribution_faithfulness).",
    )
    parser.add_argument(
        "--skip-pruning-divergence",
        action="store_true",
        help="Skip metric 3 (asymmetric_pruning_divergence).",
    )
    parser.add_argument(
        "--skip-influence-relevance-agreement",
        action="store_true",
        help="Skip metric 5 (influence_relevance_agreement).",
    )
    parser.add_argument(
        "--skip-clt-scores",
        action="store_true",
        help="Skip CLT-style replacement_score and completeness_score.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="After eval, plot completeness/replacement vs nodes (one line per alpha).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
