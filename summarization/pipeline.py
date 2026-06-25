"""End-to-end pipeline: attribution -> token weights -> prune -> cluster -> summarize.

``run_pipeline`` produces a ``circuit_tracer.Graph`` by running local attribution on a
prompt (or loading an existing ``.pt``), optionally computes SHAP token weights, prunes
the graph directly, optionally filters by activation density, clusters, and assembles
the supernode ``SummaryGraph``. Legacy spectral/agglomerative clustering is available
only through eval-owned baseline helpers.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from api import save_subgraph
from eval.legacy_cluster_baselines import cluster as legacy_cluster
from eval.legacy_cluster_baselines import find_best_k
from summarization.attr_graph import AttrGraph
from summarization.cluster import DEFAULT_THETA, cluster
from summarization.cluster_viz import supernode_graph_figure
from summarization.prune import filter_act_density, prune_attr_graph, save_prune_graph
from summarization.summarize import summarize
from summarization.utils import node_is_embedding


def _acquire_graph(args: argparse.Namespace):
    """Run local attribution on the prompt, or load an existing ``.pt`` graph."""
    from circuit_tracer.graph import Graph

    if args.graph_pt:
        return Graph.from_pt(args.graph_pt)

    if not args.prompt:
        raise ValueError("--prompt is required unless --graph-pt is set.")

    import torch
    from circuit_tracer import ReplacementModel, attribute
    from circuit_tracer.utils.demo_utils import cleanup_cuda

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    model = ReplacementModel.from_pretrained(
        args.model,
        args.transcoder,
        dtype=dtype_map[args.dtype],
        lazy_encoder=True,
        backend=args.backend,
    )
    try:
        graph = attribute(
            prompt=args.prompt,
            model=model,
            max_n_logits=args.max_n_logits,
            desired_logit_prob=args.desired_logit_prob,
            batch_size=args.batch_size,
            max_feature_nodes=args.max_feature_nodes,
            offload="cpu",
            verbose=False,
        )
    finally:
        del model
        cleanup_cuda()

    if args.graph_pt_out:
        out = Path(args.graph_pt_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        graph.to_pt(str(out))
    return graph


def _shap_token_weights(
    ag: AttrGraph,
    *,
    model_name: str,
    normalize_method: str,
    entmax_alpha: float | None,
    device: str,
) -> list[float]:
    """SHAP token attribution for the graph's prompt, mapped to embedding-node order.

    Embedding nodes are 1:1 with input-token positions (``Node.ctx_idx``), so the
    normalized per-token weight at ``ctx_idx`` is the weight for that embedding node.
    """
    from summarization.token_attribution import get_token_attribution

    prompt = str(ag.metadata.get("prompt", "") or "")
    prompt_tokens = [str(t) for t in (ag.metadata.get("prompt_tokens") or [])]
    if not prompt or not prompt_tokens:
        raise ValueError("Graph metadata lacks prompt / prompt_tokens for SHAP token attribution.")

    # Force SHAP's target Y to the graph's target logit token (aligns with logit_weights="target").
    target_token_id = next((n.feature for n in ag.nodes if n.is_target_logit), None)

    # pin_special_tokens keeps BOS / chat scaffold aligned 1:1 with prompt_tokens.
    _raw, normalized = get_token_attribution(
        prompt=prompt,
        prompt_tokens=prompt_tokens,
        model_name=model_name,
        normalize_method=normalize_method,  # type: ignore[arg-type]
        device=device,
        entmax_alpha=entmax_alpha,
        pin_special_tokens=True,
        target_token_id=target_token_id,
    )
    norm = [float(x) for x in normalized.detach().cpu().tolist()]
    weights: list[float] = []
    for node in ag.nodes:
        if node_is_embedding(node):
            ci = int(node.ctx_idx)
            weights.append(norm[ci] if 0 <= ci < len(norm) else 0.0)
    return weights


def _parse_token_weights(raw: str | None) -> list[float] | None:
    if raw is None or raw.strip() == "":
        return None
    values = json.loads(raw)
    if not isinstance(values, list):
        raise ValueError("--token-weights must be a JSON list, e.g. '[0,0.5,0.5]'")
    return [float(value) for value in values]


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: _to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(value) for value in obj]
    if isinstance(obj, tuple):
        return [_to_jsonable(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if hasattr(obj, "detach"):
        return obj.detach().cpu().tolist()
    return obj


def _save_json(path: str | None, payload: Any) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_to_jsonable(payload), indent=2), encoding="utf-8")


def _report_progress(
    args: argparse.Namespace,
    message: str,
    progress: float | None = None,
) -> None:
    callback: Callable[[str, float | None], None] | None = getattr(args, "progress_callback", None)
    if callback is not None:
        callback(message, progress)


def _supernodes_for_upload(rows: list) -> list[list[str]]:
    """Neuronpedia upload format [group_name, member_1, ...] for non-singleton supernodes."""
    return [[s.name, *s.member_node_ids()] for s in rows if len(s.features) > 1]


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    # Stage 0: produce a circuit_tracer Graph (local attribution or loaded .pt).
    _report_progress(args, "Loading attribution graph", 0.05)
    graph = _acquire_graph(args)
    device = str(getattr(args, "device", "cpu"))
    if not device.startswith("cpu") and hasattr(graph, "to"):
        graph.to(device)
    _report_progress(args, "Building attribution graph view", 0.12)
    ag = AttrGraph.from_graph(graph)

    # Stage 0b (optional): SHAP token weights from the graph's prompt.
    token_weights = _parse_token_weights(args.token_weights)
    if args.auto_token_weights and token_weights is None:
        _report_progress(args, "Computing token attribution weights", 0.22)
        shap_model = (
            args.token_attr_model or getattr(graph.cfg, "tokenizer_name", None) or args.model
        )
        token_weights = _shap_token_weights(
            ag,
            model_name=shap_model,
            normalize_method=args.token_attr_normalize,
            entmax_alpha=args.entmax_alpha if args.token_attr_normalize == "entmax" else None,
            device=args.device,
        )

    # Stage 1: prune (Graph -> PruneGraph, pure tensor math).
    _report_progress(args, "Pruning attribution graph", 0.35)
    prune_graph = prune_attr_graph(
        ag,
        logit_weights=args.logit_weights,
        token_weights=token_weights,
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
        combine_method=getattr(args, "combine_method", "geometric"),
        normalization=getattr(args, "normalization", "rank"),
        alpha=getattr(args, "alpha", 0.5),
        keep_all_tokens_and_logits=args.keep_all_tokens_and_logits,
    )

    # Stage 1b (optional): activation-density filter from the feature dashboards.
    if getattr(args, "filter_act_density", False) or getattr(args, "classify_filter", False):
        _report_progress(args, "Filtering activation density", 0.48)
        prune_graph = filter_act_density(
            prune_graph,
            act_density_lb=args.act_density_lb,
            act_density_ub=args.act_density_ub,
        )

    prune_graph_out = getattr(args, "prune_graph_out", None)
    if prune_graph_out:
        _report_progress(args, "Saving pruned graph", 0.54)
        out = Path(prune_graph_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        save_prune_graph(prune_graph, str(out))

    # Stage 2: cluster (ILP canonical; legacy baselines remain eval-owned).
    _report_progress(args, "Clustering supernodes", 0.62)
    sweep: dict[int, dict[str, Any]] = {}
    if args.method == "ilp":
        rows = cluster(
            prune_graph,
            theta=getattr(args, "theta", DEFAULT_THETA),
            max_layer_span=args.max_layer_span,
            max_sn=args.max_sn,
            eps_causal=args.eps_causal,
            ilp_time_limit=args.ilp_time_limit,
        )
        resolved_k = sum(1 for s in rows if s.type == "features")
    else:
        resolved_k = args.target_k
        if args.auto_k:
            resolved_k, sweep = find_best_k(
                prune_graph,
                max_layer_span=args.max_layer_span,
                k_min_override=args.k_min,
                k_max_override=args.k_max,
                max_sn=args.max_sn,
                mean_method=args.mean_method,
                random_state=args.random_state,
                n_init=args.n_init,
            )
        if resolved_k is None:
            resolved_k = 7

        rows = legacy_cluster(
            prune_graph,
            num_clusters=resolved_k,
            method=args.method,
            max_layer_span=args.max_layer_span,
            max_sn=args.max_sn,
            mean_method=args.mean_method,
            random_state=args.random_state,
            n_init=args.n_init,
        )
    supernode_map = {s.name: s.member_node_ids() for s in rows}

    # Stage 3: summarize.
    _report_progress(args, "Assembling summary graph", 0.78)
    sng = summarize(rows, prune_graph.pruned_adj, prune_graph.metadata)
    upload_supernodes = _supernodes_for_upload(rows)

    _report_progress(args, "Saving summary artifacts", 0.84)
    summary_graph_out = getattr(args, "summary_graph_out", None)
    if summary_graph_out:
        out = Path(summary_graph_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        sng.save(str(out))

    _save_json(args.supernodes_out, upload_supernodes)
    _save_json(args.supernode_map_out, supernode_map)
    _save_json(
        args.supernode_flow_out,
        {
            "sn_names": sng.sn_names,
            "sn_adj": sng.adj_matrix,
            "supernodes": sng.to_mapping(),
        },
    )
    _save_json(
        getattr(args, "auto_k_sweep_out", None),
        {
            str(k): {key: value for key, value in score.items() if key != "final_supernodes"}
            for k, score in sweep.items()
        },
    )

    figure_path = None
    if args.figure_html_out:
        _report_progress(args, "Rendering summary figure", 0.92)
        fig = supernode_graph_figure(
            sng=sng,
            final_supernodes=supernode_map,
            attr={n.node_id: asdict(n) for n in prune_graph.nodes},
            title="Summarization supernode graph",
        )
        out = Path(args.figure_html_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out), include_plotlyjs="cdn")
        figure_path = str(out)

    upload_status = None
    upload_body = None
    if args.upload:
        _report_progress(args, "Uploading summary graph", 0.96)
        if not args.slug or not args.display_name:
            raise ValueError("--upload requires --slug and --display-name.")
        if not 0.0 <= float(args.upload_pruning_threshold) <= 1.0:
            raise ValueError("--upload-pruning-threshold must be between 0 and 1.")
        if not 0.0 <= float(args.upload_density_threshold) <= 1.0:
            raise ValueError("--upload-density-threshold must be between 0 and 1.")
        upload_status, upload_body = save_subgraph(
            modelId=args.model_id,
            slug=args.slug,
            displayName=args.display_name,
            pinnedIds=prune_graph.node_ids,
            supernodes=upload_supernodes,
            pruningThreshold=args.upload_pruning_threshold,
            densityThreshold=args.upload_density_threshold,
        )

    _report_progress(args, "Summary pipeline complete", 1.0)
    return {
        "pruned_nodes": prune_graph.num_nodes,
        "pruned_edges": prune_graph.num_edges,
        "resolved_k": resolved_k,
        "auto_k_candidates": len(sweep),
        "supernodes": upload_supernodes,
        "supernode_map": supernode_map,
        "prune_graph_out": str(prune_graph_out) if prune_graph_out else None,
        "summary_graph_out": str(summary_graph_out) if summary_graph_out else None,
        "figure_html_out": figure_path,
        "upload_status": upload_status,
        "upload_body": upload_body,
    }
