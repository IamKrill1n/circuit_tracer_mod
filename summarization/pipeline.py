"""End-to-end summarization pipeline: attribute -> (token_attribution) -> prune -> classify -> cluster -> summarize.

``run_pipeline`` produces a ``circuit_tracer.Graph`` by running local attribution on a
prompt (or loading an existing ``.pt``), optionally computes SHAP token weights, prunes
the graph directly, optionally annotates/filters by activation density, clusters (auto-k
or fixed k), and assembles the supernode ``SummaryGraph``. ``summarization/__main__.py``
is a thin CLI over this.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from api import save_subgraph
from summarization.attr_graph import AttrGraph
from summarization.classify import filter_act_density
from summarization.cluster import cluster, find_best_k
from summarization.cluster_viz import supernode_graph_figure
from summarization.prune import prune_attr_graph
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


def _supernodes_for_upload(rows: list) -> list[list[str]]:
    """Neuronpedia upload format [label, member_1, ...] for non-singleton supernodes."""
    return [[s.name, *s.member_node_ids()] for s in rows if len(s.features) > 1]


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    # Stage 0: produce a circuit_tracer Graph (local attribution or loaded .pt).
    graph = _acquire_graph(args)
    ag = AttrGraph.from_graph(graph)

    # Stage 0b (optional): SHAP token weights from the graph's prompt.
    token_weights = _parse_token_weights(args.token_weights)
    if args.auto_token_weights and token_weights is None:
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
    prune_graph = prune_attr_graph(
        ag,
        logit_weights=args.logit_weights,
        token_weights=token_weights,
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
        keep_all_tokens_and_logits=args.keep_all_tokens_and_logits,
    )

    # Stage 1b (optional): activation-density filter from the feature dashboards.
    if args.classify_filter:
        prune_graph = filter_act_density(
            prune_graph,
            act_density_lb=args.act_density_lb,
            act_density_ub=args.act_density_ub,
        )

    # Stage 2: cluster (auto-k captures the per-k sweep for reporting).
    sweep: dict[int, dict[str, Any]] = {}
    if args.method == "ilp":
        # ILP picks K endogenously by minimising the linearised paper objective.
        rows = cluster(
            prune_graph,
            method="ilp",
            max_layer_span=args.max_layer_span,
            max_sn=args.max_sn,
            lambda_causal=args.lambda_causal,
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
                lambda_causal=args.lambda_causal,
            )
        if resolved_k is None:
            resolved_k = 7

        rows = cluster(
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
    sng = summarize(rows, prune_graph.pruned_adj)
    labelled_supernodes = _supernodes_for_upload(rows)

    _save_json(args.supernodes_out, labelled_supernodes)
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
        args.auto_k_sweep_out,
        {
            str(k): {key: value for key, value in score.items() if key != "final_supernodes"}
            for k, score in sweep.items()
        },
    )

    figure_path = None
    if args.figure_html_out:
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
            supernodes=labelled_supernodes,
            pruningThreshold=args.upload_pruning_threshold,
            densityThreshold=args.upload_density_threshold,
        )

    return {
        "pruned_nodes": prune_graph.num_nodes,
        "pruned_edges": prune_graph.num_edges,
        "resolved_k": resolved_k,
        "auto_k_candidates": len(sweep),
        "supernodes": labelled_supernodes,
        "supernode_map": supernode_map,
        "figure_html_out": figure_path,
        "upload_status": upload_status,
        "upload_body": upload_body,
    }
