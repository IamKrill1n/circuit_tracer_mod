from __future__ import annotations

import logging
import os
import time


import torch
from transformers import AutoTokenizer

from circuit_tracer.frontend.graph_models import Metadata, Model, Node, QParams
from circuit_tracer.frontend.utils import add_graph_metadata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from circuit_tracer.graph import Graph


logger = logging.getLogger(__name__)


def load_graph_data(file_path) -> Graph:
    """Load graph data from a PyTorch file."""
    from circuit_tracer.graph import Graph

    start_time = time.time()
    graph = Graph.from_pt(file_path)
    time_ms = (time.time() - start_time) * 1000
    logger.info(f"Loading graph data: {time_ms=:.2f} ms")
    return graph


def create_nodes(graph: Graph, node_mask, tokenizer, cumulative_scores):
    """Create all nodes for the graph."""
    start_time = time.time()

    nodes = {}

    n_features = len(graph.selected_features)
    layers = graph.cfg.n_layers
    error_end_idx = n_features + graph.n_pos * layers
    token_end_idx = error_end_idx + len(graph.input_tokens)

    for node_idx in node_mask.nonzero().squeeze().tolist():
        if node_idx in range(n_features):
            layer, pos, feat_idx = graph.active_features[graph.selected_features[node_idx]].tolist()
            nodes[node_idx] = Node.feature_node(
                layer,
                pos,
                feat_idx,
                influence=cumulative_scores[node_idx],
                activation=graph.activation_values[graph.selected_features[node_idx]].item(),
            )
        elif node_idx in range(n_features, error_end_idx):
            layer, pos = divmod(node_idx - n_features, graph.n_pos)
            nodes[node_idx] = Node.error_node(layer, pos, influence=cumulative_scores[node_idx])
        elif node_idx in range(error_end_idx, token_end_idx):
            pos = node_idx - error_end_idx
            nodes[node_idx] = Node.token_node(
                pos, graph.input_tokens[pos], influence=cumulative_scores[node_idx]
            )
        elif node_idx in range(token_end_idx, len(cumulative_scores)):
            pos = node_idx - token_end_idx

            # vocab_idx can be either a valid token_id (< vocab_size) or a virtual
            # index (>= vocab_size) for arbitrary strings/functions thereof. The virtual indices
            # encode the position in the logit_targets list as: vocab_size + position.
            token, vocab_idx = graph.logit_targets[pos]

            nodes[node_idx] = Node.logit_node(
                pos=graph.n_pos - 1,
                vocab_idx=vocab_idx,
                token=token,
                target_logit=pos == 0,
                token_prob=graph.logit_probabilities[pos].item(),
                num_layers=layers,
            )

    total_time = (time.time() - start_time) * 1000
    logger.info(f"Total node creation: {total_time=:.2f} ms")

    return nodes


def create_used_nodes_and_edges(graph: Graph, nodes, edge_mask):
    """Filter to only used nodes and create edges."""
    start_time = time.time()
    edges = edge_mask.numpy()
    dsts, srcs = edges.nonzero()
    weights = graph.adjacency_matrix.numpy()[dsts, srcs].tolist()

    used_edges = [
        {"source": nodes[src].node_id, "target": nodes[dst].node_id, "weight": weight}
        for src, dst, weight in zip(srcs, dsts, weights)
        if src in nodes and dst in nodes
    ]

    connected_ids = set()
    for edge in used_edges:
        connected_ids.add(edge["source"])
        connected_ids.add(edge["target"])

    nodes_before = len(nodes)
    used_nodes = [
        node
        for node in nodes.values()
        if node.node_id in connected_ids or node.feature_type in ["embedding", "logit"]
    ]
    nodes_after = len(used_nodes)
    logger.info(f"Filtered {nodes_before - nodes_after} nodes")

    time_ms = (time.time() - start_time) * 1000
    logger.info(f"Creating used nodes and edges: {time_ms=:.2f} ms")
    logger.info(f"Used nodes: {len(used_nodes)}, Used edges: {len(used_edges)}")

    return used_nodes, used_edges


def build_model(graph: Graph, used_nodes, used_edges, slug, scan, node_threshold, tokenizer):
    """Build the full model object."""
    start_time = time.time()

    if isinstance(scan, list):
        transcoder_list = scan
        transcoder_list_str = "-".join(transcoder_list)
        transcoder_list_hash = hash(transcoder_list_str)
        scan = "custom-" + str(transcoder_list_hash)
    else:
        transcoder_list = []

    meta = Metadata(
        slug=slug,
        scan=scan,
        transcoder_list=transcoder_list,
        prompt_tokens=[tokenizer.decode(t) for t in graph.input_tokens],
        prompt=graph.input_string,
        node_threshold=node_threshold,
    )

    qparams = QParams(
        pinnedIds=[],
        supernodes=[],
        linkType="both",
        clickedId="",
        sg_pos="",
    )

    full_model = Model(
        metadata=meta,
        qParams=qparams,
        nodes=used_nodes,
        links=used_edges,
    )

    time_ms = (time.time() - start_time) * 1000
    logger.info(f"Building model: {time_ms=:.2f} ms")

    return full_model


def create_graph_files_from_prune_graph(
    prune_graph,
    slug: str,
    output_path,
):
    """Write frontend JSON files from a PruneGraph (e.g. produced by prune_pt_graph).

    Mirrors create_graph_files but skips the pruning step since PruneGraph is
    already pruned.
    """
    from summarization.prune import PruneGraph

    if not isinstance(prune_graph, PruneGraph):
        raise TypeError(f"expected PruneGraph, got {type(prune_graph)!r}")

    if os.path.exists(output_path):
        assert os.path.isdir(output_path)
    else:
        os.makedirs(output_path, exist_ok=True)

    meta_raw = prune_graph.metadata
    scan = meta_raw.get("scan") or ""
    if isinstance(scan, list):
        transcoder_list = scan
        scan = "-".join(scan)
    else:
        transcoder_list = []

    # Convert summarization Nodes → frontend Nodes (same fields, different class)
    frontend_nodes = [
        Node(
            node_id=n.node_id,
            feature=n.feature,
            layer=n.layer,
            ctx_idx=n.ctx_idx,
            feature_type=n.feature_type,
            token_prob=n.token_prob,
            is_target_logit=n.is_target_logit,
            run_idx=n.run_idx,
            reverse_ctx_idx=n.reverse_ctx_idx,
            jsNodeId=n.jsNodeId,
            clerp=n.clerp,
            influence=n.influence,
            activation=n.activation,
        )
        for n in prune_graph.nodes
    ]

    # Build links from pruned_adj (adj[dst, src] = weight)
    adj = prune_graph.pruned_adj.cpu()
    dsts, srcs = adj.nonzero(as_tuple=True)
    id_map = {n.node_idx: n.node_id for n in prune_graph.nodes}
    used_edges = [
        {
            "source": id_map[int(srcs[i])],
            "target": id_map[int(dsts[i])],
            "weight": float(adj[int(dsts[i]), int(srcs[i])].item()),
        }
        for i in range(len(srcs))
    ]

    meta = Metadata(
        slug=slug,
        scan=scan,
        transcoder_list=transcoder_list,
        prompt_tokens=meta_raw.get("prompt_tokens", []),
        prompt=meta_raw.get("prompt", ""),
        node_threshold=None,
    )
    qparams = QParams(pinnedIds=[], supernodes=[], linkType="both", clickedId="", sg_pos="")
    full_model = Model(metadata=meta, qParams=qparams, nodes=frontend_nodes, links=used_edges)

    with open(os.path.join(output_path, f"{slug}.json"), "w") as f:
        f.write(full_model.model_dump_json(indent=2))
    add_graph_metadata(full_model.metadata.model_dump(), output_path)
    logger.info("Graph data written to %s", output_path)


def create_graph_files(
    graph_or_path: Graph | str,
    slug: str,
    output_path,
    scan=None,
    node_threshold=0.8,
    edge_threshold=0.98,
):
    # Import Graph/prune_graph locally to avoid circular import at module import time
    from circuit_tracer.graph import Graph, prune_graph

    total_start_time = time.time()

    if isinstance(graph_or_path, Graph):
        graph = graph_or_path
    else:
        graph = load_graph_data(graph_or_path)

    if os.path.exists(output_path):
        assert os.path.isdir(output_path)
    else:
        os.makedirs(output_path, exist_ok=True)

    if scan is None:
        if graph.scan is None:
            raise ValueError(
                "Neither scan nor graph.scan was set. One must be set to identify "
                "which transcoders were used when creating the graph."
            )
        scan = graph.scan

    device = "cuda" if torch.cuda.is_available() else "cpu"
    graph.to(device)
    node_mask, edge_mask, cumulative_scores = (
        el.cpu() for el in prune_graph(graph, node_threshold, edge_threshold)
    )
    graph.to("cpu")

    tokenizer = AutoTokenizer.from_pretrained(graph.cfg.tokenizer_name)
    nodes = create_nodes(graph, node_mask, tokenizer, cumulative_scores)
    used_nodes, used_edges = create_used_nodes_and_edges(graph, nodes, edge_mask)
    model = build_model(graph, used_nodes, used_edges, slug, scan, node_threshold, tokenizer)

    # Write the output locally
    with open(os.path.join(output_path, f"{slug}.json"), "w") as f:
        f.write(model.model_dump_json(indent=2))
    add_graph_metadata(model.metadata.model_dump(), output_path)
    logger.info(f"Graph data written to {output_path}")

    total_time_ms = (time.time() - total_start_time) * 1000
    logger.info(f"Total execution time: {total_time_ms=:.2f} ms")
