from __future__ import annotations

from argparse import Namespace

import torch

from summarization.__main__ import build_parser
from summarization.prune import PruneGraph, load_prune_graph, prune_attr_graph, prune_combined
from summarization.summarize import Node, SummaryGraph, Supernode


def test_pipeline_cli_prune_defaults() -> None:
    args = build_parser().parse_args([])

    assert args.logit_weights == "target"
    assert args.token_attr_normalize == "entmax"
    assert args.entmax_alpha == 1.25
    assert args.combine_method == "geometric"
    assert args.normalization == "rank"
    assert args.node_threshold == 0.02
    assert args.edge_threshold == 0.9
    assert args.alpha == 0.5
    assert args.keep_all_tokens_and_logits is False
    assert args.filter_act_density is True


def test_pipeline_cli_can_disable_act_density_filter() -> None:
    args = build_parser().parse_args(["--no-filter-act-density"])

    assert args.filter_act_density is False


def test_pipeline_cli_accepts_prune_graph_out() -> None:
    args = build_parser().parse_args(
        [
            "--prune-graph-out",
            "out/000_prune_graph.pt",
            "--summary-graph-out",
            "out/000_summary_graph.pt",
        ]
    )

    assert args.prune_graph_out == "out/000_prune_graph.pt"
    assert args.summary_graph_out == "out/000_summary_graph.pt"


def test_prune_api_defaults_match_pipeline_defaults() -> None:
    combined_defaults = prune_combined.__defaults__
    attr_graph_defaults = prune_attr_graph.__defaults__

    assert combined_defaults[0] == "target"
    assert combined_defaults[4] == 0.02
    assert combined_defaults[5] == 0.9
    assert combined_defaults[6] == "geometric"
    assert combined_defaults[7] == "rank"
    assert combined_defaults[8] == 0.5
    assert combined_defaults[9] is False

    assert attr_graph_defaults[0] == "target"
    assert attr_graph_defaults[4] == 0.02
    assert attr_graph_defaults[5] == 0.9
    assert attr_graph_defaults[6] == "geometric"
    assert attr_graph_defaults[7] == "rank"
    assert attr_graph_defaults[8] == 0.5
    assert attr_graph_defaults[9] is False


def test_run_pipeline_saves_prune_graph(monkeypatch, tmp_path) -> None:
    from summarization import pipeline

    nodes = [
        Node(
            node_id="E_1_0",
            node_idx=0,
            feature=0,
            layer="E",
            ctx_idx=0,
            feature_type="embedding",
        ),
        Node(
            node_id="1_10_0",
            node_idx=1,
            feature=10,
            layer="1",
            ctx_idx=0,
            feature_type="cross layer transcoder",
        ),
        Node(
            node_id="3_2_0",
            node_idx=2,
            feature=2,
            layer="3",
            ctx_idx=0,
            feature_type="logit",
            is_target_logit=True,
        ),
    ]
    prune_graph = PruneGraph(
        nodes=nodes,
        pruned_adj=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        metadata={"prompt": "a", "prompt_tokens": ["a"]},
        node_influence=torch.ones(3),
        node_relevance=torch.ones(3),
    )
    rows = [
        Supernode("SN_0", [nodes[1]], "features", 1, 1),
        Supernode("SN_EMB_0", [nodes[0]], "emb", 0, 0),
        Supernode("SN_LOGIT_0", [nodes[2]], "logit", 3, 3),
    ]
    summary_graph = SummaryGraph(rows, prune_graph.pruned_adj, prune_graph.metadata)

    monkeypatch.setattr(pipeline, "_acquire_graph", lambda _args: object())
    monkeypatch.setattr(pipeline.AttrGraph, "from_graph", lambda _graph: object())
    monkeypatch.setattr(pipeline, "prune_attr_graph", lambda *_args, **_kwargs: prune_graph)
    monkeypatch.setattr(pipeline, "cluster", lambda *_args, **_kwargs: rows)
    monkeypatch.setattr(pipeline, "summarize", lambda *_args, **_kwargs: summary_graph)

    prune_path = tmp_path / "000_prune_graph.pt"
    summary_path = tmp_path / "000_summary_graph.pt"
    args = Namespace(
        token_weights=None,
        auto_token_weights=False,
        logit_weights="target",
        node_threshold=0.02,
        edge_threshold=0.9,
        combine_method="geometric",
        normalization="rank",
        alpha=0.5,
        keep_all_tokens_and_logits=False,
        filter_act_density=False,
        classify_filter=False,
        prune_graph_out=str(prune_path),
        method="ilp",
        theta="p65",
        max_layer_span=4,
        max_sn=None,
        eps_causal=None,
        ilp_time_limit=30.0,
        supernodes_out=None,
        supernode_map_out=None,
        supernode_flow_out=None,
        auto_k_sweep_out=None,
        summary_graph_out=str(summary_path),
        figure_html_out=None,
        upload=False,
    )

    result = pipeline.run_pipeline(args)

    saved = load_prune_graph(str(prune_path))
    assert saved.num_nodes == prune_graph.num_nodes
    assert saved.num_edges == prune_graph.num_edges
    assert result["prune_graph_out"] == str(prune_path)
    assert result["summary_graph_out"] == str(summary_path)
