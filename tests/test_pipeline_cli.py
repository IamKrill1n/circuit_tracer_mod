from __future__ import annotations

from summarization.__main__ import build_parser
from summarization.prune import prune_attr_graph, prune_combined


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
