from __future__ import annotations

import re

import pytest
import torch

from summarization.cluster_viz import _supernode_layout, supernode_graph_figure
from summarization.summarize import Node, SummaryGraph, Supernode


def _node(
    node_id: str,
    node_idx: int,
    layer: str,
    ctx_idx: int,
    feature_type: str,
    *,
    is_target_logit: bool = False,
) -> Node:
    return Node(
        node_id=node_id,
        node_idx=node_idx,
        feature=node_idx,
        layer=layer,
        ctx_idx=ctx_idx,
        feature_type=feature_type,
        token_prob=0.7 if is_target_logit else 0.0,
        is_target_logit=is_target_logit,
        clerp='Output "red" (p=0.700)' if is_target_logit else node_id,
    )


def _summary_graph() -> SummaryGraph:
    emb = Supernode("SN_EMB_0", [_node("E_0", 0, "E", 0, "embedding")], "emb", -1, -1)
    mid = Supernode("SN_0", [_node("1_0", 1, "1", 4, "sae_feature")], "features", 1, 1)
    logit = Supernode(
        "SN_LOGIT_0",
        [_node("27_0", 2, "27", 0, "logit", is_target_logit=True)],
        "logit",
        27,
        27,
    )
    pruned_adj = torch.zeros((3, 3), dtype=torch.float32)
    pruned_adj[1, 0] = 1.0
    pruned_adj[2, 1] = 1.0
    return SummaryGraph([emb, mid, logit], pruned_adj)


def test_logit_layout_uses_output_anchor_not_logit_ctx() -> None:
    sng = _summary_graph()
    mapping = sng.to_mapping()
    pos, _top_y = _supernode_layout(
        sng.sn_names,
        mapping,
        attr=None,
        node_by_name=sng.node_by_name(),
        right_x=8.0,
    )

    assert pos["SN_LOGIT_0"][0] == pytest.approx(8.0)
    assert pos["SN_EMB_0"][0] == pytest.approx(0.0)
    assert pos["SN_0"][0] == pytest.approx(4.0)


def test_edge_paths_are_complete_within_axis_range() -> None:
    fig = supernode_graph_figure(
        _summary_graph(),
        prompt_tokens=["<bos>", "The", "saying", "goes", ":", "celery", "is", "green"],
        prompt="<bos>The saying goes: celery is green",
    )

    edge_shapes = [
        shape
        for shape in fig.layout.shapes
        if isinstance(shape.path, str) and shape.path.startswith("M ")
    ]
    assert len(edge_shapes) >= 2

    x_min, x_max = fig.layout.xaxis.range
    y_min, y_max = fig.layout.yaxis.range
    for shape in edge_shapes:
        coords = [float(raw) for raw in re.findall(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?", shape.path)]
        xs = coords[0::2]
        ys = coords[1::2]
        assert min(xs) >= x_min
        assert max(xs) <= x_max
        assert min(ys) >= y_min
        assert max(ys) <= y_max
