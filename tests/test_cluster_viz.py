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


def test_hover_text_includes_supernode_label_role_and_description() -> None:
    sng = _summary_graph()
    sng.supernodes[1].name = "Color relation"
    sng.supernodes[1].role = "Abstract"
    sng.supernodes[1].description = "Combines object and color evidence."

    fig = supernode_graph_figure(
        sng,
        final_supernodes=sng.to_mapping(),
        use_supernode_names=True,
    )

    hover_trace = next(trace for trace in fig.data if getattr(trace, "hovertext", None))
    hover_text = "<br>".join(str(item) for item in hover_trace.hovertext)

    assert "Label: Color relation" in hover_text
    assert "Role: Abstract" in hover_text
    assert "Description: Combines object and color evidence." in hover_text


def test_steering_overlay_adds_intervention_ratio_outputs_and_stored_notes() -> None:
    fig = supernode_graph_figure(
        _summary_graph(),
        prompt_tokens=["<bos>", "The", "sky", "is"],
        prompt="<bos>The sky is",
        steering_factors={"SN_0": -1.0},
        activation_ratios={"SN_0": 0.20},
        top_outputs=[{"token": " blue", "probability": 0.42}],
        stored_interventions=[
            {
                "record_id": "donor:0",
                "label": "Stored color",
                "factor": 2.0,
                "target_pos": 2,
                "n_features": 3,
            }
        ],
    )

    annotation_text = "\n".join(str(annotation.text) for annotation in fig.layout.annotations)
    assert "-1x" in annotation_text
    assert "20%" in annotation_text
    assert "blue 0.420" in annotation_text
    assert "External interventions" in annotation_text
    assert "Stored color: 2x at pos 2 (3 feats)" in annotation_text

    hover_trace = next(trace for trace in fig.data if getattr(trace, "hovertext", None))
    hover_text = "<br>".join(str(item) for item in hover_trace.hovertext)
    assert "Intervention: -1x" in hover_text
    assert "Activation ratio: 20%" in hover_text
