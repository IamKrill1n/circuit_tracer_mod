from __future__ import annotations

import re

import numpy as np
import pytest
import torch

from summarization.cluster_viz import (
    _CARD_W,
    _CARD_H,
    _label_card_size,
    _select_edge_indices,
    _supernode_layout,
    _wrap_label,
    supernode_graph_figure,
)
from summarization.summarize import Node, SummaryGraph, Supernode


def _node(
    node_id: str,
    node_idx: int,
    layer: str,
    ctx_idx: int,
    feature_type: str,
    *,
    is_target_logit: bool = False,
    clerp: str | None = None,
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
        clerp=clerp or ('Output "red" (p=0.700)' if is_target_logit else node_id),
    )


def _summary_graph() -> SummaryGraph:
    emb = Supernode(
        "SN_EMB_0",
        [_node("E_0", 0, "E", 0, "embedding", clerp="Emb: The")],
        "emb",
        -1,
        -1,
    )
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


def test_connected_logit_layout_uses_barycentric_structure_not_output_anchor() -> None:
    sng = _summary_graph()
    mapping = sng.to_mapping()
    pos, _top_y, _layer_y = _supernode_layout(
        sng.sn_names,
        mapping,
        attr=None,
        node_by_name=sng.node_by_name(),
        right_x=8.0,
        visible_edges=[
            ("SN_EMB_0", "SN_0", 1.0),
            ("SN_0", "SN_LOGIT_0", 1.0),
        ],
    )

    assert pos["SN_EMB_0"][0] == pytest.approx(0.0)
    assert pos["SN_0"][0] == pytest.approx(0.0)
    assert pos["SN_LOGIT_0"][0] == pytest.approx(0.0)


def test_same_layer_supernodes_have_extra_horizontal_spacing() -> None:
    first = Supernode("SN_A", [_node("1_0", 0, "1", 1, "sae_feature")], "features", 1, 1)
    second = Supernode("SN_B", [_node("1_1", 1, "1", 1, "sae_feature")], "features", 1, 1)
    sng = SummaryGraph([first, second], torch.zeros((2, 2), dtype=torch.float32))

    pos, _top_y, _layer_y = _supernode_layout(
        sng.sn_names,
        sng.to_mapping(),
        attr=None,
        node_by_name=sng.node_by_name(),
    )

    assert pos["SN_B"][0] - pos["SN_A"][0] > _CARD_W


def test_barycentric_layout_places_middle_node_between_embedding_anchors() -> None:
    nodes = [
        Supernode("SN_EMB_0", [_node("E_0", 0, "E", 0, "embedding")], "emb", -1, -1),
        Supernode("SN_EMB_6", [_node("E_6", 1, "E", 6, "embedding")], "emb", -1, -1),
        Supernode("SN_MID", [_node("1_0", 2, "1", 4, "sae_feature")], "features", 1, 1),
    ]
    sng = SummaryGraph(nodes, torch.zeros((3, 3), dtype=torch.float32))

    pos, _top_y, _rank_y = _supernode_layout(
        sng.sn_names,
        sng.to_mapping(),
        attr=None,
        node_by_name=sng.node_by_name(),
        visible_edges=[
            ("SN_EMB_0", "SN_MID", 1.0),
            ("SN_EMB_6", "SN_MID", 3.0),
        ],
    )

    assert pos["SN_EMB_0"][0] == pytest.approx(0.0)
    assert pos["SN_EMB_6"][0] == pytest.approx(6.0)
    assert pos["SN_MID"][0] == pytest.approx(4.5)


def test_topological_layout_orders_chain_bottom_to_top() -> None:
    sng = _summary_graph()

    pos, _top_y, _rank_y = _supernode_layout(
        sng.sn_names,
        sng.to_mapping(),
        attr=None,
        node_by_name=sng.node_by_name(),
        visible_edges=[
            ("SN_EMB_0", "SN_0", 1.0),
            ("SN_0", "SN_LOGIT_0", 1.0),
        ],
    )

    assert pos["SN_EMB_0"][1] < pos["SN_0"][1] < pos["SN_LOGIT_0"][1]


def test_skip_edge_preserves_longest_path_topological_ordering() -> None:
    sng = _summary_graph()

    pos, _top_y, _rank_y = _supernode_layout(
        sng.sn_names,
        sng.to_mapping(),
        attr=None,
        node_by_name=sng.node_by_name(),
        visible_edges=[
            ("SN_EMB_0", "SN_0", 1.0),
            ("SN_0", "SN_LOGIT_0", 1.0),
            ("SN_EMB_0", "SN_LOGIT_0", 0.5),
        ],
    )

    assert pos["SN_EMB_0"][1] < pos["SN_0"][1] < pos["SN_LOGIT_0"][1]


def test_cycle_visible_edges_layout_without_validation() -> None:
    first = Supernode("SN_A", [_node("1_0", 0, "1", 1, "sae_feature")], "features", 1, 1)
    second = Supernode("SN_B", [_node("2_0", 1, "2", 2, "sae_feature")], "features", 2, 2)
    logit = Supernode(
        "SN_LOGIT_0",
        [_node("27_0", 2, "27", 0, "logit", is_target_logit=True)],
        "logit",
        27,
        27,
    )
    sng = SummaryGraph([first, second, logit], torch.zeros((3, 3), dtype=torch.float32))
    visible_edges = [
        ("SN_A", "SN_B", 1.0),
        ("SN_B", "SN_A", 1.0),
        ("SN_B", "SN_LOGIT_0", 1.0),
    ]

    pos, _top_y, _rank_y = _supernode_layout(
        sng.sn_names,
        sng.to_mapping(),
        attr=None,
        node_by_name=sng.node_by_name(),
        visible_edges=visible_edges,
    )

    assert set(pos) == set(sng.sn_names)


def test_legacy_cyclic_sn_adj_renders_without_validation() -> None:
    legacy = {
        "sn_names": ["SN_A", "SN_B"],
        "sn_adj": np.array([[0.0, 1.0], [1.0, 0.0]]),
    }

    fig = supernode_graph_figure(legacy, final_supernodes={"SN_A": ["a"], "SN_B": ["b"]})

    assert fig.data


def test_figure_uses_summary_adj_when_raw_pruned_adj_is_empty() -> None:
    first = Supernode("SN_A", [_node("1_0", 0, "1", 1, "sae_feature")], "features", 1, 1)
    second = Supernode("SN_B", [_node("2_0", 1, "2", 2, "sae_feature")], "features", 2, 2)
    sng = SummaryGraph(
        [first, second],
        torch.zeros((2, 2), dtype=torch.float32),
        adj=np.array([[0.0, 0.0], [2.0, 0.0]]),
    )

    fig = supernode_graph_figure(sng)
    edge_hovertemplates = [
        str(trace.hovertemplate)
        for trace in fig.data
        if getattr(trace, "hovertemplate", None) and "->" in str(trace.hovertemplate)
    ]

    assert any("SN_A -> SN_B" in hovertemplate for hovertemplate in edge_hovertemplates)


def test_select_edge_indices_keeps_top_k_per_source_and_positive_only() -> None:
    edges = [
        ("A", "B", 1.0),
        ("A", "C", -4.0),
        ("A", "D", 2.0),
        ("B", "C", 3.0),
        ("B", "D", -5.0),
    ]

    assert _select_edge_indices(edges, top_k=2) == {1, 2, 3, 4}
    assert _select_edge_indices(edges, top_k=1, positive_only=True) == {2, 3}


def test_supernode_figure_adds_edge_filter_controls() -> None:
    emb = Supernode(
        "SN_EMB_0",
        [_node("E_0", 0, "E", 0, "embedding", clerp="Emb: The")],
        "emb",
        -1,
        -1,
    )
    first = Supernode("SN_A", [_node("1_0", 1, "1", 1, "sae_feature")], "features", 1, 1)
    second = Supernode("SN_B", [_node("2_0", 2, "2", 2, "sae_feature")], "features", 2, 2)
    third = Supernode("SN_C", [_node("3_0", 3, "3", 3, "sae_feature")], "features", 3, 3)
    adj = np.zeros((4, 4), dtype=np.float64)
    adj[1, 0] = 1.0
    adj[2, 0] = 2.0
    adj[3, 0] = -3.0
    sng = SummaryGraph([emb, first, second, third], torch.zeros((4, 4)), adj=adj)

    fig = supernode_graph_figure(sng)

    assert fig.layout.sliders
    assert fig.layout.updatemenus
    assert fig.layout.sliders[0].currentvalue.prefix == "Top k edges per node: "
    assert fig.layout.sliders[0].active == 3
    assert fig.layout.sliders[0].y < 0
    assert fig.layout.updatemenus[0].y < 0
    button_labels = [button.label for button in fig.layout.updatemenus[0].buttons]
    assert button_labels == ["All edges", "Positive only"]


def test_supernode_figure_initially_shows_top_three_edges_per_source() -> None:
    emb = Supernode(
        "SN_EMB_0",
        [_node("E_0", 0, "E", 0, "embedding", clerp="Emb: The")],
        "emb",
        -1,
        -1,
    )
    targets = [
        Supernode(
            f"SN_{idx}",
            [_node(f"{idx}_0", idx, str(idx), idx, "sae_feature")],
            "features",
            idx,
            idx,
        )
        for idx in range(1, 5)
    ]
    adj = np.zeros((5, 5), dtype=np.float64)
    for idx, weight in enumerate([1.0, 4.0, -3.0, 2.0], start=1):
        adj[idx, 0] = weight
    sng = SummaryGraph([emb, *targets], torch.zeros((5, 5)), adj=adj)

    fig = supernode_graph_figure(sng)

    edge_traces = [
        trace
        for trace in fig.data
        if getattr(trace, "hovertemplate", None) and "->" in str(trace.hovertemplate)
    ]
    assert len(edge_traces) == 4
    assert sum(trace.visible is True for trace in edge_traces) == 3
    assert sum(trace.visible is False for trace in edge_traces) == 1


def test_duplicate_supernode_labels_do_not_create_render_self_loops() -> None:
    first = Supernode(
        "Unrelated text",
        [_node("1_0", 0, "1", 1, "sae_feature")],
        "features",
        1,
        1,
    )
    second = Supernode(
        "Unrelated text",
        [_node("2_0", 1, "2", 2, "sae_feature")],
        "features",
        2,
        2,
    )
    sng = SummaryGraph(
        [first, second],
        torch.zeros((2, 2), dtype=torch.float32),
        adj=np.array([[0.0, 0.0], [1.0, 0.0]]),
    )

    fig = supernode_graph_figure(sng)

    edge_hovertemplates = [
        str(trace.hovertemplate)
        for trace in fig.data
        if getattr(trace, "hovertemplate", None) and "->" in str(trace.hovertemplate)
    ]
    assert any("Unrelated text -> Unrelated text (2)" in text for text in edge_hovertemplates)


def test_long_role_label_wraps_for_wider_card() -> None:
    wrapped = _wrap_label("Abstract: National associations")

    assert wrapped == "Abstract: National<br>associations"
    assert _CARD_W > 2.0


def test_card_height_grows_to_contain_wrapped_label() -> None:
    width, height = _label_card_size("Abstract: Political geography", member_count=1)

    assert width >= _CARD_W
    assert height > _CARD_H


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


def test_skip_edge_uses_fixed_port_orthogonal_route_under_cards() -> None:
    emb = Supernode(
        "SN_EMB_0",
        [_node("E_0", 0, "E", 0, "embedding", clerp="Emb: The")],
        "emb",
        -1,
        -1,
    )
    middle = Supernode("SN_MID", [_node("1_0", 1, "1", 0, "sae_feature")], "features", 1, 1)
    logit = Supernode(
        "SN_LOGIT_0",
        [_node("27_0", 2, "27", 0, "logit", is_target_logit=True)],
        "logit",
        27,
        27,
    )
    adj = np.zeros((3, 3), dtype=np.float64)
    adj[1, 0] = 1.0
    adj[2, 1] = 1.0
    adj[2, 0] = 0.5
    sng = SummaryGraph([emb, middle, logit], torch.zeros((3, 3)), adj=adj)
    pos, _top_y, _rank_y = _supernode_layout(
        sng.sn_names,
        sng.to_mapping(),
        attr=None,
        node_by_name=sng.node_by_name(),
        visible_edges=[
            ("SN_EMB_0", "SN_MID", 1.0),
            ("SN_MID", "SN_LOGIT_0", 1.0),
            ("SN_EMB_0", "SN_LOGIT_0", 0.5),
        ],
    )

    fig = supernode_graph_figure(sng)

    edge_traces = [
        trace
        for trace in fig.data
        if getattr(trace, "hovertemplate", None)
        and "SN_EMB_0 -> SN_LOGIT_0" in str(trace.hovertemplate)
    ]
    assert len(edge_traces) == 1
    edge_trace_index = list(fig.data).index(edge_traces[0])
    edge_shapes = [
        shape
        for shape in fig.layout.shapes
        if str(getattr(shape.line, "color", "")).startswith("#")
    ]
    path = str(edge_shapes[edge_trace_index].path)
    coords = [float(raw) for raw in re.findall(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?", path)]
    xs = coords[0::2]

    assert " L " in path
    assert " C " not in path
    assert xs[0] == pytest.approx(pos["SN_EMB_0"][0])
    assert xs[-1] == pytest.approx(pos["SN_LOGIT_0"][0])


def test_prompt_tokens_render_as_top_strip_without_old_bar_labels() -> None:
    prompt_tokens = ["<bos>", "The", "saying", "goes", ":", "ab", "uja"]
    fig = supernode_graph_figure(
        _summary_graph(),
        prompt_tokens=prompt_tokens,
        prompt="<bos>The saying goes: abuja",
    )

    annotations = list(fig.layout.annotations)
    annotation_text = "\n".join(str(annotation.text) for annotation in annotations)

    assert "Input Tokens" not in annotation_text
    assert "Outputs / Logits" not in annotation_text

    token_annotations = [
        annotation
        for annotation in annotations
        if str(annotation.text) in {"<bos>", "saying", "ab"}
    ]
    assert len(token_annotations) == 3
    token_y = {float(annotation.y) for annotation in token_annotations}
    assert len(token_y) == 1
    assert next(iter(token_y)) == max(
        float(annotation.y) for annotation in annotations if annotation.text
    )
    assert next(iter(token_y)) > 0.0


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


def test_supernode_figure_html_uses_draggable_svg_renderer() -> None:
    fig = supernode_graph_figure(_summary_graph())

    page = fig.to_html(include_plotlyjs="cdn", full_html=True)

    assert fig.payload["scale"] <= 60.0
    assert "data-cluster-viz" in page
    assert "data-svg" in page
    assert "data-edge-panel" in page
    assert "pointerdown" in page
    assert "node.rankPy" in page
    assert "selectedNodeId" in page
    assert "stroke-dasharray" in page
    assert "node.py + node.heightPx / 2 + 4" in page
    assert "function renderEdges()" in page
    assert "pathFromRoute" in page
    assert "Plotly.newPlot" not in page


def test_initial_edge_paths_are_orthogonal_not_curved() -> None:
    fig = supernode_graph_figure(_summary_graph())

    paths = [str(shape.path) for shape in fig.layout.shapes]

    assert paths
    assert all(" L " in path for path in paths)
    assert all(" C " not in path for path in paths)


def test_display_labels_use_node_kind_role_and_render_steering_overlays() -> None:
    sng = _summary_graph()
    sng.supernodes[1].name = "Hue"
    sng.supernodes[1].role = "Abstract"

    fig = supernode_graph_figure(
        sng,
        prompt_tokens=["<bos>", "The", "sky", "is"],
        prompt="<bos>The sky is",
        steering_factors={"Hue": -1.0},
        activation_ratios={"Hue": 0.20},
        top_outputs=[
            {
                "token": " blue",
                "probability": 0.42,
                "clean_probability": 0.10,
                "probability_delta": 0.32,
                "logit_delta": 1.25,
            }
        ],
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
    assert "Emb: The" in annotation_text
    assert "Abstract: Hue" in annotation_text
    assert "Logit: red" in annotation_text
    assert "\u0394 Logit: blue" in annotation_text
    assert "red" in annotation_text
    assert "-1x" not in annotation_text
    assert "20%" not in annotation_text
    assert "Stored color" in annotation_text

    hover_trace = next(trace for trace in fig.data if getattr(trace, "hovertext", None))
    hover_text = "<br>".join(str(item) for item in hover_trace.hovertext)
    assert "Intervention: -1x" in hover_text
    assert "Activation ratio: 20%" in hover_text
    assert "Stored: Stored color 2x @ pos 2" in hover_text

    page = fig.to_html(full_html=False)
    assert "ct-intervention-summary" in page
    assert '"active": true' in page
    assert '"steeredCount": 1' in page
    assert '"storedCount": 1' in page
    assert '"text": "-1x"' in page
    assert '"text": "20%"' in page
    assert '"text": "Stored"' in page
    assert '"kind": "delta"' in page
    assert '"text": "\u0394 +1.25"' in page
    assert "\u0394 Logit: blue" in page
    assert "Clean probability: 10%" in page
    assert "\u0394 probability: +32.0%" in page
    assert " blue" in page
    assert "0.42" in page
    assert "Stored color" in page

    nodes = fig.payload["nodes"]
    synthetic_nodes = [
        node for node in nodes if node["kind"] in {"intervention", "output_delta"}
    ]
    graph_nodes = [
        node for node in nodes if node["kind"] not in {"intervention", "output_delta"}
    ]
    assert len(synthetic_nodes) == 2
    assert synthetic_nodes[0]["x"] == pytest.approx(synthetic_nodes[1]["x"])
    assert synthetic_nodes[0]["x"] > max(node["x"] for node in graph_nodes)
    stored_node = next(node for node in synthetic_nodes if node["kind"] == "intervention")
    output_node = next(node for node in synthetic_nodes if node["kind"] == "output_delta")
    assert output_node["y"] > stored_node["y"]


def test_non_steering_summary_graph_keeps_intervention_payload_inactive() -> None:
    fig = supernode_graph_figure(_summary_graph())

    page = fig.to_html(full_html=False)

    assert '"active": false' in page
    assert '"steeredCount": 0' in page
    assert '"storedCount": 0' in page
    assert '"topOutputs": []' in page
    assert '"badges": []' in page
    assert "Intervention: -1x" not in page
