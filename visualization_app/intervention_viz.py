from __future__ import annotations

from dataclasses import dataclass, field
import html
import math
from typing import Any

import numpy as np

_GRAPH_OFFSET_X = 70
_GRAPH_WIDTH = 960
_GRAPH_HEIGHT = 560
_NODE_WIDTH = 170
_NODE_HEIGHT = 54
_COL_GAP = 95
_PROMPT_SEPARATOR_Y = 650


@dataclass(frozen=True)
class Feature:
    layer: int
    pos: int
    feature_idx: int


@dataclass(eq=False)
class InterventionNode:
    name: str
    label: str
    features: list[Feature] = field(default_factory=list)
    activation: float | None = None
    children: list["InterventionNode"] = field(default_factory=list)
    intervention: str | None = None
    replacement_nodes: list["InterventionNode"] = field(default_factory=list)


@dataclass
class InterventionGraph:
    ordered_nodes: list[list[InterventionNode]]
    prompt: str
    nodes: dict[str, InterventionNode] = field(default_factory=dict)


def summary_graph_to_intervention_graph(
    sng: Any,
    *,
    prompt: str,
    steering_factors: dict[str, float],
    activation_ratios: dict[str, float | None],
    stored_interventions: list[dict[str, Any]],
    prompt_tokens: list[str] | None = None,
    edge_threshold: float,
) -> InterventionGraph:
    sn_names = list(sng.sn_names)
    graph_nodes = [
        InterventionNode(
            name=supernode.name,
            label=_display_label(supernode, prompt_tokens or []),
            features=_features_for_supernode(supernode),
            activation=activation_ratios.get(supernode.name),
            intervention=_format_factor(steering_factors[supernode.name])
            if supernode.name in steering_factors
            else None,
        )
        for supernode in sng.supernodes
    ]
    node_by_name = {node.name: node for node in graph_nodes}

    for source, target in _visible_edges(sn_names, np.asarray(sng.adj), edge_threshold):
        node_by_name[source].children.append(node_by_name[target])

    for stored in stored_interventions:
        host = _closest_host_node(sng.supernodes, graph_nodes, stored)
        if host is None:
            continue
        donor = InterventionNode(
            name=str(stored.get("label") or stored.get("record_id") or "Stored intervention"),
            label=str(stored.get("label") or stored.get("record_id") or "Stored intervention"),
            activation=None,
            intervention=_format_factor(float(stored.get("factor", 0.0))),
            children=list(host.children),
        )
        host.replacement_nodes.append(donor)

    graph = InterventionGraph(
        ordered_nodes=_ordered_nodes(sn_names, np.asarray(sng.adj), graph_nodes),
        prompt=prompt,
    )
    graph.nodes = node_by_name
    return graph


def render_intervention_svg(
    graph: InterventionGraph,
    top_outputs: list[dict[str, Any]],
) -> str:
    graph_width = _layout_width(graph.ordered_nodes)
    svg_width = graph_width + _GRAPH_OFFSET_X + 70
    node_data = _calculate_node_positions(graph.ordered_nodes, graph_width)
    connections = _build_connections_data(graph.ordered_nodes)
    connections_svg = _create_connection_svg(node_data, connections)
    nodes_svg = _create_nodes_svg(node_data)
    prompt_svg, prompt_height = _prompt_svg(graph.prompt)
    outputs_y = _PROMPT_SEPARATOR_Y + 80 + prompt_height
    outputs_svg = _outputs_svg(top_outputs, outputs_y)
    height = max(760, outputs_y + 55)

    return f"""<svg width="{svg_width}" height="{height}" viewBox="0 0 {svg_width} {height}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Steering intervention graph">
  <rect width="{svg_width}" height="{height}" fill="#f5f5f5"/>
  <rect x="30" y="20" width="{svg_width - 60}" height="{height - 40}" fill="white" stroke="none" rx="12"/>
  <text x="40" y="45" fill="#666" font-family="Arial, sans-serif" font-size="14" font-weight="bold">Graph &amp; Interventions</text>
  <g transform="translate({_GRAPH_OFFSET_X}, 0)">
    {connections_svg}
    {nodes_svg}
  </g>
  <line x1="40" y1="{_PROMPT_SEPARATOR_Y}" x2="{svg_width - 40}" y2="{_PROMPT_SEPARATOR_Y}" stroke="#ddd" stroke-width="1"/>
  <text x="40" y="{_PROMPT_SEPARATOR_Y + 20}" fill="#666" font-family="Arial, sans-serif" font-size="12" font-weight="bold">Prompt</text>
  {prompt_svg}
  {outputs_svg}
</svg>"""


def create_intervention_svg(
    sng: Any,
    *,
    prompt: str,
    steering_factors: dict[str, float],
    activation_ratios: dict[str, float | None],
    top_outputs: list[dict[str, Any]],
    stored_interventions: list[dict[str, Any]],
    prompt_tokens: list[str] | None = None,
    edge_threshold: float,
) -> str:
    graph = summary_graph_to_intervention_graph(
        sng,
        prompt=prompt,
        steering_factors=steering_factors,
        activation_ratios=activation_ratios,
        stored_interventions=stored_interventions,
        prompt_tokens=prompt_tokens,
        edge_threshold=edge_threshold,
    )
    return render_intervention_svg(graph, top_outputs)


def _display_label(supernode: Any, prompt_tokens: list[str]) -> str:
    if supernode.type == "emb":
        token = _token_for_supernode(supernode, prompt_tokens)
        return f"Emb: {token}" if token else str(supernode.name)
    if supernode.type == "logit":
        token = _token_for_supernode(supernode, prompt_tokens)
        return f"Logit: {token}" if token else str(supernode.name)
    return str(supernode.name)


def _token_for_supernode(supernode: Any, prompt_tokens: list[str]) -> str:
    for node in supernode.features:
        clerp = str(getattr(node, "clerp", "") or "").strip()
        if clerp:
            return _token_from_clerp(clerp)
    if supernode.type == "emb" and supernode.features:
        ctx_idx = int(supernode.features[0].ctx_idx)
        if 0 <= ctx_idx < len(prompt_tokens):
            return _clean_token(prompt_tokens[ctx_idx])
    return ""


def _token_from_clerp(text: str) -> str:
    if '"' in text:
        parts = text.split('"')
        if len(parts) >= 3:
            return _clean_token(parts[1])
    if ":" in text:
        return _clean_token(text.split(":", 1)[1])
    return _clean_token(text)


def _clean_token(token: str) -> str:
    cleaned = token.replace("Ġ", " ").replace("▁", " ").replace("\n", "\\n")
    cleaned = cleaned.strip()
    return cleaned if cleaned else "·"


def _features_for_supernode(supernode: Any) -> list[Feature]:
    features: list[Feature] = []
    for node in supernode.features:
        if node.feature_type != "cross layer transcoder":
            continue
        layer, feature_idx = _parse_clt_node_id(str(node.node_id))
        if layer is None or feature_idx is None:
            continue
        features.append(Feature(layer=layer, pos=int(node.ctx_idx), feature_idx=feature_idx))
    return features


def _parse_clt_node_id(node_id: str) -> tuple[int | None, int | None]:
    parts = node_id.split("_")
    if len(parts) < 2:
        return None, None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None, None


def _visible_edges(
    sn_names: list[str],
    adj: np.ndarray,
    edge_threshold: float,
) -> list[tuple[str, str]]:
    if adj.size == 0:
        return []
    max_abs = float(np.max(np.abs(adj)))
    cutoff = max(0.0, float(edge_threshold)) * max_abs
    edges: list[tuple[str, str]] = []
    for target_idx, source_idx in zip(*np.nonzero(adj)):
        weight = float(adj[target_idx, source_idx])
        if abs(weight) < cutoff:
            continue
        edges.append((sn_names[source_idx], sn_names[target_idx]))
    return edges


def _ordered_nodes(
    sn_names: list[str],
    adj: np.ndarray,
    nodes: list[InterventionNode],
) -> list[list[InterventionNode]]:
    if adj.size == 0:
        return [nodes]
    name_to_idx = {name: idx for idx, name in enumerate(sn_names)}
    depth = {name: 0 for name in sn_names}
    for _ in range(len(sn_names)):
        changed = False
        for target_idx, source_idx in zip(*np.nonzero(adj)):
            source = sn_names[source_idx]
            target = sn_names[target_idx]
            next_depth = depth[source] + 1
            if next_depth > depth[target]:
                depth[target] = next_depth
                changed = True
        if not changed:
            break
    else:
        for target_idx, source_idx in zip(*np.nonzero(adj)):
            source = sn_names[source_idx]
            target = sn_names[target_idx]
            depth[target] = max(depth[target], depth[source])
    rows: dict[int, list[InterventionNode]] = {}
    for node in nodes:
        rows.setdefault(depth.get(node.name, 0), []).append(node)
    for row in rows.values():
        row.sort(key=lambda node: name_to_idx.get(node.name, 0))
    return [rows[rank] for rank in sorted(rows)]


def _closest_host_node(
    supernodes: list[Any],
    graph_nodes: list[InterventionNode],
    stored: dict[str, Any],
) -> InterventionNode | None:
    candidates: list[tuple[tuple[float, float, str], InterventionNode]] = []
    target_layer = float(stored.get("layer", 0))
    target_pos = float(stored.get("target_pos", 0))
    for supernode, graph_node in zip(supernodes, graph_nodes):
        if supernode.type != "features":
            continue
        layer_center = (float(supernode.layer_min) + float(supernode.layer_max)) / 2.0
        positions = [float(node.ctx_idx) for node in supernode.features]
        pos_center = float(np.mean(positions)) if positions else 0.0
        key = (abs(layer_center - target_layer), abs(pos_center - target_pos), graph_node.name)
        candidates.append((key, graph_node))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def _format_factor(factor: float) -> str:
    if float(factor).is_integer():
        return f"{int(factor)}x"
    return f"{factor:g}x"


def _layout_width(nodes: list[list[InterventionNode]]) -> int:
    max_row_width = max(
        (len(row) * _NODE_WIDTH + max(0, len(row) - 1) * _COL_GAP for row in nodes),
        default=_GRAPH_WIDTH,
    )
    return max(_GRAPH_WIDTH, max_row_width)


def _calculate_node_positions(
    nodes: list[list[InterventionNode]],
    graph_width: int,
) -> dict[str, dict[str, Any]]:
    node_data: dict[str, dict[str, Any]] = {}

    for row_index, row in enumerate(nodes):
        row_y = _GRAPH_HEIGHT - (row_index * (_GRAPH_HEIGHT / (len(nodes) + 0.5)))
        row_width = len(row) * _NODE_WIDTH + max(0, len(row) - 1) * _COL_GAP
        start_x = (graph_width - row_width) / 2
        for col_index, node in enumerate(row):
            node_x = start_x + col_index * (_NODE_WIDTH + _COL_GAP)
            node_data[node.name] = {"x": node_x, "y": row_y, "node": node}
            for replacement_index, replacement in enumerate(node.replacement_nodes):
                node_data[replacement.name] = {
                    "x": node_x + 45 + replacement_index * 45,
                    "y": row_y - 72 - replacement_index * 10,
                    "node": replacement,
                }

    return node_data


def _get_node_center(node_data: dict[str, dict[str, Any]], node_name: str) -> dict[str, float]:
    node = node_data.get(node_name)
    if not node:
        return {"x": 0, "y": 0}
    return {"x": float(node["x"]) + _NODE_WIDTH / 2, "y": float(node["y"]) + _NODE_HEIGHT / 2}


def _build_connections_data(nodes: list[list[InterventionNode]]) -> list[dict[str, Any]]:
    connections: list[dict[str, Any]] = []
    all_nodes: list[InterventionNode] = []
    replacement_names: set[str] = set()

    def visit(node: InterventionNode) -> None:
        if node in all_nodes:
            return
        all_nodes.append(node)
        for replacement in node.replacement_nodes:
            replacement_names.add(replacement.name)
            visit(replacement)
        for child in node.children:
            visit(child)

    for row in nodes:
        for node in row:
            visit(node)

    for node in all_nodes:
        if node.replacement_nodes:
            continue
        for child in node.children:
            connection = {"from": node.name, "to": child.name}
            if node.name in replacement_names:
                connection["replacement"] = True
            connections.append(connection)
    return connections


def _create_connection_svg(
    node_data: dict[str, dict[str, Any]],
    connections: list[dict[str, Any]],
) -> str:
    svg_parts: list[str] = []
    for conn in connections:
        from_center = _get_node_center(node_data, str(conn["from"]))
        to_center = _get_node_center(node_data, str(conn["to"]))
        if from_center["x"] == 0 or to_center["x"] == 0:
            continue
        stroke_color = "#D2691E" if conn.get("replacement") else "#8B4513"
        stroke_width = "4" if conn.get("replacement") else "3"
        svg_parts.append(
            f'<line x1="{from_center["x"]}" y1="{from_center["y"]}" '
            f'x2="{to_center["x"]}" y2="{to_center["y"]}" '
            f'stroke="{stroke_color}" stroke-width="{stroke_width}"/>'
        )
        svg_parts.append(_arrow_polygon(from_center, to_center, stroke_color))
    return "\n".join(svg_parts)


def _arrow_polygon(
    from_center: dict[str, float],
    to_center: dict[str, float],
    stroke_color: str,
) -> str:
    dx = to_center["x"] - from_center["x"]
    dy = to_center["y"] - from_center["y"]
    length = math.sqrt(dx * dx + dy * dy)
    if length == 0:
        return ""
    dx_norm = dx / length
    dy_norm = dy / length
    arrow_size = 8
    base_x = to_center["x"] - arrow_size * dx_norm
    base_y = to_center["y"] - arrow_size * dy_norm
    perp_x = -dy_norm * (arrow_size / 2)
    perp_y = dx_norm * (arrow_size / 2)
    left_x = base_x + perp_x
    left_y = base_y + perp_y
    right_x = base_x - perp_x
    right_y = base_y - perp_y
    return (
        f'<polygon points="{to_center["x"]},{to_center["y"]} '
        f'{left_x},{left_y} {right_x},{right_y}" fill="{stroke_color}"/>'
    )


def _create_nodes_svg(node_data: dict[str, dict[str, Any]]) -> str:
    svg_parts: list[str] = []
    replacement_names = {
        replacement.name
        for data in node_data.values()
        for replacement in data["node"].replacement_nodes
    }
    for name, data in node_data.items():
        node = data["node"]
        x = float(data["x"])
        y = float(data["y"])
        is_low_activation = node.activation is not None and node.activation <= 0.25
        has_negative_intervention = bool(node.intervention and "-" in node.intervention)
        is_replacement = name in replacement_names
        if is_low_activation or has_negative_intervention:
            fill_color = "#f0f0f0"
            text_color = "#777" if has_negative_intervention else "#bbb"
            stroke_color = "#ddd"
        elif is_replacement:
            fill_color = "#FFF8DC"
            text_color = "#333"
            stroke_color = "#D2691E"
        else:
            fill_color = "#e8e8e8"
            text_color = "#333"
            stroke_color = "#999"
        svg_parts.append(
            f'<rect x="{x}" y="{y}" width="{_NODE_WIDTH}" height="{_NODE_HEIGHT}" '
            f'fill="{fill_color}" stroke="{stroke_color}" stroke-width="2" rx="8"/>'
        )
        escaped_label = html.escape(str(node.label))
        label_lines = _wrap_text_for_svg(escaped_label, max_width=19)[:3]
        first_y = y + (_NODE_HEIGHT / 2) - ((len(label_lines) - 1) * 7) + 4
        svg_parts.append(
            f'<text x="{x + _NODE_WIDTH / 2}" y="{first_y}" text-anchor="middle" '
            f'fill="{text_color}" font-family="Arial, sans-serif" font-size="12" '
            f'font-weight="bold">{_node_label_tspans(label_lines, x + _NODE_WIDTH / 2)}</text>'
        )
        if node.activation is not None:
            activation_pct = round(node.activation * 100)
            label_x = x - 12
            label_y = y - 5
            svg_parts.append(
                f'<rect x="{label_x}" y="{label_y}" width="34" height="16" '
                f'fill="white" stroke="#ccc" stroke-width="1" rx="4"/>'
            )
            svg_parts.append(
                f'<text x="{label_x + 17}" y="{label_y + 12}" text-anchor="middle" '
                f'fill="#8B4513" font-family="Arial, sans-serif" font-size="10" '
                f'font-weight="bold">{activation_pct}%</text>'
            )
        if node.intervention:
            escaped_intervention = html.escape(node.intervention)
            text_width = len(node.intervention) * 8 + 10
            svg_parts.append(
                f'<rect x="{x - 18}" y="{y - 5}" width="{text_width}" height="16" '
                f'fill="#D2691E" stroke="none" rx="12"/>'
            )
            svg_parts.append(
                f'<text x="{x - 18 + text_width / 2}" y="{y + 7}" text-anchor="middle" '
                f'fill="white" font-family="Arial, sans-serif" font-size="10" '
                f'font-weight="bold">{escaped_intervention}</text>'
            )
    return "\n".join(svg_parts)


def _prompt_svg(prompt: str) -> tuple[str, int]:
    lines = _wrap_text_for_svg(html.escape(prompt), max_width=130)
    svg_lines = [
        f'<text x="40" y="{_PROMPT_SEPARATOR_Y + 45 + i * 15}" fill="#333" '
        f'font-family="Arial, sans-serif" font-size="12">{line}</text>'
        for i, line in enumerate(lines)
    ]
    return "\n".join(svg_lines), max(0, (len(lines) - 1) * 15)


def _outputs_svg(top_outputs: list[dict[str, Any]], y: int) -> str:
    if not top_outputs:
        return ""
    svg_parts = [
        f'<text x="40" y="{y}" fill="#666" font-family="Arial, sans-serif" '
        'font-size="10" font-weight="bold">Top Outputs</text>'
    ]
    current_x = 40
    for item in top_outputs[:6]:
        token = str(item.get("token") or "(empty)")
        probability = float(item.get("probability", 0.0))
        percentage_text = f"{round(probability * 100)}%"
        item_width = len(token) * 8 + len(percentage_text) * 6 + 26
        escaped_token = html.escape(token)
        svg_parts.append(
            f'<rect x="{current_x}" y="{y + 8}" width="{item_width}" height="22" '
            f'fill="#e8e8e8" stroke="none" rx="6"/>'
        )
        svg_parts.append(
            f'<text x="{current_x + 8}" y="{y + 23}" fill="#333" '
            f'font-family="Arial, sans-serif" font-size="11" font-weight="bold">'
            f'{escaped_token} <tspan fill="#555" font-size="10">{percentage_text}</tspan></text>'
        )
        current_x += item_width + 10
    return "\n".join(svg_parts)


def _node_label_tspans(lines: list[str], x: float) -> str:
    tspans = []
    for index, line in enumerate(lines):
        dy = 0 if index == 0 else 14
        tspans.append(f'<tspan x="{x}" dy="{dy}">{line}</tspan>')
    return "".join(tspans)


def _wrap_text_for_svg(text: str, max_width: int = 80) -> list[str]:
    if len(text) <= max_width:
        return [text]
    words = text.split()
    lines: list[str] = []
    current_line = ""
    for word in words:
        candidate = f"{current_line} {word}".strip()
        if len(candidate) <= max_width:
            current_line = candidate
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines
