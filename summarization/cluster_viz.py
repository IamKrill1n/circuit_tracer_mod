"""Plotly visualization helpers for clustered supernode graphs.

Renders the supernode graph in the style of the Anthropic attribution-graph
figure: supernodes as rounded "card" boxes (with a stacked-paper shadow for
composites), curved green/red connectors, and a tokenized prompt strip above
the graph.
"""

from __future__ import annotations

import html
import math
from typing import Any, cast

import numpy as np
import plotly.graph_objects as go

from summarization.utils import layer_index_from_node, layer_index_from_node_id
from summarization.summarize import SummaryGraph, Supernode

# Beige card fill (paper palette) with a thin kind-colored border accent.
_KIND_STYLE = {
    "emb": ("#EDE9DD", "#4CAF50"),
    "logit": ("#EDE9DD", "#FF9800"),
    "middle": ("#EDE9DD", "#5B6B7B"),
}
_CARD_W = 2.35
_CARD_H = 0.92
_BAR_H = 0.52
_MIN_CARD_X_GAP = 2.75
_RANK_Y_GAP = 1.55


def _sn_kind(sn_name: str, node_by_name: dict[str, Supernode]) -> str:
    row = node_by_name.get(sn_name)
    if row is not None:
        return "middle" if row.type == "features" else row.type
    if "EMB" in sn_name:
        return "emb"
    if "LOGIT" in sn_name:
        return "logit"
    return "middle"


def _sn_title(
    sn: str,
    members: list[str],
    attr: dict[str, dict[str, Any]] | None,
    supernode: Supernode | None = None,
) -> str:
    lines = [f"Label: {html.escape(str(supernode.name if supernode is not None else sn))}"]
    if supernode is not None and supernode.role:
        lines.append(f"Role: {html.escape(supernode.role)}")
    if supernode is not None and supernode.description:
        lines.append(f"Description: {html.escape(supernode.description)}")
    if not members:
        return "<br>".join(lines)

    lines.append(f"Members: {len(members)} nodes")
    if attr is None:
        return "<br>".join(lines)

    previews: list[str] = []
    for nid in members[:5]:
        clerp = str(attr.get(nid, {}).get("clerp", "") or "").strip()
        if clerp:
            previews.append(f"{html.escape(str(nid))}: {html.escape(clerp[:80])}")
        else:
            previews.append(html.escape(str(nid)))
    more = f" … +{len(members) - 5} more" if len(members) > 5 else ""
    lines.extend(previews)
    if more:
        lines.append(more)
    return "<br>".join(lines)


def _logit_prob(
    sn: str,
    members: list[str],
    node_by_name: dict[str, Supernode],
    attr: dict[str, dict[str, Any]] | None,
) -> float:
    """Representative token probability of a logit supernode (max over members)."""
    row = node_by_name.get(sn)
    if row is not None and row.features:
        return max(float(n.token_prob) for n in row.features)
    if attr is not None and members:
        return max(float(attr.get(nid, {}).get("token_prob", 0.0)) for nid in members)
    return 0.0


def _parse_ctx_idx(attr: dict[str, dict[str, Any]] | None, node_id: str) -> int:
    if attr is None:
        return 0
    raw = attr.get(node_id, {}).get("ctx_idx", 0)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def _layer_and_ctx_for_supernode(
    sn: str,
    members: list[str],
    attr: dict[str, dict[str, Any]] | None,
    node_by_name: dict[str, Supernode] | None = None,
) -> tuple[int, float]:
    row = (node_by_name or {}).get(sn)
    if row is not None and row.features:
        layers = [layer_index_from_node(node) for node in row.features]
        ctx_idx = [node.ctx_idx for node in row.features]
        return (min(layers) if layers else 0, float(np.mean(ctx_idx) if ctx_idx else 0.0))

    items = members if members else [sn]
    layers = [
        layer_index_from_node_id(
            nid, layer=(attr or {}).get(nid, {}).get("layer") if attr else None
        )
        for nid in items
    ]
    ctx_idx = [_parse_ctx_idx(attr, nid) for nid in items]
    return (min(layers) if layers else 0, float(np.mean(ctx_idx) if ctx_idx else 0.0))


def _member_clerp(
    members: list[str],
    attr: dict[str, dict[str, Any]] | None,
    supernode: Supernode | None = None,
) -> str:
    if attr is not None:
        for nid in members:
            clerp = str(attr.get(nid, {}).get("clerp", "") or "").strip()
            if clerp:
                return clerp
    if supernode is not None:
        for node in supernode.features:
            clerp = str(node.clerp or "").strip()
            if clerp:
                return clerp
    return ""


def _token_from_clerp(text: str) -> str:
    if '"' in text:
        parts = text.split('"')
        if len(parts) >= 3:
            return _clean_token(parts[1])
    if ":" in text:
        return _clean_token(text.split(":", 1)[1])
    return _clean_token(text)


def _node_label(
    sn: str,
    members: list[str],
    attr: dict[str, dict[str, Any]] | None,
    supernode: Supernode | None,
) -> str:
    kind = supernode.type if supernode is not None else _sn_kind(sn, {})
    clerp = _member_clerp(members, attr, supernode)
    if kind == "emb":
        token = _token_from_clerp(clerp) if clerp else sn
        return f"Emb: {token}"
    if kind == "logit":
        token = _token_from_clerp(clerp) if clerp else sn
        return f"Logit: {token}"

    label = supernode.name if supernode is not None else sn
    role = supernode.role if supernode is not None else ""
    return f"{role}: {label}" if role else str(label)


def _wrap_label(text: str, width: int = 18, max_lines: int = 5) -> str:
    """Hard-wrap a label with <br> (Plotly annotations don't auto-wrap)."""
    words = str(text).split()
    lines: list[str] = []
    cur = ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[: max_lines - 1] + [lines[max_lines - 1] + "…"]
    return "<br>".join(lines)


def _clean_token(tok: str) -> str:
    """Make a raw tokenizer token printable (strip subword markers, show spaces)."""
    cleaned = tok.replace("Ġ", " ").replace("▁", " ").replace("\n", "\\n")
    cleaned = cleaned.strip()
    return cleaned if cleaned else "·"


def _edge_style(weight: float, max_abs_w: float) -> tuple[float, str]:
    scale = abs(weight) / max(max_abs_w, 1e-9)
    width = 0.5 + 9.0 * scale
    alpha = 0.15 + 0.80 * scale
    if weight >= 0:
        color = f"rgba(33,120,78,{alpha:.3f})"
    else:
        color = f"rgba(203,24,29,{alpha:.3f})"
    return width, color


def _rounded_rect_path(x0: float, y0: float, x1: float, y1: float, rx: float, ry: float) -> str:
    """SVG path string for a rounded rectangle (y0 < y1). Quadratic corners."""
    return (
        f"M {x0 + rx},{y0} L {x1 - rx},{y0} Q {x1},{y0} {x1},{y0 + ry} "
        f"L {x1},{y1 - ry} Q {x1},{y1} {x1 - rx},{y1} "
        f"L {x0 + rx},{y1} Q {x0},{y1} {x0},{y1 - ry} "
        f"L {x0},{y0 + ry} Q {x0},{y0} {x0 + rx},{y0} Z"
    )


def _add_card(
    fig: go.Figure,
    cx: float,
    cy: float,
    w: float,
    h: float,
    fill: str,
    line_color: str,
    stacked: bool,
) -> None:
    rx, ry = 0.10, 0.10
    x0, x1 = cx - w / 2, cx + w / 2
    y0, y1 = cy - h / 2, cy + h / 2
    if stacked:
        # Two offset rects behind (down-right) for the stacked-paper effect.
        for off in (0.10, 0.05):
            fig.add_shape(
                type="path",
                path=_rounded_rect_path(x0 + off, y0 - off, x1 + off, y1 - off, rx, ry),
                fillcolor="#DCD6C4",
                line=dict(color=line_color, width=1),
            )
    fig.add_shape(
        type="path",
        path=_rounded_rect_path(x0, y0, x1, y1, rx, ry),
        fillcolor=fill,
        line=dict(color=line_color, width=1.6),
    )


def _rendered_edges(
    sn_names: list[str],
    sn_adj: np.ndarray,
    visible_names: list[str],
    edge_threshold: float,
) -> tuple[list[tuple[str, str, float]], float]:
    """Return visible source -> target edges after logit and weight filtering."""
    max_abs_w = float(np.max(np.abs(sn_adj))) if sn_adj.size else 1.0
    visible = set(visible_names)
    rendered: list[tuple[str, str, float]] = []
    k = len(sn_names)
    for target_idx in range(k):
        for source_idx in range(k):
            if target_idx == source_idx:
                continue
            weight = float(sn_adj[target_idx, source_idx])
            if weight == 0.0 or abs(weight) < edge_threshold * max_abs_w:
                continue
            source, target = sn_names[source_idx], sn_names[target_idx]
            if source in visible and target in visible:
                rendered.append((source, target, weight))
    return rendered, max_abs_w


def _fallback_rank_rows(
    sn_names: list[str],
    mapping: dict[str, list[str]],
    attr: dict[str, dict[str, Any]] | None,
    node_by_name: dict[str, Supernode],
) -> dict[int, list[str]]:
    ranked: list[tuple[tuple[int, int, float, str], str]] = []
    for sn in sn_names:
        layer, ctx_mean = _layer_and_ctx_for_supernode(sn, mapping.get(sn, []), attr, node_by_name)
        kind = _sn_kind(sn, node_by_name)
        kind_rank = {"emb": 0, "middle": 1, "logit": 2}.get(kind, 1)
        ranked.append(((kind_rank, layer, ctx_mean, sn), sn))

    rows: dict[int, list[str]] = {}
    last_key: tuple[int, int] | None = None
    rank = -1
    for (kind_rank, layer, _ctx_mean, _sn), sn in sorted(ranked):
        key = (kind_rank, layer)
        if key != last_key:
            rank += 1
            last_key = key
        rows.setdefault(rank, []).append(sn)
    return rows


def _strongly_connected_components(
    sn_names: list[str],
    outgoing: dict[str, list[str]],
) -> list[list[str]]:
    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    components: list[list[str]] = []

    def visit(sn: str) -> None:
        nonlocal index
        indices[sn] = index
        lowlink[sn] = index
        index += 1
        stack.append(sn)
        on_stack.add(sn)

        for target in outgoing[sn]:
            if target not in indices:
                visit(target)
                lowlink[sn] = min(lowlink[sn], lowlink[target])
            elif target in on_stack:
                lowlink[sn] = min(lowlink[sn], indices[target])

        if lowlink[sn] != indices[sn]:
            return

        component: list[str] = []
        while stack:
            member = stack.pop()
            on_stack.remove(member)
            component.append(member)
            if member == sn:
                break
        components.append(component)

    for sn in sn_names:
        if sn not in indices:
            visit(sn)
    return components


def _topological_rank_rows(
    sn_names: list[str],
    edges: list[tuple[str, str, float]],
    mapping: dict[str, list[str]],
    attr: dict[str, dict[str, Any]] | None,
    node_by_name: dict[str, Supernode],
) -> dict[int, list[str]]:
    """Group visible supernodes by source-to-sink longest-path depth."""
    if not edges:
        return _fallback_rank_rows(sn_names, mapping, attr, node_by_name)

    name_set = set(sn_names)
    outgoing = {sn: [] for sn in sn_names}
    for source, target, _weight in edges:
        if source not in name_set or target not in name_set:
            continue
        outgoing[source].append(target)

    components = _strongly_connected_components(sn_names, outgoing)
    component_idx: dict[str, int] = {}
    for idx, component in enumerate(components):
        for sn in component:
            component_idx[sn] = idx

    component_outgoing = {idx: set() for idx in range(len(components))}
    component_indegree = {idx: 0 for idx in range(len(components))}
    for source, targets in outgoing.items():
        source_component = component_idx[source]
        for target in targets:
            target_component = component_idx[target]
            if source_component == target_component:
                continue
            if target_component not in component_outgoing[source_component]:
                component_outgoing[source_component].add(target_component)
                component_indegree[target_component] += 1

    def component_key(idx: int) -> tuple[float, int, str]:
        layers_and_ctx = [
            _layer_and_ctx_for_supernode(sn, mapping.get(sn, []), attr, node_by_name)
            for sn in components[idx]
        ]
        ctx_mean = float(np.mean([ctx for _layer, ctx in layers_and_ctx]))
        min_layer = min(layer for layer, _ctx in layers_and_ctx)
        return (ctx_mean, min_layer, min(components[idx]))

    component_depth = {idx: 0 for idx in range(len(components))}
    ready = sorted(
        [idx for idx, degree in component_indegree.items() if degree == 0],
        key=component_key,
    )
    while ready:
        source = ready.pop(0)
        for target in sorted(component_outgoing[source], key=component_key):
            component_depth[target] = max(component_depth[target], component_depth[source] + 1)
            component_indegree[target] -= 1
            if component_indegree[target] == 0:
                ready.append(target)
                ready.sort(key=lambda idx: (component_depth[idx], *component_key(idx)))

    depth = {sn: component_depth[component_idx[sn]] for sn in sn_names}
    non_logit_depths = [
        rank for sn, rank in depth.items() if _sn_kind(sn, node_by_name) != "logit"
    ]
    if non_logit_depths:
        min_logit_depth = max(non_logit_depths) + 1
        for sn in sn_names:
            if _sn_kind(sn, node_by_name) == "logit" and not outgoing[sn]:
                depth[sn] = max(depth[sn], min_logit_depth)

    rows: dict[int, list[str]] = {}
    for sn, rank in depth.items():
        rows.setdefault(rank, []).append(sn)
    return rows


def _supernode_layout(
    sn_names: list[str],
    mapping: dict[str, list[str]],
    attr: dict[str, dict[str, Any]] | None,
    node_by_name: dict[str, Supernode],
    right_x: float | None = None,
    visible_edges: list[tuple[str, str, float]] | None = None,
) -> tuple[dict[str, tuple[float, float]], float, dict[int, float]]:
    """x = token position (ctx mean); y = topological rank below the prompt strip."""
    rows = _topological_rank_rows(sn_names, visible_edges or [], mapping, attr, node_by_name)
    non_logit_ctx: list[float] = []
    for sn in sn_names:
        if _sn_kind(sn, node_by_name) == "logit":
            continue
        _layer, ctx_mean = _layer_and_ctx_for_supernode(sn, mapping.get(sn, []), attr, node_by_name)
        non_logit_ctx.append(ctx_mean)

    logit_x = right_x
    if logit_x is None:
        logit_x = (max(non_logit_ctx) + 1.5) if non_logit_ctx else 0.0

    ordered_ranks = sorted(rows)
    rank_y = {rank: (rank + 1) * _RANK_Y_GAP for rank in ordered_ranks}

    pos: dict[str, tuple[float, float]] = {}
    for rank in ordered_ranks:
        items: list[tuple[str, float, int]] = []
        for sn in rows[rank]:
            layer, ctx_mean = _layer_and_ctx_for_supernode(sn, mapping.get(sn, []), attr, node_by_name)
            x0 = float(logit_x) if _sn_kind(sn, node_by_name) == "logit" else ctx_mean
            items.append((sn, x0, layer))
        items = sorted(items, key=lambda p: (p[1], p[2], p[0]))
        last_x: float | None = None
        for sn, ctx, _layer in items:
            x = float(ctx)
            if last_x is not None and x - last_x < _MIN_CARD_X_GAP:
                x = last_x + _MIN_CARD_X_GAP
            last_x = x
            pos[sn] = (x, float(rank_y[rank]))
    max_rank = max(ordered_ranks, default=-1)
    top_y = (max_rank + 2) * _RANK_Y_GAP
    return pos, top_y, rank_y


def _rect_boundary_point(
    cx: float,
    cy: float,
    w: float,
    h: float,
    toward_x: float,
    toward_y: float,
) -> tuple[float, float]:
    """Point where the center-to-target ray leaves a card rectangle."""
    dx = toward_x - cx
    dy = toward_y - cy
    if dx == 0.0 and dy == 0.0:
        return cx, cy + h / 2
    sx = (w / 2) / abs(dx) if dx != 0.0 else math.inf
    sy = (h / 2) / abs(dy) if dy != 0.0 else math.inf
    scale = min(sx, sy)
    return cx + dx * scale, cy + dy * scale


def _prompt_token_strip(
    fig: go.Figure,
    prompt_tokens: list[str],
    emb_ctx: set[int],
    y: float,
) -> None:
    cellw, cellh = 0.9, _BAR_H
    for i, tok in enumerate(prompt_tokens):
        highlight = i in emb_ctx
        fill = "#D8D2BF" if highlight else "#F4F2EB"
        x0, x1 = i - cellw / 2, i + cellw / 2
        fig.add_shape(
            type="path",
            path=_rounded_rect_path(x0, y - cellh / 2, x1, y + cellh / 2, 0.06, 0.08),
            fillcolor=fill,
            line=dict(color="#B9B29A", width=1),
        )
        fig.add_annotation(
            x=i,
            y=y,
            text=_clean_token(tok),
            showarrow=False,
            font=dict(size=9, color="#333"),
        )


def supernode_graph_figure(
    sng: SummaryGraph | dict[str, Any],
    final_supernodes: dict[str, list[str]] | None = None,
    attr: dict[str, dict[str, Any]] | None = None,
    title: str = "Cluster graph (supernodes)",
    seed: int = 42,
    prompt_tokens: list[str] | None = None,
    prompt: str | None = None,
    use_supernode_names: bool = False,
    edge_threshold: float = 0.0,
    top_k_logits: int | None = None,
    steering_factors: dict[str, float] | None = None,
    activation_ratios: dict[str, float | None] | None = None,
    top_outputs: list[Any] | None = None,
    stored_interventions: list[dict[str, Any]] | None = None,
) -> go.Figure:
    """
    Build an interactive Plotly figure in the Anthropic attribution-graph style:
    supernodes as rounded cards, directed edges as curved green/red arrows, and
    (when ``prompt_tokens`` is given) a tokenized prompt strip above the graph.

    `sng` may be a `SummaryGraph` instance or the legacy dict.

    ``edge_threshold`` (0-1) hides edges whose magnitude is below that fraction of
    the largest edge weight. ``top_k_logits`` keeps only the k highest-probability
    logit supernodes (and their edges); ``None`` shows all.
    """
    # Kept for callers that still pass the old display/steering overlay options.
    _ = (use_supernode_names, steering_factors, activation_ratios, top_outputs, stored_interventions)

    # Duck-typing rather than isinstance so this survives Streamlit hot-reload,
    # which re-imports SummaryGraph and breaks isinstance on session-state objects.
    if hasattr(sng, "sn_names") and hasattr(sng, "adj_matrix"):
        graph = cast(Any, sng)
        sn_names = graph.sn_names
        sn_adj = np.asarray(graph.adj_matrix, dtype=np.float64)
        mapping = final_supernodes if final_supernodes is not None else graph.to_mapping()
        node_by_name = graph.node_by_name()
    else:
        if final_supernodes is None:
            raise ValueError("final_supernodes is required when sng is a plain dict.")
        legacy = cast(dict[str, Any], sng)
        sn_names = list(legacy["sn_names"])
        sn_adj = np.asarray(legacy["sn_adj"], dtype=np.float64)
        mapping = final_supernodes
        node_by_name = {}

    # Optionally keep only the top-k logit supernodes by token probability. Hidden
    # logits are dropped from the layout, so their cards and edges never render.
    hidden: set[str] = set()
    if top_k_logits is not None:
        logit_sns = [sn for sn in sn_names if _sn_kind(sn, node_by_name) == "logit"]
        ranked = sorted(
            logit_sns,
            key=lambda sn: _logit_prob(sn, mapping.get(sn, []), node_by_name, attr),
            reverse=True,
        )
        hidden = set(ranked[max(top_k_logits, 0) :])

    layout_names = [sn for sn in sn_names if sn not in hidden]
    output_x = float(len(prompt_tokens)) if prompt_tokens else None
    rendered_edges, max_abs_w = _rendered_edges(sn_names, sn_adj, layout_names, edge_threshold)
    pos, top_y, _layer_y = _supernode_layout(
        layout_names,
        mapping,
        attr,
        node_by_name,
        right_x=output_x,
        visible_edges=rendered_edges,
    )

    # Per-card geometry, colors, kinds.
    geom: dict[str, tuple[float, float, float, float]] = {}  # sn -> (cx, cy, w, h)
    kinds: dict[str, str] = {}
    emb_ctx: set[int] = set()
    for sn in sn_names:
        if sn not in pos:
            continue
        cx, cy = pos[sn]
        members = mapping.get(sn, [])
        m = max(len(members), 1)
        grow = min(0.18, 0.03 * (m - 1))
        geom[sn] = (cx, cy, _CARD_W + grow, _CARD_H + grow)
        kind = _sn_kind(sn, node_by_name)
        kinds[sn] = kind
        if kind == "emb":
            emb_ctx.add(int(round(cx)))

    fig = go.Figure()

    # --- Curved edges (drawn first so cards sit on top) ---
    edge_xs: list[float] = []
    edge_ys: list[float] = []
    for u, v, w in rendered_edges:
        if u not in geom or v not in geom:
            continue
        uc, vc = geom[u], geom[v]
        xs, ys = _rect_boundary_point(uc[0], uc[1], uc[2], uc[3], vc[0], vc[1])
        xt, yt = _rect_boundary_point(vc[0], vc[1], vc[2], vc[3], uc[0], uc[1])
        dx, dy = xt - xs, yt - ys
        length = math.hypot(dx, dy) or 1.0
        px, py = -dy / length, dx / length  # perpendicular
        bow = 0.18 * length
        cxm, cym = (xs + xt) / 2 + px * bow, (ys + yt) / 2 + py * bow
        width, color = _edge_style(w, max_abs_w)
        edge_xs.extend((xs, cxm, xt))
        edge_ys.extend((ys, cym, yt))
        fig.add_shape(
            type="path",
            path=f"M {xs},{ys} Q {cxm},{cym} {xt},{yt}",
            line=dict(width=width, color=color),
        )
        fig.add_trace(
            go.Scatter(
                x=[xs, cxm, xt],
                y=[ys, cym, yt],
                mode="markers",
                marker=dict(size=16, color="rgba(0,0,0,0)"),
                hovertemplate=f"{u} -> {v}<br>weight={w:.4f}<extra></extra>",
                showlegend=False,
            )
        )
        # Arrowhead at the target end, tangent to the incoming curve.
        tangent_len = math.hypot(xt - cxm, yt - cym) or 1.0
        ax = xt - (xt - cxm) / tangent_len * min(0.28, 0.18 * length)
        ay = yt - (yt - cym) / tangent_len * min(0.28, 0.18 * length)
        fig.add_annotation(
            x=xt,
            y=yt,
            ax=ax,
            ay=ay,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.0,
            arrowwidth=max(1.0, width * 0.6),
            arrowcolor=color,
            text="",
        )

    # --- Cards + labels ---
    hover_x: list[float] = []
    hover_y: list[float] = []
    hover_text: list[str] = []
    for sn in sn_names:
        if sn not in geom:
            continue
        cx, cy, w, h = geom[sn]
        members = mapping.get(sn, [])
        fill, line_color = _KIND_STYLE.get(kinds[sn], _KIND_STYLE["middle"])
        _add_card(fig, cx, cy, w, h, fill, line_color, stacked=len(members) > 1)
        label_text = _node_label(sn, members, attr, node_by_name.get(sn))
        fig.add_annotation(
            x=cx,
            y=cy,
            text=_wrap_label(label_text),
            showarrow=False,
            font=dict(size=9, color="#1a1a1a"),
            align="center",
        )
        hover_x.append(cx)
        hover_y.append(cy)
        hover_title = _sn_title(sn, members, attr, node_by_name.get(sn))
        hover_text.append(hover_title)

    # Invisible markers carry the rich hover (composite members).
    fig.add_trace(
        go.Scatter(
            x=hover_x,
            y=hover_y,
            mode="markers",
            marker=dict(size=18, color="rgba(0,0,0,0)"),
            hovertext=hover_text,
            hoverinfo="text",
            showlegend=False,
        )
    )

    # --- Tokenized prompt strip ---
    if prompt_tokens:
        _prompt_token_strip(fig, prompt_tokens, emb_ctx, y=float(top_y))

    # --- Layout / ranges ---
    token_xs = [float(i) for i in range(len(prompt_tokens or []))]
    xs_for_range = [g[0] for g in geom.values()] + edge_xs + token_xs + [0.0]
    ys_for_range = [g[1] for g in geom.values()] + edge_ys
    if prompt_tokens:
        ys_for_range.append(float(top_y))
    if not ys_for_range:
        ys_for_range.append(0.0)
    x_min = min(xs_for_range) - 1.5
    x_max = max(xs_for_range) + 1.5
    y_min = min(ys_for_range) - 0.7
    y_max = max(ys_for_range) + 0.7
    fig.update_layout(
        title=title,
        showlegend=False,
        xaxis=dict(visible=False, range=[x_min, x_max]),
        yaxis=dict(visible=False, range=[y_min, y_max]),
        margin=dict(l=30, r=30, t=50, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=760,
    )
    return fig
