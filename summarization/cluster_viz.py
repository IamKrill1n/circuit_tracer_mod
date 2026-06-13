"""Plotly visualization helpers for clustered supernode graphs.

Renders the supernode graph in the style of the Anthropic attribution-graph
figure: supernodes as rounded "card" boxes (with a stacked-paper shadow for
composites), curved green/red connectors, and full-width Input-Tokens (bottom)
and Outputs/Logits (top) bars showing the prompt.
"""

from __future__ import annotations

import html
import math
from typing import Any

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
_CARD_W = 0.92
_CARD_H = 0.58
_BAR_H = 0.52


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


def _sn_label(sn: str, members: list[str], attr: dict[str, dict[str, Any]] | None) -> str:
    if attr is None:
        return sn
    for nid in members:
        clerp = str(attr.get(nid, {}).get("clerp", "") or "").strip()
        if clerp:
            return clerp[:70] + ("..." if len(clerp) > 70 else "")
    return sn


def _wrap_label(text: str, width: int = 14, max_lines: int = 4) -> str:
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


def _supernode_layout(
    sn_names: list[str],
    mapping: dict[str, list[str]],
    attr: dict[str, dict[str, Any]] | None,
    node_by_name: dict[str, Supernode],
    right_x: float | None = None,
) -> tuple[dict[str, tuple[float, float]], int]:
    """x = token position (ctx mean); y = layer rank (row 0 reserved for token bar)."""
    rows: dict[int, list[tuple[str, float]]] = {}
    non_logit_ctx: list[float] = []
    for sn in sn_names:
        layer, ctx_mean = _layer_and_ctx_for_supernode(sn, mapping.get(sn, []), attr, node_by_name)
        if _sn_kind(sn, node_by_name) == "logit":
            continue
        non_logit_ctx.append(ctx_mean)
        rows.setdefault(layer, []).append((sn, ctx_mean))

    logit_x = right_x
    if logit_x is None:
        logit_x = (max(non_logit_ctx) + 1.5) if non_logit_ctx else 0.0
    for sn in sn_names:
        if _sn_kind(sn, node_by_name) != "logit":
            continue
        layer, _ctx_mean = _layer_and_ctx_for_supernode(sn, mapping.get(sn, []), attr, node_by_name)
        rows.setdefault(layer, []).append((sn, float(logit_x)))

    ordered_layers = sorted(rows)
    layer_y = {layer: idx + 1 for idx, layer in enumerate(ordered_layers)}

    pos: dict[str, tuple[float, float]] = {}
    min_gap = 1.0
    for layer in ordered_layers:
        items = sorted(rows[layer], key=lambda p: (p[1], p[0]))
        last_x: float | None = None
        for sn, ctx in items:
            x = float(ctx)
            if last_x is not None and x - last_x < min_gap:
                x = last_x + min_gap
            last_x = x
            pos[sn] = (x, float(layer_y[layer]))
    top_y = len(ordered_layers) + 1
    return pos, top_y


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


def _token_bar(
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
    fig.add_annotation(
        xref="paper",
        x=0.0,
        xanchor="left",
        yref="y",
        y=y,
        text="<b>Input Tokens →</b>",
        showarrow=False,
        font=dict(size=11, color="#C2570C"),
    )


def _output_bar(
    fig: go.Figure,
    prompt_tokens: list[str],
    prompt: str | None,
    out_label: str,
    y: float,
) -> None:
    n = len(prompt_tokens)
    cellh = _BAR_H
    text = (prompt or "".join(_clean_token(t) for t in prompt_tokens)).strip()
    if len(text) > 70:
        text = text[:67] + "…"
    # Wide prompt bar.
    fig.add_shape(
        type="path",
        path=_rounded_rect_path(-0.5, y - cellh / 2, n - 1 + 0.5, y + cellh / 2, 0.06, 0.08),
        fillcolor="#F4F2EB",
        line=dict(color="#B9B29A", width=1),
    )
    fig.add_annotation(
        x=-0.4,
        xanchor="left",
        y=y,
        text=text,
        showarrow=False,
        font=dict(size=10, color="#333"),
    )
    # Highlighted predicted-token cell at the right.
    if out_label:
        fig.add_shape(
            type="path",
            path=_rounded_rect_path(n - 0.5, y - cellh / 2, n + 0.5, y + cellh / 2, 0.06, 0.08),
            fillcolor="#D8D2BF",
            line=dict(color="#B9B29A", width=1),
        )
        fig.add_annotation(
            x=n,
            y=y,
            text=_clean_token(out_label),
            showarrow=False,
            font=dict(size=10, color="#333"),
        )
    fig.add_annotation(
        xref="paper",
        x=0.0,
        xanchor="left",
        yref="y",
        y=y,
        text="<b>Outputs / Logits →</b>",
        showarrow=False,
        font=dict(size=11, color="#C2570C"),
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
) -> go.Figure:
    """
    Build an interactive Plotly figure in the Anthropic attribution-graph style:
    supernodes as rounded cards, directed edges as curved green/red arrows, and
    (when ``prompt_tokens`` is given) Input-Tokens / Outputs-Logits prompt bars.

    `sng` may be a `SummaryGraph` instance or the legacy dict.

    ``edge_threshold`` (0-1) hides edges whose magnitude is below that fraction of
    the largest edge weight. ``top_k_logits`` keeps only the k highest-probability
    logit supernodes (and their edges); ``None`` shows all.
    """
    # Duck-typing rather than isinstance so this survives Streamlit hot-reload,
    # which re-imports SummaryGraph and breaks isinstance on session-state objects.
    if hasattr(sng, "sn_names") and hasattr(sng, "adj_matrix"):
        sn_names = sng.sn_names
        sn_adj = np.asarray(sng.adj_matrix, dtype=np.float64)
        mapping = final_supernodes if final_supernodes is not None else sng.to_mapping()
        node_by_name = sng.node_by_name()
    else:
        if final_supernodes is None:
            raise ValueError("final_supernodes is required when sng is a plain dict.")
        sn_names = list(sng["sn_names"])
        sn_adj = np.asarray(sng["sn_adj"], dtype=np.float64)
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
    pos, top_y = _supernode_layout(layout_names, mapping, attr, node_by_name, right_x=output_x)
    k = len(sn_names)

    # Per-card geometry, colors, kinds.
    geom: dict[str, tuple[float, float, float, float]] = {}  # sn -> (cx, cy, w, h)
    kinds: dict[str, str] = {}
    emb_ctx: set[int] = set()
    logit_labels: list[str] = []
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
        elif kind == "logit":
            logit_labels.append(_sn_label(sn, members, attr))

    fig = go.Figure()

    # --- Curved edges (drawn first so cards sit on top) ---
    max_abs_w = float(np.max(np.abs(sn_adj))) if sn_adj.size else 1.0
    edge_xs: list[float] = []
    edge_ys: list[float] = []
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            w = float(sn_adj[i, j])  # source j -> target i
            if w == 0.0 or abs(w) < edge_threshold * max_abs_w:
                continue
            u, v = sn_names[j], sn_names[i]
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
        # Labeled middle cards show the LLM-generated supernode name; emb/logit keep
        # their clerp ("Emb: France", 'Output "Paris"') since those aren't relabeled.
        if use_supernode_names and kinds[sn] == "middle":
            label_text = sn
        else:
            label_text = _sn_label(sn, members, attr)
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
        hover_text.append(_sn_title(sn, members, attr, node_by_name.get(sn)))

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

    # --- Prompt bars + input/output arrows ---
    if prompt_tokens:
        _token_bar(fig, prompt_tokens, emb_ctx, y=0.0)
        out_label = logit_labels[0] if logit_labels else ""
        _output_bar(fig, prompt_tokens, prompt, out_label, y=float(top_y))
        # Token cell -> embedding card.
        for sn, kind in kinds.items():
            if kind != "emb":
                continue
            cx, cy, _w, h = geom[sn]
            fig.add_annotation(
                x=cx,
                y=cy - h / 2,
                ax=cx,
                ay=_BAR_H / 2,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=0.9,
                arrowwidth=1.0,
                arrowcolor="#9a9a9a",
                text="",
            )
        # Logit card -> output bar.
        for sn, kind in kinds.items():
            if kind != "logit":
                continue
            cx, cy, _w, h = geom[sn]
            fig.add_annotation(
                x=cx,
                y=float(top_y) - _BAR_H / 2,
                ax=cx,
                ay=cy + h / 2,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=0.9,
                arrowwidth=1.0,
                arrowcolor="#9a9a9a",
                text="",
            )

    # --- Layout / ranges ---
    n_tokens = len(prompt_tokens) if prompt_tokens else 0
    xs_for_range = [g[0] for g in geom.values()] + edge_xs + [0.0, float(n_tokens)]
    ys_for_range = [g[1] for g in geom.values()] + edge_ys + [0.0, float(top_y)]
    x_min = min(xs_for_range) - 1.5
    x_max = max(xs_for_range) + 1.5
    y_min = min(ys_for_range) - 0.7
    y_max = max(ys_for_range) + 0.7
    fig.update_layout(
        title=title,
        showlegend=False,
        xaxis=dict(visible=False, range=[x_min, x_max]),
        yaxis=dict(visible=False, range=[y_min, y_max]),
        margin=dict(l=120, r=30, t=50, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=760,
    )
    return fig
