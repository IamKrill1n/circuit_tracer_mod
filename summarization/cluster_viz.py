"""Visualization helpers for clustered supernode graphs.

Renders the supernode graph in the style of the Anthropic attribution-graph
figure: supernodes as rounded "card" boxes (with a stacked-paper shadow for
composites), green/red connectors routed around cards, and a tokenized prompt
strip above the graph.
"""

from __future__ import annotations

from dataclasses import dataclass
import html
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np

from summarization.utils import layer_index_from_node, layer_index_from_node_id
from summarization.summarize import SummaryGraph, Supernode

# Beige card fill (paper palette) with a thin kind-colored border accent.
_KIND_STYLE = {
    "emb": ("#EDE9DD", "#4CAF50"),
    "logit": ("#EDE9DD", "#FF9800"),
    "middle": ("#EDE9DD", "#5B6B7B"),
}
_CARD_W = 2.75
_CARD_H = 1.05
_BAR_H = 0.52
_MIN_CARD_X_GAP = 3.45
_RANK_Y_GAP = 2.05
_DEFAULT_EDGE_TOP_K = 3
_EDGE_TOP_K_VALUES = (1, 2, 3, 5, 10)
_LOW_ACTIVATION_RATIO = 0.25


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


def _label_card_size(label: str, member_count: int) -> tuple[float, float]:
    wrapped = _wrap_label(label)
    lines = wrapped.split("<br>") if wrapped else [""]
    max_line_chars = max(len(line) for line in lines)
    grow = min(0.18, 0.03 * (max(member_count, 1) - 1))
    width = max(_CARD_W + grow, min(3.30, 0.24 + 0.13 * max_line_chars))
    height = max(_CARD_H + grow, 0.48 + 0.31 * len(lines))
    return width, height


def _clean_token(tok: str) -> str:
    """Make a raw tokenizer token printable (strip subword markers, show spaces)."""
    cleaned = tok.replace("Ġ", " ").replace("▁", " ").replace("\n", "\\n")
    cleaned = cleaned.strip()
    return cleaned if cleaned else "·"


def _edge_style(weight: float, max_abs_w: float) -> tuple[float, str, float, str]:
    linear_scale = abs(weight) / max(max_abs_w, 1e-9)
    scale = math.log1p(9.0 * linear_scale) / math.log1p(9.0)
    width = 0.7 + 4.8 * scale
    alpha = 0.20 + 0.75 * scale
    if weight >= 0:
        color = "#21784e"
        dash = ""
    else:
        color = "#cb181d"
        dash = "7 5"
    return width, color, alpha, dash


def _format_factor(factor: float) -> str:
    value = float(factor)
    if value.is_integer():
        return f"{int(value)}x"
    return f"{value:g}x"


def _format_percent(value: float) -> str:
    return f"{round(float(value) * 100)}%"


def _format_probability_delta(value: float) -> str:
    return f"{float(value) * 100:+.1f}%"


def _format_logit_delta(value: float) -> str:
    return f"\u0394 {float(value):+.2f}"


def _format_output(item: Any) -> dict[str, Any]:
    token = str(getattr(item, "token", "") if not isinstance(item, dict) else item.get("token", ""))
    probability = (
        getattr(item, "probability", 0.0)
        if not isinstance(item, dict)
        else item.get("probability", 0.0)
    )
    out = {
        "token": token or "(empty)",
        "probability": float(probability or 0.0),
    }
    for key in ("clean_probability", "probability_delta", "logit_delta"):
        raw = getattr(item, key, None) if not isinstance(item, dict) else item.get(key)
        if raw is not None:
            out[key] = float(raw)
    return out


def _stored_intervention_host(
    stored: dict[str, Any],
    candidates: list[str],
    mapping: dict[str, list[str]],
    attr: dict[str, dict[str, Any]] | None,
    node_by_name: dict[str, Supernode],
) -> str | None:
    if not candidates:
        return None

    target_layer = float(stored.get("layer", 0))
    target_pos = float(stored.get("target_pos", 0))
    ranked: list[tuple[tuple[float, float, str], str]] = []
    for sn in candidates:
        layer, ctx_mean = _layer_and_ctx_for_supernode(sn, mapping.get(sn, []), attr, node_by_name)
        key = (abs(float(layer) - target_layer), abs(ctx_mean - target_pos), sn)
        ranked.append((key, sn))
    return min(ranked, key=lambda item: item[0])[1]


def _rounded_rect_path(x0: float, y0: float, x1: float, y1: float, rx: float, ry: float) -> str:
    """SVG path string for a rounded rectangle (y0 < y1). Quadratic corners."""
    return (
        f"M {x0 + rx},{y0} L {x1 - rx},{y0} Q {x1},{y0} {x1},{y0 + ry} "
        f"L {x1},{y1 - ry} Q {x1},{y1} {x1 - rx},{y1} "
        f"L {x0 + rx},{y1} Q {x0},{y1} {x0},{y1 - ry} "
        f"L {x0},{y0 + ry} Q {x0},{y0} {x0 + rx},{y0} Z"
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


def _select_edge_indices(
    edges: list[tuple[str, str, float]],
    *,
    top_k: int | None = None,
    positive_only: bool = False,
) -> set[int]:
    """Select visible edges, keeping each source node's top-|weight| outgoing edges."""
    by_source: dict[str, list[tuple[int, str, str, float]]] = {}
    for idx, (source, target, weight) in enumerate(edges):
        if positive_only and weight <= 0.0:
            continue
        by_source.setdefault(source, []).append((idx, source, target, weight))

    selected: set[int] = set()
    for source_edges in by_source.values():
        ranked = sorted(
            source_edges,
            key=lambda item: (-abs(item[3]), item[1], item[2], item[0]),
        )
        keep = ranked if top_k is None else ranked[: max(int(top_k), 0)]
        selected.update(idx for idx, _source, _target, _weight in keep)
    return selected


def _edge_filter_k_values(edges: list[tuple[str, str, float]]) -> list[int | None]:
    max_outdegree = 0
    counts: dict[str, int] = {}
    for source, _target, _weight in edges:
        counts[source] = counts.get(source, 0) + 1
        max_outdegree = max(max_outdegree, counts[source])
    values: list[int | None] = [None]
    values.extend(k for k in _EDGE_TOP_K_VALUES if k < max_outdegree)
    if max_outdegree > 0:
        values.append(max_outdegree)
    return values


@dataclass
class ClusterGraphFigure:
    """Small HTML figure wrapper with Plotly-like serialization methods."""

    payload: dict[str, Any]

    @property
    def nodes(self) -> list[dict[str, Any]]:
        return cast(list[dict[str, Any]], self.payload["nodes"])

    @property
    def edges(self) -> list[dict[str, Any]]:
        return cast(list[dict[str, Any]], self.payload["edges"])

    @property
    def data(self) -> tuple[SimpleNamespace, ...]:
        selected = set(self.payload["initialEdgeIndices"])
        edge_traces = [
            SimpleNamespace(
                hovertemplate=f"{edge['source']} -> {edge['target']}<br>weight={edge['weight']:.4f}<extra></extra>",
                visible=edge["index"] in selected,
            )
            for edge in self.edges
        ]
        hover_trace = SimpleNamespace(
            hovertext=[node["hover"].replace("\n", "<br>") for node in self.nodes]
        )
        return (*edge_traces, hover_trace)

    @property
    def layout(self) -> SimpleNamespace:
        selected = set(self.payload["initialEdgeIndices"])
        edge_shapes = [
            SimpleNamespace(
                path=edge["initialPath"],
                line=SimpleNamespace(
                    color=edge["color"],
                    width=edge["width"],
                    dash=edge["dash"],
                ),
            )
            for edge in self.edges
            if edge["index"] in selected
        ]
        annotations = [
            SimpleNamespace(text=line, x=node["x"], y=node["y"])
            for node in self.nodes
            for line in ["<br>".join(node["labelLines"])]
        ]
        annotations.extend(
            SimpleNamespace(text=token["text"], x=token["x"], y=token["y"])
            for token in self.payload["promptTokens"]
        )
        top_k_values = self.payload["topKValues"]
        active = top_k_values.index(self.payload["defaultTopK"])
        slider = SimpleNamespace(
            currentvalue=SimpleNamespace(prefix="Top k edges per node: "),
            active=active,
            y=-0.12,
        )
        buttons = [
            SimpleNamespace(label="All edges"),
            SimpleNamespace(label="Positive only"),
        ]
        updatemenu = SimpleNamespace(buttons=buttons, y=-0.16)
        return SimpleNamespace(
            shapes=tuple(edge_shapes),
            annotations=tuple(annotations),
            xaxis=SimpleNamespace(range=tuple(self.payload["xRange"])),
            yaxis=SimpleNamespace(range=tuple(self.payload["yRange"])),
            sliders=(slider,),
            updatemenus=(updatemenu,),
        )

    def to_html(
        self,
        include_plotlyjs: str | bool | None = None,
        full_html: bool = True,
        config: dict[str, Any] | None = None,
    ) -> str:
        _ = (include_plotlyjs, config)
        return _cluster_graph_html(self.payload, full_html=full_html)

    def write_html(
        self,
        file: str | Path,
        include_plotlyjs: str | bool | None = None,
        full_html: bool = True,
        config: dict[str, Any] | None = None,
    ) -> None:
        Path(file).write_text(
            self.to_html(include_plotlyjs=include_plotlyjs, full_html=full_html, config=config),
            encoding="utf-8",
        )


def _cluster_graph_html(payload: dict[str, Any], *, full_html: bool) -> str:
    payload_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    title = html.escape(str(payload.get("title", "Cluster graph")))
    fragment = """
<div class="ct-cluster-viz" data-cluster-viz>
  <div class="ct-cluster-toolbar">
    <strong class="ct-cluster-title"></strong>
    <label>
      Top k edges per node
      <select data-top-k></select>
    </label>
    <label>
      <input type="checkbox" data-positive-only />
      Positive only
    </label>
    <button type="button" data-reset-layout>Reset layout</button>
    <div class="ct-intervention-summary" data-intervention-summary hidden></div>
  </div>
  <div class="ct-cluster-canvas" data-canvas>
    <svg data-svg role="img" aria-label="Summary graph"></svg>
  </div>
  <div class="ct-edge-panel" data-edge-panel hidden></div>
</div>
<script>
(function () {
  const graphData = __DATA__;
  const root = document.currentScript.previousElementSibling;
  const title = root.querySelector(".ct-cluster-title");
  const topKSelect = root.querySelector("[data-top-k]");
  const positiveOnly = root.querySelector("[data-positive-only]");
  const resetButton = root.querySelector("[data-reset-layout]");
  const interventionSummary = root.querySelector("[data-intervention-summary]");
  const edgePanel = root.querySelector("[data-edge-panel]");
  const svg = root.querySelector("[data-svg]");
  const scale = graphData.scale;
  const margin = graphData.margin;
  const nodeById = new Map();
  const nodeElements = new Map();
  let renderedEdges = [];
  let hoverNodeId = null;
  let selectedNodeId = null;
  let dragState = null;
  let suppressNextClick = false;

  title.textContent = graphData.title;
  graphData.nodes.forEach((node) => {
    node.autoPx = margin.left + (node.x - graphData.xRange[0]) * scale;
    node.rankPy = margin.top + (graphData.yRange[1] - node.y) * scale;
    node.userOffsetPx = 0;
    node.px = node.autoPx;
    node.py = node.rankPy;
    node.widthPx = Math.max(148, node.width * scale);
    node.heightPx = Math.max(58, node.height * scale);
    nodeById.set(node.id, node);
  });

  svg.setAttribute("viewBox", `0 0 ${graphData.canvasWidth} ${graphData.canvasHeight}`);
  svg.style.width = "100%";
  svg.style.height = "100%";
  svg.setAttribute("preserveAspectRatio", "xMidYMid meet");

  const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
  const edgeLayer = document.createElementNS("http://www.w3.org/2000/svg", "g");
  const tokenLayer = document.createElementNS("http://www.w3.org/2000/svg", "g");
  const nodeLayer = document.createElementNS("http://www.w3.org/2000/svg", "g");
  edgeLayer.setAttribute("class", "edge-layer");
  tokenLayer.setAttribute("class", "token-layer");
  nodeLayer.setAttribute("class", "node-layer");
  svg.append(defs, edgeLayer, tokenLayer, nodeLayer);

  function svgEl(tag, attrs = {}) {
    const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, String(value)));
    return el;
  }

  function edgeRoute(edge) {
    const source = nodeById.get(edge.source);
    const target = nodeById.get(edge.target);
    const start = { x: source.px, y: source.py - source.heightPx / 2 };
    const end = { x: target.px, y: target.py + target.heightPx / 2 };
    const midY = (start.y + end.y) / 2;
    if (Math.abs(start.x - end.x) < 1e-9) {
      return [start, end];
    }
    return [start, { x: start.x, y: midY }, { x: end.x, y: midY }, end];
  }

  function pathFromRoute(route) {
    const [head, ...tail] = route;
    return [
      `M ${head.x.toFixed(1)} ${head.y.toFixed(1)}`,
      ...tail.map((point) => `L ${point.x.toFixed(1)} ${point.y.toFixed(1)}`),
    ].join(" ");
  }

  function svgPoint(event) {
    const point = svg.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    const ctm = svg.getScreenCTM();
    if (!ctm) {
      return { x: event.clientX, y: event.clientY };
    }
    const transformed = point.matrixTransform(ctm.inverse());
    return { x: transformed.x, y: transformed.y };
  }

  function visibleEdgeIndices() {
    const topK = topKSelect.value === "all" ? null : Number(topKSelect.value);
    const positive = positiveOnly.checked;
    const bySource = new Map();
    graphData.edges.forEach((edge, idx) => {
      if (edge.alwaysVisible) {
        return;
      }
      if (positive && edge.weight <= 0) {
        return;
      }
      if (!bySource.has(edge.source)) {
        bySource.set(edge.source, []);
      }
      bySource.get(edge.source).push({ idx, edge });
    });
    const visible = new Set();
    graphData.edges.forEach((edge, idx) => {
      if (edge.alwaysVisible) {
        visible.add(idx);
      }
    });
    bySource.forEach((items) => {
      items.sort((a, b) => Math.abs(b.edge.weight) - Math.abs(a.edge.weight) || a.edge.target.localeCompare(b.edge.target));
      const kept = topK === null ? items : items.slice(0, topK);
      kept.forEach((item) => visible.add(item.idx));
    });
    return visible;
  }

  function localNeighborhood(nodeId) {
    const nodes = new Set([nodeId]);
    const edges = new Set();
    graphData.edges.forEach((edge) => {
      if (edge.source === nodeId || edge.target === nodeId) {
        nodes.add(edge.source);
        nodes.add(edge.target);
        edges.add(edge.index);
      }
    });
    return { nodes, edges };
  }

  function activeNodeId() {
    return selectedNodeId || hoverNodeId;
  }

  function updateEdgePanel() {
    if (!selectedNodeId) {
      edgePanel.hidden = true;
      edgePanel.replaceChildren();
      return;
    }
    const visible = visibleEdgeIndices();
    const rows = graphData.edges
      .filter((edge) => visible.has(edge.index))
      .filter((edge) => edge.source === selectedNodeId || edge.target === selectedNodeId)
      .sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight) || a.source.localeCompare(b.source));
    const title = document.createElement("strong");
    title.textContent = selectedNodeId;
    const selectedNode = nodeById.get(selectedNodeId);
    const details = document.createElement("div");
    details.className = "ct-node-details";
    if (selectedNode && selectedNode.interventionDetails && selectedNode.interventionDetails.length) {
      selectedNode.interventionDetails.forEach((text) => {
        const item = document.createElement("span");
        item.textContent = text;
        details.appendChild(item);
      });
    }
    const list = document.createElement("ul");
    rows.forEach((edge) => {
      const item = document.createElement("li");
      item.textContent = `${edge.source} -> ${edge.target}: ${edge.weight.toFixed(4)}`;
      list.appendChild(item);
    });
    edgePanel.replaceChildren(title);
    if (details.childNodes.length) {
      edgePanel.appendChild(details);
    }
    edgePanel.appendChild(list);
    edgePanel.hidden = false;
  }

  function renderInterventionSummary() {
    const summary = graphData.interventionSummary;
    if (!summary || !summary.active) {
      interventionSummary.hidden = true;
      interventionSummary.replaceChildren();
      return;
    }
    const counts = document.createElement("span");
    counts.className = "ct-intervention-counts";
    counts.textContent = `${summary.steeredCount} steered · ${summary.storedCount} stored`;
    interventionSummary.appendChild(counts);
    if (summary.topOutputs.length) {
      const outputLabel = document.createElement("span");
      outputLabel.className = "ct-output-label";
      outputLabel.textContent = "Top outputs";
      interventionSummary.appendChild(outputLabel);
    }
    summary.topOutputs.forEach((item) => {
      const pill = document.createElement("span");
      pill.className = "ct-output-pill";
      pill.textContent = `${item.token} ${Number(item.probability).toFixed(3)}`;
      interventionSummary.appendChild(pill);
    });
    interventionSummary.hidden = false;
  }

  function applyHighlight() {
    const nodeId = activeNodeId();
    const neighborhood = nodeId ? localNeighborhood(nodeId) : null;
    nodeElements.forEach((group, id) => {
      group.classList.toggle("is-dimmed", Boolean(neighborhood) && !neighborhood.nodes.has(id));
      group.classList.toggle("is-active", id === nodeId);
      group.classList.toggle(
        "is-neighbor",
        Boolean(neighborhood) && id !== nodeId && neighborhood.nodes.has(id),
      );
    });
    renderedEdges.forEach(({ edge, path }) => {
      const highlighted = Boolean(neighborhood) && neighborhood.edges.has(edge.index);
      path.classList.toggle("is-dimmed", Boolean(neighborhood) && !highlighted);
      path.classList.toggle("is-highlighted", highlighted);
      path.classList.toggle("is-selected", Boolean(selectedNodeId) && highlighted);
      path.setAttribute("stroke-width", String(Boolean(selectedNodeId) && highlighted ? edge.width + 1.2 : edge.width));
      if (highlighted) {
        edgeLayer.appendChild(path);
      }
    });
  }

  function renderEdges() {
    edgeLayer.replaceChildren();
    defs.replaceChildren();
    renderedEdges = [];
    const visible = visibleEdgeIndices();
    graphData.edges.forEach((edge, idx) => {
      if (!visible.has(idx)) {
        return;
      }
      const marker = svgEl("marker", {
        id: `arrow-${idx}`,
        viewBox: "0 0 10 10",
        refX: "9",
        refY: "5",
        markerWidth: "7",
        markerHeight: "7",
        orient: "auto-start-reverse",
      });
      marker.appendChild(svgEl("path", { d: "M 0 0 L 10 5 L 0 10 z", fill: edge.color }));
      defs.appendChild(marker);

      const route = edgeRoute(edge);
      const path = svgEl("path", {
        class: "ct-edge",
        d: pathFromRoute(route),
        stroke: edge.color,
        "stroke-width": edge.width,
        "stroke-opacity": edge.opacity,
        "stroke-dasharray": edge.dash,
        "marker-end": `url(#arrow-${idx})`,
      });
      path.dataset.edgeIndex = String(edge.index);
      const title = svgEl("title");
      title.textContent = `${edge.source} -> ${edge.target}\\nweight=${edge.weight.toFixed(4)}`;
      path.appendChild(title);
      edgeLayer.appendChild(path);
      renderedEdges.push({ edge, path });
    });
    applyHighlight();
    updateEdgePanel();
  }

  function renderTokens() {
    tokenLayer.replaceChildren();
    graphData.promptTokens.forEach((token) => {
      const x = margin.left + (token.x - graphData.xRange[0]) * scale;
      const y = margin.top + (graphData.yRange[1] - token.y) * scale;
      const w = 0.9 * scale;
      const h = graphData.tokenHeight * scale;
      const group = svgEl("g");
      group.appendChild(svgEl("rect", {
        x: x - w / 2,
        y: y - h / 2,
        width: w,
        height: h,
        rx: 6,
        fill: token.highlight ? "#D8D2BF" : "#F4F2EB",
        stroke: "#B9B29A",
      }));
      const text = svgEl("text", {
        x,
        y: y + 4,
        "text-anchor": "middle",
        class: "ct-token-label",
      });
      text.textContent = token.text;
      group.appendChild(text);
      tokenLayer.appendChild(group);
    });
  }

  function renderNodes() {
    nodeLayer.replaceChildren();
    nodeElements.clear();
    graphData.nodes.forEach((node) => {
      const classes = ["ct-node"];
      if (node.lowActivation) {
        classes.push("has-low-activation");
      }
      if (node.badges && node.badges.length) {
        classes.push("has-intervention");
      }
      const group = svgEl("g", { class: classes.join(" "), tabindex: "0" });
      group.dataset.nodeId = node.id;
      group.style.cursor = "grab";
      if (node.stacked) {
        [10, 5].forEach((offset) => {
          group.appendChild(svgEl("rect", {
            x: node.px - node.widthPx / 2 + offset,
            y: node.py - node.heightPx / 2 + offset,
            width: node.widthPx,
            height: node.heightPx,
            rx: 9,
            fill: "#DCD6C4",
            stroke: node.border,
          }));
        });
      }
      group.appendChild(svgEl("rect", {
        x: node.px - node.widthPx / 2,
        y: node.py - node.heightPx / 2,
        width: node.widthPx,
        height: node.heightPx,
        rx: 9,
        fill: node.fill,
        stroke: node.border,
        "stroke-width": 1.6,
      }));
      const text = svgEl("text", {
        x: node.px,
        y: node.py - ((node.labelLines.length - 1) * 7),
        "text-anchor": "middle",
        class: "ct-node-label",
      });
      node.labelLines.forEach((line, idx) => {
        const tspan = svgEl("tspan", {
          x: node.px,
          dy: idx === 0 ? 0 : 15,
        });
        tspan.textContent = line;
        text.appendChild(tspan);
      });
      const title = svgEl("title");
      title.textContent = node.hover;
      group.appendChild(text);
      if (node.badges && node.badges.length) {
        let badgeX = node.px - node.widthPx / 2 + 7;
        const badgeY = node.py + node.heightPx / 2 + 4;
        node.badges.forEach((badge) => {
          const textWidth = Math.max(32, badge.text.length * 7 + 14);
          const badgeGroup = svgEl("g", { class: `ct-badge ct-badge-${badge.kind}` });
          badgeGroup.appendChild(svgEl("rect", {
            x: badgeX,
            y: badgeY,
            width: textWidth,
            height: 18,
            rx: 5,
          }));
          const badgeText = svgEl("text", {
            x: badgeX + textWidth / 2,
            y: badgeY + 13,
            "text-anchor": "middle",
          });
          badgeText.textContent = badge.text;
          badgeGroup.appendChild(badgeText);
          group.appendChild(badgeGroup);
          badgeX += textWidth + 5;
        });
      }
      group.appendChild(title);
      group.addEventListener("pointerdown", (event) => {
        const point = svgPoint(event);
        dragState = {
          node,
          startX: point.x,
          px: node.px,
          moved: false,
        };
        group.setPointerCapture(event.pointerId);
        group.style.cursor = "grabbing";
      });
      group.addEventListener("pointerup", (event) => {
        if (dragState) {
          suppressNextClick = dragState.moved;
          group.releasePointerCapture(event.pointerId);
        }
        dragState = null;
        group.style.cursor = "grab";
      });
      group.addEventListener("mouseenter", () => {
        hoverNodeId = node.id;
        applyHighlight();
      });
      group.addEventListener("mouseleave", () => {
        if (hoverNodeId === node.id) {
          hoverNodeId = null;
          applyHighlight();
        }
      });
      group.addEventListener("click", (event) => {
        event.stopPropagation();
        if (suppressNextClick) {
          suppressNextClick = false;
          return;
        }
        selectedNodeId = node.id;
        updateEdgePanel();
        applyHighlight();
      });
      nodeElements.set(node.id, group);
      nodeLayer.appendChild(group);
    });
    applyHighlight();
  }

  function rerenderGraph() {
    renderEdges();
    renderNodes();
  }

  svg.addEventListener("pointermove", (event) => {
    if (!dragState) {
      return;
    }
    const point = svgPoint(event);
    const dx = point.x - dragState.startX;
    dragState.moved = dragState.moved || Math.abs(dx) > 3;
    dragState.node.userOffsetPx = dragState.px - dragState.node.autoPx + dx;
    dragState.node.px = dragState.node.autoPx + dragState.node.userOffsetPx;
    dragState.node.py = dragState.node.rankPy;
    rerenderGraph();
  });

  graphData.topKValues.forEach((value) => {
    const option = document.createElement("option");
    option.value = value === null ? "all" : String(value);
    option.textContent = value === null ? "all" : String(value);
    if (value === graphData.defaultTopK) {
      option.selected = true;
    }
    topKSelect.appendChild(option);
  });
  topKSelect.addEventListener("change", renderEdges);
  positiveOnly.addEventListener("change", renderEdges);
  resetButton.addEventListener("click", () => {
    graphData.nodes.forEach((node) => {
      node.userOffsetPx = 0;
      node.px = node.autoPx;
      node.py = node.rankPy;
    });
    rerenderGraph();
  });
  svg.addEventListener("click", (event) => {
    if (event.target !== svg) {
      return;
    }
    selectedNodeId = null;
    updateEdgePanel();
    applyHighlight();
  });

  renderTokens();
  renderInterventionSummary();
  rerenderGraph();
})();
</script>
"""
    fragment = fragment.replace("__DATA__", payload_json)
    styles = """
<style>
  html,
  body {
    height: 100%;
    margin: 0;
    overflow: hidden;
  }
  .ct-cluster-viz {
    color: #1f2933;
    display: grid;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    grid-template-rows: auto minmax(0, 1fr) auto;
    height: 100vh;
    min-height: 0;
  }
  .ct-cluster-toolbar {
    align-items: center;
    background: #ffffff;
    border-bottom: 1px solid #d9dee7;
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    padding: 12px 14px;
    position: sticky;
    top: 0;
    z-index: 2;
  }
  .ct-cluster-title {
    font-size: 15px;
    margin-right: auto;
  }
  .ct-cluster-toolbar label {
    align-items: center;
    display: inline-flex;
    font-size: 12px;
    gap: 6px;
  }
  .ct-cluster-toolbar select,
  .ct-cluster-toolbar button {
    background: #ffffff;
    border: 1px solid #bac2cf;
    border-radius: 6px;
    color: #243044;
    font: inherit;
    padding: 5px 8px;
  }
  .ct-intervention-summary {
    align-items: center;
    display: inline-flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-left: 4px;
  }
  .ct-intervention-counts,
  .ct-output-label,
  .ct-output-pill {
    background: #f2f5f8;
    border: 1px solid #cbd5e1;
    border-radius: 999px;
    color: #334155;
    font-size: 11px;
    font-weight: 650;
    padding: 4px 8px;
  }
  .ct-output-pill {
    background: #fff7ed;
    border-color: #fed7aa;
    color: #9a3412;
  }
  .ct-output-label {
    background: transparent;
    border-color: transparent;
    color: #64748b;
    padding-left: 2px;
    padding-right: 0;
  }
  .ct-cluster-canvas {
    background: #ffffff;
    min-height: 0;
    overflow: auto;
  }
  .ct-cluster-canvas svg {
    display: block;
    height: 100%;
    width: 100%;
  }
  .edge-layer {
    z-index: 1;
  }
  .node-layer {
    z-index: 2;
  }
  .ct-edge {
    fill: none;
    stroke-linecap: round;
    stroke-linejoin: round;
    transition: opacity 120ms ease, stroke-opacity 120ms ease, stroke-width 120ms ease;
  }
  .ct-edge.is-dimmed {
    opacity: 0.12;
  }
  .ct-edge.is-highlighted {
    stroke-opacity: 0.95;
  }
  .ct-node {
    touch-action: none;
    transition: opacity 120ms ease;
    user-select: none;
  }
  .ct-node.is-dimmed {
    opacity: 0.24;
  }
  .ct-node.is-active > rect:last-of-type {
    stroke-width: 2.6;
  }
  .ct-node.is-neighbor > rect:last-of-type {
    stroke-width: 2.1;
  }
  .ct-node.has-low-activation > rect:last-of-type {
    fill: #f3f4f6;
    stroke: #cbd5e1;
  }
  .ct-node.has-low-activation .ct-node-label {
    fill: #64748b;
  }
  .ct-node-label {
    fill: #1a1a1a;
    font-size: 13px;
    pointer-events: none;
  }
  .ct-badge {
    pointer-events: none;
  }
  .ct-badge rect {
    stroke-width: 0;
  }
  .ct-badge text {
    fill: #ffffff;
    font-size: 11px;
    font-weight: 700;
  }
  .ct-badge-factor rect {
    fill: #d2691e;
  }
  .ct-badge-activation rect {
    fill: #64748b;
  }
  .ct-badge-stored rect {
    fill: #8b5cf6;
  }
  .ct-badge-delta rect {
    fill: #d2691e;
  }
  .ct-token-label {
    fill: #333333;
    font-size: 11px;
    pointer-events: none;
  }
  .ct-edge-panel {
    background: #ffffff;
    border-top: 1px solid #d9dee7;
    font-size: 12px;
    max-height: 160px;
    overflow: auto;
    padding: 10px 14px;
  }
  .ct-edge-panel ul {
    columns: 2 240px;
    list-style: none;
    margin: 6px 0 0;
    padding: 0;
  }
  .ct-node-details {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 7px;
  }
  .ct-node-details span {
    background: #f8fafc;
    border: 1px solid #d8e0ea;
    border-radius: 999px;
    color: #334155;
    padding: 3px 8px;
  }
  .ct-edge-panel li {
    break-inside: avoid;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    padding: 2px 0;
  }
</style>
"""
    if not full_html:
        return styles + fragment
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  {styles}
</head>
<body>
  {fragment}
</body>
</html>
"""


def _unique_names(names: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    out: list[str] = []
    for name in names:
        count = counts.get(name, 0) + 1
        counts[name] = count
        out.append(name if count == 1 else f"{name} ({count})")
    return out


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
    outgoing = {sn: set() for sn in sn_names}
    indegree = {sn: 0 for sn in sn_names}
    for source, target, _weight in edges:
        if source not in name_set or target not in name_set:
            continue
        if target not in outgoing[source]:
            outgoing[source].add(target)
            indegree[target] += 1

    def node_key(sn: str) -> tuple[float, int, str]:
        layer, ctx_mean = _layer_and_ctx_for_supernode(sn, mapping.get(sn, []), attr, node_by_name)
        return (ctx_mean, layer, sn)

    depth = {sn: 0 for sn in sn_names}
    seen: set[str] = set()
    ready = sorted([sn for sn, degree in indegree.items() if degree == 0], key=node_key)
    while ready:
        source = ready.pop(0)
        seen.add(source)
        for target in sorted(outgoing[source], key=node_key):
            depth[target] = max(depth[target], depth[source] + 1)
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
                ready.sort(key=lambda sn: (depth[sn], *node_key(sn)))

    non_logit_depths = [rank for sn, rank in depth.items() if _sn_kind(sn, node_by_name) != "logit"]
    if non_logit_depths:
        min_logit_depth = max(non_logit_depths) + 1
        for sn in sn_names:
            if _sn_kind(sn, node_by_name) == "logit" and not outgoing[sn]:
                depth[sn] = max(depth[sn], min_logit_depth)

    rows: dict[int, list[str]] = {}
    for sn, rank in depth.items():
        rows.setdefault(rank, []).append(sn)
    return rows


def _layout_sort_key(
    sn: str,
    mapping: dict[str, list[str]],
    attr: dict[str, dict[str, Any]] | None,
    node_by_name: dict[str, Supernode],
) -> tuple[float, int, str]:
    layer, ctx_mean = _layer_and_ctx_for_supernode(sn, mapping.get(sn, []), attr, node_by_name)
    return (ctx_mean, layer, sn)


def _resolve_rank_x_collisions(
    row: list[str],
    x_by_name: dict[str, float],
    mapping: dict[str, list[str]],
    attr: dict[str, dict[str, Any]] | None,
    node_by_name: dict[str, Supernode],
) -> None:
    if len(row) <= 1:
        return

    ordered = sorted(
        row, key=lambda sn: (x_by_name[sn], *_layout_sort_key(sn, mapping, attr, node_by_name))
    )
    desired_mean = sum(x_by_name[sn] for sn in ordered) / len(ordered)

    adjusted: dict[str, float] = {}
    last_x: float | None = None
    for sn in ordered:
        x = x_by_name[sn]
        if last_x is not None and x - last_x < _MIN_CARD_X_GAP:
            x = last_x + _MIN_CARD_X_GAP
        adjusted[sn] = x
        last_x = x

    adjusted_mean = sum(adjusted.values()) / len(adjusted)
    shift = desired_mean - adjusted_mean
    for sn, x in adjusted.items():
        x_by_name[sn] = x + shift


def _barycentric_x_layout(
    rows: dict[int, list[str]],
    edges: list[tuple[str, str, float]],
    mapping: dict[str, list[str]],
    attr: dict[str, dict[str, Any]] | None,
    node_by_name: dict[str, Supernode],
    right_x: float | None,
) -> dict[str, float]:
    sn_names = [sn for rank in sorted(rows) for sn in rows[rank]]
    name_set = set(sn_names)
    predecessors: dict[str, list[tuple[str, float]]] = {sn: [] for sn in sn_names}
    successors: dict[str, list[tuple[str, float]]] = {sn: [] for sn in sn_names}
    for source, target, weight in edges:
        if source not in name_set or target not in name_set or source == target:
            continue
        abs_weight = abs(float(weight))
        if abs_weight == 0.0:
            continue
        successors[source].append((target, abs_weight))
        predecessors[target].append((source, abs_weight))

    emb_x: list[float] = []
    for sn in sn_names:
        if _sn_kind(sn, node_by_name) != "emb":
            continue
        _layer, ctx_mean = _layer_and_ctx_for_supernode(sn, mapping.get(sn, []), attr, node_by_name)
        emb_x.append(float(ctx_mean))
    fallback_center = (
        float(np.mean(emb_x)) if emb_x else float(right_x if right_x is not None else 0.0)
    )

    x_by_name: dict[str, float] = {}
    for rank in sorted(rows):
        ordered = sorted(
            rows[rank], key=lambda sn: _layout_sort_key(sn, mapping, attr, node_by_name)
        )
        for idx, sn in enumerate(ordered):
            _layer, ctx_mean = _layer_and_ctx_for_supernode(
                sn, mapping.get(sn, []), attr, node_by_name
            )
            if _sn_kind(sn, node_by_name) == "emb":
                x_by_name[sn] = float(ctx_mean)
            elif (
                _sn_kind(sn, node_by_name) == "logit"
                and right_x is not None
                and not predecessors[sn]
                and not successors[sn]
            ):
                x_by_name[sn] = float(right_x)
            else:
                row_offset = (idx - (len(ordered) - 1) / 2) * _MIN_CARD_X_GAP
                x_by_name[sn] = fallback_center + row_offset

    ordered_ranks = sorted(rows)

    def update_node(sn: str) -> None:
        if _sn_kind(sn, node_by_name) == "emb":
            return
        weighted_sum = 0.0
        total_weight = 0.0
        for neighbor, weight in predecessors[sn]:
            weighted_sum += weight * x_by_name[neighbor]
            total_weight += weight
        for neighbor, weight in successors[sn]:
            weighted_sum += weight * x_by_name[neighbor]
            total_weight += weight
        if total_weight > 0.0:
            x_by_name[sn] = weighted_sum / total_weight

    for _sweep in range(8):
        for rank in ordered_ranks:
            for sn in sorted(rows[rank], key=lambda name: x_by_name[name]):
                update_node(sn)
        for rank in reversed(ordered_ranks):
            for sn in sorted(rows[rank], key=lambda name: x_by_name[name]):
                update_node(sn)

    for rank in ordered_ranks:
        _resolve_rank_x_collisions(rows[rank], x_by_name, mapping, attr, node_by_name)
    return x_by_name


def _supernode_layout(
    sn_names: list[str],
    mapping: dict[str, list[str]],
    attr: dict[str, dict[str, Any]] | None,
    node_by_name: dict[str, Supernode],
    right_x: float | None = None,
    visible_edges: list[tuple[str, str, float]] | None = None,
) -> tuple[dict[str, tuple[float, float]], float, dict[int, float]]:
    """Barycentric x layout with embedding nodes anchored to token position."""
    rows = _topological_rank_rows(sn_names, visible_edges or [], mapping, attr, node_by_name)
    ordered_ranks = sorted(rows)
    rank_y = {rank: (rank + 1) * _RANK_Y_GAP for rank in ordered_ranks}
    x_by_name = _barycentric_x_layout(
        rows,
        visible_edges or [],
        mapping,
        attr,
        node_by_name,
        right_x,
    )

    pos: dict[str, tuple[float, float]] = {}
    for rank in ordered_ranks:
        for sn in rows[rank]:
            pos[sn] = (x_by_name[sn], float(rank_y[rank]))
    max_rank = max(ordered_ranks, default=-1)
    top_y = (max_rank + 2) * _RANK_Y_GAP
    return pos, top_y, rank_y


def _dedupe_route_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    cleaned: list[tuple[float, float]] = []
    for point in points:
        if not cleaned or math.hypot(point[0] - cleaned[-1][0], point[1] - cleaned[-1][1]) > 1e-9:
            cleaned.append(point)
    return cleaned


def _orthogonal_path(points: list[tuple[float, float]]) -> str:
    route = _dedupe_route_points(points)
    head, *tail = route
    chunks = [f"M {head[0]},{head[1]}"]
    chunks.extend(f"L {x},{y}" for x, y in tail)
    return " ".join(chunks)


def _routed_edge_points(
    source: str,
    target: str,
    geom: dict[str, tuple[float, float, float, float]],
) -> list[tuple[float, float]]:
    """Orthogonal source-top -> target-bottom route in graph coordinates."""
    sx, sy, sw, sh = geom[source]
    tx, ty, tw, th = geom[target]
    source_port = (sx, sy + sh / 2)
    target_port = (tx, ty - th / 2)
    mid_y = (source_port[1] + target_port[1]) / 2

    if abs(source_port[0] - target_port[0]) < 1e-9:
        return _dedupe_route_points([source_port, target_port])
    return _dedupe_route_points(
        [
            source_port,
            (source_port[0], mid_y),
            (target_port[0], mid_y),
            target_port,
        ]
    )


def _resolve_synthetic_column(
    synthetic_nodes: list[dict[str, Any]],
    geom: dict[str, tuple[float, float, float, float]],
    column_x: float,
) -> None:
    if not synthetic_nodes:
        return

    previous_top: float | None = None
    for node in sorted(synthetic_nodes, key=lambda item: (float(item["y"]), str(item["id"]))):
        node_id = str(node["id"])
        _cx, cy, w, h = geom[node_id]
        if previous_top is not None:
            min_cy = previous_top + 0.35 + h / 2
            cy = max(cy, min_cy)
        geom[node_id] = (column_x, cy, w, h)
        node["x"] = column_x
        node["y"] = cy
        previous_top = cy + h / 2


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
) -> ClusterGraphFigure:
    """
    Build an interactive HTML/SVG figure in the Anthropic attribution-graph style:
    supernodes as draggable rounded cards, directed green/red arrows, and
    (when ``prompt_tokens`` is given) a tokenized prompt strip above the graph.

    `sng` may be a `SummaryGraph` instance or the legacy dict.

    ``edge_threshold`` (0-1) hides edges whose magnitude is below that fraction of
    the largest edge weight. ``top_k_logits`` keeps only the k highest-probability
    logit supernodes (and their edges); ``None`` shows all.
    """
    _ = (
        seed,
        prompt,
        use_supernode_names,
    )
    steering_factors = steering_factors or {}
    activation_ratios = activation_ratios or {}
    top_output_payload = [_format_output(item) for item in (top_outputs or [])]
    stored_interventions = stored_interventions or []

    # Duck-typing rather than isinstance so this survives Streamlit hot-reload,
    # which re-imports SummaryGraph and breaks isinstance on session-state objects.
    if hasattr(sng, "sn_names") and (hasattr(sng, "adj") or hasattr(sng, "adj_matrix")):
        graph = cast(Any, sng)
        sn_names = _unique_names(list(graph.sn_names))
        raw_adj = graph.adj if hasattr(graph, "adj") else graph.adj_matrix
        sn_adj = np.asarray(raw_adj, dtype=np.float64)
        if hasattr(graph, "supernodes"):
            supernodes = list(graph.supernodes)
            mapping = {
                sn_name: supernode.member_node_ids()
                for sn_name, supernode in zip(sn_names, supernodes)
            }
            node_by_name = {sn_name: supernode for sn_name, supernode in zip(sn_names, supernodes)}
        else:
            mapping = final_supernodes if final_supernodes is not None else graph.to_mapping()
            node_by_name = graph.node_by_name()
    else:
        if final_supernodes is None:
            raise ValueError("final_supernodes is required when sng is a plain dict.")
        legacy = cast(dict[str, Any], sng)
        sn_names = _unique_names(list(legacy["sn_names"]))
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
        kind = _sn_kind(sn, node_by_name)
        kinds[sn] = kind
        label_text = _node_label(sn, members, attr, node_by_name.get(sn))
        card_w, card_h = _label_card_size(label_text, len(members))
        geom[sn] = (cx, cy, card_w, card_h)
        if kind == "emb":
            emb_ctx.add(int(round(cx)))

    visible_feature_names = [
        sn for sn in layout_names if sn in geom and _sn_kind(sn, node_by_name) == "middle"
    ]
    stored_by_host: dict[str, list[dict[str, Any]]] = {}
    for stored in stored_interventions:
        host = _stored_intervention_host(stored, visible_feature_names, mapping, attr, node_by_name)
        if host is not None:
            stored_by_host.setdefault(host, []).append(stored)

    synthetic_nodes: list[dict[str, Any]] = []
    synthetic_edges: list[tuple[str, str, float, str]] = []
    synthetic_column_x = max((cx for cx, _cy, _w, _h in geom.values()), default=0.0)
    synthetic_column_x += _MIN_CARD_X_GAP
    intervention_source_ids = [sn for sn in steering_factors if sn in geom]
    for host, stored_items in stored_by_host.items():
        _hx, hy, _hw, _hh = geom[host]
        for stored in stored_items:
            label = str(stored.get("label") or stored.get("record_id") or "Stored intervention")
            factor = float(stored.get("factor", 0.0))
            factor_text = _format_factor(factor)
            synthetic_id = f"__stored_intervention_{len(synthetic_nodes)}"
            card_w, card_h = _label_card_size(label, int(stored.get("n_features", 1)))
            sx = synthetic_column_x
            sy = hy
            geom[synthetic_id] = (sx, sy, card_w, card_h)
            hover = "\n".join(
                [
                    f"Stored intervention: {label}",
                    f"Factor: {factor_text}",
                    f"Target position: {int(stored.get('target_pos', 0))}",
                ]
            )
            synthetic_nodes.append(
                {
                    "id": synthetic_id,
                    "x": sx,
                    "y": sy,
                    "width": card_w,
                    "height": card_h,
                    "kind": "intervention",
                    "fill": "#FFF7ED",
                    "border": "#D2691E",
                    "stacked": False,
                    "label": label,
                    "labelLines": _wrap_label(label).split("<br>"),
                    "hover": hover,
                    "badges": [{"kind": "factor", "text": factor_text}],
                    "lowActivation": False,
                    "interventionDetails": [
                        f"Stored intervention: {label}",
                        f"Factor: {factor_text}",
                    ],
                }
            )
            synthetic_edges.append((synthetic_id, host, factor if factor != 0.0 else 1.0, "#D2691E"))
            intervention_source_ids.append(synthetic_id)

    output_change = top_output_payload[0] if top_output_payload else None
    if output_change is not None and ("logit_delta" in output_change or steering_factors):
        token = _clean_token(str(output_change["token"]))
        label = f"\u0394 Logit: {token}"
        card_w, card_h = _label_card_size(label, 1)
        logit_hosts = [
            sn for sn in layout_names if sn in geom and _sn_kind(sn, node_by_name) == "logit"
        ]
        token_key = token.casefold()
        matching_logit_hosts = [
            sn
            for sn in logit_hosts
            if token_key
            and token_key in _node_label(sn, mapping.get(sn, []), attr, node_by_name.get(sn)).casefold()
        ]
        ranked_logit_hosts = sorted(
            matching_logit_hosts or logit_hosts,
            key=lambda sn: (geom[sn][1], -abs(geom[sn][0]), sn),
            reverse=True,
        )
        output_id = "__output_logit_delta_0"
        if ranked_logit_hosts:
            logit_host = ranked_logit_hosts[0]
            _lx, ly, _lw, lh = geom[logit_host]
            ox = synthetic_column_x
            oy = ly + lh / 2 + card_h / 2 + 0.52
        else:
            ox = synthetic_column_x
            oy = max((cy for _cx, cy, _w, _h in geom.values()), default=0.0) + _RANK_Y_GAP
        geom[output_id] = (ox, oy, card_w, card_h)
        badges = [{"kind": "activation", "text": _format_percent(output_change["probability"])}]
        if "logit_delta" in output_change:
            badges.insert(0, {"kind": "delta", "text": _format_logit_delta(output_change["logit_delta"])})
        details = [f"Output probability: {_format_percent(output_change['probability'])}"]
        if "clean_probability" in output_change:
            details.append(f"Clean probability: {_format_percent(output_change['clean_probability'])}")
        if "probability_delta" in output_change:
            details.append(
                f"\u0394 probability: {_format_probability_delta(output_change['probability_delta'])}"
            )
        if "logit_delta" in output_change:
            details.append(f"\u0394 logit: {_format_logit_delta(output_change['logit_delta'])}")
        synthetic_nodes.append(
            {
                "id": output_id,
                "x": ox,
                "y": oy,
                "width": card_w,
                "height": card_h,
                "kind": "output_delta",
                "fill": "#FFF7ED",
                "border": "#D2691E",
                "stacked": False,
                "label": label,
                "labelLines": _wrap_label(label).split("<br>"),
                "hover": "\n".join(details),
                "badges": badges,
                "lowActivation": False,
                "interventionDetails": details,
            }
        )
        delta_weight = float(output_change.get("logit_delta", 1.0))
        for source in intervention_source_ids:
            if source in geom:
                synthetic_edges.append(
                    (source, output_id, delta_weight if delta_weight != 0.0 else 1.0, "#D2691E")
                )

    _resolve_synthetic_column(synthetic_nodes, geom, synthetic_column_x)

    # --- Edges ---
    top_k_values = _edge_filter_k_values(rendered_edges)
    default_top_k = _DEFAULT_EDGE_TOP_K if _DEFAULT_EDGE_TOP_K in top_k_values else None
    initial_edge_indices = _select_edge_indices(
        rendered_edges,
        top_k=default_top_k,
        positive_only=False,
    )
    edge_payload: list[dict[str, Any]] = []
    edge_xs: list[float] = []
    edge_ys: list[float] = []
    for edge_idx, (u, v, w) in enumerate(rendered_edges):
        if u not in geom or v not in geom:
            continue
        width, color, opacity, dash = _edge_style(w, max_abs_w)
        route = _routed_edge_points(u, v, geom)
        if edge_idx in initial_edge_indices:
            edge_xs.extend(x for x, _y in route)
            edge_ys.extend(y for _x, y in route)
        edge_payload.append(
            {
                "index": edge_idx,
                "source": u,
                "target": v,
                "weight": w,
                "width": width,
                "color": color,
                "opacity": opacity,
                "dash": dash,
                "initialPath": _orthogonal_path(route),
            }
        )
    for u, v, w, color in synthetic_edges:
        if u not in geom or v not in geom:
            continue
        edge_idx = len(edge_payload)
        route = _routed_edge_points(u, v, geom)
        edge_xs.extend(x for x, _y in route)
        edge_ys.extend(y for _x, y in route)
        edge_payload.append(
            {
                "index": edge_idx,
                "source": u,
                "target": v,
                "weight": w,
                "width": 3.2,
                "color": color,
                "opacity": 0.92,
                "dash": "",
                "initialPath": _orthogonal_path(route),
                "alwaysVisible": True,
            }
        )

    # --- Cards + labels ---
    node_payload: list[dict[str, Any]] = []
    for sn in sn_names:
        if sn not in geom:
            continue
        cx, cy, w, h = geom[sn]
        members = mapping.get(sn, [])
        fill, line_color = _KIND_STYLE.get(kinds[sn], _KIND_STYLE["middle"])
        label_text = _node_label(sn, members, attr, node_by_name.get(sn))
        hover_title = html.unescape(_sn_title(sn, members, attr, node_by_name.get(sn)))
        badges: list[dict[str, str]] = []
        intervention_details: list[str] = []
        if sn in steering_factors:
            factor_text = _format_factor(float(steering_factors[sn]))
            badges.append({"kind": "factor", "text": factor_text})
            intervention_details.append(f"Intervention: {factor_text}")
        activation_ratio = activation_ratios.get(sn)
        if activation_ratio is not None:
            activation_text = _format_percent(float(activation_ratio))
            badges.append({"kind": "activation", "text": activation_text})
            intervention_details.append(f"Activation ratio: {activation_text}")
        hosted_stored = stored_by_host.get(sn, [])
        if hosted_stored:
            badges.append({"kind": "stored", "text": "Stored"})
            for stored in hosted_stored:
                label = str(stored.get("label") or stored.get("record_id") or "Stored intervention")
                factor = _format_factor(float(stored.get("factor", 0.0)))
                target_pos = int(stored.get("target_pos", 0))
                intervention_details.append(f"Stored: {label} {factor} @ pos {target_pos}")
        if intervention_details:
            hover_title = "\n".join([hover_title, *intervention_details])
        node_payload.append(
            {
                "id": sn,
                "x": cx,
                "y": cy,
                "width": w,
                "height": h,
                "kind": kinds[sn],
                "fill": fill,
                "border": line_color,
                "stacked": len(members) > 1,
                "label": label_text,
                "labelLines": _wrap_label(label_text).split("<br>"),
                "hover": hover_title.replace("<br>", "\n"),
                "badges": badges,
                "lowActivation": (
                    activation_ratio is not None
                    and float(activation_ratio) <= _LOW_ACTIVATION_RATIO
                ),
                "interventionDetails": intervention_details,
            }
        )
    node_payload.extend(synthetic_nodes)

    # --- Tokenized prompt strip ---
    prompt_payload: list[dict[str, Any]] = []
    prompt_y = float(top_y)
    if synthetic_nodes:
        prompt_y = max(prompt_y, max(float(node["y"]) for node in synthetic_nodes) + _RANK_Y_GAP)
    if prompt_tokens:
        prompt_payload = [
            {
                "x": float(i),
                "y": prompt_y,
                "text": _clean_token(tok),
                "highlight": i in emb_ctx,
            }
            for i, tok in enumerate(prompt_tokens)
        ]

    # --- Layout / ranges ---
    token_xs = [float(i) for i in range(len(prompt_tokens or []))]
    xs_for_range = [g[0] for g in geom.values()] + edge_xs + token_xs + [0.0]
    ys_for_range = [g[1] for g in geom.values()] + edge_ys
    if prompt_tokens:
        ys_for_range.append(prompt_y)
    if not ys_for_range:
        ys_for_range.append(0.0)
    x_min = min(xs_for_range) - 1.5
    x_max = max(xs_for_range) + 1.5
    y_min = min(ys_for_range) - 0.7
    y_max = max(ys_for_range) + 0.7
    scale = 58.0
    margin = {"left": 72.0, "right": 72.0, "top": 72.0, "bottom": 72.0}
    canvas_width = max(960.0, (x_max - x_min) * scale + margin["left"] + margin["right"])
    canvas_height = max(620.0, (y_max - y_min) * scale + margin["top"] + margin["bottom"])
    return ClusterGraphFigure(
        {
            "title": title,
            "prompt": prompt or "",
            "nodes": node_payload,
            "edges": edge_payload,
            "initialEdgeIndices": sorted(initial_edge_indices),
            "promptTokens": prompt_payload,
            "topKValues": top_k_values,
            "defaultTopK": default_top_k,
            "xRange": [x_min, x_max],
            "yRange": [y_min, y_max],
            "scale": scale,
            "margin": margin,
            "canvasWidth": canvas_width,
            "canvasHeight": canvas_height,
            "tokenHeight": _BAR_H,
            "interventionSummary": {
                "active": bool(steering_factors or activation_ratios or stored_interventions or top_outputs),
                "steeredCount": len(steering_factors),
                "storedCount": len(stored_interventions),
                "topOutputs": top_output_payload[:6],
            },
        }
    )
