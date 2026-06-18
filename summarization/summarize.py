"""Supernode / summary-graph types and the summarize stage (clusters -> SummaryGraph)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch

if TYPE_CHECKING:
    from summarization.prune import PruneGraph

SupernodeType = Literal["emb", "feature", "features", "logit"]


def cluster_kind_to_supernode_type(kind: str) -> SupernodeType:
    """Map raw cluster role strings to ``Supernode.type``."""
    if kind == "emb":
        return "emb"
    if kind == "logit":
        return "logit"
    return "features"


@dataclass
class Node:
    """Summarization-node view aligned with frontend node fields + relevance."""

    node_id: str
    node_idx: int
    feature: int
    layer: str
    ctx_idx: int
    feature_type: str
    token_prob: float = 0.0
    is_target_logit: bool = False
    run_idx: int = 0
    reverse_ctx_idx: int = 0
    jsNodeId: str = ""
    clerp: str = ""
    influence: float | None = None
    activation: float | None = None
    relevance: float | None = None

    def set_clerp(self, clerp: str) -> None:
        self.clerp = clerp

    def set_node_idx(self, node_idx: int) -> None:
        self.node_idx = node_idx


def _tensor_value_at(values: Any, idx: int) -> float | None:
    if values is None:
        return None
    try:
        raw = values[idx]
    except (IndexError, TypeError, KeyError):
        return None
    if hasattr(raw, "detach"):
        raw = raw.detach().cpu().item()
    return float(raw)


def node_from_prune_graph(
    prune_graph: "PruneGraph",
    node_id: str,
    id_to_idx: dict[str, int] | None = None,
) -> Node:
    """Build a typed summarization node from ``PruneGraph.nodes`` plus score tensors."""
    nodes: list[Node] = prune_graph.nodes
    lookup = {n.node_id: n for n in nodes}
    base = lookup.get(node_id)
    if base is None:
        raise KeyError(f"unknown node_id for PruneGraph: {node_id!r}")

    if id_to_idx is None:
        id_to_idx = {n.node_id: i for i, n in enumerate(nodes)}
    idx = id_to_idx[node_id]

    ti = _tensor_value_at(prune_graph.node_influence, idx)
    tr = _tensor_value_at(prune_graph.node_relevance, idx)

    return Node(
        node_id=base.node_id,
        node_idx=idx,
        feature=base.feature,
        layer=str(base.layer),
        ctx_idx=int(base.ctx_idx),
        feature_type=base.feature_type,
        token_prob=float(base.token_prob),
        is_target_logit=base.is_target_logit,
        run_idx=int(base.run_idx),
        reverse_ctx_idx=int(base.reverse_ctx_idx),
        jsNodeId=str(base.jsNodeId or base.node_id),
        clerp=str(base.clerp),
        influence=ti if ti is not None else base.influence,
        activation=base.activation,
        relevance=tr if tr is not None else base.relevance,
    )


def compute_sn_adj(
    index_lists: list[list[int]],
    pruned_adj: torch.Tensor,
) -> np.ndarray:
    """Block-sum supernode adjacency. ``block[t, s] = sum_{u in S_s, v in S_t} adj[v, u]``.

    May contain antiparallel pairs and longer SCCs; ``get_adj`` resolves these.
    """
    adj = pruned_adj.detach().cpu().numpy().astype(np.float64)  # [tgt, src]
    n_total = adj.shape[0]
    n_sn = len(index_lists)
    indicator = np.zeros((n_sn, n_total), dtype=np.float64)
    for sn_idx, idxs in enumerate(index_lists):
        for i in idxs:
            if 0 <= i < n_total:
                indicator[sn_idx, i] = 1.0
    block = indicator @ adj @ indicator.T  # block[t, s]
    np.fill_diagonal(block, 0.0)
    return block


@dataclass
class Supernode:
    """One grouped supernode: display name, typed members, role, and layer span."""

    name: str
    features: list[Node]
    type: SupernodeType
    layer_min: int
    layer_max: int
    role: str = ""  # LLM-assigned cluster role ("Input" | "Abstract" | "Output" | "Trash")
    description: str = ""  # LLM-assigned one-sentence description (two-pass scheme only)

    def member_node_ids(self) -> list[str]:
        return [node.node_id for node in self.features]


def _compute_depths(supernodes: list[Supernode]) -> np.ndarray:
    """Median member-layer per supernode. emb → −∞, logit → +∞ (forced sources/sinks)."""
    from summarization.utils import layer_index_from_node

    depths = np.empty(len(supernodes), dtype=np.float64)
    for k, sn in enumerate(supernodes):
        if sn.type == "emb":
            depths[k] = -np.inf
        elif sn.type == "logit":
            depths[k] = np.inf
        else:
            layers = [layer_index_from_node(n) for n in sn.features]
            depths[k] = float(np.median(layers)) if layers else 0.0
    return depths


def _stable_argsort(supernodes: list[Supernode], depths: np.ndarray) -> np.ndarray:
    """Sort by depth; ties → mean layer → min ctx_idx → supernode index."""
    from summarization.utils import layer_index_from_node

    keys: list[tuple[float, float, int, int]] = []
    for k, sn in enumerate(supernodes):
        if sn.features:
            layers = [layer_index_from_node(n) for n in sn.features]
            mean_layer = float(np.mean(layers))
            min_ctx = min(n.ctx_idx for n in sn.features)
        else:
            mean_layer = 0.0
            min_ctx = 0
        keys.append((float(depths[k]), mean_layer, int(min_ctx), k))
    return np.array(sorted(range(len(keys)), key=lambda i: keys[i]), dtype=np.int64)


def get_adj(supernodes: list[Supernode], pruned_adj: torch.Tensor) -> np.ndarray:
    """Pack supernodes into a DAG

    Block-sums ``pruned_adj`` over the supernode partition, then applies π:
    Stage A collapses each antiparallel pair to its dominant direction (magnitude
    reduced by the weaker side, sign following the survivor); Stage B removes
    back-edges against an anchor-depth ordering (emb forced source, logit forced
    sink). ``out[t, s]`` is the edge weight source ``s`` → target ``t``.
    """
    index_lists = [[n.node_idx for n in sn.features if n.node_idx >= 0] for sn in supernodes]
    M = compute_sn_adj(index_lists, pruned_adj)  # block[t, s]; may contain cycles
    n = M.shape[0]

    # --- Stage A: antiparallel collapse ---
    for i in range(n):
        for j in range(i + 1, n):
            a = float(M[i, j])
            b = float(M[j, i])
            abs_a, abs_b = abs(a), abs(b)
            if abs_a == 0.0 or abs_b == 0.0:
                continue  # unilateral edge — nothing to collapse
            if abs_a > abs_b:
                M[i, j] = (abs_a - abs_b) * (1.0 if a > 0 else -1.0)
                M[j, i] = 0.0
            elif abs_b > abs_a:
                M[j, i] = (abs_b - abs_a) * (1.0 if b > 0 else -1.0)
                M[i, j] = 0.0
            else:
                M[i, j] = 0.0
                M[j, i] = 0.0

    # --- Stage B: anchor-ordered back-edge removal ---
    depths = _compute_depths(supernodes)
    ordering = _stable_argsort(supernodes, depths)
    rank = np.empty(n, dtype=np.int64)
    rank[ordering] = np.arange(n)
    for t in range(n):
        for s in range(n):
            if M[t, s] != 0.0 and rank[s] > rank[t]:  # source ranks later → back-edge
                M[t, s] = 0.0
    return M


@dataclass
class SummaryGraph:
    supernodes: list[Supernode]
    pruned_adj: torch.Tensor
    metadata: dict = field(
        default_factory=dict
    )  # provenance: prompt, target token, scan, prompt_tokens
    adj_matrix: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.adj_matrix = get_adj(self.supernodes, self.pruned_adj)

    @property
    def sn_names(self) -> list[str]:
        return [n.name for n in self.supernodes]

    def to_mapping(self) -> dict[str, list[str]]:
        return {n.name: n.member_node_ids() for n in self.supernodes}

    def node_by_name(self) -> dict[str, Supernode]:
        return {n.name: n for n in self.supernodes}

    def save(self, path: str) -> None:
        """Pickle supernodes + pruned_adj + metadata to ``path`` (.pt). adj_matrix is re-derived on load."""
        torch.save(
            {
                "supernodes": self.supernodes,
                "pruned_adj": self.pruned_adj,
                "metadata": self.metadata,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "SummaryGraph":
        """Inverse of ``save``; ``__post_init__`` rebuilds the post-π adjacency.

        Pre-metadata ``.pt`` files load with empty metadata (see ADR 0001).
        """
        d = torch.load(path, map_location="cpu", weights_only=False)
        return cls(
            supernodes=d["supernodes"],
            pruned_adj=d["pruned_adj"],
            metadata=d.get("metadata", {}),
        )


def summarize(
    supernodes: list[Supernode],
    pruned_adj: torch.Tensor,
    metadata: dict | None = None,
) -> SummaryGraph:
    """Stage 3: assemble grouped supernodes into the post-π summary graph."""
    return SummaryGraph(supernodes=supernodes, pruned_adj=pruned_adj, metadata=metadata or {})


def steer_interventions(
    supernodes: list[Supernode],
    orig_activations: torch.Tensor,
    factors: dict[str, float] | float,
) -> list[tuple[int, int, int, float]]:
    """Steer several supernodes at once: value = factor * orig_activation per CLT feature.

    Each feature is steered at its active position (Node.ctx_idx). factor=-1 negates
    (paper steering), factor=0 ablates. ``factors`` is a per-supernode-name mapping or a
    single float applied to all. ``orig_activations``: [n_layers, n_pos, d_transcoder].
    Returns (layer, pos, feature_idx, value) tuples for ``feature_intervention``.
    """
    n_layers, n_pos, d_tc = orig_activations.shape
    out: list[tuple[int, int, int, float]] = []
    for sn in supernodes:
        factor = factors if isinstance(factors, (int, float)) else factors[sn.name]
        for n in sn.features:
            if n.feature_type != "cross layer transcoder":
                continue
            layer, feat = int(n.node_id.split("_")[0]), int(n.node_id.split("_")[1])
            pos = n.ctx_idx
            if layer < n_layers and pos < n_pos and feat < d_tc:
                out.append((layer, pos, feat, factor * orig_activations[layer, pos, feat].item()))
    return out


def constrained_window(layer: int, n_layers: int, layers_below: int, layers_above: int) -> range:
    """Constrained-patching window around a source ``layer``: [layer-below, layer+above].

    Clamped to valid layers. The paper's Fig. 9 default (below=0, above=1) gives the window
    [l, l+1]: the feature's own-layer decode (l) plus its first cross-layer write (l+1); writes
    above l+1 are dropped. CLT features decode only into layers >= their own, so any slot below
    l is empty — that is why ``layers_below`` defaults to 0.
    """
    return range(max(0, layer - layers_below), min(layer + layers_above + 1, n_layers))


def steer_interventions_constrained(
    supernodes: list[Supernode],
    orig_activations: torch.Tensor,
    factors: dict[str, float] | float,
    layers_below: int = 0,
    layers_above: int = 1,
) -> list[tuple[range, list[tuple[int, int, int, float]]]]:
    """Constrained (direct-effect) steering, grouped per activation layer.

    Same per-feature value as ``steer_interventions`` (factor * orig_activation at the
    feature's active position), but returns one group per source layer ``l`` together with
    its constrained window ``[l-layers_below, l+layers_above]`` (see ``constrained_window``).
    Each group is meant for a *separate* ``feature_intervention(..., constrained_layers=window)``
    pass — ``feature_intervention`` takes a single global range, so per-feature windows
    require per-layer passes. The default (below=0, above=1) gives the paper's Fig. 9 window
    [l, l+1]: own-layer decode plus the first cross-layer write. ``orig_activations``:
    [n_layers, n_pos, d_transcoder].
    """
    n_layers, n_pos, d_tc = orig_activations.shape
    by_layer: dict[int, list[tuple[int, int, int, float]]] = {}
    for sn in supernodes:
        factor = factors if isinstance(factors, (int, float)) else factors[sn.name]
        for n in sn.features:
            if n.feature_type != "cross layer transcoder":
                continue
            layer, feat = int(n.node_id.split("_")[0]), int(n.node_id.split("_")[1])
            pos = n.ctx_idx
            if layer < n_layers and pos < n_pos and feat < d_tc:
                value = factor * orig_activations[layer, pos, feat].item()
                by_layer.setdefault(layer, []).append((layer, pos, feat, value))
    return [
        (constrained_window(layer, n_layers, layers_below, layers_above), ivs)
        for layer, ivs in sorted(by_layer.items())
    ]
