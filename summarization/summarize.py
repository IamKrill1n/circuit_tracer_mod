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
    """Map ``cluster_graph_spectral`` role strings to ``Supernode.type``."""
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
    index_lists = [
        [n.node_idx for n in sn.features if n.node_idx >= 0]
        for sn in supernodes
    ]
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
    """Supernodes plus their post-π acyclic adjacency.

    Construct with ``SummaryGraph(supernodes=..., pruned_adj=...)``;
    ``adj_matrix`` is derived once via ``get_adj`` (block-sum + antiparallel
    collapse + back-edge removal). ``adj_matrix[t, s]`` is the supernode-level
    edge weight source ``s`` → target ``t`` — same convention as ``pruned_adj``.
    """

    supernodes: list[Supernode]
    pruned_adj: torch.Tensor
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


def summarize(supernodes: list[Supernode], pruned_adj: torch.Tensor) -> SummaryGraph:
    """Stage 3: assemble grouped supernodes into the post-π summary graph."""
    return SummaryGraph(supernodes=supernodes, pruned_adj=pruned_adj)
