"""First-class supernode and summarization (cluster) graph types."""

from __future__ import annotations

from dataclasses import dataclass
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
    """Block-sum adjacency between supernodes with dominant-direction tie-breaker.

    index_lists[i] = node indices of supernode i. Returns sn_adj[t, s] = mass
    flowing s -> t (matches pruned_adj convention). For each unordered pair
    {i, j} keeps only the stronger direction; zeros both on ties. Diagonal
    always zero.
    """
    adj = pruned_adj.detach().cpu().numpy().astype(np.float64)  # [tgt, src]
    n_total = adj.shape[0]
    n_sn = len(index_lists)
    indicator = np.zeros((n_sn, n_total), dtype=np.float64)
    for sn_idx, idxs in enumerate(index_lists):
        for i in idxs:
            if 0 <= i < n_total:
                indicator[sn_idx, i] = 1.0
    block = indicator @ adj @ indicator.T  # block[t, s] = sum_{u in S_s, v in S_t} adj[v, u]

    abs_block = np.abs(block)
    keep = abs_block > abs_block.T  # strict dominance; ties drop both
    np.fill_diagonal(keep, False)
    return block * keep


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


@dataclass
class SummarizationGraph:
    """
    Supernode-level graph
    """

    supernodes: list[Supernode]
    pruned_adj: torch.Tensor


    @property
    def sn_names(self) -> list[str]:
        return [n.name for n in self.supernodes]

    def to_mapping(self) -> dict[str, list[str]]:
        return {n.name: n.member_node_ids() for n in self.supernodes}


    @property
    def sn_adj(self) -> np.ndarray:
        index_lists = [
            [n.node_idx for n in sn.features if n.node_idx >= 0]
            for sn in self.supernodes
        ]
        return compute_sn_adj(index_lists, self.pruned_adj)

    def node_by_name(self) -> dict[str, Supernode]:
        return {n.name: n for n in self.supernodes}