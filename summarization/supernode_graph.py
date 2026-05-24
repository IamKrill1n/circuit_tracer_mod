"""First-class supernode and summarization (cluster) graph types."""

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

    May contain antiparallel pairs and longer SCCs. The post-π view lives on
    ``SummarizationGraph.sn_adj``.
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


@dataclass
class SummarizationGraph:
    """Supernode-level graph with built-in π force-DAG transform.

    ``sn_adj_raw`` returns the plain block-sum adjacency (may contain cycles).
    ``sn_adj`` returns the post-π adjacency — antiparallel pairs collapsed to
    their dominant direction, then back-edges removed against an anchor-depth
    ordering. ``l_collapse`` / ``l_back`` report the magnitude dropped at each
    stage; together they explain the gap between ``|sn_adj_raw|`` and ``|sn_adj|``.
    """

    supernodes: list[Supernode]
    pruned_adj: torch.Tensor
    # π cache (populated by _apply_pi on first access of any post-π property)
    _sn_adj_dag: np.ndarray | None = field(default=None, init=False, repr=False)
    _l_collapse: float | None = field(default=None, init=False, repr=False)
    _l_back: float | None = field(default=None, init=False, repr=False)
    _depths: np.ndarray | None = field(default=None, init=False, repr=False)
    _ordering: np.ndarray | None = field(default=None, init=False, repr=False)
    _back_edges: list[tuple[int, int]] | None = field(default=None, init=False, repr=False)

    @property
    def sn_names(self) -> list[str]:
        return [n.name for n in self.supernodes]

    def to_mapping(self) -> dict[str, list[str]]:
        return {n.name: n.member_node_ids() for n in self.supernodes}

    def node_by_name(self) -> dict[str, Supernode]:
        return {n.name: n for n in self.supernodes}

    # --- Raw (cyclic) supernode adjacency ---------------------------------
    @property
    def sn_adj_raw(self) -> np.ndarray:
        """Plain block-sum supernode adjacency. May contain cycles. Diagnostic only."""
        index_lists = [
            [n.node_idx for n in sn.features if n.node_idx >= 0]
            for sn in self.supernodes
        ]
        return compute_sn_adj(index_lists, self.pruned_adj)

    # --- π force-DAG: cached post-π adjacency + stats ---------------------
    def _apply_pi(self) -> None:
        """Stage A (antiparallel collapse) + Stage B (back-edge removal). Cached."""
        if self._sn_adj_dag is not None:
            return
        M = self.sn_adj_raw.copy()
        n = M.shape[0]

        # --- Stage A: antiparallel collapse -------------------------------
        # For each unordered pair {i, j} with mass in both directions, keep the
        # dominant direction with magnitude reduced by the weaker side; the
        # cancelled portion (2 * min(|a|, |b|)) accrues to l_collapse. Sign of
        # the survivor follows the dominant direction.
        l_collapse = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                a = float(M[i, j])
                b = float(M[j, i])
                abs_a, abs_b = abs(a), abs(b)
                if abs_a == 0.0 or abs_b == 0.0:
                    continue  # unilateral edge — nothing to collapse
                l_collapse += 2.0 * min(abs_a, abs_b)
                if abs_a > abs_b:
                    M[i, j] = (abs_a - abs_b) * (1.0 if a > 0 else -1.0)
                    M[j, i] = 0.0
                elif abs_b > abs_a:
                    M[j, i] = (abs_b - abs_a) * (1.0 if b > 0 else -1.0)
                    M[i, j] = 0.0
                else:
                    M[i, j] = 0.0
                    M[j, i] = 0.0

        # --- Stage B: anchor-ordered back-edge removal --------------------
        depths = self._compute_depths()                  # (n,) float
        ordering = self._stable_argsort(depths)          # (n,) int
        rank = np.empty(n, dtype=np.int64)
        rank[ordering] = np.arange(n)

        back_edges: list[tuple[int, int]] = []
        l_back = 0.0
        for t in range(n):
            for s in range(n):
                w = float(M[t, s])
                if w == 0.0:
                    continue
                if rank[s] > rank[t]:  # source ranks later than target → back-edge
                    l_back += abs(w)
                    back_edges.append((t, s))
                    M[t, s] = 0.0

        self._sn_adj_dag = M
        self._l_collapse = float(l_collapse)
        self._l_back = float(l_back)
        self._depths = depths
        self._ordering = ordering
        self._back_edges = back_edges

    def _compute_depths(self) -> np.ndarray:
        """Median member-layer per supernode. emb → −∞, logit → +∞ (forced sources/sinks)."""
        from summarization.utils import layer_index_from_node
        depths = np.empty(len(self.supernodes), dtype=np.float64)
        for k, sn in enumerate(self.supernodes):
            if sn.type == "emb":
                depths[k] = -np.inf
            elif sn.type == "logit":
                depths[k] = np.inf
            else:
                layers = [layer_index_from_node(n) for n in sn.features]
                depths[k] = float(np.median(layers)) if layers else 0.0
        return depths

    def _stable_argsort(self, depths: np.ndarray) -> np.ndarray:
        """Sort by depth; ties → mean layer → min ctx_idx → supernode index."""
        from summarization.utils import layer_index_from_node
        keys: list[tuple[float, float, int, int]] = []
        for k, sn in enumerate(self.supernodes):
            if sn.features:
                layers = [layer_index_from_node(n) for n in sn.features]
                mean_layer = float(np.mean(layers))
                min_ctx = min(n.ctx_idx for n in sn.features)
            else:
                mean_layer = 0.0
                min_ctx = 0
            keys.append((float(depths[k]), mean_layer, int(min_ctx), k))
        return np.array(sorted(range(len(keys)), key=lambda i: keys[i]), dtype=np.int64)

    @property
    def sn_adj(self) -> np.ndarray:
        """Post-π supernode adjacency (acyclic). Canonical view used by all evals."""
        self._apply_pi()
        assert self._sn_adj_dag is not None
        return self._sn_adj_dag

    @property
    def l_collapse(self) -> float:
        """Total magnitude dropped by Stage A (antiparallel collapse)."""
        self._apply_pi()
        assert self._l_collapse is not None
        return self._l_collapse

    @property
    def l_back(self) -> float:
        """Total magnitude dropped by Stage B (back-edge removal)."""
        self._apply_pi()
        assert self._l_back is not None
        return self._l_back

    @property
    def depths(self) -> np.ndarray:
        """Anchor depth d(S) per supernode (used as y-coordinate for visualization)."""
        self._apply_pi()
        assert self._depths is not None
        return self._depths

    @property
    def ordering(self) -> np.ndarray:
        """Permutation giving the topological order π imposes on supernodes."""
        self._apply_pi()
        assert self._ordering is not None
        return self._ordering

    @property
    def back_edges(self) -> list[tuple[int, int]]:
        """(target, source) supernode-index pairs that Stage B removed."""
        self._apply_pi()
        assert self._back_edges is not None
        return self._back_edges