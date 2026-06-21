"""Minimal summary-graph helpers needed by the entity-swap eval."""

from __future__ import annotations

from summarization.summarize import Node


def layer_index_from_node(node: Node) -> int:
    """Layer index from a typed ``Node``."""
    layer_val = node.layer
    if isinstance(layer_val, int):
        return layer_val
    if isinstance(layer_val, str) and layer_val.isdigit():
        return int(layer_val)
    node_id = node.node_id
    if node_id.startswith("E"):
        return -1
    try:
        return int(node_id.split("_")[0])
    except (ValueError, IndexError):
        return 10_000


def layer_index_from_node_id(node_id: str, *, layer: str | int | None = None) -> int:
    """Layer index when only an id string is known."""
    if isinstance(layer, int):
        return layer
    if isinstance(layer, str) and layer.isdigit():
        return int(layer)
    if node_id.startswith("E"):
        return -1
    try:
        return int(node_id.split("_")[0])
    except (ValueError, IndexError):
        return 10_000
