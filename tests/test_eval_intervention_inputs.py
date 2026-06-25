from __future__ import annotations

from pathlib import Path

import pytest
import torch

from eval.eval_intervention import (
    _baseline_k_from_summary_graph,
    _feature_supernode_count,
    _prune_graph_stem,
    _summary_graph_name,
    _summary_graph_path_for_prune_graph,
    _summary_graph_paths,
    evaluate_summary_graph,
)
from summarization.summarize import SummaryGraph, Supernode
from summarization.summarize import Node


def test_summary_graph_paths_discovers_nested_summary_graphs(tmp_path: Path) -> None:
    nested = tmp_path / "entmax" / "alpha_0.50" / "node_0.02"
    nested.mkdir(parents=True)
    first = nested / "000_summary_graph.pt"
    second = nested / "001_summary_graph.pt"
    ignored = nested / "001_prune_graph.pt"
    first.write_bytes(b"")
    second.write_bytes(b"")
    ignored.write_bytes(b"")

    assert _summary_graph_paths(None, tmp_path) == [first, second]


def test_summary_graph_paths_accepts_single_file(tmp_path: Path) -> None:
    path = tmp_path / "000_summary_graph.pt"
    path.write_bytes(b"")

    assert _summary_graph_paths(path, None) == [path]


def test_summary_graph_name_uses_relative_path_to_avoid_collisions(tmp_path: Path) -> None:
    path = tmp_path / "entmax" / "alpha_0.50" / "node_0.02" / "000_summary_graph.pt"

    assert _summary_graph_name(path, tmp_path) == "entmax__alpha_0.50__node_0.02__000"


def test_evaluate_summary_graph_requires_prompt_metadata() -> None:
    sng = SummaryGraph(supernodes=[], pruned_adj=torch.zeros((0, 0)), metadata={})

    with pytest.raises(ValueError, match="metadata\\['prompt'\\]"):
        evaluate_summary_graph(
            model=object(),  # type: ignore[arg-type]
            sng=sng,
            graph_name="missing_prompt",
            method_name="summary",
            factor=-1.0,
            layers_below=0,
            layers_above=1,
        )


def test_prune_graph_stem_strips_suffix() -> None:
    assert _prune_graph_stem(Path("000_prune_graph.pt")) == "000"
    assert _prune_graph_stem(Path("/data/042_prune_graph.pt")) == "042"


def test_summary_graph_path_for_prune_graph_pairs_by_stem(tmp_path: Path) -> None:
    summary_dir = tmp_path / "summary"
    summary_dir.mkdir()
    prune_path = tmp_path / "007_prune_graph.pt"
    prune_path.write_bytes(b"")

    assert _summary_graph_path_for_prune_graph(prune_path, summary_dir) == (
        summary_dir / "007_summary_graph.pt"
    )


def _make_clt_node(node_id: str = "0_1", ctx_idx: int = 0) -> Node:
    return Node(
        node_id=node_id,
        node_idx=0,
        feature=1,
        layer="0",
        ctx_idx=ctx_idx,
        feature_type="cross layer transcoder",
        influence=1.0,
        relevance=1.0,
    )


def test_feature_supernode_count_excludes_emb_and_logit() -> None:
    node = _make_clt_node()
    sng = SummaryGraph(
        supernodes=[
            Supernode(name="SN_0", features=[node], type="feature", layer_min=0, layer_max=0),
            Supernode(name="SN_emb", features=[], type="emb", layer_min=0, layer_max=0),
            Supernode(name="SN_logit", features=[], type="logit", layer_min=0, layer_max=0),
        ],
        pruned_adj=torch.zeros((3, 3)),
    )

    assert _feature_supernode_count(sng) == 1


def test_baseline_k_from_summary_graph_reads_feature_supernode_count(tmp_path: Path) -> None:
    summary_dir = tmp_path / "summary"
    summary_dir.mkdir()
    prune_path = tmp_path / "000_prune_graph.pt"
    prune_path.write_bytes(b"")

    node_a = _make_clt_node("0_1", 0)
    node_b = _make_clt_node("1_2", 0)
    sng = SummaryGraph(
        supernodes=[
            Supernode(name="SN_0", features=[node_a], type="feature", layer_min=0, layer_max=0),
            Supernode(name="SN_1", features=[node_b], type="feature", layer_min=1, layer_max=1),
            Supernode(name="SN_emb", features=[], type="emb", layer_min=0, layer_max=0),
        ],
        pruned_adj=torch.zeros((3, 3)),
        metadata={"prompt": "<bos> test"},
    )
    sng.save(str(summary_dir / "000_summary_graph.pt"))

    assert _baseline_k_from_summary_graph(prune_path, summary_dir) == 2


def test_baseline_k_from_summary_graph_missing_file_raises(tmp_path: Path) -> None:
    summary_dir = tmp_path / "summary"
    summary_dir.mkdir()
    prune_path = tmp_path / "000_prune_graph.pt"
    prune_path.write_bytes(b"")

    with pytest.raises(FileNotFoundError, match="000_summary_graph.pt"):
        _baseline_k_from_summary_graph(prune_path, summary_dir)
