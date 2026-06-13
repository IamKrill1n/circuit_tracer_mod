from __future__ import annotations

from pathlib import Path

import pytest
import torch

from eval.eval_intervention import _summary_graph_name, _summary_graph_paths, evaluate_summary_graph
from summarization.summarize import SummaryGraph


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
