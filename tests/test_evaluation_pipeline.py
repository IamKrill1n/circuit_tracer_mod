from __future__ import annotations

import argparse
import json

import torch

from eval.eval_cluster import run_evaluation
from summarization.cluster import clusters_to_supernodes
from summarization.prune import PruneGraph, save_prune_graph
from summarization.summarize import SummaryGraph
from summarization.utils import _node_from_json_dict


def _build_test_graph() -> PruneGraph:
    node_specs: list[tuple[str, dict]] = [
        ("E_0_0", {"feature_type": "embedding", "is_target_logit": False, "layer": "E"}),
        ("1_0_0", {"feature_type": "sae_feature", "is_target_logit": False, "layer": "1"}),
        ("1_1_0", {"feature_type": "sae_feature", "is_target_logit": False, "layer": "1"}),
        ("2_0_0", {"feature_type": "sae_feature", "is_target_logit": False, "layer": "2"}),
        ("2_1_0", {"feature_type": "sae_feature", "is_target_logit": False, "layer": "2"}),
        ("27_0_0", {"feature_type": "logit", "is_target_logit": True, "layer": "27"}),
    ]
    nodes = [_node_from_json_dict({"node_id": nid, **spec}) for nid, spec in node_specs]

    pruned_adj = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.4, 0.0, 0.1, 0.0, 0.0, 0.0],
            [0.3, 0.2, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.7, 0.6, 0.0, 0.2, 0.0],
            [0.0, 0.6, 0.7, 0.3, 0.0, 0.0],
            [0.0, 0.1, 0.2, 0.8, 0.9, 0.0],
        ],
        dtype=torch.float32,
    )

    edge_relevance = pruned_adj.clone() * 0.8
    edge_influence = pruned_adj.clone() * 1.2
    node_relevance = torch.tensor([0.0, 0.4, 0.45, 0.7, 0.75, 0.0], dtype=torch.float32)
    node_influence = torch.tensor([0.0, 0.3, 0.35, 0.8, 0.85, 0.0], dtype=torch.float32)
    return PruneGraph(
        nodes=nodes,
        pruned_adj=pruned_adj,
        metadata={},
        node_influence=node_influence,
        node_relevance=node_relevance,
        edge_influence=edge_influence,
        edge_relevance=edge_relevance,
    )


def test_evaluation_pipeline_writes_summary_and_runs_all_solvers(tmp_path) -> None:
    # eval.eval_cluster discovers files matching *_prune_graph.pt.
    graph_path = tmp_path / "toy_prune_graph.pt"
    save_prune_graph(_build_test_graph(), str(graph_path))

    output_dir = tmp_path / "eval"
    args = argparse.Namespace(
        input_path=[str(graph_path)],
        output_dir=str(output_dir),
        map_location="cpu",
        node_threshold=None,
        max_layer_span=4,
        ilp_theta="p65",
        theta_sweep=None,
        ilp_eps_causal=0.05,
        ilp_max_sn=20,
        ilp_time_limit=30.0,
        random_state=42,
        n_init=5,
    )

    result = run_evaluation(args)

    assert result["n_graphs"] == 1
    assert result["n_runs"] == 5  # ILP plus four matched-K baselines
    assert (output_dir / "summary.csv").exists()
    assert (output_dir / "method_means.csv").exists()
    assert (output_dir / "results.json").exists()
    assert (output_dir / "manifest.json").exists()

    rows = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    assert len(rows) == 5
    methods = {row["method"] for row in rows}
    assert {
        "ours-ilp",
        "baseline-spectral-cosine",
        "baseline-kmeans",
        "baseline-spectral-adj",
        "baseline-random-same-size",
    } <= methods

    for row in rows:
        # Every method row carries the plan's per-graph metrics.
        for key in (
            "matched_k",
            "role_gap",
            "signed_up_gap",
            "signed_down_gap",
            "C_causal",
            "dag_loss",
            "external_loss",
            "L_atom",
            "final_retained_mass_fraction",
        ):
            assert key in row, f"missing {key} in method row {row.get('method')}"
        assert row["result_path"]

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["methods"]) == methods
    assert manifest["theta_sweep"] == []


def test_evaluation_pipeline_discovers_numeric_prune_graph_files(tmp_path) -> None:
    graph_path = tmp_path / "000.pt"
    save_prune_graph(_build_test_graph(), str(graph_path))

    output_dir = tmp_path / "eval"
    args = argparse.Namespace(
        input_path=[str(tmp_path)],
        output_dir=str(output_dir),
        map_location="cpu",
        node_threshold=None,
        max_layer_span=4,
        ilp_theta="p65",
        theta_sweep=None,
        ilp_eps_causal=0.05,
        ilp_max_sn=20,
        ilp_time_limit=30.0,
        random_state=42,
        n_init=5,
        summary_graphs_dir=None,
    )

    result = run_evaluation(args)

    assert result["n_graphs"] == 1
    rows = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    assert {row["method"] for row in rows} == {
        "ours-ilp",
        "baseline-spectral-cosine",
        "baseline-kmeans",
        "baseline-spectral-adj",
        "baseline-random-same-size",
    }


def test_evaluation_pipeline_uses_saved_summary_graph_for_ours_ilp(tmp_path) -> None:
    prune_graph = _build_test_graph()
    prune_dir = tmp_path / "pruned"
    summary_dir = tmp_path / "summary"
    prune_dir.mkdir()
    summary_dir.mkdir()
    graph_path = prune_dir / "000.pt"
    save_prune_graph(prune_graph, str(graph_path))

    clusters = [
        ["1_0_0", "1_1_0"],
        ["2_0_0"],
        ["2_1_0"],
        ["E_0_0"],
        ["27_0_0"],
    ]
    saved = SummaryGraph(
        supernodes=clusters_to_supernodes(prune_graph, clusters),
        pruned_adj=prune_graph.pruned_adj,
        metadata=prune_graph.metadata,
    )
    saved.save(str(summary_dir / "000.pt"))

    output_dir = tmp_path / "eval"
    args = argparse.Namespace(
        input_path=[str(prune_dir)],
        output_dir=str(output_dir),
        map_location="cpu",
        node_threshold=None,
        max_layer_span=4,
        ilp_theta="p65",
        theta_sweep=None,
        ilp_eps_causal=0.05,
        ilp_max_sn=20,
        ilp_time_limit=0.001,
        random_state=42,
        n_init=5,
        summary_graphs_dir=str(summary_dir),
    )

    result = run_evaluation(args)

    assert result["n_graphs"] == 1
    assert result["n_runs"] == 5
    rows = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    ours = next(row for row in rows if row["method"] == "ours-ilp")
    assert ours["matched_k"] == 3
    assert all(row["matched_k"] == 3 for row in rows)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["summary_graphs_dir"] == str(summary_dir.resolve())
