from __future__ import annotations

import argparse
import json

import torch

from eval.eval_cluster import run_evaluation
from summarization.prune import PruneGraph, save_prune_graph
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
        enforce_dag=False,
        ilp_theta="p65",
        ilp_eps_causal=0.05,
        ilp_max_sn=20,
        ilp_time_limit=30.0,
        random_state=42,
        n_init=5,
        lambda_causal=1.0,
    )

    result = run_evaluation(args)

    assert result["n_graphs"] == 1
    assert result["n_runs"] == 5  # ILP plus four matched-K baselines
    assert (output_dir / "summary.csv").exists()
    assert (output_dir / "results.json").exists()
    assert (output_dir / "manifest.json").exists()

    rows = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    assert len(rows) == 5
    solvers = {row["solver"] for row in rows}
    assert {
        "ours-ilp",
        "baseline-spectral-cosine",
        "baseline-kmeans",
        "baseline-spectral-adj",
        "baseline-random-same-size",
    } <= solvers

    for row in rows:
        # Every solver row should carry the eval-local Stage-2 objective terms.
        for key in (
            "matched_k",
            "L",
            "L_atom",
            "L_atom_norm",
            "sil_norm",
            "C_causal",
            "internalized_mass_fraction",
            "dag_removed_mass_fraction",
            "final_retained_mass_fraction",
            "prune_loss",
        ):
            assert key in row, f"missing {key} in solver row {row.get('solver')}"
        assert row["result_path"]

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    lambdas = manifest["lambdas"]
    assert abs(lambdas["lambda_causal"] - 1.0) < 1e-9
