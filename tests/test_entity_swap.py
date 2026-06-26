from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

import eval.eval_entity_swap as entity_swap
from eval.eval_entity_swap import (
    GraphRecord,
    RELATION_NAMES,
    build_parser,
    _dedup_donor_features,
    _donor_interventions,
    _eligible_ordered_pairs,
    _filter_ordered_pairs,
    _load_pair_list,
    _numeric_summary_paths,
    _relation_idx,
    _sample_ordered_pairs,
    _select_output_clt_nodes,
    _skip_pair_reason,
    _source_interventions,
    _summary_rows,
    _token_matches,
)
from summarization.summarize import Node, SummaryGraph, Supernode


def test_help_does_not_import_torch(tmp_path: Path) -> None:
    (tmp_path / "sitecustomize.py").write_text(
        """
import builtins

_real_import = builtins.__import__

def _blocked_import(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise ModuleNotFoundError("No module named 'torch'")
    return _real_import(name, *args, **kwargs)

builtins.__import__ = _blocked_import
""",
        encoding="utf-8",
    )
    pythonpath = str(tmp_path)
    if os.environ.get("PYTHONPATH"):
        pythonpath = f"{pythonpath}{os.pathsep}{os.environ['PYTHONPATH']}"

    result = subprocess.run(
        [sys.executable, "eval/eval_entity_swap.py", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONPATH": pythonpath},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--dtype {float32,float16,bfloat16}" in result.stdout


def _node(
    node_id: str,
    node_idx: int,
    *,
    feature_type: str = "cross layer transcoder",
    ctx_idx: int | None = None,
    feature: int = 0,
    activation: float | None = None,
    is_target_logit: bool = False,
) -> Node:
    parts = node_id.split("_")
    inferred_ctx = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else 0
    return Node(
        node_id=node_id,
        node_idx=node_idx,
        feature=feature,
        layer=parts[0],
        ctx_idx=inferred_ctx if ctx_idx is None else ctx_idx,
        feature_type=feature_type,
        activation=activation,
        is_target_logit=is_target_logit,
    )


def _sng(supernodes: list[Supernode]) -> SummaryGraph:
    max_idx = max((node.node_idx for sn in supernodes for node in sn.features), default=-1)
    return SummaryGraph(
        supernodes,
        torch.zeros((max_idx + 1, max_idx + 1)),
        metadata={"prompt": "A is to B as C is to", "prompt_tokens": ["A", " B", " C"]},
    )


def _record(
    idx: int,
    target_id: int,
    *,
    output_status: str = "ok",
    output_clt_nodes: list[Node] | None = None,
    donor_features: dict[tuple[int, int], float] | None = None,
) -> GraphRecord:
    return GraphRecord(
        idx=idx,
        relation_idx=_relation_idx(idx),
        relation_name=RELATION_NAMES[_relation_idx(idx)],
        raw_analogy=f"raw {idx}",
        path=Path(f"{idx:03d}.sng.pt"),
        sng=_sng([]),
        prompt=f"prompt {idx}",
        prompt_tokens=["a", " b"],
        target_id=target_id,
        target_clerp=f"target {target_id}",
        output_clt_nodes=[] if output_clt_nodes is None else output_clt_nodes,
        output_status=output_status,
        donor_features={} if donor_features is None else donor_features,
    )


def test_numeric_summary_paths_exclude_unnumbered_summaries(tmp_path: Path) -> None:
    for name in ["001.sng.pt", "Austin.sng.pt", "000.sng.pt", "002.pt", "abc.sng.pt"]:
        (tmp_path / name).write_bytes(b"")

    assert [path.name for path in _numeric_summary_paths(tmp_path)] == [
        "000.sng.pt",
        "001.sng.pt",
    ]


def test_relation_idx_uses_ten_item_blocks() -> None:
    assert _relation_idx(0) == 0
    assert _relation_idx(9) == 0
    assert _relation_idx(10) == 1
    assert _relation_idx(99) == 9


def test_output_selection_combines_output_supernodes_and_skips_missing() -> None:
    output_a = _node("1_10_0", 0)
    output_b = _node("2_20_1", 1)
    abstract = _node("3_30_1", 2)
    logit = _node("30_999_0", 3, feature_type="logit", feature=999, is_target_logit=True)
    sng = _sng(
        [
            Supernode("Output A", [output_a], "features", 1, 1, role="Output"),
            Supernode("Abstract", [abstract], "features", 3, 3, role="Abstract"),
            Supernode("Output B", [output_b], "features", 2, 2, role="Output"),
            Supernode("Target", [logit], "logit", 30, 30, role="Output"),
        ]
    )

    nodes, status = _select_output_clt_nodes(sng)

    assert status == "ok"
    assert [node.node_id for node in nodes] == ["1_10_0", "2_20_1"]

    missing_sng = _sng([Supernode("Abstract", [abstract], "features", 3, 3, role="Abstract")])
    missing_nodes, missing_status = _select_output_clt_nodes(missing_sng)
    assert missing_nodes == []
    assert missing_status == "missing_output_role"


def test_source_interventions_use_negative_clean_activation() -> None:
    clean_activations = torch.zeros((3, 4, 32))
    clean_activations[1, 0, 10] = 1.5
    clean_activations[2, 3, 20] = -2.0
    nodes = [_node("1_10_0", 0), _node("2_20_3", 1)]

    interventions = _source_interventions(nodes, clean_activations, source_factor=-4.0)

    assert interventions == [(1, 0, 10, -6.0), (2, 3, 20, 8.0)]


def test_donor_interventions_use_stored_activation_at_recipient_final_position() -> None:
    nodes = [
        _node("1_10_0", 0, activation=1.0),
        _node("1_10_1", 1, activation=-3.0),
        _node("2_5_0", 2, activation=0.5),
        _node("2_7_0", 3, activation=None),
    ]

    donor_features = _dedup_donor_features(nodes)
    interventions = _donor_interventions(donor_features, target_pos=9, donor_factor=2.0)

    assert donor_features == {(1, 10): -3.0, (2, 5): 0.5}
    assert interventions == [(1, 9, 10, -6.0), (2, 9, 5, 1.0)]


def test_same_target_pairs_are_skipped() -> None:
    source = _record(0, target_id=123, donor_features={(1, 10): 1.0})
    donor = _record(1, target_id=123, donor_features={(2, 20): 2.0})

    assert _skip_pair_reason(source, donor) == "same_target_token"


def test_token_match_accepts_uppercase_variant() -> None:
    assert _token_matches(10, " Australia", 11, " australia")
    assert _token_matches(10, " australia", 10, " greece")
    assert not _token_matches(10, " Lebanon", 11, " le")


def test_default_factor_grid_excludes_plus_minus_one() -> None:
    args = build_parser().parse_args([])

    assert args.negation_coefficients == "-2"
    assert args.addition_coefficients == "2,4,8"
    assert args.sample_pairs_per_relation is None
    assert args.random_state == 42


def test_sample_ordered_pairs_uses_only_eligible_pairs_and_is_stable_by_relation() -> None:
    records = [
        _record(30, target_id=10, donor_features={(1, 10): 1.0}),
        _record(31, target_id=11, donor_features={(1, 11): 1.0}),
        _record(32, target_id=11, donor_features={(1, 12): 1.0}),
        _record(33, target_id=13, output_status="missing_output_role"),
    ]

    pairs = _eligible_ordered_pairs(records)

    assert [(source.idx, donor.idx) for source, donor in pairs] == [
        (30, 31),
        (30, 32),
        (31, 30),
        (32, 30),
    ]
    assert _sample_ordered_pairs(pairs, None, random_state=42, relation_idx=3) == pairs
    assert _sample_ordered_pairs(pairs, 20, random_state=42, relation_idx=3) == pairs
    sample = _sample_ordered_pairs(pairs, 2, random_state=42, relation_idx=3)

    assert len(sample) == 2
    assert sample == _sample_ordered_pairs(pairs, 2, random_state=42, relation_idx=3)
    assert sample != _sample_ordered_pairs(pairs, 2, random_state=42, relation_idx=4)


def test_pair_list_filters_eligible_ordered_pairs(tmp_path: Path) -> None:
    path = tmp_path / "pairs.csv"
    path.write_text("source_idx,donor_idx\n0,2\n2,0\n9,8\n", encoding="utf-8")
    pair_list = _load_pair_list(path)
    records = [
        _record(0, target_id=4, donor_features={(0, 1): 1.0}),
        _record(1, target_id=5, donor_features={(0, 2): 1.0}),
        _record(2, target_id=6, donor_features={(0, 3): 1.0}),
    ]

    filtered = _filter_ordered_pairs(_eligible_ordered_pairs(records), pair_list)

    assert [(source.idx, donor.idx) for source, donor in filtered] == [(0, 2), (2, 0)]


def test_summary_rows_group_by_relation_and_coefficient() -> None:
    result_rows = [
        {
            "relation_idx": 0,
            "relation_name": RELATION_NAMES[0],
            "source_factor": -1.0,
            "donor_factor": 1.0,
            "clean_top1_is_source": 1,
            "clean_top1_is_source_exact": 1,
            "success": 1,
            "success_exact": 1,
            "top1_is_donor": 1,
            "top1_is_donor_exact": 1,
            "top5_has_donor": 1,
            "top5_has_donor_exact": 1,
            "p_source_clean": 0.8,
            "p_source_steered": 0.2,
            "p_donor_clean": 0.1,
            "p_donor_steered": 0.7,
        },
        {
            "relation_idx": 0,
            "relation_name": RELATION_NAMES[0],
            "source_factor": -1.0,
            "donor_factor": 1.0,
            "clean_top1_is_source": 1,
            "clean_top1_is_source_exact": 1,
            "success": 0,
            "success_exact": 0,
            "top1_is_donor": 0,
            "top1_is_donor_exact": 0,
            "top5_has_donor": 1,
            "top5_has_donor_exact": 0,
            "p_source_clean": 0.6,
            "p_source_steered": 0.5,
            "p_donor_clean": 0.2,
            "p_donor_steered": 0.3,
        },
        {
            "relation_idx": 0,
            "relation_name": RELATION_NAMES[0],
            "source_factor": -1.0,
            "donor_factor": 1.0,
            "clean_top1_is_source": 0,
            "clean_top1_is_source_exact": 0,
            "success": 0,
            "success_exact": 0,
            "top1_is_donor": 1,
            "top1_is_donor_exact": 0,
            "top5_has_donor": 1,
            "top5_has_donor_exact": 0,
            "p_source_clean": 0.4,
            "p_source_steered": 0.1,
            "p_donor_clean": 0.3,
            "p_donor_steered": 0.6,
        },
    ]
    skip_rows = [
        {
            "relation_idx": 0,
            "relation_name": RELATION_NAMES[0],
            "source_factor": -1.0,
            "donor_factor": 1.0,
            "reason": "same_target_token",
        },
        {
            "relation_idx": 0,
            "relation_name": RELATION_NAMES[0],
            "source_factor": -1.0,
            "donor_factor": 1.0,
            "reason": "donor_missing_output_role",
        },
    ]

    summary = _summary_rows(result_rows, skip_rows)

    assert len(summary) == 1
    row = summary[0]
    assert row["n_attempted"] == 3
    assert row["n_eligible"] == 2
    assert row["n_success"] == 1
    assert row["n_top1_hits"] == 2
    assert row["n_top1_hits_exact"] == 1
    assert row["n_top5_hits"] == 3
    assert row["n_top5_hits_exact"] == 1
    assert row["success_rate"] == pytest.approx(1 / 3)
    assert row["eligible_success_rate"] == pytest.approx(1 / 2)
    assert row["top1_hit_rate"] == pytest.approx(2 / 3)
    assert row["top5_hit_rate"] == pytest.approx(1.0)
    assert row["eligible_top5_hit_rate"] == pytest.approx(1.0)
    assert row["top1_is_donor_rate"] == pytest.approx(2 / 3)
    assert row["mean_p_source_clean"] == pytest.approx(0.6)
    assert row["mean_p_donor_steered"] == pytest.approx(0.5333333333333333)
    assert row["n_skipped"] == 2
    assert row["n_skipped_same_target_token"] == 1
    assert row["n_skipped_donor_missing_output_role"] == 1


def test_summary_rows_include_skip_only_groups() -> None:
    summary = _summary_rows(
        [],
        [
            {
                "relation_idx": 1,
                "relation_name": RELATION_NAMES[1],
                "source_factor": -2.0,
                "donor_factor": 2.0,
                "reason": "source_no_usable_clt_features",
            }
        ],
    )

    assert summary[0]["n_attempted"] == 0
    assert math.isnan(summary[0]["success_rate"])
    assert summary[0]["n_skipped_source_no_usable_clt_features"] == 1


def test_run_entity_swap_uses_tokenized_inputs_for_model_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _record(
        0,
        target_id=4,
        output_clt_nodes=[_node("1_3_1", 0)],
        donor_features={(1, 3): 4.0},
    )
    donor = _record(
        1,
        target_id=5,
        output_clt_nodes=[_node("0_2_1", 1)],
        donor_features={(0, 2): 1.5},
    )
    monkeypatch.setattr(entity_swap, "_load_graph_records", lambda *_args: [source, donor])

    class FakeModel:
        def __init__(self) -> None:
            self.tokenizer = self
            self.ensure_prompts: list[str] = []
            self.activation_inputs: list[torch.Tensor] = []
            self.intervention_inputs: list[torch.Tensor] = []
            self.constrained_layers: list[list[int]] = []
            self.activations = torch.zeros((2, 2, 8))
            self.activations[1, 1, 3] = 4.0
            self.activations[0, 1, 2] = 3.0

        def ensure_tokenized(self, prompt: str) -> torch.Tensor:
            self.ensure_prompts.append(prompt)
            return torch.tensor([0, len(self.ensure_prompts)])

        def decode(self, token_ids: list[int]) -> str:
            return f" tok{token_ids[0]}"

        def get_activations(
            self, inputs: torch.Tensor, sparse: bool = False
        ) -> tuple[torch.Tensor, torch.Tensor]:
            assert sparse is False
            self.activation_inputs.append(inputs)
            logits = torch.zeros((1, 2, 8))
            logits[0, -1, 4] = 5.0
            return logits, self.activations.clone()

        def feature_intervention(
            self,
            inputs: torch.Tensor,
            interventions: list[tuple[int, int, int, float]],
            constrained_layers: range | None = None,
            freeze_attention: bool = True,
            return_activations: bool = False,
        ) -> tuple[torch.Tensor, None]:
            assert freeze_attention is True
            assert return_activations is False
            assert isinstance(inputs, torch.Tensor)
            assert constrained_layers is not None
            self.intervention_inputs.append(inputs)
            self.constrained_layers.append(list(constrained_layers))
            assert interventions
            logits = torch.zeros((1, 2, 8))
            logits[0, -1, 5] = 6.0
            return logits, None

    model = FakeModel()
    args = argparse.Namespace(
        negation_coefficients="-2",
        addition_coefficients="2",
        relations="0",
        graph_dir=tmp_path,
        analogies_file=tmp_path / "bats.txt",
        output_dir=tmp_path / "out",
        layers_below=0,
        layers_above=1,
    )

    entity_swap.run_entity_swap(model, args)  # type: ignore[arg-type]

    assert model.ensure_prompts == ["prompt 0", "prompt 1"]
    assert all(isinstance(inputs, torch.Tensor) for inputs in model.activation_inputs)
    assert all(isinstance(inputs, torch.Tensor) for inputs in model.intervention_inputs)
    assert model.constrained_layers
    assert all(window for window in model.constrained_layers)


def test_run_entity_swap_samples_eligible_pairs_before_interventions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [
        _record(
            0,
            target_id=4,
            output_clt_nodes=[_node("0_1_1", 0)],
            donor_features={(0, 1): 1.0},
        ),
        _record(
            1,
            target_id=5,
            output_clt_nodes=[_node("0_2_1", 1)],
            donor_features={(0, 2): 1.0},
        ),
        _record(
            2,
            target_id=6,
            output_clt_nodes=[_node("0_3_1", 2)],
            donor_features={(0, 3): 1.0},
        ),
    ]
    monkeypatch.setattr(entity_swap, "_load_graph_records", lambda *_args: records)

    class FakeModel:
        def __init__(self) -> None:
            self.tokenizer = self
            self.intervention_count = 0
            self.activations = torch.ones((1, 2, 8))

        def ensure_tokenized(self, prompt: str) -> torch.Tensor:
            idx = int(prompt.split()[-1])
            return torch.tensor([0, idx])

        def decode(self, token_ids: list[int]) -> str:
            return f" tok{token_ids[0]}"

        def get_activations(
            self, inputs: torch.Tensor, sparse: bool = False
        ) -> tuple[torch.Tensor, torch.Tensor]:
            logits = torch.zeros((1, 2, 8))
            logits[0, -1, int(inputs[-1]) + 4] = 5.0
            return logits, self.activations.clone()

        def feature_intervention(
            self,
            inputs: torch.Tensor,
            interventions: list[tuple[int, int, int, float]],
            constrained_layers: range | None = None,
            freeze_attention: bool = True,
            return_activations: bool = False,
        ) -> tuple[torch.Tensor, None]:
            self.intervention_count += 1
            logits = torch.zeros((1, 2, 8))
            logits[0, -1, 5] = 6.0
            return logits, None

    model = FakeModel()
    args = argparse.Namespace(
        negation_coefficients="-2",
        addition_coefficients="2",
        relations="0",
        graph_dir=tmp_path,
        analogies_file=tmp_path / "bats.txt",
        output_dir=tmp_path / "out",
        layers_below=0,
        layers_above=1,
        sample_pairs_per_relation=1,
        random_state=42,
    )

    entity_swap.run_entity_swap(model, args)  # type: ignore[arg-type]

    result_lines = (tmp_path / "out" / "swap_results.csv").read_text(encoding="utf-8").splitlines()
    assert len(result_lines) == 2
    assert model.intervention_count == 1


def test_run_entity_swap_pair_list_restricts_interventions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [
        _record(
            0,
            target_id=4,
            output_clt_nodes=[_node("0_1_1", 0)],
            donor_features={(0, 1): 1.0},
        ),
        _record(
            1,
            target_id=5,
            output_clt_nodes=[_node("0_2_1", 1)],
            donor_features={(0, 2): 1.0},
        ),
        _record(
            2,
            target_id=6,
            output_clt_nodes=[_node("0_3_1", 2)],
            donor_features={(0, 3): 1.0},
        ),
    ]
    monkeypatch.setattr(entity_swap, "_load_graph_records", lambda *_args: records)
    pair_list = tmp_path / "pairs.csv"
    pair_list.write_text("source_idx,donor_idx\n0,2\n", encoding="utf-8")

    class FakeModel:
        def __init__(self) -> None:
            self.tokenizer = self
            self.intervention_count = 0
            self.activations = torch.ones((1, 2, 8))

        def ensure_tokenized(self, prompt: str) -> torch.Tensor:
            idx = int(prompt.split()[-1])
            return torch.tensor([0, idx])

        def decode(self, token_ids: list[int]) -> str:
            return f" tok{token_ids[0]}"

        def get_activations(
            self, inputs: torch.Tensor, sparse: bool = False
        ) -> tuple[torch.Tensor, torch.Tensor]:
            logits = torch.zeros((1, 2, 8))
            logits[0, -1, int(inputs[-1]) + 4] = 5.0
            return logits, self.activations.clone()

        def feature_intervention(
            self,
            inputs: torch.Tensor,
            interventions: list[tuple[int, int, int, float]],
            constrained_layers: range | None = None,
            freeze_attention: bool = True,
            return_activations: bool = False,
        ) -> tuple[torch.Tensor, None]:
            self.intervention_count += 1
            logits = torch.zeros((1, 2, 8))
            logits[0, -1, 6] = 6.0
            return logits, None

    model = FakeModel()
    args = argparse.Namespace(
        negation_coefficients="-2",
        addition_coefficients="2",
        relations="0",
        graph_dir=tmp_path,
        analogies_file=tmp_path / "bats.txt",
        output_dir=tmp_path / "out",
        layers_below=0,
        layers_above=1,
        pair_list=pair_list,
    )

    entity_swap.run_entity_swap(model, args)  # type: ignore[arg-type]

    result_lines = (tmp_path / "out" / "swap_results.csv").read_text(encoding="utf-8").splitlines()
    assert len(result_lines) == 2
    assert model.intervention_count == 1
