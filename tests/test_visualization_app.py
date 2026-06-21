from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from summarization.summarize import Node, SummaryGraph, Supernode
from visualization_app import services
from visualization_app.server import app


def _node(
    node_id: str,
    node_idx: int,
    feature_type: str = "cross layer transcoder",
    activation: float | None = None,
) -> Node:
    return Node(
        node_id=node_id,
        node_idx=node_idx,
        feature=node_idx,
        layer=str(node_idx),
        ctx_idx=0,
        feature_type=feature_type,
        activation=activation,
    )


def test_slugify_and_graph_paths(tmp_path: Path) -> None:
    slug = services.slugify("Gemma graph: Austin!")

    assert slug == "Gemma-graph--Austin"
    assert services.graph_dir(slug, tmp_path) == tmp_path / slug
    assert services.graph_json_path(slug, tmp_path) == tmp_path / slug / f"{slug}.json"
    assert services.sidecar_pt_path(slug, tmp_path) == tmp_path / f"{slug}.pt"


def test_list_graphs_reads_viewer_directories(tmp_path: Path) -> None:
    graph_root = tmp_path / "graph_files"
    pt_root = tmp_path / "generated_graphs"
    graph_dir = graph_root / "austin"
    graph_dir.mkdir(parents=True)
    pt_root.mkdir()
    (pt_root / "Austin.pt").write_bytes(b"pt")
    (pt_root / "austin.sng.pt").write_bytes(b"sng")
    (graph_dir / "austin.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "slug": "austin",
                    "scan": "CLT-HP",
                    "prompt": "The capital of Texas is",
                    "prompt_tokens": ["The", " capital"],
                },
                "nodes": [{"node_id": "E_0_0"}],
                "links": [{"source": "E_0_0", "target": "1_0_0"}],
            }
        ),
        encoding="utf-8",
    )

    records = services.list_graphs(graph_root, pt_root)

    assert len(records) == 1
    assert records[0].slug == "austin"
    assert records[0].has_pt is True
    assert records[0].has_summary is True
    assert records[0].node_count == 1
    assert records[0].link_count == 1


def test_summary_request_defaults_match_graph_workflow() -> None:
    from visualization_app.server import SummaryRequest

    req = SummaryRequest()

    assert req.token_weights_source == "shap"
    assert req.token_attr_normalize == "entmax"
    assert req.shap_values_path == "dataset/analogies/shap_values.json"
    assert req.node_threshold == 0.02
    assert req.edge_threshold == 0.9
    assert req.filter_act_density is True
    assert req.cluster_method == "ilp"
    assert req.max_layer_span == 7
    assert req.max_sn == 20
    assert req.eps_causal == 0.05
    assert req.theta == 0.0
    assert req.label_model == "gemma-4-31b-it"


def test_steering_request_defaults_match_streamlit_workflow() -> None:
    from visualization_app.server import SteeringRequest

    req = SteeringRequest(factors={"SN 1": -1.0})

    assert req.factors == {"SN 1": -1.0}
    assert req.model_name == ""
    assert req.transcoder == ""
    assert req.dtype == "bfloat16"
    assert req.backend == "transformerlens"
    assert req.freeze_attention is True
    assert req.layers_below == 0
    assert req.layers_above == 1
    assert req.edge_threshold == 0.1
    assert req.top_k == 5
    assert req.stored_supernodes == []

    stored_req = SteeringRequest(
        stored_supernodes=[{"record_id": "austin:0", "factor": 2.0, "target_pos": 3}]
    )
    assert stored_req.factors == {}
    assert stored_req.stored_supernodes[0].record_id == "austin:0"
    assert stored_req.stored_supernodes[0].factor == 2.0
    assert stored_req.stored_supernodes[0].target_pos == 3


def test_load_steering_model_suppresses_dependency_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenWriter:
        def write(self, _text: str) -> int:
            raise BrokenPipeError(32, "Broken pipe")

        def flush(self) -> None:
            raise BrokenPipeError(32, "Broken pipe")

    expected_model = object()
    captured_args: tuple[object, ...] | None = None
    captured_kwargs: dict[str, object] | None = None

    def fake_from_pretrained(*args, **kwargs):
        nonlocal captured_args, captured_kwargs
        print("dependency stdout")
        print("dependency stderr", file=sys.stderr)
        captured_args = args
        captured_kwargs = kwargs
        return expected_model

    services._load_steering_model.cache_clear()
    monkeypatch.setattr("circuit_tracer.ReplacementModel.from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(sys, "stdout", BrokenWriter())
    monkeypatch.setattr(sys, "stderr", BrokenWriter())

    try:
        model = services._load_steering_model(
            "model",
            "transcoder",
            "float32",
            "transformerlens",
        )
    finally:
        services._load_steering_model.cache_clear()

    assert model is expected_model
    assert captured_args == ("model", "transcoder")
    assert captured_kwargs is not None
    assert captured_kwargs["lazy_encoder"] is True
    assert captured_kwargs["backend"] == "transformerlens"


def test_steering_options_reads_feature_supernodes(tmp_path: Path) -> None:
    graph_root = tmp_path / "graph_files"
    pt_root = tmp_path / "generated_graphs"
    graph_dir = graph_root / "austin"
    graph_dir.mkdir(parents=True)
    pt_root.mkdir()
    (graph_dir / "austin.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "slug": "austin",
                    "scan": "CLT-HP",
                    "prompt": "viewer prompt",
                    "prompt_tokens": ["viewer"],
                },
                "nodes": [],
                "links": [],
            }
        ),
        encoding="utf-8",
    )
    sng = SummaryGraph(
        supernodes=[
            Supernode(
                name="State feature",
                features=[_node("1_10", 0), _node("2_20", 1, feature_type="embedding")],
                type="features",
                layer_min=1,
                layer_max=2,
                role="Abstract",
                description="Tracks the state.",
            ),
            Supernode(
                name="Target logit",
                features=[_node("L_0", 2, feature_type="logit")],
                type="logit",
                layer_min=99,
                layer_max=99,
            ),
        ],
        pruned_adj=torch.zeros((3, 3)),
        metadata={"prompt": "summary prompt"},
    )
    sng.save(str(pt_root / "austin.sng.pt"))

    options = services.steering_options("austin", graph_root, pt_root)

    assert options["prompt"] == "summary prompt"
    assert options["transcoder"] == "CLT-HP"
    assert options["model_name"] == ""
    assert options["supernodes"] == [
        {
            "name": "State feature",
            "role": "Abstract",
            "description": "Tracks the state.",
            "layer_min": 1,
            "layer_max": 2,
            "feature_count": 1,
        }
    ]


def test_supernode_storage_indexes_feature_supernodes_only(tmp_path: Path) -> None:
    graph_root = tmp_path / "graph_files"
    pt_root = tmp_path / "generated_graphs"
    pt_root.mkdir()
    sng = SummaryGraph(
        supernodes=[
            Supernode(
                name="Entity label",
                features=[
                    _node("1_10_0", 0, activation=3.0),
                    _node("2_20_0", 1, feature_type="embedding"),
                ],
                type="features",
                layer_min=1,
                layer_max=2,
                role="Input",
                description="Tracks the source entity.",
            ),
            Supernode(
                name="Output logit",
                features=[_node("L_0", 2, feature_type="logit")],
                type="logit",
                layer_min=99,
                layer_max=99,
            ),
        ],
        pruned_adj=torch.zeros((3, 3)),
        metadata={"prompt": "prompt", "prompt_tokens": ["prompt"], "scan": "CLT-HP"},
    )
    sng.save(str(pt_root / "austin.sng.pt"))

    payload = services.rebuild_supernode_storage(graph_root, pt_root)

    assert services.supernode_storage_path(pt_root).exists()
    assert payload["version"] == 1
    assert len(payload["records"]) == 1
    record = payload["records"][0]
    assert record["record_id"] == "austin:0"
    assert record["label"] == "Entity label"
    assert record["role"] == "Input"
    assert record["description"] == "Tracks the source entity."
    assert record["feature_count"] == 1
    assert record["transcoder"] == "CLT-HP"


def test_supernode_storage_filters_by_label_role_and_description(tmp_path: Path) -> None:
    pt_root = tmp_path / "generated_graphs"
    pt_root.mkdir()
    payload = {
        "version": 1,
        "records": [
            {
                "record_id": "a:0",
                "label": "Entity label",
                "role": "Input",
                "description": "Tracks the source entity.",
                "source_slug": "austin",
                "model_name": "google/gemma-2-2b",
                "transcoder": "mntss/clt-gemma-2-2b-2.5M",
            },
            {
                "record_id": "b:0",
                "label": "Relation label",
                "role": "Abstract",
                "description": "Combines relation evidence.",
                "source_slug": "boston",
                "model_name": "Qwen/Qwen3-4B",
                "transcoder": "mwhanna/qwen3-4b-transcoders",
            },
        ],
    }
    services.supernode_storage_path(pt_root).write_text(json.dumps(payload), encoding="utf-8")

    by_label = services.list_supernode_storage(label="entity", pt_root=pt_root)
    by_role = services.list_supernode_storage(role="abstract", pt_root=pt_root)
    by_description = services.list_supernode_storage(description="source", pt_root=pt_root)
    by_model = services.list_supernode_storage(model_name="google/gemma-2-2b", pt_root=pt_root)
    by_transcoder = services.list_supernode_storage(
        transcoder="mwhanna/qwen3-4b-transcoders", pt_root=pt_root
    )

    assert [record["record_id"] for record in by_label["records"]] == ["a:0"]
    assert [record["record_id"] for record in by_role["records"]] == ["b:0"]
    assert [record["record_id"] for record in by_description["records"]] == ["a:0"]
    assert [record["record_id"] for record in by_model["records"]] == ["a:0"]
    assert [record["record_id"] for record in by_transcoder["records"]] == ["b:0"]


def test_supernode_storage_endpoint_forwards_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_list_supernode_storage(**kwargs):
        captured.update(kwargs)
        return {
            "version": 1,
            "count": 1,
            "records": [
                {
                    "record_id": "austin:0",
                    "label": "Entity label",
                    "role": "Input",
                    "description": "Tracks the source entity.",
                    "source_slug": "austin",
                    "layer_min": 1,
                    "layer_max": 2,
                    "feature_count": 3,
                }
            ],
        }

    monkeypatch.setattr(services, "list_supernode_storage", fake_list_supernode_storage)
    client = TestClient(app)

    response = client.get(
        "/api/supernode-storage?label=Entity&role=Input&description=source"
        "&source_slug=austin&model_name=google%2Fgemma-2-2b"
        "&transcoder=mntss%2Fclt-gemma-2-2b-2.5M"
    )

    assert response.status_code == 200
    assert captured == {
        "label": "Entity",
        "role": "Input",
        "description": "source",
        "source_slug": "austin",
        "model_name": "google/gemma-2-2b",
        "transcoder": "mntss/clt-gemma-2-2b-2.5M",
    }
    assert response.json()["records"][0]["label"] == "Entity label"


def test_stored_supernode_intervention_rejects_model_mismatch(tmp_path: Path) -> None:
    pt_root = tmp_path / "generated_graphs"
    pt_root.mkdir()
    payload = {
        "version": 1,
        "records": [
            {
                "record_id": "donor:0",
                "source_slug": "donor",
                "source_path": str(pt_root / "donor.sng.pt"),
                "supernode_index": 0,
                "label": "Stored donor",
                "model_name": "model-a",
                "transcoder": "tc-a",
            }
        ],
    }
    services.supernode_storage_path(pt_root).write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="indexed for model"):
        services._stored_supernode_intervention_groups(
            [{"record_id": "donor:0", "factor": 1.0, "target_pos": 0}],
            n_pos=1,
            n_layers=1,
            d_transcoder=1,
            model_name="model-b",
            transcoder="tc-a",
            layers_below=0,
            layers_above=1,
            pt_root=pt_root,
        )


def test_token_weights_from_shap_uses_graph_target_token(monkeypatch: pytest.MonkeyPatch) -> None:
    from summarization.summarize import Node

    captured: dict[str, int | None] = {}

    def fake_get_token_attribution(**kwargs):
        captured["target_token_id"] = kwargs["target_token_id"]
        return torch.ones(2), torch.tensor([0.25, 0.75])

    def fake_build_index_sets(_nodes):
        return {"embedding": [0, 1]}

    monkeypatch.setattr(
        "summarization.token_attribution.get_token_attribution",
        fake_get_token_attribution,
    )
    monkeypatch.setattr("summarization.utils._build_index_sets", fake_build_index_sets)
    ag = type(
        "FakeAttrGraph",
        (),
        {
            "metadata": {"prompt": "A B", "prompt_tokens": ["A", " B"]},
            "nodes": [
                Node("E_0_0", 0, 0, "E", 0, "embedding"),
                Node("E_0_1", 1, 0, "E", 1, "embedding"),
                Node("L_0", 2, 1234, "L", 1, "logit", is_target_logit=True),
            ],
        },
    )()

    weights = services._token_weights_from_shap(
        ag,
        model_name="model",
        normalize_method="entmax",
        entmax_alpha=1.25,
        device="cpu",
    )

    assert captured["target_token_id"] == 1234
    assert weights == [0.25, 0.75]


def test_token_attribution_passes_explicit_target_to_shap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from summarization import token_attribution

    captured: dict[str, tuple] = {}

    class FakeTokenizer:
        def decode(self, ids):
            assert ids == [7]
            return " target"

        def __call__(self, text, add_special_tokens):
            assert text == " target"
            assert add_special_tokens is False
            return {"input_ids": [7]}

    class FakeExplanation:
        values = [2.0]
        feature_names = ["A"]

    class FakeExplainer:
        def __call__(self, *args, batch_size):
            captured["args"] = args
            captured["batch_size"] = batch_size
            return FakeExplanation()

    monkeypatch.setattr(token_attribution, "_cached_tokenizer", lambda _model_name: FakeTokenizer())
    monkeypatch.setattr(
        token_attribution,
        "_build_shap_lm_explainer",
        lambda **_kwargs: FakeExplainer(),
    )

    _raw, normalized = token_attribution.get_token_attribution(
        prompt="A",
        prompt_tokens=["A"],
        model_name="model",
        normalize_method="softmax",
        device="cpu",
        target_token_id=7,
    )

    assert captured["args"] == (["A"], [" target"])
    assert captured["batch_size"] == 1
    assert normalized.tolist() == [1.0]


def test_token_attribution_strips_graph_bos_before_pinned_shap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from summarization import token_attribution

    captured: dict[str, tuple] = {}

    class FakeTokenizer:
        bos_token = "<bos>"

    class FakeExplanation:
        values = [0.0, 2.0]
        feature_names = ["", "A"]

    class FakeExplainer:
        def __call__(self, *args, batch_size):
            captured["args"] = args
            captured["batch_size"] = batch_size
            return FakeExplanation()

    monkeypatch.setattr(token_attribution, "_cached_tokenizer", lambda _model_name: FakeTokenizer())
    monkeypatch.setattr(
        token_attribution,
        "_build_shap_lm_explainer",
        lambda **_kwargs: FakeExplainer(),
    )

    raw, normalized = token_attribution.get_token_attribution(
        prompt="<bos>A",
        prompt_tokens=["<bos>", "A"],
        model_name="model",
        normalize_method="softmax",
        device="cpu",
        pin_special_tokens=True,
    )

    assert captured["args"] == (["A"],)
    assert captured["batch_size"] == 1
    assert raw.tolist() == [0.0, 2.0]
    assert normalized.tolist() == [0.0, 1.0]


def test_format_generation_prompt_formats_qwen_with_chat_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_format_qwen_with_tokenizer(
        messages,
        *,
        model_name,
        add_generation_prompt,
        enable_thinking,
    ):
        captured["messages"] = messages
        captured["model_name"] = model_name
        captured["add_generation_prompt"] = add_generation_prompt
        captured["enable_thinking"] = enable_thinking
        return "formatted qwen prompt"

    monkeypatch.setattr(
        "attribute_utils.format_qwen_with_tokenizer",
        fake_format_qwen_with_tokenizer,
    )

    formatted = services._format_generation_prompt(
        prompt="Solve this.",
        model_name="Qwen/Qwen3-4B",
        qwen_system="System message.",
        qwen_enable_thinking=True,
    )

    assert formatted == "formatted qwen prompt"
    assert captured == {
        "messages": [
            {"role": "system", "content": "System message."},
            {"role": "user", "content": "Solve this."},
        ],
        "model_name": "Qwen/Qwen3-4B",
        "add_generation_prompt": True,
        "enable_thinking": True,
    }


def test_format_generation_prompt_uses_existing_qwen_assistant_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_format_qwen_with_tokenizer(messages, **kwargs):
        captured["messages"] = messages
        captured.update(kwargs)
        return "formatted qwen prompt"

    monkeypatch.setattr(
        "attribute_utils.format_qwen_with_tokenizer",
        fake_format_qwen_with_tokenizer,
    )

    services._format_generation_prompt(
        prompt="Question",
        model_name="qwen-local",
        qwen_system="",
        qwen_assistant="Partial answer",
    )

    assert captured["messages"] == [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Partial answer"},
    ]
    assert captured["add_generation_prompt"] is False
    assert captured["enable_thinking"] is False


def test_format_generation_prompt_leaves_non_qwen_prompt_unchanged() -> None:
    assert (
        services._format_generation_prompt(
            prompt="The capital of France is",
            model_name="google/gemma-2-2b",
            qwen_system="Ignored",
            qwen_assistant="Ignored",
            qwen_enable_thinking=True,
        )
        == "The capital of France is"
    )


def test_run_summary_delegates_core_work_to_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pt_root = tmp_path / "generated_graphs"
    pt_root.mkdir()
    pt_path = services.sidecar_pt_path("austin", pt_root)
    pt_path.write_bytes(b"placeholder")
    captured = {}

    def fake_run_pipeline(args):
        captured["graph_pt"] = args.graph_pt
        captured["method"] = args.method
        captured["auto_token_weights"] = args.auto_token_weights
        captured["summary_graph_out"] = args.summary_graph_out
        sng = SummaryGraph(
            [
                Supernode(
                    "SN_0",
                    [_node("0_0_0", 0), _node("0_1_0", 1)],
                    "features",
                    0,
                    0,
                )
            ],
            torch.zeros((2, 2)),
        )
        sng.save(args.summary_graph_out)
        return {
            "pruned_nodes": 2,
            "pruned_edges": 1,
            "resolved_k": 1,
            "auto_k_candidates": 0,
            "supernodes": [["SN_0", "0_0_0", "0_1_0"]],
            "supernode_map": {"SN_0": ["0_0_0", "0_1_0"]},
            "figure_html_out": None,
            "upload_status": None,
            "upload_body": None,
        }

    monkeypatch.setattr("summarization.pipeline.run_pipeline", fake_run_pipeline)

    result = services.run_summary(
        slug="austin",
        settings={
            "token_weights_source": "shap",
            "cluster_method": "ilp",
            "label_supernodes": False,
        },
        pt_root=pt_root,
    )

    assert captured == {
        "graph_pt": str(pt_path),
        "method": "ilp",
        "auto_token_weights": True,
        "summary_graph_out": str(services.summary_path("austin", pt_root)),
    }
    assert result["summary_path"] == str(services.summary_path("austin", pt_root))
    assert result["pruned_nodes"] == 2
    assert result["viewer"]["supernodes"] == [["SN_0", "0_0_0", "0_1_0"]]
    storage = json.loads(services.supernode_storage_path(pt_root).read_text(encoding="utf-8"))
    assert [record["label"] for record in storage["records"]] == ["SN_0"]


def test_run_steering_uses_current_and_stored_supernodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph_root = tmp_path / "graph_files"
    pt_root = tmp_path / "generated_graphs"
    current_dir = graph_root / "current"
    current_dir.mkdir(parents=True)
    pt_root.mkdir()
    (current_dir / "current.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "slug": "current",
                    "scan": "CLT-HP",
                    "prompt": "current prompt",
                    "prompt_tokens": ["current", " prompt", "."],
                },
                "nodes": [],
                "links": [],
            }
        ),
        encoding="utf-8",
    )
    current_sng = SummaryGraph(
        [
            Supernode(
                "Current SN",
                [_node("0_1_0", 0, activation=1.0)],
                "features",
                0,
                0,
            )
        ],
        torch.zeros((1, 1)),
        metadata={"prompt": "current prompt", "prompt_tokens": ["current", " prompt", "."]},
    )
    current_sng.save(str(pt_root / "current.sng.pt"))

    donor_sng = SummaryGraph(
        [
            Supernode(
                "Stored entity",
                [
                    _node("1_10_0", 0, activation=3.0),
                    _node("1_10_1", 1, activation=5.0),
                    _node("2_20_0", 2, feature_type="embedding"),
                ],
                "features",
                1,
                1,
                role="Input",
                description="Stored donor entity.",
            )
        ],
        torch.zeros((3, 3)),
        metadata={"prompt": "donor prompt", "prompt_tokens": ["donor"], "scan": "CLT-HP"},
    )
    donor_sng.save(str(pt_root / "donor.sng.pt"))
    services.rebuild_supernode_storage(graph_root, pt_root)

    class FakeTokenizer:
        def decode(self, token_ids):
            return f"tok{token_ids[0]}"

    class FakeModel:
        def __init__(self):
            self.tokenizer = FakeTokenizer()
            self.calls = []
            self.orig_activations = torch.zeros((3, 3, 32))
            self.orig_activations[0, 0, 1] = 5.0

        def ensure_tokenized(self, prompt):
            assert prompt == "current prompt"
            return torch.tensor([[0, 1, 2]])

        def get_activations(self, _tokens):
            return torch.zeros((1, 3, 8)), self.orig_activations.clone()

        def feature_intervention(
            self,
            _tokens,
            interventions,
            constrained_layers=None,
            freeze_attention=True,
            return_activations=False,
        ):
            self.calls.append(
                {
                    "interventions": list(interventions),
                    "constrained_layers": list(constrained_layers or []),
                    "freeze_attention": freeze_attention,
                    "return_activations": return_activations,
                }
            )
            logits = torch.zeros((1, 3, 8))
            logits[0, -1, 0] = 1.0
            return logits, None

    fake_model = FakeModel()
    monkeypatch.setattr(services, "_load_steering_model", lambda *_args: fake_model)

    result = services.run_steering(
        slug="current",
        factors={"Current SN": -1.0},
        stored_supernodes=[{"record_id": "donor:0", "factor": -1.0, "target_pos": 2}],
        model_name="model",
        transcoder="CLT-HP",
        top_k=2,
        root=graph_root,
        pt_root=pt_root,
    )

    intervention_calls = [
        call["interventions"] for call in fake_model.calls if call["interventions"]
    ]
    assert [(0, 0, 1, -5.0)] in intervention_calls
    assert [(1, 2, 10, -5.0)] in intervention_calls
    assert result["stored_supernodes"] == [
        {
            "record_id": "donor:0",
            "label": "Stored entity",
            "source_slug": "donor",
            "factor": -1.0,
            "target_pos": 2,
            "n_features": 1,
        }
    ]
    assert "figure_html" in result
    assert "Steering intervention graph" in result["figure_html"]
    assert "Current SN" in result["figure_html"]


def test_summary_graph_viewer_payload_respects_200_node_limit() -> None:
    supernodes = []
    node_idx = 0
    for sn_idx in range(3):
        members = []
        for member_idx in range(90):
            members.append(_node(f"{sn_idx}_{member_idx}_0", node_idx))
            node_idx += 1
        supernodes.append(Supernode(f"SN_{sn_idx}", members, "features", 0, 1))
    singleton = Supernode("SN_SINGLE", [_node("E_0_0", node_idx, "embedding")], "emb", -1, -1)
    supernodes.append(singleton)

    sng = SummaryGraph(supernodes, torch.zeros((node_idx + 1, node_idx + 1)))

    pinned_ids, grouped, stats = services.summary_graph_viewer_payload(sng, max_nodes=200)

    assert len(pinned_ids) == 181
    assert [group[0] for group in grouped] == ["SN_0", "SN_1"]
    assert stats["dropped_supernodes"] == 1
    assert stats["dropped_members"] == 90


def test_summary_graph_viewer_payload_includes_singleton_logit_supernodes() -> None:
    feature = Supernode(
        "SN_0",
        [_node("0_0_0", 0), _node("0_1_0", 1)],
        "features",
        0,
        0,
    )
    emb = Supernode("SN_EMB_0", [_node("E_0_0", 2, "embedding")], "emb", -1, -1)
    logit = Supernode(
        "SN_LOGIT_0",
        [_node("L_0", 3, "logit")],
        "logit",
        99,
        99,
    )
    sng = SummaryGraph([feature, emb, logit], torch.zeros((4, 4)))

    pinned_ids, grouped, stats = services.summary_graph_viewer_payload(sng, max_nodes=200)

    assert pinned_ids == ["L_0", "0_0_0", "0_1_0", "E_0_0"]
    assert grouped == [["SN_0", "0_0_0", "0_1_0"], ["SN_LOGIT_0", "L_0"]]
    assert stats["supernodes"] == 2

    capped_ids, capped_grouped, capped_stats = services.summary_graph_viewer_payload(
        sng, max_nodes=2
    )

    assert "L_0" in capped_ids
    assert capped_grouped == [["SN_LOGIT_0", "L_0"]]
    assert capped_stats["dropped_supernodes"] == 1


def test_summary_job_reports_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_load_graph_record(_slug):
        return object()

    def fake_run_summary(*, progress, **_kwargs):
        progress("Pruning attribution graph", 0.35)
        return {"ok": True}

    monkeypatch.setattr(services, "load_graph_record", fake_load_graph_record)
    monkeypatch.setattr(services, "run_summary", fake_run_summary)
    client = TestClient(app)

    response = client.post("/api/graphs/austin/summary", json={"label_supernodes": False})

    assert response.status_code == 200
    job_id = response.json()["job"]["id"]
    for _ in range(20):
        job = client.get(f"/api/jobs/{job_id}").json()["job"]
        if job["status"] == "completed":
            break
        time.sleep(0.05)
    assert job["status"] == "completed"
    assert job["progress"] == 1.0
    assert job["result"] == {"ok": True}


def test_preview_job_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_preview_prompt(**_kwargs):
        return {"tokens": ["A"], "next_tokens": [{"token": "B", "probability": 0.9}]}

    monkeypatch.setattr(services, "preview_prompt", fake_preview_prompt)
    client = TestClient(app)

    response = client.post("/api/graphs/preview", json={"prompt": "A"})

    assert response.status_code == 200
    job_id = response.json()["job"]["id"]
    for _ in range(20):
        job = client.get(f"/api/jobs/{job_id}").json()["job"]
        if job["status"] == "completed":
            break
        time.sleep(0.05)
    assert job["status"] == "completed"
    assert job["result"]["tokens"] == ["A"]


def test_upload_rejects_non_pt_file() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/graphs/upload",
        files={"file": ("graph.txt", b"not a graph", "text/plain")},
    )

    assert response.status_code == 400
    assert "Only .pt" in response.json()["detail"]
