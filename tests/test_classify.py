import json
import pytest
import torch

from summarization.classify import (
    normalize_text,
    heuristic_classify,
    classify_features,
    filter_act_density,
)
from summarization.prune import PruneGraph
from summarization.utils import _node_from_json_dict


def test_normalize_text_basic():
    assert normalize_text("Hello, WORLD!!") == "hello world"
    assert normalize_text("_Dallas-123") == "dallas123"


def test_heuristic_classify_say_prefix_is_output():
    out = heuristic_classify(
        clerp="Say hello",
        top_tokens=["a", "b", "c"],
        top_next_tokens=["x", "y", "z"],
        top_logits=["l1", "l2", "l3"],
    )
    assert out == "output"


def test_heuristic_classify_input_when_clerp_matches_top_tokens():
    out = heuristic_classify(
        clerp="dog",
        top_tokens=["dog"] * 10,
        top_next_tokens=["x", "y", "z", "a", "b", "c", "d", "e", "f", "g"],
        top_logits=["l1", "l2", "l3"],
    )
    assert out == "input"


def test_heuristic_classify_output_when_next_token_dominates():
    out = heuristic_classify(
        clerp="animal concept",
        top_tokens=["dog", "cat", "bird", "fish"],
        top_next_tokens=["the", "the", "the", "the", "the", "the", "the", "the", "the", "x", "y"],
        top_logits=["l1", "l2"],
    )
    assert out == "output"


def test_heuristic_classify_output_when_logit_repeats():
    out = heuristic_classify(
        clerp="animal concept",
        top_tokens=["dog", "cat", "bird", "fish"],
        top_next_tokens=["a", "b", "c", "d", "e", "f"],
        top_logits=["tokA", "tokA", "tokA"],
    )
    assert out == "output"


def test_heuristic_classify_abstract_fallback():
    out = heuristic_classify(
        clerp="topic about travel",
        top_tokens=["airport", "bus", "train", "road"],
        top_next_tokens=["to", "from", "at", "near"],
        top_logits=["alpha", "beta"],
    )
    assert out == "abstract"


def test_classify_features_embedding_and_logit_passthrough():
    node_ids = ["E_0_0", "L_1"]
    attr = {
        "E_0_0": {"feature_type": "embedding"},
        "L_1": {"feature_type": "logit"},
    }
    metadata = {"scan": "gemma-2-2b", "info": {"neuronpedia_source_set": "clt-hp"}}

    out = classify_features(node_ids=node_ids, attr=attr, metadata=metadata)
    assert out["E_0_0"] == "embedding"
    assert out["L_1"] == "logit"


def test_classify_features_trash_when_high_act_density(monkeypatch):
    def fake_get_feature(modelId, layer, index):
        data = {
            "explanations": [{"description": "anything"}],
            "frac_nonzero": 0.2,  # > 0.1 => trash
            "activations": [],
            "pos_str": [],
        }
        return 200, json.dumps(data)

    monkeypatch.setattr("summarization.classify.get_feature", fake_get_feature)

    node_ids = ["25_9974_1"]
    attr = {"25_9974_1": {"feature_type": "cross layer transcoder"}}
    metadata = {"scan": "gemma-2-2b", "info": {"neuronpedia_source_set": "gemmascope-transcoder-16k"}}

    out = classify_features(node_ids=node_ids, attr=attr, metadata=metadata)
    assert out["25_9974_1"] == "trash"


def test_classify_features_calls_heuristic_path(monkeypatch):
    def fake_get_feature(modelId, layer, index):
        # Build activations so top_tokens all "dog" -> heuristic input
        activations = []
        for _ in range(10):
            activations.append(
                {
                    "tokens": ["x", "dog", "next"],
                    "maxValueTokenIndex": 1,
                }
            )
        data = {
            "explanations": [{"description": "dog"}],
            "frac_nonzero": 0.01,
            "activations": activations,
            "pos_str": ["a", "b", "c"],
        }
        return 200, json.dumps(data)

    monkeypatch.setattr("summarization.classify.get_feature", fake_get_feature)

    node_ids = ["3_42_0"]
    attr = {"3_42_0": {"feature_type": "cross layer transcoder"}}
    metadata = {"scan": "gemma-2-2b", "info": {"neuronpedia_source_set": "clt-hp"}}

    out = classify_features(node_ids=node_ids, attr=attr, metadata=metadata)
    assert out["3_42_0"] == "input"


def test_classify_features_skip_on_non_200(monkeypatch):
    def fake_get_feature(modelId, layer, index):
        return 500, "error"

    monkeypatch.setattr("summarization.classify.get_feature", fake_get_feature)

    node_ids = ["1_2_3"]
    attr = {"1_2_3": {"feature_type": "cross layer transcoder"}}
    metadata = {"scan": "gemma-2-2b", "info": {"neuronpedia_source_set": "clt-hp"}}

    out = classify_features(node_ids=node_ids, attr=attr, metadata=metadata)
    assert "1_2_3" not in out


def test_filter_act_density_drops_out_of_band_and_sets_clerp(monkeypatch):
    def fake_get_feature(modelId, layer, index):
        # index 10 -> in band (keep); index 20 -> too frequent (drop)
        desc = "in band" if index == 10 else "too frequent"
        frac = 0.01 if index == 10 else 0.5
        return 200, json.dumps({"explanations": [{"description": desc}], "frac_nonzero": frac})

    monkeypatch.setattr("summarization.classify.get_feature", fake_get_feature)

    node_ids = ["E_0_0", "0_10_0", "1_20_0", "L_1"]
    raw = {
        "E_0_0": {"feature_type": "embedding", "ctx_idx": 1},
        "0_10_0": {"feature_type": "cross layer transcoder", "layer": 0, "ctx_idx": 0},
        "1_20_0": {"feature_type": "cross layer transcoder", "layer": 1, "ctx_idx": 0},
        "L_1": {"feature_type": "logit", "is_target_logit": True, "token_prob": 0.9},
    }
    nodes = [
        _node_from_json_dict({"node_id": nid, "node_idx": i, **raw[nid]})
        for i, nid in enumerate(node_ids)
    ]
    # adj[target, source]: E->feat1, E->feat2, feat1->L, feat2->L
    adj = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.8, 0.3, 0.0],
        ],
        dtype=torch.float32,
    )
    pg = PruneGraph(
        nodes=nodes,
        pruned_adj=adj,
        metadata={
            "scan": "gemma-2-2b",
            "info": {"neuronpedia_source_set": "clt-hp"},
            "prompt_tokens": ["<bos>", "hi"],
        },
    )

    out = filter_act_density(pg)
    ids = {n.node_id for n in out.nodes}
    assert "1_20_0" not in ids  # out-of-band density dropped
    assert "0_10_0" in ids  # in-band feature kept
    assert {"E_0_0", "L_1"}.issubset(ids)  # boundary nodes always kept

    by_id = {n.node_id: n for n in out.nodes}
    assert by_id["0_10_0"].clerp == "in band"  # clerp from Neuronpedia explanation
    assert by_id["E_0_0"].clerp == "Emb: hi"  # embedding clerp from prompt_tokens[ctx_idx]
    assert out.pruned_adj.shape[0] == len(out.nodes)


def test_filter_act_density_requires_source_set():
    node = _node_from_json_dict(
        {"node_id": "0_10_0", "node_idx": 0, "feature_type": "cross layer transcoder", "layer": 0, "ctx_idx": 0}
    )
    pg = PruneGraph(nodes=[node], pruned_adj=torch.zeros((1, 1)), metadata={})
    with pytest.raises(ValueError, match="source_set"):
        filter_act_density(pg)
