import json

import torch

from summarization.group_llm import (
    LabelScheme,
    _build_feature_block,
    _build_graph_user_message,
    _build_single_supernode_user_message,
    _parse_graph_label_response,
    resolve_model,
)
from summarization.summarize import Node, SummaryGraph, Supernode


def test_resolve_model_routes_gemini_with_google_api_key(tmp_path, monkeypatch) -> None:
    registry_path = tmp_path / "llm_models.json"
    registry_path.write_text(
        json.dumps(
            {
                "gemini-test": {
                    "provider": "gemini",
                    "base_url": None,
                    "api_key_env": "GOOGLE_API_KEY",
                    "defaults": {"temperature": 0.2, "thinking_effort": None},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")

    route = resolve_model("gemini-test", registry_path)

    assert route.provider == "gemini"
    assert route.api_key == "google-key"


def test_resolve_model_rejects_unknown_provider(tmp_path) -> None:
    registry_path = tmp_path / "llm_models.json"
    registry_path.write_text(
        json.dumps(
            {
                "gemini-test": {
                    "provider": "google",
                    "base_url": None,
                    "api_key_env": "GEMINI_API_KEY",
                    "defaults": {"temperature": 0.2, "thinking_effort": None},
                }
            }
        ),
        encoding="utf-8",
    )

    try:
        resolve_model("gemini-test", registry_path)
    except ValueError as exc:
        assert "unsupported provider" in str(exc)
    else:
        raise AssertionError("expected unsupported provider to raise ValueError")


def test_parse_graph_label_response_accepts_prompt_contract() -> None:
    response = """
    ```json
    {
      "supernodes": [
        {
          "id": "0",
          "role": "Input",
          "label": "Country token",
          "description": "Tracks the country named in the prompt."
        }
      ]
    }
    ```
    """

    labels = _parse_graph_label_response(response)

    assert labels == {
        0: ("Country token", "Input", "Tracks the country named in the prompt.")
    }


def test_parse_graph_label_response_keeps_legacy_cluster_fields() -> None:
    response = '{"clusters": [{"id": 1, "type": "Output", "name": "Paris support"}]}'

    labels = _parse_graph_label_response(response)

    assert labels == {1: ("Paris support", "Output", "")}


def test_feature_block_uses_layer_context_without_node_id() -> None:
    node = Node(
        node_id="12_345_6",
        node_idx=0,
        feature=345,
        layer="12",
        ctx_idx=1,
        feature_type="cross layer transcoder",
        clerp="capital relation",
    )

    info = {
        "top_tokens": ["city"],
        "top_next_tokens": ["capital"],
        "top_logits": ["Paris"],
        "contexts": ["The <<capital>> of France"],
    }

    block = _build_feature_block(node, info, model_n_layers=26)

    assert "Layer: 12 of 26 (middle reasoning stage)" in block
    assert "Position: token index 1" in block
    assert "<MAX_ACTIVATING_TOKENS>" in block
    assert "<TOKENS_AFTER_MAX_ACTIVATING_TOKEN>" in block
    assert "<TOP_POSITIVE_LOGITS>" in block
    assert "<TOP_ACTIVATING_TEXTS>" in block
    assert "capital relation" not in block
    assert "12_345_6" not in block
    assert "345" not in block
    assert " capital" not in block


def test_graph_user_message_uses_layer_not_feature_ids(monkeypatch) -> None:
    feature = Node(
        node_id="20_777_1",
        node_idx=0,
        feature=777,
        layer="20",
        ctx_idx=1,
        feature_type="cross layer transcoder",
        clerp="answer token support",
    )
    logit = Node(
        node_id="27_42_0",
        node_idx=1,
        feature=42,
        layer="27",
        ctx_idx=0,
        feature_type="logit",
        is_target_logit=True,
        clerp='Output " Paris" (p=0.9)',
    )
    sng = SummaryGraph(
        supernodes=[
            Supernode("SN_0", [feature], "features", 20, 20),
            Supernode("SN_LOGIT_0", [logit], "logit", 27, 27),
        ],
        pruned_adj=torch.zeros((2, 2)),
        metadata={"scan": "scan", "prompt": "The capital is", "prompt_tokens": ["The", " capital"]},
    )
    monkeypatch.setattr("summarization.group_llm._fetch_feature_context", lambda *args, **kwargs: None)

    user_message, ordered = _build_graph_user_message(sng, " Paris", LabelScheme())

    assert ordered == [sng.supernodes[0]]
    assert "Layer: 20 of 26 (late reasoning stage)" in user_message
    assert "Position: token index 1" in user_message
    assert "[Feature 0]" not in user_message
    assert "[Feature]" in user_message
    assert "answer token support" not in user_message
    assert "20_777_1" not in user_message
    assert "777" not in user_message
    assert '1: " capital"' not in user_message


def test_single_supernode_message_has_no_supernode_context() -> None:
    user_message = _build_single_supernode_user_message(
        {"prompt": "The capital is"},
        " Paris",
        ["Layer: 20 of 26 (late reasoning stage)\nPosition: token index 1"],
    )

    assert "Feature evidence in this supernode:" in user_message
    assert "Layer span" not in user_message
    assert "Active prompt-token positions" not in user_message
    assert "First-pass interpretation" not in user_message
    assert "Graph context" not in user_message
