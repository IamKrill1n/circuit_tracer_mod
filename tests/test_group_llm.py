import json
import sys
import types

import torch

from summarization.group_llm import (
    LabelScheme,
    ModelRoute,
    ModelSettings,
    _build_feature_block,
    _build_graph_user_message,
    _build_single_supernode_user_message,
    _gemini_generate,
    _merge_settings,
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
                    "model": "gemini-2.5-flash",
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
    assert route.supports_thinking_budget


def test_resolve_model_allows_registry_to_disable_gemini_thinking_budget(
    tmp_path, monkeypatch
) -> None:
    registry_path = tmp_path / "llm_models.json"
    registry_path.write_text(
        json.dumps(
            {
                "gemma-test": {
                    "provider": "gemini",
                    "model": "gemma-4-31b-it",
                    "base_url": None,
                    "api_key_env": "GOOGLE_API_KEY",
                    "supports_thinking_budget": False,
                    "defaults": {"temperature": 0.2, "thinking_effort": "medium"},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")

    route = resolve_model("gemma-test", registry_path)

    assert route.provider == "gemini"
    assert not route.supports_thinking_budget


def test_merge_settings_can_disable_registry_thinking_default() -> None:
    defaults = ModelSettings(temperature=1.0, thinking_effort="medium")

    merged = _merge_settings(
        ModelSettings(temperature=0.2, use_default_thinking_effort=False),
        defaults,
    )

    assert merged.temperature == 0.2
    assert merged.thinking_effort is None


def test_gemini_generate_omits_thinking_config_for_unsupported_model(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class FakeThinkingConfig:
        def __init__(self, thinking_budget: int) -> None:
            calls["thinking_budget"] = thinking_budget

    class FakeGenerateContentConfig:
        def __init__(self, **kwargs) -> None:
            calls["config_kwargs"] = kwargs

    class FakeModels:
        def generate_content(self, *, model: str, config, contents: str):
            calls["model"] = model
            calls["config"] = config
            calls["contents"] = contents
            return types.SimpleNamespace(text="ok")

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            calls["api_key"] = api_key
            self.models = FakeModels()

    google_module = types.ModuleType("google")
    google_module.__path__ = []
    genai_module = types.ModuleType("google.genai")
    genai_types_module = types.ModuleType("google.genai.types")
    genai_errors_module = types.ModuleType("google.genai.errors")

    class FakeClientError(Exception):
        pass

    class FakeServerError(Exception):
        pass

    genai_module.Client = FakeClient
    genai_types_module.GenerateContentConfig = FakeGenerateContentConfig
    genai_types_module.ThinkingConfig = FakeThinkingConfig
    genai_errors_module.ClientError = FakeClientError
    genai_errors_module.ServerError = FakeServerError
    google_module.genai = genai_module
    genai_module.types = genai_types_module
    genai_module.errors = genai_errors_module

    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)
    monkeypatch.setitem(sys.modules, "google.genai.types", genai_types_module)
    monkeypatch.setitem(sys.modules, "google.genai.errors", genai_errors_module)

    route = ModelRoute(
        provider="gemini",
        model="gemma-4-31b-it",
        base_url=None,
        api_key="google-key",
        defaults=ModelSettings(),
        supports_thinking_budget=False,
    )
    settings = ModelSettings(temperature=0.2, thinking_effort="medium")

    text = _gemini_generate(route, settings, "system prompt", "user prompt")

    assert text == "ok"
    assert calls["api_key"] == "google-key"
    assert calls["model"] == "gemma-4-31b-it"
    assert calls["contents"] == "user prompt"
    assert calls["config_kwargs"] == {
        "system_instruction": "system prompt",
        "temperature": 0.2,
    }
    assert "thinking_budget" not in calls


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

    assert labels == {0: ("Country token", "Input", "Tracks the country named in the prompt.")}


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
    monkeypatch.setattr(
        "summarization.group_llm._fetch_feature_context", lambda *args, **kwargs: None
    )

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
