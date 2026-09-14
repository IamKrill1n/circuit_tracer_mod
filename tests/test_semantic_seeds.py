from __future__ import annotations

import json

import pytest
import torch

from summarization.attr_graph import AttrGraph
from summarization.label import ModelRoute, ModelSettings
from summarization.semantic_seeds import embedding_weights, select_semantic_seeds, span_weights
from summarization.summarize import Node


def graph() -> AttrGraph:
    tokens = ["<bos>", "France", " capital", " France", " Ger", "many"]
    nodes = [Node(f"E_{i}", i, i, "E", i, "embedding") for i in [4, 0, 1, 2, 3, 5]]
    return AttrGraph(nodes, torch.zeros(6, 6), {
        "prompt": "France capital France Germany", "prompt_tokens": tokens,
        "special_token_indices": [0], "tokens_decoded": True,
    })


def test_equal_mass_per_span_and_exact_embedding_mapping() -> None:
    ag = graph()
    payload = {"spans": [{"start": 2, "end": 3, "reason": "relation"},
                         {"start": 4, "end": 6, "reason": "country, including both subwords"}]}
    weights = span_weights(payload, ag.metadata["prompt_tokens"], [0])
    assert weights == [0, 0, 0.5, 0, 0.25, 0.25]
    assert embedding_weights(ag, weights) == [0.25, 0, 0, 0.5, 0, 0.25]
    assert sum(weights) == pytest.approx(1)


@pytest.mark.parametrize("spans, message", [
    ([], "No relevant span"),
    ([{"start": 0, "end": 1, "reason": "x"}], "special"),
    ([{"start": -1, "end": 2, "reason": "x"}], "indices"),
    ([{"start": True, "end": 2, "reason": "x"}], "indices"),
    ([{"start": 1, "end": 7, "reason": "x"}], "indices"),
    ([{"start": 2, "end": 2, "reason": "x"}], "indices"),
    ([{"start": 1, "end": 2, "reason": " "}], "reason"),
    ([{"start": 1, "end": 3, "reason": "x"}, {"start": 2, "end": 4, "reason": "y"}], "overlap"),
])
def test_reject_invalid_spans(spans, message) -> None:
    with pytest.raises(ValueError, match=message):
        span_weights({"spans": spans}, graph().metadata["prompt_tokens"], [0])


def test_selector_saves_configuration_and_repeated_token_position(monkeypatch) -> None:
    from summarization import semantic_seeds

    ag = graph()
    route = ModelRoute("openai", "wire-model", None, "secret-not-for-metadata", ModelSettings())
    monkeypatch.setattr(semantic_seeds, "resolve_model", lambda _: route)
    def generate(route, settings, system, user):
        assert settings.temperature == 0
        assert json.loads(user)["tokens"][3]["text"] == " France"
        return '{"spans": [{"start": 3, "end": 4, "reason": "subject"}]}'
    monkeypatch.setattr(semantic_seeds, "generate_text", generate)
    assert select_semantic_seeds(ag, "claim", "selector") == [0, 0, 0, 0, 1, 0]
    assert ag.metadata["semantic_seed"]["wire_model"] == "wire-model"
    assert "secret-not-for-metadata" not in json.dumps(ag.metadata)


def test_missing_tokenizer_metadata_fails_before_provider_call() -> None:
    ag = graph()
    del ag.metadata["special_token_indices"]
    with pytest.raises(ValueError, match="tokenizer-derived"):
        select_semantic_seeds(ag, "claim", "selector")


def test_duplicate_embedding_positions_rejected() -> None:
    ag = graph()
    ag.nodes[0].ctx_idx = 0
    with pytest.raises(ValueError, match="exactly once"):
        embedding_weights(ag, [0] * 6)


def test_shuffling_preserves_mass_and_special_tokens() -> None:
    from eval.prune_graphs import seed_conditions
    ag = graph()
    weights = [0.25, 0, 0, 0.5, 0, 0.25]
    conditions = seed_conditions(ag, weights, 17)
    assert conditions == seed_conditions(ag, weights, 17)
    assert sorted(conditions["shuffled"]) == sorted(weights)
    assert conditions["shuffled"][1] == 0
    assert sum(conditions["uniform"]) == pytest.approx(1)


def test_pipeline_rejects_conflicting_modes_before_loading(monkeypatch) -> None:
    from summarization.__main__ import build_parser
    from summarization.pipeline import run_pipeline
    args = build_parser().parse_args(["--claim", "claim", "--selector-model", "selector", "--token-weights", "[1]"])
    with pytest.raises(ValueError, match="mutually exclusive"):
        run_pipeline(args)
