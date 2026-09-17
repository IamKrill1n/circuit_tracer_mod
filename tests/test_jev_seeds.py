from __future__ import annotations

import threading
from argparse import Namespace
from types import SimpleNamespace

import httpx2
import pytest
import torch
from typesafe_sdk import TypeSafeBadRequestError

from summarization.attr_graph import AttrGraph
from summarization.jev_seeds import judge_token_relevance, select_jev_semantic_seeds
from summarization.summarize import Node

TOKENS = ["<bos>", "France", " capital", " France", " Ger", "many"]
PROMPT = "France capital France Germany"


class _FakeResponse:
    def __init__(self, scores: dict[str, float]) -> None:
        self.nouls = {index: SimpleNamespace(noul=score) for index, score in scores.items()}
        self.model = "jev-test-1.0"
        self.usage = SimpleNamespace(input_tokens=100, output_tokens=5)


class _FakeClient:
    """Answers from a {token_index: noul} table; ids in ``missing`` get no answer."""

    def __init__(self, scores: dict[str, float], missing: set[str] | None = None) -> None:
        self.scores = scores
        self.missing = missing or set()
        self.calls: list[dict] = []
        self._lock = threading.Lock()

    def system_one(self, state, questions, model=None):
        with self._lock:
            self.calls.append({"state": state, "questions": questions, "model": model})
        answered = {
            index: self.scores[index]
            for index in questions
            if index not in self.missing and index in self.scores
        }
        return _FakeResponse(answered)


def graph(tokens: list[str] | None = None, special: list[int] | None = None) -> AttrGraph:
    tokens = TOKENS if tokens is None else tokens
    special = [0] if special is None else special
    nodes = [Node(f"E_{i}", i, i, "E", i, "embedding") for i in [4, 0, 1, 2, 3, 5]]
    return AttrGraph(
        nodes,
        torch.zeros(len(tokens), len(tokens)),
        {
            "prompt": PROMPT,
            "prompt_tokens": tokens,
            "special_token_indices": special,
            "tokens_decoded": True,
        },
    )


def test_selects_tokens_and_allocates_equal_mass_per_run() -> None:
    client = _FakeClient({"1": 0.9, "2": 0.4, "3": 0.8, "4": 0.7, "5": 0.2})
    ag = graph()

    weights = select_jev_semantic_seeds(ag, "claim X", client=client)

    assert weights == [0.25, 0, 0.5, 0, 0.25, 0]
    record = ag.metadata["semantic_seed"]
    assert record["selector"] == "jev"
    assert record["fallback"] is None
    assert record["runs"] == [[1], [3, 4]]
    assert record["judged"] == 5
    assert record["unjudged"] == []
    assert record["threshold"] == 0.5
    assert sum(record["token_weights"]) == pytest.approx(1)

    call = client.calls[0]
    assert call["model"] == "jev-latest"
    assert call["state"]["claim"] == "claim X"
    assert call["state"]["prompt"] == PROMPT
    assert call["state"]["tokens"][0] == {"index": 1, "text": "France"}
    assert set(call["questions"]) == {"1", "2", "3", "4", "5"}
    question = call["questions"]["1"]
    assert "`tokens`" in question.instructions
    assert "`claim`" in question.instructions
    assert set(question.criteria) == {"true", "false"}


def test_special_token_never_seeded() -> None:
    client = _FakeClient({"0": 0.99, "1": 0.9})
    ag = graph()

    weights = select_jev_semantic_seeds(ag, "claim X", client=client)

    assert weights is not None
    assert weights[2] == 1.0
    assert sum(weights) == pytest.approx(1)
    assert ag.metadata["semantic_seed"]["runs"] == [[1]]
    assert ag.metadata["semantic_seed"]["token_weights"][0] == 0
    assert "0" not in client.calls[0]["questions"]


def test_below_threshold_returns_none_and_records_fallback() -> None:
    client = _FakeClient({str(index): 0.1 for index in range(6)})
    ag = graph()

    assert select_jev_semantic_seeds(ag, "claim X", client=client) is None

    record = ag.metadata["semantic_seed"]
    assert record["fallback"] == "output_only"
    assert record["runs"] == []
    assert record["embedding_weights"] is None
    assert record["token_weights"] == [0.0] * 6


def test_no_token_judgments_raises() -> None:
    client = _FakeClient({})

    with pytest.raises(RuntimeError, match="no token judgments"):
        select_jev_semantic_seeds(graph(), "claim X", client=client)


def test_unjudged_tokens_are_recorded_and_not_seeded() -> None:
    client = _FakeClient({"1": 0.9, "2": 0.9, "4": 0.9, "5": 0.9}, missing={"3"})
    ag = graph()

    select_jev_semantic_seeds(ag, "claim X", client=client)

    record = ag.metadata["semantic_seed"]
    assert record["unjudged"] == [3]
    assert record["runs"] == [[1, 2], [4, 5]]


def test_tokens_per_request_splits_batches() -> None:
    client = _FakeClient({str(index): 0.9 for index in range(6)})

    scores, usage = judge_token_relevance(
        "claim X",
        TOKENS,
        PROMPT,
        [0],
        tokens_per_request=2,
        max_concurrent_requests=1,
        client=client,
    )

    assert scores == {str(index): 0.9 for index in range(1, 6)}
    assert usage["requests"] == 3
    assert [len(call["questions"]) for call in client.calls] == [2, 2, 1]


def test_state_char_budget_forces_smaller_batches() -> None:
    client = _FakeClient({str(index): 0.9 for index in range(6)})

    _, usage = judge_token_relevance(
        "claim X",
        TOKENS,
        PROMPT,
        [0],
        tokens_per_request=64,
        max_concurrent_requests=1,
        max_state_chars=1,
        client=client,
    )

    assert usage["requests"] == 5
    assert all(len(call["questions"]) == 1 for call in client.calls)


class _SplittingClient(_FakeClient):
    """Rejects multi-token chunks the way Jev does when the state is too long."""

    def system_one(self, state, questions, model=None):
        if len(questions) > 1:
            with self._lock:
                self.calls.append({"state": state, "questions": questions, "model": model})
            raise TypeSafeBadRequestError(
                400, {"detail": {"error_type": "max_tokens_exceeded"}}, httpx2.Headers()
            )
        return super().system_one(state, questions, model=model)


def test_max_tokens_exceeded_splits_and_retries() -> None:
    client = _SplittingClient({str(index): 0.9 for index in range(6)})

    scores, usage = judge_token_relevance(
        "claim X",
        TOKENS,
        PROMPT,
        [0],
        tokens_per_request=4,
        max_concurrent_requests=1,
        client=client,
    )

    assert scores == {str(index): 0.9 for index in range(1, 6)}
    assert usage["requests"] == 5
    assert usage["input_tokens"] == 500


def test_missing_tokenizer_metadata_fails_before_provider_call() -> None:
    ag = graph()
    del ag.metadata["special_token_indices"]
    client = _FakeClient({})

    with pytest.raises(ValueError, match="tokenizer-derived"):
        select_jev_semantic_seeds(ag, "claim X", client=client)
    assert client.calls == []


def test_embedding_position_alignment_fails_before_provider_call() -> None:
    ag = graph()
    ag.nodes[0].ctx_idx = 0
    client = _FakeClient({"1": 0.9})

    with pytest.raises(ValueError, match="exactly once"):
        select_jev_semantic_seeds(ag, "claim X", client=client)
    assert client.calls == []


def test_validation_errors() -> None:
    ag = graph()
    with pytest.raises(ValueError, match="threshold"):
        select_jev_semantic_seeds(ag, "claim X", threshold=1.5, client=_FakeClient({}))
    with pytest.raises(ValueError, match="nonempty claim"):
        select_jev_semantic_seeds(ag, "  ", client=_FakeClient({}))
    with pytest.raises(ValueError, match="at least 1"):
        judge_token_relevance(
            "claim X", TOKENS, PROMPT, [0], tokens_per_request=0, client=_FakeClient({})
        )
    with pytest.raises(ValueError, match="prompt token"):
        judge_token_relevance("claim X", [], PROMPT, [], client=_FakeClient({}))


def test_claim_seed_selector_jev_fallback_and_success(monkeypatch: pytest.MonkeyPatch) -> None:
    from summarization import jev_seeds, pipeline

    args = Namespace(
        seed_selector="jev",
        jev_model="jev-test",
        jev_threshold=0.5,
        jev_tokens_per_request=64,
        jev_max_concurrent_requests=4,
        jev_max_state_chars=48_000,
    )
    monkeypatch.setattr(jev_seeds, "select_jev_semantic_seeds", lambda *a, **k: None)
    assert pipeline._select_claim_seeds(args, graph(), "claim X") == (
        None,
        "arithmetic",
        1.0,
        "output_only",
    )

    weights = [1.0, 0, 0, 0, 0, 0]
    monkeypatch.setattr(jev_seeds, "select_jev_semantic_seeds", lambda *a, **k: weights)
    assert pipeline._select_claim_seeds(args, graph(), "claim X") == (weights, None, None, None)


def test_claim_seed_selector_llm_path(monkeypatch: pytest.MonkeyPatch) -> None:
    from summarization import pipeline, semantic_seeds

    monkeypatch.setattr(
        semantic_seeds, "select_semantic_seeds", lambda ag, claim, model: [0, 0, 1.0, 0, 0, 0]
    )
    args = Namespace(seed_selector="llm", selector_model="selector")

    assert pipeline._select_claim_seeds(args, graph(), "claim X") == (
        [0, 0, 1.0, 0, 0, 0],
        None,
        None,
        None,
    )
