import threading
from types import SimpleNamespace

import httpx2
import pytest
import torch
from typesafe_sdk import TypeSafeBadRequestError

from summarization.feature_source import FeatureInfo
from summarization.jev_relevance import (
    feature_digest,
    filter_by_jev_relevance,
    judge_feature_relevance,
)
from summarization.prune import PruneGraph
from summarization.utils import _node_from_json_dict


class _FakeResponse:
    def __init__(self, scores: dict[str, float]) -> None:
        self.nouls = {node_id: SimpleNamespace(noul=score) for node_id, score in scores.items()}
        self.model = "jev-test-1.0"
        self.usage = SimpleNamespace(input_tokens=100, output_tokens=5)


class _FakeClient:
    """Answers from a {node_id: noul} table; ids in ``missing`` get no answer."""

    def __init__(self, scores: dict[str, float], missing: set[str] | None = None) -> None:
        self.scores = scores
        self.missing = missing or set()
        self.calls: list[dict] = []
        self._lock = threading.Lock()

    def system_one(self, state, questions, model=None):
        with self._lock:
            self.calls.append({"state": state, "questions": questions, "model": model})
        answered = {
            node_id: self.scores[node_id]
            for node_id in questions
            if node_id not in self.missing and node_id in self.scores
        }
        return _FakeResponse(answered)


def _info(
    act_density: float = 0.01,
    top_logits: list[str] | None = None,
    contexts: list[str] | None = None,
) -> FeatureInfo:
    return FeatureInfo(
        act_density=act_density,
        top_logits=["word"] if top_logits is None else top_logits,
        contexts=["a <<b>> c"] if contexts is None else contexts,
        top_tokens=["b"],
        top_next_tokens=["c"],
    )


def _empty_info() -> FeatureInfo:
    return FeatureInfo(
        act_density=0.0, top_logits=[], contexts=[], top_tokens=[], top_next_tokens=[]
    )


def _base_graph() -> PruneGraph:
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
    adj = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.8, 0.3, 0.0],
        ],
        dtype=torch.float32,
    )
    return PruneGraph(
        nodes=nodes,
        pruned_adj=adj,
        metadata={"scan": "gemma-2-2b", "prompt_tokens": ["<bos>", "hi"]},
    )


def _install_fetch(monkeypatch: pytest.MonkeyPatch, info_by_node: dict) -> None:
    def fake_fetch(scan, node_id, **kwargs):
        return info_by_node.get(node_id)

    monkeypatch.setattr("summarization.jev_relevance.fetch_feature_info", fake_fetch)


@pytest.fixture(autouse=True)
def _no_fetch_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("summarization.jev_relevance.time.sleep", lambda _seconds: None)


def test_transient_fetch_failure_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"0_10_0": 0, "1_20_0": 0}

    def flaky_fetch(scan, node_id, **kwargs):
        calls[node_id] += 1
        if node_id == "0_10_0" and calls[node_id] == 1:
            return None
        return _info()

    monkeypatch.setattr("summarization.jev_relevance.fetch_feature_info", flaky_fetch)
    client = _FakeClient({"0_10_0": 0.9, "1_20_0": 0.9})

    out = filter_by_jev_relevance(_base_graph(), "claim X", client=client)

    assert {"0_10_0", "1_20_0"}.issubset({n.node_id for n in out.nodes})
    assert out.metadata["jev_relevance"]["unjudged"] == []
    assert out.metadata["jev_relevance"]["judged"] == 2


def test_filter_drops_irrelevant_and_records_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fetch(monkeypatch, {"0_10_0": _info(), "1_20_0": _info()})
    pg = _base_graph()
    client = _FakeClient({"0_10_0": 0.9, "1_20_0": 0.1})

    out = filter_by_jev_relevance(pg, "the claim and what I want to see", client=client)

    assert {n.node_id for n in out.nodes} == {"E_0_0", "0_10_0", "L_1"}
    assert out.pruned_adj.shape == (3, 3)
    assert out.num_edges == 2
    record = out.metadata["jev_relevance"]
    assert record["query"] == "the claim and what I want to see"
    assert record["model"] == "jev-latest"
    assert record["response_models"] == ["jev-test-1.0"]
    assert record["threshold"] == 0.5
    assert record["judged"] == 2
    assert record["scores"] == {"0_10_0": 0.9, "1_20_0": 0.1}
    assert record["dropped"] == ["1_20_0"]
    assert record["unjudged"] == []
    assert record["requests"] == 1
    assert record["input_tokens"] == 100
    assert "jev_relevance" not in pg.metadata


def test_state_and_question_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fetch(monkeypatch, {"0_10_0": _info(), "1_20_0": _info()})
    client = _FakeClient({"0_10_0": 0.9, "1_20_0": 0.9})

    filter_by_jev_relevance(_base_graph(), "claim X", client=client)

    call = client.calls[0]
    assert call["model"] == "jev-latest"
    assert call["state"]["query"] == "claim X"
    assert set(call["state"]["features"]) == {"0_10_0", "1_20_0"}
    digest = call["state"]["features"]["0_10_0"]
    assert digest == {
        "activation_frequency": 0.01,
        "top_logits": ["word"],
        "examples": [{"context": "a <<b>> c", "peak_token": "b", "next_token": "c"}],
    }
    question = call["questions"]["0_10_0"]
    assert "`0_10_0`" in question.instructions
    assert "`query`" in question.instructions
    assert set(question.criteria) == {"true", "false"}


def test_empty_digest_is_dropped_as_unjudged(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fetch(monkeypatch, {"0_10_0": _empty_info(), "1_20_0": _info()})
    client = _FakeClient({"1_20_0": 0.9})

    out = filter_by_jev_relevance(_base_graph(), "claim X", client=client)

    assert {n.node_id for n in out.nodes} == {"E_0_0", "1_20_0", "L_1"}
    record = out.metadata["jev_relevance"]
    assert record["unjudged"] == ["0_10_0"]
    assert record["scores"] == {"1_20_0": 0.9}
    assert [set(call["questions"]) for call in client.calls] == [{"1_20_0"}]


def test_missing_answer_is_dropped_as_unjudged(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fetch(monkeypatch, {"0_10_0": _info(), "1_20_0": _info()})
    client = _FakeClient({"0_10_0": 0.9, "1_20_0": 0.9}, missing={"1_20_0"})

    out = filter_by_jev_relevance(_base_graph(), "claim X", client=client)

    assert {n.node_id for n in out.nodes} == {"E_0_0", "0_10_0", "L_1"}
    assert out.metadata["jev_relevance"]["unjudged"] == ["1_20_0"]


def test_unfetchable_dashboard_is_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fetch(monkeypatch, {"1_20_0": _info()})
    client = _FakeClient({"1_20_0": 0.9})

    out = filter_by_jev_relevance(_base_graph(), "claim X", client=client)

    assert {n.node_id for n in out.nodes} == {"E_0_0", "1_20_0", "L_1"}
    assert out.metadata["jev_relevance"]["unjudged"] == ["0_10_0"]


def test_all_unfetchable_raises_instead_of_collapsing(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fetch(monkeypatch, {})
    client = _FakeClient({})

    with pytest.raises(RuntimeError, match="no fetchable feature dashboard"):
        filter_by_jev_relevance(_base_graph(), "claim X", client=client)
    assert client.calls == []


def test_chunks_are_issued_per_batch() -> None:
    digests = {f"n{i}": feature_digest(_info(act_density=0.1 * i)) for i in range(3)}
    client = _FakeClient({f"n{i}": 0.7 for i in range(3)})

    scores, usage = judge_feature_relevance(
        "claim X",
        digests,
        features_per_request=1,
        max_concurrent_requests=3,
        client=client,
    )

    assert scores == {"n0": 0.7, "n1": 0.7, "n2": 0.7}
    assert usage["requests"] == 3
    assert len(client.calls) == 3
    assert all(len(call["questions"]) == 1 for call in client.calls)
    assert usage["input_tokens"] == 300
    assert usage["output_tokens"] == 15


def test_state_char_budget_forces_smaller_batches() -> None:
    digests = {f"n{i}": feature_digest(_info(act_density=0.1 * i)) for i in range(3)}
    client = _FakeClient({f"n{i}": 0.7 for i in range(3)})

    _, usage = judge_feature_relevance(
        "claim X",
        digests,
        features_per_request=64,
        max_concurrent_requests=1,
        max_state_chars=1,
        client=client,
    )

    assert usage["requests"] == 3
    assert all(len(call["questions"]) == 1 for call in client.calls)


class _SplittingClient(_FakeClient):
    """Rejects multi-feature chunks the way Jev does when the state is too long."""

    def system_one(self, state, questions, model=None):
        if len(questions) > 1:
            with self._lock:
                self.calls.append({"state": state, "questions": questions, "model": model})
            raise TypeSafeBadRequestError(
                400, {"detail": {"error_type": "max_tokens_exceeded"}}, httpx2.Headers()
            )
        return super().system_one(state, questions, model=model)


def test_max_tokens_exceeded_splits_and_retries() -> None:
    digests = {f"n{i}": feature_digest(_info(act_density=0.1 * i)) for i in range(4)}
    client = _SplittingClient({f"n{i}": 0.7 for i in range(4)})

    scores, usage = judge_feature_relevance(
        "claim X",
        digests,
        features_per_request=4,
        max_concurrent_requests=1,
        client=client,
    )

    assert scores == {f"n{i}": 0.7 for i in range(4)}
    assert (
        usage["requests"] == 4
    )  # the rejected 4- and 2-feature attempts are not billed in metadata
    assert usage["input_tokens"] == 400


def test_target_logit_survives_losing_all_parents(monkeypatch: pytest.MonkeyPatch) -> None:
    node_ids = ["E_0_0", "0_10_0", "L_1"]
    raw = {
        "E_0_0": {"feature_type": "embedding", "ctx_idx": 0},
        "0_10_0": {"feature_type": "cross layer transcoder", "layer": 0, "ctx_idx": 0},
        "L_1": {"feature_type": "logit", "is_target_logit": True, "token_prob": 0.9},
    }
    nodes = [
        _node_from_json_dict({"node_id": nid, "node_idx": i, **raw[nid]})
        for i, nid in enumerate(node_ids)
    ]
    adj = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.8, 0.0],
        ],
        dtype=torch.float32,
    )
    pg = PruneGraph(nodes=nodes, pruned_adj=adj, metadata={"scan": "gemma-2-2b"})
    _install_fetch(monkeypatch, {"0_10_0": _info()})
    client = _FakeClient({"0_10_0": 0.0})

    out = filter_by_jev_relevance(pg, "claim X", client=client)

    assert {n.node_id for n in out.nodes} == {"L_1"}
    assert out.metadata["jev_relevance"]["dropped"] == ["0_10_0"]


def test_threshold_is_inclusive(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fetch(monkeypatch, {"0_10_0": _info(), "1_20_0": _info()})
    client = _FakeClient({"0_10_0": 0.5, "1_20_0": 0.4999})

    out = filter_by_jev_relevance(_base_graph(), "claim X", threshold=0.5, client=client)

    assert "0_10_0" in {n.node_id for n in out.nodes}
    assert out.metadata["jev_relevance"]["dropped"] == ["1_20_0"]


def test_validation_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fetch(monkeypatch, {"0_10_0": _info()})
    pg = _base_graph()
    with pytest.raises(ValueError, match="threshold"):
        filter_by_jev_relevance(pg, "claim X", threshold=1.5)
    with pytest.raises(ValueError, match="nonempty query"):
        filter_by_jev_relevance(pg, "  ")
    with pytest.raises(ValueError, match="at least 1"):
        filter_by_jev_relevance(pg, "claim X", features_per_request=0)
    with pytest.raises(ValueError, match="scan"):
        filter_by_jev_relevance(
            PruneGraph(nodes=[], pruned_adj=torch.zeros((0, 0)), metadata={}),
            "claim X",
        )
