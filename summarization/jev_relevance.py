"""Query-conditioned feature pruning with TypeSafe Jev Noul judgments.

Stage 1c: after influence pruning (and the optional activation-density filter), each
remaining CLT feature's dashboard digest is written as JSON into one TypeSafe state
under ``features.<node_id>``, and a batched Noul asks whether that feature is relevant
to the query (mechanistic claim plus what the user wants to see). Features Jev judges
irrelevant are dropped with their edges. Features whose dashboard cannot be fetched, is
empty, or whose answer is missing are dropped too (fail closed); the target logit is
never dropped. Dashboard digests, thresholds, and raw probabilities are recorded under
``metadata["jev_relevance"]`` for audit.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from typesafe_sdk import Noul, NoulCriteria, TypeSafeClient

from summarization.feature_source import FeatureInfo, fetch_feature_info
from summarization.prune import PruneGraph, _subset_prune_graph, remove_dangling_nodes
from summarization.summarize import Node
from summarization.typesafe_batch import BatchResult, ask_with_split, merge_results
from summarization.utils import _build_index_sets

DEFAULT_MODEL = "jev-latest"
DEFAULT_THRESHOLD = 0.5
DEFAULT_FEATURES_PER_REQUEST = 64
DEFAULT_MAX_CONCURRENT_REQUESTS = 4
DEFAULT_MAX_CONCURRENT_FETCHES = 4
_FETCH_ATTEMPTS = 3
_FETCH_BACKOFF_S = 0.5
# Jev takes roughly a 32k-token request; cap each state well below it and let the
# adaptive split in _ask_chunk absorb any remaining overflow.
DEFAULT_MAX_STATE_CHARS = 48_000
_QUESTION_CHARS = 450

_RELEVANCE_CRITERIA: NoulCriteria = {
    "true": (
        "A reader explaining the query's mechanistic claim would cite this feature's "
        "activating contexts, top logits, or activation frequency as directly related "
        "evidence."
    ),
    "false": (
        "The feature concerns something the query does not ask about, or its connection "
        "to the claim is only incidental (generic syntax, boilerplate, or an unrelated "
        "topic)."
    ),
}


def relevance_question(node_id: str) -> Noul:
    """One Noul asking whether a feature's dashboard is relevant to the query."""
    return Noul(
        instructions=(
            f"In `features`, the key `{node_id}` holds one feature's dashboard. "
            "Is that feature relevant to the query in `query`?"
        ),
        criteria=_RELEVANCE_CRITERIA,
    )


def feature_digest(info: FeatureInfo) -> dict[str, Any]:
    """Compact JSON view of a feature dashboard: activation stats, top logits, examples."""
    return {
        "activation_frequency": info.act_density,
        "top_logits": info.top_logits,
        "examples": [
            {"context": context, "peak_token": token, "next_token": next_token}
            for context, token, next_token in zip(
                info.contexts, info.top_tokens, info.top_next_tokens, strict=True
            )
        ],
    }


def _digest_is_empty(digest: dict[str, Any]) -> bool:
    return not digest["examples"] and not digest["top_logits"]


def _ask_chunk(
    client: Any,
    model: str,
    query: str,
    digests: dict[str, dict[str, Any]],
) -> BatchResult:
    """One batched TypeSafe call, halving and retrying when Jev rejects it as too long."""
    state = {"query": query, "features": digests}
    questions = {node_id: relevance_question(node_id) for node_id in digests}
    return ask_with_split(
        client,
        model,
        state,
        questions,
        lambda full_state, ids: {
            "query": full_state["query"],
            "features": {node_id: full_state["features"][node_id] for node_id in ids},
        },
    )


def _chunk_features(
    digests: dict[str, dict[str, Any]],
    features_per_request: int,
    max_state_chars: int,
) -> list[list[str]]:
    """Greedy batches bounded by feature count and serialized-state characters."""
    chunks: list[list[str]] = []
    current: list[str] = []
    used = 0
    for node_id, digest in digests.items():
        size = len(json.dumps(digest, ensure_ascii=False)) + _QUESTION_CHARS
        if current and (len(current) >= features_per_request or used + size > max_state_chars):
            chunks.append(current)
            current, used = [], 0
        current.append(node_id)
        used += size
    if current:
        chunks.append(current)
    return chunks


def _validate_judge_inputs(
    query: str, features_per_request: int, max_concurrent_requests: int, max_state_chars: int
) -> None:
    if not query.strip():
        raise ValueError("Jev relevance requires a nonempty query.")
    if features_per_request < 1:
        raise ValueError(f"features_per_request must be at least 1, got {features_per_request}")
    if max_concurrent_requests < 1:
        raise ValueError(
            f"max_concurrent_requests must be at least 1, got {max_concurrent_requests}"
        )
    if max_state_chars < 1:
        raise ValueError(f"max_state_chars must be at least 1, got {max_state_chars}")


def judge_feature_relevance(
    query: str,
    digests: dict[str, dict[str, Any]],
    *,
    model: str = DEFAULT_MODEL,
    features_per_request: int = DEFAULT_FEATURES_PER_REQUEST,
    max_concurrent_requests: int = DEFAULT_MAX_CONCURRENT_REQUESTS,
    max_state_chars: int = DEFAULT_MAX_STATE_CHARS,
    client: Any = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Ask Jev for p(relevant) per feature digest, in concurrently issued batches.

    Returns the scores keyed by node id plus request metadata. Features whose answer is
    missing from a response get no score. ``client`` accepts a ``TypeSafeClient``
    substitute for tests; when omitted a client is built from ``TYPESAFE_API_KEY``.
    """
    _validate_judge_inputs(query, features_per_request, max_concurrent_requests, max_state_chars)
    if not digests:
        return {}, {
            "model": model,
            "response_models": [],
            "requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }

    chunks = _chunk_features(digests, features_per_request, max_state_chars)

    def ask_all(active_client: Any) -> list[BatchResult]:
        workers = min(max_concurrent_requests, len(chunks))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    _ask_chunk,
                    active_client,
                    model,
                    query,
                    {node_id: digests[node_id] for node_id in chunk},
                )
                for chunk in chunks
            ]
            return [future.result() for future in futures]

    if client is None:
        with TypeSafeClient(model=model) as owned:
            results = ask_all(owned)
    else:
        results = ask_all(client)

    return merge_results(results, model)


def filter_by_jev_relevance(
    prune_graph: PruneGraph,
    query: str,
    *,
    model: str = DEFAULT_MODEL,
    threshold: float = DEFAULT_THRESHOLD,
    scan: str | None = None,
    features_dir: str | Path | None = None,
    features_per_request: int = DEFAULT_FEATURES_PER_REQUEST,
    max_concurrent_requests: int = DEFAULT_MAX_CONCURRENT_REQUESTS,
    max_state_chars: int = DEFAULT_MAX_STATE_CHARS,
    max_concurrent_fetches: int = DEFAULT_MAX_CONCURRENT_FETCHES,
    client: Any = None,
) -> PruneGraph:
    """Drop CLT features Jev judges irrelevant to ``query``.

    The query carries the mechanistic claim and what the user wants to see out of the
    circuit. Feature dashboards are fetched concurrently and judged in parallel batches
    bounded by ``features_per_request`` and ``max_state_chars``; a batch Jev rejects as
    too long is halved and retried. Logit/embedding/error nodes are never judged; the
    target logit survives even if every feature feeding it is dropped. Raises when no
    feature dashboard could be fetched at all, rather than silently collapsing the graph.
    """
    _validate_judge_inputs(query, features_per_request, max_concurrent_requests, max_state_chars)
    if max_concurrent_fetches < 1:
        raise ValueError(f"max_concurrent_fetches must be at least 1, got {max_concurrent_fetches}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    scan = scan or prune_graph.metadata.get("scan", "")
    if not scan:
        raise ValueError(
            "filter_by_jev_relevance requires a transcoder scan. "
            "Pass scan=... when the graph metadata lacks one."
        )

    nodes = [replace(n) for n in prune_graph.nodes]
    adj = prune_graph.pruned_adj
    feature_nodes = [n for n in nodes if n.feature_type == "cross layer transcoder"]
    n_features = len(feature_nodes)

    def fetch_one(node: Node) -> dict[str, Any] | None:
        for attempt in range(_FETCH_ATTEMPTS):
            try:
                info = fetch_feature_info(scan, node.node_id, features_dir=features_dir)
            except IndexError:  # dashboard validates but carries no quantile examples
                return None
            if info is not None:
                return feature_digest(info)
            if attempt + 1 < _FETCH_ATTEMPTS:  # transient HF/Cloudfront failure
                time.sleep(_FETCH_BACKOFF_S * (attempt + 1))
        return None

    digests: dict[str, dict[str, Any]] = {}
    unjudged: list[str] = []
    with ThreadPoolExecutor(max_workers=min(max_concurrent_fetches, max(n_features, 1))) as pool:
        for node, digest in zip(feature_nodes, pool.map(fetch_one, feature_nodes), strict=True):
            if digest is None or _digest_is_empty(digest):
                unjudged.append(node.node_id)
            else:
                digests[node.node_id] = digest
    if n_features and not digests:
        raise RuntimeError(
            f"Jev relevance found no fetchable feature dashboard for scan {scan!r}; "
            "refusing to drop every feature."
        )

    scores, usage = judge_feature_relevance(
        query,
        digests,
        model=model,
        features_per_request=features_per_request,
        max_concurrent_requests=max_concurrent_requests,
        max_state_chars=max_state_chars,
        client=client,
    )
    unjudged.extend(node_id for node_id in digests if node_id not in scores)

    node_mask = torch.ones(len(nodes), dtype=torch.bool, device=adj.device)
    edge_mask = adj != 0
    dropped: list[str] = []
    for i, node in enumerate(nodes):
        if node.feature_type != "cross layer transcoder":
            continue
        score = scores.get(node.node_id)
        if score is not None and score >= threshold:
            continue
        node_mask[i] = False
        edge_mask[i, :] = False
        edge_mask[:, i] = False
        if score is not None:
            dropped.append(node.node_id)

    idx = _build_index_sets(nodes)
    require_in = torch.tensor(idx["feature"] + idx["logit"], dtype=torch.long, device=adj.device)
    require_out = torch.tensor(
        idx["feature"] + idx["error"] + idx["embedding"],
        dtype=torch.long,
        device=adj.device,
    )
    node_mask = remove_dangling_nodes(node_mask, edge_mask, require_in, require_out)
    for i in idx["target_logit"]:
        node_mask[i] = True

    kept = node_mask.nonzero(as_tuple=True)[0]
    filtered = _subset_prune_graph(prune_graph, nodes, kept)
    filtered.metadata = {
        **filtered.metadata,
        "jev_relevance": {
            "query": query,
            **usage,
            "threshold": threshold,
            "features_per_request": features_per_request,
            "max_state_chars": max_state_chars,
            "judged": len(scores),
            "scores": scores,
            "dropped": dropped,
            "unjudged": unjudged,
        },
    }
    return filtered
