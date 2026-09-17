"""Claim-conditioned embedding seeds from TypeSafe Jev per-token Nouls.

Seed selection stage: each prompt token gets one Noul asking whether it belongs to a
minimal complete prompt span the claim refers to (subject entity, queried relation).
Tokens whose Noul clears ``threshold`` are grouped into contiguous runs and given
equal mass per run, matching the selector-LLM span weighting; the resulting
embedding weights seed the relevance half of pruning.

Special tokens are never judged or seeded. Tokens Jev gives no answer for count as
below threshold and are recorded as unjudged; if Jev answers no token at all the
function raises rather than degrading silently. When no token clears the threshold
the selector returns ``None`` and records ``fallback="output_only"`` so the caller
can prune by influence only, explicitly.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from typesafe_sdk import Noul, NoulCriteria, TypeSafeClient

from summarization.attr_graph import AttrGraph
from summarization.semantic_seeds import embedding_weights
from summarization.typesafe_batch import BatchResult, ask_with_split, merge_results

DEFAULT_MODEL = "jev-latest"
DEFAULT_THRESHOLD = 0.5
DEFAULT_TOKENS_PER_REQUEST = 64
DEFAULT_MAX_CONCURRENT_REQUESTS = 4
DEFAULT_MAX_STATE_CHARS = 48_000
_QUESTION_CHARS = 450

_RELEVANCE_CRITERIA: NoulCriteria = {
    "true": (
        "The token belongs to a minimal complete prompt span the claim refers to, such as "
        "the subject entity or the queried relation, including all subword tokens of a "
        "selected word."
    ),
    "false": (
        "The token is unrelated context or a distractor, a special token, or chat scaffolding."
    ),
}


def token_relevance_question(index: int) -> Noul:
    """One Noul asking whether prompt token ``index`` belongs to a claim-relevant span."""
    return Noul(
        instructions=(
            f"In `tokens`, the entry whose `index` is {index} holds one token of the prompt "
            "in `prompt`. Is that token part of a minimal complete prompt span the claim in "
            "`claim` refers to?"
        ),
        criteria=_RELEVANCE_CRITERIA,
    )


def _token_entries(tokens: list[str], special_indices: list[int]) -> list[dict[str, Any]]:
    """Judgeable prompt tokens; special tokens are excluded from the state."""
    special = set(special_indices)
    return [
        {"index": index, "text": text} for index, text in enumerate(tokens) if index not in special
    ]


def _chunk_indices(
    entries: list[dict[str, Any]], tokens_per_request: int, max_state_chars: int
) -> list[list[int]]:
    """Greedy batches bounded by token count and serialized-state characters."""
    chunks: list[list[int]] = []
    current: list[int] = []
    used = 0
    for entry in entries:
        size = len(json.dumps(entry, ensure_ascii=False)) + _QUESTION_CHARS
        if current and (len(current) >= tokens_per_request or used + size > max_state_chars):
            chunks.append(current)
            current, used = [], 0
        current.append(int(entry["index"]))
        used += size
    if current:
        chunks.append(current)
    return chunks


def _validate_judge_inputs(
    claim: str, tokens_per_request: int, max_concurrent_requests: int, max_state_chars: int
) -> None:
    if not claim.strip():
        raise ValueError("Jev token seeds require a nonempty claim.")
    if tokens_per_request < 1:
        raise ValueError(f"tokens_per_request must be at least 1, got {tokens_per_request}")
    if max_concurrent_requests < 1:
        raise ValueError(
            f"max_concurrent_requests must be at least 1, got {max_concurrent_requests}"
        )
    if max_state_chars < 1:
        raise ValueError(f"max_state_chars must be at least 1, got {max_state_chars}")


def judge_token_relevance(
    claim: str,
    tokens: list[str],
    prompt: str,
    special_indices: list[int],
    *,
    model: str = DEFAULT_MODEL,
    tokens_per_request: int = DEFAULT_TOKENS_PER_REQUEST,
    max_concurrent_requests: int = DEFAULT_MAX_CONCURRENT_REQUESTS,
    max_state_chars: int = DEFAULT_MAX_STATE_CHARS,
    client: Any = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Ask Jev for p(claim-relevant) per prompt token, in concurrently issued batches.

    Special tokens are neither included in the state nor judged. Returns the scores
    keyed by stringified token index plus request metadata. Tokens whose answer is
    missing from a response get no score. ``client`` accepts a ``TypeSafeClient``
    substitute for tests; when omitted a client is built from ``TYPESAFE_API_KEY``.
    """
    _validate_judge_inputs(claim, tokens_per_request, max_concurrent_requests, max_state_chars)
    if not tokens:
        raise ValueError("judge_token_relevance requires at least one prompt token.")
    entries = _token_entries(tokens, special_indices)
    entries_by_index = {str(entry["index"]): entry for entry in entries}
    chunks = _chunk_indices(entries, tokens_per_request, max_state_chars)

    def subset_state(state: dict[str, Any], ids: list[str]) -> dict[str, Any]:
        return {
            "claim": state["claim"],
            "prompt": state["prompt"],
            "tokens": [entries_by_index[question_id] for question_id in ids],
        }

    def ask_all(active_client: Any) -> list[BatchResult]:
        workers = min(max_concurrent_requests, len(chunks))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    ask_with_split,
                    active_client,
                    model,
                    {
                        "claim": claim,
                        "prompt": prompt,
                        "tokens": [entries_by_index[str(index)] for index in chunk],
                    },
                    {str(index): token_relevance_question(index) for index in chunk},
                    subset_state,
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


def select_jev_semantic_seeds(
    ag: AttrGraph,
    claim: str,
    *,
    model: str = DEFAULT_MODEL,
    threshold: float = DEFAULT_THRESHOLD,
    tokens_per_request: int = DEFAULT_TOKENS_PER_REQUEST,
    max_concurrent_requests: int = DEFAULT_MAX_CONCURRENT_REQUESTS,
    max_state_chars: int = DEFAULT_MAX_STATE_CHARS,
    client: Any = None,
) -> list[float] | None:
    """Judge claim-relevant prompt tokens with Jev and return embedding weights.

    Returns ``None`` when no non-special token clears ``threshold``; the graph metadata
    records ``fallback="output_only"`` so the caller can prune by influence only.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    tokens = ag.metadata.get("prompt_tokens")
    special = ag.metadata.get("special_token_indices")
    if not isinstance(tokens, list) or not tokens or not all(isinstance(t, str) for t in tokens):
        raise ValueError("Jev token seeds require decoded prompt tokens.")
    if ag.metadata.get("tokens_decoded") is False or not isinstance(special, list):
        raise ValueError(
            "Jev token seeds require tokenizer-derived special_token_indices; reload a .pt graph."
        )
    if any(type(index) is not int or not 0 <= index < len(tokens) for index in special):
        raise ValueError("Invalid special_token_indices.")
    # Check alignment before a paid provider call.
    embedding_weights(ag, [0.0] * len(tokens))
    prompt = ag.metadata.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("Jev token seeds require the exact prompt.")

    scores, usage = judge_token_relevance(
        claim,
        tokens,
        prompt,
        special,
        model=model,
        tokens_per_request=tokens_per_request,
        max_concurrent_requests=max_concurrent_requests,
        max_state_chars=max_state_chars,
        client=client,
    )
    if not scores:
        raise RuntimeError("Jev token seeds received no token judgments; refusing to continue.")

    special_set = set(special)
    selected = [
        index
        for index in range(len(tokens))
        if index not in special_set and str(index) in scores and scores[str(index)] >= threshold
    ]
    runs: list[list[int]] = []
    for index in selected:
        if runs and index == runs[-1][-1] + 1:
            runs[-1].append(index)
        else:
            runs.append([index])

    weights = [0.0] * len(tokens)
    for run in runs:
        mass = 1.0 / len(runs) / len(run)
        for index in run:
            weights[index] = mass

    result = embedding_weights(ag, weights) if runs else None
    ag.metadata["semantic_seed"] = {
        "claim": claim,
        "selector": "jev",
        "model": model,
        **usage,
        "threshold": threshold,
        "tokens_per_request": tokens_per_request,
        "max_state_chars": max_state_chars,
        "judged": len(scores),
        "scores": scores,
        "runs": runs,
        "unjudged": [
            index
            for index in range(len(tokens))
            if index not in special_set and str(index) not in scores
        ],
        "token_weights": weights,
        "embedding_weights": result,
        "weighting": "equal_span_mass",
        "fallback": None if runs else "output_only",
    }
    return result
