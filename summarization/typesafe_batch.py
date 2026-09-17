"""Shared TypeSafe batching for the Jev pruning stages.

``jev_relevance`` and ``jev_seeds`` both submit many Noul judgments over one shared
state. ``ask_with_split`` issues one request and, when Jev rejects it as too long,
halves the questions and the matching state entries and retries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from typesafe_sdk import TypeSafeBadRequestError


@dataclass
class BatchResult:
    scores: dict[str, float]
    response_model: str
    input_tokens: int
    output_tokens: int
    requests: int

    def merge(self, other: BatchResult) -> BatchResult:
        return BatchResult(
            scores={**self.scores, **other.scores},
            response_model=self.response_model or other.response_model,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            requests=self.requests + other.requests,
        )


def is_max_tokens_error(exc: TypeSafeBadRequestError) -> bool:
    body = exc.body
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, dict) and detail.get("error_type") == "max_tokens_exceeded":
            return True
    return "max_tokens_exceeded" in str(exc)


def ask_with_split(
    client: Any,
    model: str,
    state: dict[str, Any],
    questions: dict[str, Any],
    subset_state: Callable[[dict[str, Any], list[str]], dict[str, Any]],
) -> BatchResult:
    """One TypeSafe call, halving questions and state when Jev rejects it as too long."""
    try:
        response = client.system_one(state, questions, model=model)
    except TypeSafeBadRequestError as exc:
        if not is_max_tokens_error(exc) or len(questions) < 2:
            raise
        question_ids = list(questions)
        mid = len(question_ids) // 2
        left_ids = question_ids[:mid]
        right_ids = question_ids[mid:]
        left = ask_with_split(
            client,
            model,
            subset_state(state, left_ids),
            {question_id: questions[question_id] for question_id in left_ids},
            subset_state,
        )
        right = ask_with_split(
            client,
            model,
            subset_state(state, right_ids),
            {question_id: questions[question_id] for question_id in right_ids},
            subset_state,
        )
        return left.merge(right)
    scores = {
        question_id: float(response.nouls[question_id].noul)
        for question_id in questions
        if question_id in response.nouls
    }
    return BatchResult(
        scores=scores,
        response_model=response.model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        requests=1,
    )


def merge_results(
    results: list[BatchResult], model: str
) -> tuple[dict[str, float], dict[str, Any]]:
    """Combine batch results into score and usage records for stage metadata."""
    scores: dict[str, float] = {}
    response_models: set[str] = set()
    input_tokens = 0
    output_tokens = 0
    requests = 0
    for result in results:
        scores.update(result.scores)
        response_models.add(result.response_model)
        input_tokens += result.input_tokens
        output_tokens += result.output_tokens
        requests += result.requests

    usage = {
        "model": model,
        "response_models": sorted(response_models),
        "requests": requests,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
    return scores, usage
