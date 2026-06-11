# 1. Metadata travels on the SummaryGraph

Status: accepted

## Context

Labeling a supernode needs the computation's provenance — the prompt, the predicted target
token, the transcoder `scan`, and `prompt_tokens`. Historically `SummaryGraph` was a pure
structural type (`supernodes` + `pruned_adj`, with `adj_matrix` derived), and callers passed
`prune_graph.metadata` as a separate argument to every labeling call.

We wanted the labeling entry point to take only (graph, model name, settings, scheme) without a
loose metadata dict threaded through every call site.

## Decision

`SummaryGraph` carries a `metadata: dict` field. `save`/`load` persist it, and `summarize()`
accepts it. Labeling reads provenance off the graph.

## Consequences

- The `.pt` save format changes: a saved `SummaryGraph` now stores `metadata` alongside
  `supernodes` and `pruned_adj`. Old `.pt` files load with empty metadata.
- `SummaryGraph` is no longer a purely structural object — it also carries provenance.
- Every construction site (`summarize()`, `app.py`, `run_llm_labels.py`) must supply metadata.
- Reversing this means re-introducing a metadata argument across all labeling call sites.
