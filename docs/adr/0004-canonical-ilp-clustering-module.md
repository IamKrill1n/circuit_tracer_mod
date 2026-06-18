# 4. Canonical clustering uses the ILP stage module

Status: accepted

## Context

The summarization package is being narrowed to three primary stages: pruning, clustering, and
labeling. The current `summarization.cluster` module mixes production-facing entry points with
older spectral/agglomerative clustering baselines, auto-k scoring helpers, and conversion
utilities. Meanwhile, `summarization.ilp_cluster` contains the clustering algorithm used by the
main research workflow.

Keeping the old baseline code under the package's canonical `cluster` name makes the stage
boundary hard to read: future callers cannot tell whether `cluster` means the active
summarization algorithm or legacy evaluation code.

## Decision

`summarization.cluster` is the canonical package module for the clustering stage, and its active
implementation is the ILP clustering path.

Older spectral/agglomerative/auto-k clustering code is retained outside the package as
evaluation-only legacy baseline code. It may still be used by eval scripts, but it is not part of
the production summarization package surface.

Compatibility shims may temporarily preserve old import paths while callers migrate.

## Consequences

- The package surface matches the domain pipeline: prune → cluster → label.
- The canonical clustering stage points at the algorithm used by the main workflow.
- Evaluation scripts that compare old baselines must import them from eval-owned modules.
- Existing callers of old `summarization.cluster` helpers need migration or temporary shims.
- The compatibility shim can be removed once app, eval, and tests no longer depend on the old
  import paths.
