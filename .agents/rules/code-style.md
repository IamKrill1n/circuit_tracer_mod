# Code Style for Research Code

This is a research codebase. Optimize for readability, traceability of assumptions, and fast iteration. Prioritize in this order: **readability → reproducibility → maintainability → correctness on real inputs**. Don't sacrifice the first three for exhaustive defensive programming.

## Clarity over cleverness

Prefer explicit, flat code over abstractions. A function that does one clear thing beats a general one that handles hypothetical cases. Name variables after their mathematical meaning (`phi_vectors`, `influence_scores`, `adj`), not their type (`matrix`, `array`, `data`). Inline small helpers when naming them adds no clarity. One concept per function. Flat is better than nested.

## Trace assumptions explicitly

When a function depends on a non-obvious invariant (tensor shape, adjacency convention, node ordering, threshold semantics), state it in a short inline comment — this is the one place comments are expected. Example: `# adj[target, source] — same convention as AttrGraph`. Otherwise, comments explain *why*: a non-obvious algorithmic choice, a known numerical pitfall, a deliberate deviation from the paper. Never comment on what the code obviously does.

## No speculative robustness

Don't add `try/except` blocks, fallback branches, or type coercions for scenarios that can't happen in the current research workflow. If a tensor is always 2-D here, don't guard against 1-D. Don't write `if x is None` guards for values that are never None in practice. Trust callers within the same codebase.

## Validate only at research boundaries

Raise early on user-supplied inputs (CLI args, config values, files loaded from disk, API responses). Don't re-check types or shapes mid-pipeline when the caller already guarantees them.

## Numeric code

- Prefer named intermediate tensors over long chained expressions — one operation per line when the shape or meaning changes.
- Add a `# (N, d)` shape comment on a tensor the first time it appears in a non-trivial computation, not on every line.
- `eps` guards on divisions and `clamp` calls are fine; don't add them preemptively where the denominator is guaranteed nonzero by construction.

## Reproducibility

Every stochastic operation (`torch.manual_seed`, `random_state` in sklearn, etc.) must accept an explicit seed parameter — never hardcode or silently default to a random seed. Eval scripts must be runnable end-to-end from a single command with fixed seeds.

## Experiments, not frameworks

Don't create base classes, registries, or plugin systems speculatively. Three similar blocks of research code beats a premature abstraction that obscures what each experiment does differently. Eval scripts in `eval/` are scripts, not libraries — inline logic is fine, and duplication across two eval scripts is fine. Introduce an abstraction only when at least three concrete cases already exist.

## Tests cover behavior, not coverage

Write tests that verify a concrete research invariant (e.g., pruning retains the target logit, clustering produces non-overlapping supernodes). Skip tests for internal helpers unless they encode a subtle invariant worth pinning down.

## Style

- Type hints on function signatures; skip them on obvious one-liners.
- No multi-line docstrings. A single-line summary is enough if the name isn't self-explanatory.
- Modern Python type syntax: `X | Y`, `X | None`, `list[T]`, `dict[K, V]` — not `typing.*` forms.
