# Code Style for Research Code

This is a research codebase. Prioritize in this order: **readability → reproducibility → maintainability → correctness on real inputs**. Do not sacrifice the first three for exhaustive defensive programming.

## Clarity over cleverness

Name variables after their mathematical meaning (`phi_vectors`, `influence_scores`, `adj`) not their type (`matrix`, `array`, `data`). One concept per function. Flat is better than nested.

## No speculative robustness

Don't add `try/except` blocks, fallback branches, or type coercions for inputs that won't occur in practice. If a tensor is always 2-D here, don't guard against 1-D. Trust callers within the same codebase.

## Validate only at research boundaries

Raise early on user-supplied inputs (CLI args, config values, files loaded from disk). Don't validate internal function arguments passed between modules you control.

## Numeric code

- Prefer named intermediate tensors over long chained expressions — one operation per line when the shape or meaning changes.
- Add a `# (N, d)` shape comment on a tensor the first time it appears in a non-trivial computation, not on every line.
- `eps` guards on divisions and `clamp` calls are fine; don't add them preemptively where the denominator is guaranteed nonzero by construction.

## Reproducibility

Every stochastic operation (`torch.manual_seed`, `random_state` in sklearn, etc.) must accept an explicit seed parameter — never hardcode or silently default to a random seed. Eval scripts must be runnable end-to-end from a single command with fixed seeds.

## Experiments, not frameworks

Eval scripts in `eval/` are scripts, not libraries. Avoid introducing abstractions (base classes, registries, plugin systems) unless at least three concrete cases already exist. Inline logic is fine; duplication across two eval scripts is fine.

## Comments explain intent, not mechanics

Only comment on the *why*: a non-obvious algorithmic choice, a known numerical pitfall, a deliberate deviation from the paper. Don't comment on what the code obviously does.
