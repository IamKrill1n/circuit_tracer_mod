# Running the pipeline

Full attribution requires a GPU with CUDA:

```bash
python -m summarization --prompt "..." --model google/gemma-2-2b --transcoder mntss/clt-gemma-2-2b-2.5M
```

Use `--graph-pt` to load a saved graph instead of generating attribution, and `--graph-pt-out` to save a graph.

Add `--jev-query` to drop features TypeSafe Jev judges irrelevant to a mechanistic claim plus
what you want to see out of the circuit (requires `TYPESAFE_API_KEY`):

```bash
python -m summarization --graph-pt demos/000.pt --jev-query "The model identifies the country, then retrieves its capital. Show country and capital features."
```

Tune with `--jev-threshold` (default 0.5), `--jev-model`, `--jev-features-per-request`,
`--jev-max-state-chars`, and `--jev-max-concurrent-requests`/`--jev-max-concurrent-fetches`.

For claim-conditioned token weights, use `--claim` with either `--selector-model` (selector LLM spans,
default) or `--seed-selector jev` (per-token Jev Nouls, requires `TYPESAFE_API_KEY`):

```bash
python -m summarization --graph-pt demos/000.pt --claim "..." --seed-selector jev
```

With `--seed-selector jev`, tokens above `--jev-threshold` are grouped into runs with equal mass per
run; tune batching with `--jev-tokens-per-request` (default 64). When no token clears the threshold,
the run prunes by influence only (`arithmetic`, alpha=1) and records `fallback="output_only"` in the
graph metadata and the pipeline result.
