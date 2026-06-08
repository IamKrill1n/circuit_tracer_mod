---
name: circuit-methods
description: Look up methodology details from the Attribution Graphs paper (transformer-circuits.pub/2025/attribution-graphs/methods.html). Use when asked about cross-layer transcoders, replacement models, attribution graph construction, graph pruning, supernode grouping, intervention validation, faithfulness evaluation, or any technical details from the circuit-tracing methods paper.
---

# Circuit Methods Reference

When this skill is invoked, fetch the methods paper and answer the user's question from it.

## Source

URL: https://transformer-circuits.pub/2025/attribution-graphs/methods.html

## What this page covers

The paper describes the full methodology for circuit tracing in transformer language models:

- **Replacement model** — substituting MLPs with cross-layer transcoders (CLTs) to produce an interpretable forward pass
- **Attribution graph construction** — computing a per-prompt directed graph where nodes are features/tokens/logits and edges are direct attribution scores
- **Graph pruning** — reducing the dense graph to a sparse, interpretable circuit
- **Supernode grouping** — clustering features into functional groups
- **Intervention / validation** — testing mechanistic hypotheses by patching activations
- **Faithfulness evaluation** — measuring how well the replacement model approximates the original
- **Case studies** — acronyms, factual recall, arithmetic addition

## Workflow

1. Fetch the page with a targeted prompt that extracts the section relevant to the user's question.
2. Answer the user directly and concisely, citing section names where helpful.
3. If the user's question spans multiple sections, make multiple focused fetches rather than one broad one.

## Fetch instructions

Use `WebFetch` with:
- url: `https://transformer-circuits.pub/2025/attribution-graphs/methods.html`
- prompt: a specific extraction prompt matching the user's question (e.g. "Explain how graph pruning works, including the scoring formula and threshold selection.")

Keep fetches targeted — ask for one section at a time to get precise answers.
