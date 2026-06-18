# Context Glossary

Canonical domain language for the summarization pipeline. Glossary only — no
implementation details. See `docs/adr/` for decisions.

## Supernode

A grouped cluster of circuit nodes treated as one unit in the summary graph. Carries
a display **name**, a **role**, and (after labeling) a one-sentence **description**.

## Role

The function a supernode plays in producing the prediction. Fixed vocabulary:

- **Input** — detects information directly present in the prompt (entities, tokens, syntax).
- **Abstract** — an intermediate concept, relation, or reasoning step combining multiple sources.
- **Output** — promotes candidate next-token(s); directly useful for the prediction.
- **Trash** — no consistent interpretable pattern.

## Summary graph

The post-π directed acyclic graph over supernodes. Beyond structure, it also carries the
**provenance** of the computation it summarizes: the prompt, the predicted target token, the
transcoder scan, and the prompt tokens.

## Labeling scheme

The strategy by which an LLM assigns each supernode its name/role/description. The current
labeling scheme is **one-pass**: a single whole-graph request labels every supernode together
from feature evidence.

## Labeling

The act of assigning display names, roles, and descriptions to supernodes. Labeling happens
after supernodes exist; it is not per-feature classification.

## Feature classification

Deprecated language for assigning roles to individual features. The summarization pipeline uses
supernode labeling instead.

## Activation-density filter

A pruning refinement that removes circuit nodes whose feature activation frequency is outside
the chosen interpretability band. It changes which evidence remains before supernodes are
formed, but does not assign roles or labels.

## Clustering

The stage that partitions a pruned attribution graph into supernodes.

## Role vector

A vector representation of one circuit node's upstream and downstream contribution pattern,
used to decide which nodes play similar roles in the computation.

## Feature evidence

The local evidence used to interpret a feature within the current computation. Feature
evidence identifies where the feature appears in the computation and gives activation/logit
signals, but it is distinct from the supernode's final display label.

## Model registry

The catalogue mapping a **model name** to its **provider**, endpoint, credential source, and
default generation settings. The single place that knows how to reach and configure a model.

## Provider

The backend family serving a model: official OpenAI, Google Gemini, or a generic
OpenAI-compatible endpoint (self-hosted or third-party).
