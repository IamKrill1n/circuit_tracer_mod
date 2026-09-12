# Pruning to test a mechanistic claim

## Decision

Replace SHAP-guided token relevance with a target tied to the scientific question. Start with one graph per example containing multiple output targets; joint summarization across graphs is unnecessary.

“Is the model lying?” names a behavior to detect. A testable mechanistic claim is narrower:

> In this run, the model represents the correct fact, but an incentive-sensitive computation changes its report toward the false alternative. Replacing that computation with its honest-condition state restores accurate reporting while preserving access to the fact.

This claim can fail. The graph might instead reflect confusion, an explicit opposite-answer instruction, or mere recognition of a deception-related prompt. Retain evidence against the claim rather than pruning until the desired story appears.

## Can a known lying vector be enough?

**Enough to define a detection baseline: possibly. Enough to establish the mechanism: no.** A vector is a direction in a particular model's internal representation. A linear detector scores a state using s = vᵀh + b, where h is the activation, v the fitted direction, and b a constant offset.

Attributing to s answers “what makes this detector fire?” It does not by itself answer “what causes the model to misreport?” A detector may read the incentive instruction, the topic of deception, uncertainty, or a consequence of deciding on an answer. Altering its score without altering behavior would expose that gap.

There is evidence that simple directions detect defection in constructed sleeper agents, but the authors explicitly leave generalization to natural deception open. Treat this as motivation for a baseline, not evidence for a universal lying direction. See [Simple probes can catch sleeper agents](https://www.anthropic.com/research/probes-catch-sleeper-agents).

Fit or validate the direction on the exact model revision, layer, token position, and normalization convention used for tracing. A middle-layer direction cannot simply become a final-layer target because its dimensionality matches. Separate discovery and held-out evaluation; balance answer identity and include benign deception-related language. The generated pilot prompts do not yet provide observed behavior labels or a validated vector.

## What the current code permits

Verified from local source:

- `summarization/pipeline.py::_shap_token_weights` estimates input-token importance for the graph's target token. `prune_combined` combines input relevance with backward output influence. This optimizes a different question from preservation of a misreporting mechanism; it does not prove SHAP is intrinsically unsuitable for every instruction model.
- `summarization/prune.py::normalize_matrix` takes absolute edge values before influence propagation. Signed adjacency remains in the retained graph, but ranking uses unsigned influence. Feeding +1 and −1 output seeds through this normalization is not signed mechanistic attribution.
- `circuit_tracer.attribution.attribute` accepts `attribution_targets`. `CustomTarget(token_str, prob, vec)` in `circuit_tracer/attribution/targets.py` supports arbitrary output directions. The pipeline's `_acquire_graph` does not currently expose them.
- In the TransformerLens backend, custom targets are injected at the final position and final layer, with final-normalized residual activations cached. They are not an arbitrary-layer probe interface. Check the other backend's semantics before claiming equivalent support.
- Custom targets use virtual vocabulary IDs. Existing evaluations that index token probabilities by those IDs must be adapted: a probe score or logit difference is not a token probability. The target's `prob` metadata must not be presented as an empirical probability of lying.

## First implementation: answer contrast, optional detector target

For a verified single-token reporting task, define:

**m = logit(false answer) − logit(true answer).**

A logit is the model's score before probability normalization. This margin measures preference for the false report; it does not establish knowledge or intent.

Use the difference of the corresponding unembedding directions as a custom target, while retaining true and false token targets separately. An unembedding direction maps the final representation to a token score. Check model-specific final transformations, such as score soft-capping, and validate the custom target against the actual measured margin: their numerical equivalence cannot be assumed for every model.

If a validated final-representation detector exists, add its direction as a fourth target. All targets belong to the same prompt computation and can live in one graph. Keep target scales separate; a larger vector norm must not automatically receive a larger graph budget.

**Minimum comparison:** ordinary output influence without SHAP; answer-contrast pruning; detector-target pruning; union of answer-contrast and detector candidates. Retain the current SHAP method as a baseline at matched node budgets.

## Proposed pruning procedure

1. **Construct the right graph first.** Request the true answer, false answer, and contrast explicitly. Target selection and `max_feature_nodes` can exclude useful features before this repository's pruning begins. Pruning cannot recover absent nodes.
2. **Select candidates per target.** Use backward influence magnitude to propose candidates separately for each target, with a minimum budget per target. This is a heuristic for search, not an estimate of causal mediation. Preserve signs in the retained edges and report support and opposition separately.
3. **Take the union, not only the intersection.** A truth-retrieval group might matter without affecting the lying detector; an incentive group might affect the detector without affecting reporting. Keeping both allows the claim to be tested. Overlap may prioritize inspection but should not be a hard gate.
4. **Preserve paths and alternatives.** Keep required connecting nodes and both answer boundaries. If a detector becomes an internal sink in a later implementation, update dangling-node rules: they currently expect feature nodes to lead onward. Do not force every claim-relevant feature to connect to the emitted token alone.
5. **Shrink by measured intervention fidelity.** Iteratively propose removing low-priority nodes or small groups. Accept removal only if the retained computation preserves the measured answer scores and the effects of predefined interventions within validation-selected tolerances. Use zeroing and matched activation replacement as separate experimental conditions. A zeroed feature is not necessarily its honest state.
6. **Summarize the resulting graph normally.** Preserve target identity, signs, and intervention provenance in the evidence supplied for interpretation. Names such as “lying” must not become ground-truth labels.

For step 5, record J, the change in m produced by each predefined intervention. Seek a small node set S satisfying both |m(S) − m(full)| ≤ ε and |J(S) − J(full)| ≤ δ on validation examples. Here m(S) requires an explicitly defined model intervention outside S; it is not merely a sum over displayed edges. If removing edges is also claimed to preserve computation, test edge-specific interventions separately. These constraints are a proposed operational objective, not a guaranteed minimal circuit or proof of deception.

Recheck candidate groups jointly because redundant groups may each look individually dispensable. Report absolute errors when the original margin/effect is near zero. Also measure both answer probabilities and unrelated task performance so preserving a margin by damaging everything else does not pass.

## Causal checks that decide whether the vector helps

| Result | Interpretation and next step |
|---|---|
| Probe distinguishes conditions; text does equally well | Likely little additional detection value; improve matched controls. |
| Patching probe-related features changes s but not m | Explains the detector, with no demonstrated effect on reporting. |
| Patching changes m but destroys factual recall | May disrupt knowledge or general performance rather than reporting policy. |
| Patching changes m selectively, with recall intact | Supports a reporting-related mechanism; test reverse patches and held-out scenarios. |
| Probe-only pruning misses groups needed for intervention fidelity | Add behavioral targets; a single direction was insufficient for this use case. |
| Probe-only matches richer pruning on held-out fidelity and detection | Prefer the simpler target for this bounded setting; do not claim universality. |

Matched honest and misleading runs are used to collect intervention values and validate hypotheses. Each run still produces its own summary; no cross-graph clustering is required. Align patched token positions by task meaning rather than blindly using the same index when prompts differ.

## Important limit: suppressed evidence

Active-feature graphs can omit a truth-related feature that is fully suppressed. Changing the pruning rule alone cannot reveal it. The [methods paper's discussion of inactive features and inhibitory circuits](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) identifies this limitation and motivates comparing matched prompts. Use an honest run to identify candidates for additional tracing or intervention; absence from the misleading graph is not evidence of absent knowledge.

## Next work, in order

1. Run a behavior pilot on the candidate prompts and controls. Record outputs, refusals, correctness, and separate truth-access checks. No automatic deception labels from incentive conditions.
2. Expose existing explicit/custom targets through the summarization boundary, with metadata distinguishing scores from token probabilities. Start with the answer contrast; add a vector only once its measurement location and validity are established.
3. Add per-target candidate budgets and union selection without SHAP gating. Compare equal-size graphs against ordinary output pruning and current SHAP pruning.
4. Add intervention-fidelity measurements and use them to select pruning settings. Validate final claims in the original model, stating any frozen-attention or replacement-model restrictions.
5. Only then decide whether the vector alone is enough. Success means preserving and explaining the behavior under interventions, not merely producing a graph that looks deception-related.

Candidate data live in [deception_pilot/](deception_pilot/). They are authored synthetic inputs, not model-generated experimental outcomes. Run outputs should go to `experiments/runs/<timestamp>/` with explicit seeds, parameters, model/transcoder revisions, and git commit hash. Keep discovery scenarios separate from validation/test scenarios; the small pilot is for feasibility, not a reliable detector-performance estimate.
