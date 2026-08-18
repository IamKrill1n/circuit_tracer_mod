# Steering the attribution-graph project toward AI-safety benchmarks

Research date: 2026-08-15

## Executive answer

Attribution graphs are a credible tool for AI-safety research, but the publishable contribution
should not be another attractive graph for one prompt. Anthropic has already shown qualitative
attribution-graph case studies for hallucination, refusal, a jailbreak, chain-of-thought (CoT)
faithfulness, and a hidden-goal model. The open opportunity is **benchmark-scale, held-out,
causally validated graph analysis**: define a safety-relevant behavioral contrast, discover
recurring mechanisms over many controlled examples, and show that the graph predicts selective
interventions in the underlying model. This framing follows both the demonstrated safety use cases
and the stated limitations in [On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html).

The best first flagship is **SycoBench-600** on an instruction-tuned model with a qualified
transcoder. It supplies baseline, misleading-pressure, and correct-correction conditions, so the
mechanistic claim can distinguish undesirable capitulation from legitimate updating. **MASK** is
the strongest follow-up because it explicitly separates belief accuracy from honesty under
pressure. **WMDP**, the 2025 binary version of **TruthfulQA**, and **BBQ** are useful immediate
engineering pilots with the current base-model setup, but they are respectively about hazardous
knowledge, truthfulness, and social bias rather than the richer aligned behavior of an assistant.
**HarmBench**, CoT faithfulness, and backdoor/model-organism tasks are high-value later projects
that require sequence-level targets and population-level graphs.

A single prompt-specific, final-position next-token graph is useful when the claim is exactly a
local decision such as “why did the model choose answer B rather than C?” It is not sufficient to
explain refusal, honesty, a jailbreak, a generated rationale, or a hidden policy in general. The
minimum unit for most safety claims is a **contrastive, paired-condition graph collection**; for
generated behavior it must also be **multi-token**, and for a mechanism claim it must be
**population-level and validated on held-out examples**.

## What the repository currently measures

The following are verified from the current code, not assumptions:

| Capability | Current behavior | Consequence for safety research |
|---|---|---|
| Attribution target | Both attribution backends construct targets from the final input position; the NNSight path explicitly uses `ctx.logits[0, -1]` ([source](../circuit_tracer/attribution/attribute_nnsight.py#L175)). | There is no native sequence-level behavioral target. |
| Contrastive direction | `CustomTarget` accepts an arbitrary residual/unembedding direction, including a logit-difference direction ([source](../circuit_tracer/attribution/targets.py#L29)). String targets themselves must tokenize to exactly one token ([source](../circuit_tracer/attribution/targets.py#L293)). | A correct-vs-wrong answer-margin graph is already possible at one position; a multi-token answer is not. The present target-probability faithfulness evaluator assumes a real vocabulary ID, so it must be generalized before validating a virtual custom target. |
| Output-side pruning | Influence is seeded from the marked target logit, output probabilities, or a caller-provided full node seed ([source](../summarization/prune.py#L463)). | The plumbing can use a task target, but the default “predicted token” seed is usually the wrong safety estimand. |
| Input-side pruning | Relevance is seeded uniformly over prompt embeddings or by caller-provided token/full-node weights, then combined with influence by rank or min-max normalization ([source](../summarization/prune.py#L476)). | The input boundary should be defined by the controlled manipulation, not generic SHAP relevance, for paired safety experiments. |
| Signs | Path scoring normalizes absolute edge weights, although the retained adjacency remains signed ([source](../summarization/prune.py#L33)). | Positive and suppressive safety pathways can rank identically or cancel after aggregation unless scored separately. |
| Pruning validation | The current causal pruning evaluation zeros omitted CLT features and measures `P(target | pruned) / P(target | baseline)` ([source](../eval/eval_faithfulness.py#L1)). | This must become retention of a task contrast, plus behavioral specificity controls. |
| Summary scope | `SummaryGraph` metadata records prompt, target token, scan, and prompt tokens for one graph ([source](../summarization/summarize.py#L216)); there is no cross-example graph object. | A dataset-level circuit needs a new aggregation/evaluation layer rather than repeated independent prose summaries. |
| Model paths | Tests exercise base Gemma-2-2B and Llama-3.2-1B CLTs ([Gemma](../tests/test_transformerlens_nnsight_same_gemma_clts.py#L24), [Llama](../tests/test_transformerlens_nnsight_same_llama_clts.py#L21)); they also exercise Gemma-3-1B-IT with public per-layer transcoders, while the tested Gemma-3 CLT is a 270M pretrained model ([source](../tests/test_attributions_gemma3_nnsight.py#L529)). | Software support for chat formatting exists, but it is not evidence that a sufficiently capable instruction model has a faithful CLT. |

The original method intentionally builds prompt-specific local causal graphs. Its default pruning
preserves paths influencing output logits, and its authors stress that graph hypotheses require
interventions because the replacement model can differ from the underlying model. They also report
that perturbation agreement deteriorates over multiple downstream layers and that the method omits
the causes of attention-pattern formation ([Attribution Graphs methods and limitations](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)). Those limitations become more important on long chat prompts, adversarial suffixes, and CoT.

## Redefine relevance around a safety claim

Before pruning a graph, specify four objects:

1. **Behavioral outcome `J`**: the quantity the safety claim is about, such as the correct-vs-wrong
   answer margin, consistency with an elicited belief, refusal-vs-compliance sequence score, or a
   benchmark classifier score.
2. **Controlled input contrast `C`**: the exact intervention that defines the claim, such as wrong
   pressure vs correct correction, trigger absent vs present, harmful vs minimally matched benign,
   or ambiguous vs disambiguated context.
3. **Graph unit `G`**: one answer-position graph, a teacher-forced graph at every response position,
   a matched pair/difference of graphs, or a cohort of graph pairs.
4. **Validation intervention `do(V)`**: a predeclared feature or supernode edit in the underlying
   model and the predicted direction of change in `J`.

This yields a task-specific replacement for the current generic pruning score:

- Seed output influence with the gradient/direction of `J`, not with the probability of the sampled
  next token. For a single-token multiple-choice task, use `logit(correct) − logit(wrong)` through
  `CustomTarget`.
- Seed input relevance from `C`. For SycoBench this is the added pressure/correction turn; for BBQ
  it is the disambiguating evidence; for a backdoor it is the trigger. Uniform token relevance or
  SHAP over the whole prompt answers a different question.
- Score helpful and suppressive paths separately. A safety mechanism often contains a detector
  that raises refusal and a jailbreak route that suppresses it. Absolute path mass is useful for
  finding both, but the retained report must preserve direction and should maintain separate
  positive and negative budgets.
- Add population stability to the pruning criterion. A node or edge should receive a support rate,
  signed effect distribution, and condition-specific occurrence rate over examples. Do not average
  raw edges before aligning features, semantic token roles, and output positions.
- Select the pruning point on a causal frontier: graph size versus retained task contrast and
  held-out intervention-effect prediction. The original paper similarly evaluates pruning as a
  size-versus-completeness/replacement frontier, rather than treating a threshold as intrinsically
  meaningful ([Attribution Graphs, graph pruning](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)).

For a matched pair, a useful graph proxy is `D(v) = effect(v on J | condition 1) − effect(v on J |
condition 0)`. The causal target is stronger: the difference between the effect of intervening on
`v` in the two conditions. Rank with the graph proxy, then test the causal difference on held-out
items. This prevents “a feature activates in unsafe examples” from being mistaken for “the feature
causes the unsafe behavior.”

## Is one graph for one next token sufficient?

| Claim | Minimum useful graph unit | Is the current unit enough? |
|---|---|---|
| Why this prompt chose answer `B` over `C` | One contrastive final-position graph | Yes, if both answer tokens are single tokens and option order is controlled. |
| Why user pressure flipped an answer | Baseline/pressure/correction graph triplet, aligned by item and semantic role | No. One graph cannot separate the question circuit from the pressure-induced change. |
| Why the model refused or complied | Paired harmful/benign or attacked/unattacked conditions plus graphs over the response prefix | Only as an early-token proxy; refusal is a sequence-level behavior. |
| Whether a rationale faithfully mediated the answer | Graphs over the hint, CoT trajectory, and answer, with a mediation intervention | No. A final answer graph omits how the generated rationale changed the later state. |
| Whether a trigger implements a backdoor | Triggered/clean pairs across many tasks, paraphrases, and response positions | No. A single trigger example cannot establish a reusable policy. |
| Whether a mechanism is general | A stratified cohort graph with held-out causal tests | No. Independent summaries provide anecdotes, not a population claim. |

Recommended hierarchy:

1. **Local token graph**: one position and one contrastive target.
2. **Trajectory graph**: teacher-force the generated response and build a graph for each scored
   position; aggregate node/edge effects with explicit position weights.
3. **Paired delta graph**: align local or trajectory graphs across a controlled condition pair and
   report shared, gained, lost, and sign-flipped mechanisms.
4. **Cohort circuit**: group recurring features/supernodes across examples; store support and effect
   distributions rather than only a mean graph.

The first level remains valuable for hypothesis generation. It should not be presented as an
explanation of an entire safety benchmark. The original methods paper itself says that its graphs
describe mechanisms on a particular prompt and that naive global weights are difficult to
interpret because of interference ([Attribution Graphs, global circuits](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)).

## Benchmark candidates

### 1. SycoBench-600: best first benchmark-scale mechanistic study

**Why this benchmark.** SycoBench-600 contains 600 controlled English multiple-choice instances
across eight domains and three difficulty tiers. Each item is evaluated under doubt, authority,
explicit wrong suggestion, and a matched correct-suggestion condition. Its correction-selectivity
metric explicitly separates accepting valid correction from capitulating to wrong pressure. The
dataset, raw logs, prompt variants, metrics, and validation code are public
([paper](https://aclanthology.org/2026.findings-acl.1759/),
[official artifact](https://github.com/debu-sinha/sycobench-600)).

- **Mechanistic hypothesis:** evidence/answer features support the baseline answer; social-pressure
  features can route into an answer-update mechanism. Selective models gate that update on evidence
  quality, whereas sycophantic flips use the pressure route without corrective evidence.
- **Graph unit:** for each normalized question, a triplet consisting of baseline, one wrong-pressure
  variant, and the correct-correction variant. Build a contrastive graph at the answer marker for
  `logit(correct letter) − logit(pressure-favored wrong letter)`. Counterbalance answer letters and
  align tokens by semantic roles, not absolute positions.
- **Pruning/relevance:** output influence is the answer margin. Input relevance begins at the added
  pressure/correction turn. Population relevance is the difference between wrong-pressure flip
  effects and correct-correction update effects. Preserve separate paths that promote the proposed
  answer and suppress the previous answer.
- **Validation intervention:** ablate or patch pressure-exclusive supernodes and predict that wrong
  flips decrease while valid corrections remain. Patch evidence-quality supernodes from the correct
  condition into the wrong-pressure condition and predict restoration of the correct answer. Test
  all choices before inspecting held-out effects.
- **Metrics:** official baseline accuracy, pressure-robust accuracy, flip-to-wrong rate, no-change
  rate under correct correction, and correction selectivity; add held-out intervention direction
  accuracy, effect-size rank correlation, retained answer margin after pruning, and collateral
  damage on unpressured questions.
- **Feasibility:** the answer decision can be reduced to one token and the repository already
  supports custom logit directions. The meaningful behavior requires an instruction-tuned model;
  Gemma-3-1B-IT plus its public per-layer transcoders is an engineering path, but model accuracy and
  sycophantic variation must be screened before graph generation.
- **Risks/confounds:** option-letter circuits, prompt-length/position shifts, generic agreement
  language, baseline errors, and a model that is too weak to distinguish evidence from pressure.
  Condition all flip metrics on baseline-correct items exactly as the benchmark does. Treat
  per-layer-transcoder results separately from CLT results.

### 2. MASK: honesty versus belief accuracy

**Why this benchmark.** MASK is a public, human-collected benchmark designed to test whether a model
contradicts its own elicited beliefs when pressured to lie, rather than conflating honesty with
factual accuracy. Its official release includes multiple elicitation prompts and pressure
scenarios ([project page](https://www.mask-benchmark.ai/),
[official code](https://github.com/centerforaisafety/mask),
[dataset](https://huggingface.co/datasets/cais/MASK)).

- **Mechanistic hypothesis:** a proposition/belief circuit is present in both conditions, but
  pressure or role-following pathways alter the reporting circuit. An honest model routes belief
  evidence to the answer despite pressure; a dishonest response suppresses or bypasses that route.
- **Graph unit:** a matched set containing belief elicitation, neutral reporting, and pressured
  reporting for the same proposition. For binary items, target the reported yes/no margin; define
  its sign relative to the model's elicited belief, not the external ground truth.
- **Pruning/relevance:** output influence is consistency with elicited belief. Input relevance is
  seeded on the incentive, system-role, or pressure span. A second analysis can target factual
  correctness, making the honesty and accuracy circuits directly comparable.
- **Validation intervention:** patch belief-supporting supernodes from elicitation into pressured
  reporting, or ablate pressure-specific routes. The predicted result is greater belief-consistent
  reporting without changing the separately measured belief answer. Include reverse patches and
  unrelated proposition controls.
- **Metrics:** MASK honesty and accuracy; causal change in belief consistency; unchanged belief
  elicitation; task specificity; graph completeness/replacement; and held-out effect prediction.
- **Feasibility:** binary final answers fit the current target machinery, but realistic pressure is
  chat/instruction behavior. A base model is a poor scientific substitute even if it can emit yes
  or no.
- **Risks/confounds:** an elicited answer is still a behavioral proxy for belief, prompt wording can
  change the belief itself, and a pressure intervention may alter epistemic state rather than only
  reporting. Require consistency across the benchmark's multiple belief elicitations and avoid
  claiming direct access to “true internal belief.”

### 3. WMDP: hazardous knowledge and unlearning for an immediate base-model pilot

**Why this benchmark.** WMDP publicly releases 3,668 multiple-choice questions in biosecurity,
cybersecurity, and chemical security, along with forget corpora, RMU code, and several unlearned
models. It is explicitly intended both as a hazardous-knowledge proxy and an unlearning benchmark
([official repository and data links](https://github.com/centerforaisafety/wmdp),
[paper](https://arxiv.org/abs/2403.03218)).

- **Mechanistic hypothesis:** correct hazardous answers depend on recurring domain-knowledge
  retrieval paths. A successful unlearning method should selectively disrupt those paths rather
  than merely add an answer-suppression or refusal route.
- **Graph unit:** one graph per question for `logit(correct choice) − logit(strongest distractor)`,
  repeated under option permutations. At dataset scale, build domain-stratified cohorts. For
  unlearning, compare the original and edited checkpoints on matched items.
- **Pruning/relevance:** target the answer margin. Seed input relevance on the question concepts
  and answer-content spans, not the output letter alone. For model diffing, rank mechanisms by
  changed causal effect on the margin while requiring stable effects on matched retain questions.
- **Validation intervention:** ablate candidate knowledge supernodes in the original model and
  predict a selective WMDP margin loss; restore/patch them into the unlearned model only if feature
  identity is justified. Measure effects on retain-domain questions and ordinary language-model
  loss.
- **Metrics:** WMDP accuracy/margin by domain, general-capability retention, intervention
  selectivity, option-permutation robustness, graph completeness/replacement, and cross-question
  recurrence.
- **Feasibility:** this is the cleanest safety dataset for the existing base Gemma/Llama CLTs,
  provided the chosen small model performs meaningfully above chance. Run a black-box capability
  screen before any expensive graphs.
- **Risks/confounds:** WMDP measures a proxy for hazardous knowledge, not end-to-end harmful
  capability; multiple-choice shortcuts and contamination are possible; the official dataset has
  had formatting/content revisions; and small models may have no mechanism worth studying. A CLT
  trained for the original checkpoint is not automatically valid after unlearning. Comparing
  checkpoints requires a qualified decomposition per checkpoint or a demonstrated reconstruction
  and feature-alignment procedure.

### 4. HarmBench: refusal and jailbreak robustness

**Why this benchmark.** HarmBench provides public harmful behaviors, precomputed attack test cases,
completion generation, and classifiers in a standardized pipeline covering attacks and defenses
([official repository](https://github.com/centerforaisafety/HarmBench),
[paper](https://arxiv.org/abs/2402.04249)). Refusal also has falsifiable mechanistic baselines:
Arditi et al. found a direction whose removal suppresses refusal and whose addition induces refusal
([paper and code](https://github.com/andyrdt/refusal_direction)), while newer work reports distinct
directions across refusal categories despite a shared refusal/over-refusal control trade-off
([Joad et al., 2026](https://arxiv.org/abs/2602.02132)).

- **Mechanistic hypothesis:** specific harm detectors feed a more general refusal policy; jailbreaks
  can prevent harm recognition, suppress refusal propagation, or strengthen competing compliance
  and continuation pathways. Which failure occurs should vary by attack and harm category.
- **Graph unit:** a four-way matched set where possible: harmful/refused, harmful/jailbroken,
  benign/complied, and benign/over-refused. For each prompt, trace the response trajectory, not only
  its first token. A first-token refusal-vs-compliance token-set margin is an exploratory proxy.
- **Pruning/relevance:** use a teacher-forced refusal-vs-compliance sequence log-likelihood margin,
  or a validated differentiable readout of the eventual behavior. Seed input relevance on the
  harmful semantic span and separately on the adversarial suffix. Maintain positive refusal,
  suppression, and compliance path budgets.
- **Validation intervention:** restore harm-detection supernodes from the clean harmful prompt into
  a successful jailbreak; ablate proposed refusal-execution supernodes; patch attack-specific
  compliance routes; and test effects on both harmful and benign held-out prompts. Compare graph
  supernodes against the simple refusal-direction baseline.
- **Metrics:** HarmBench attack success/classifier score, refusal rate on harmful inputs,
  over-refusal on benign inputs, per-category generalization, effect prediction, and graph
  compression at fixed behavioral fidelity.
- **Feasibility:** requires an instruction-tuned model with nontrivial refusal behavior and a
  faithful dictionary on chat/adversarial prompts. The repository can run formatted instruction
  prompts, but the current base CLTs do not supply the safety policy to analyze.
- **Risks/confounds:** refusal templates are surface-correlated with safety; classifiers can confuse
  disclaimers with non-compliance; harmful/benign prompts are not automatically minimal pairs;
  attacks shift length and token positions; and attention-pattern causes are omitted. Anthropic's
  jailbreak study specifically found competing refusal/compliance circuits, but also warns that
  different jailbreaks may use different mechanisms
  ([Biology, jailbreak case study](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)).

### 5. TruthfulQA and BBQ: controlled shakedown tasks, not the final flagship

**TruthfulQA.** The official repository now recommends a binary version pairing the best answer
with the best incorrect answer; it also retains generation, MC1, and MC2 tasks
([dataset and evaluation code](https://github.com/sylinrl/TruthfulQA),
[paper](https://aclanthology.org/2022.acl-long.229/)).

- **Mechanistic claim and graph:** contrast misconception retrieval with truth-supporting evidence
  using `logit(best true option) − logit(best false option)`, with option order randomized. Aggregate
  by misconception category.
- **Intervention and metrics:** ablate misconception or truth supernodes and predict direction of
  answer-margin change; report binary accuracy, calibration/margin, permutation robustness,
  causal specificity, and recurrence.
- **Fit and risk:** ideal for testing custom contrastive targets on current base CLTs, but it
  measures truthfulness/knowledge rather than honesty. The repository notes that the 2025 revision
  removed outdated items but has not fully revalidated all remaining questions. Generated
  truthfulness/informativeness is multi-token and should wait for trajectory support.

**BBQ.** BBQ provides ambiguous and disambiguated question-answering conditions across social-bias
categories. The ambiguous setting tests reliance on stereotypes when evidence is insufficient; the
disambiguated setting tests whether stereotypes override an informative context
([official dataset](https://github.com/nyu-mll/BBQ),
[paper](https://aclanthology.org/2022.findings-acl.165/)).

- **Mechanistic claim and graph:** compare a stereotype-prior route, an uncertainty/unknown route,
  and an evidence route across matched ambiguous/disambiguated examples. Target the stereotyped-vs-
  unknown margin in ambiguous cases and correct-vs-stereotyped margin in disambiguated cases.
- **Intervention and metrics:** patch disambiguating-evidence supernodes into the ambiguous graph,
  ablate stereotype-associated routes, and test whether bias falls without reducing accuracy on
  counter-stereotypical evidence. Report official accuracy/bias metrics plus causal specificity,
  counterbalanced names/options, and held-out category transfer.
- **Fit and risk:** multiple choice and controlled pairs fit the current base-model machinery. It is
  valuable responsible-AI work but a weaker fit to catastrophic-risk questions. Names, identities,
  answer positions, and template artifacts can masquerade as mechanisms; report category-specific
  results and inspect examples with care.

### 6. CoT faithfulness: high value after multi-token graph support

Turpin et al. demonstrate unfaithful CoT by adding biasing features such as a suggested answer or
systematic answer-order cues that change the answer without being acknowledged in the rationale
([paper](https://arxiv.org/abs/2305.04388)). A complementary intervention-based benchmark changes,
paraphrases, truncates, or corrupts the CoT and measures how answers depend on it
([Lanham et al.](https://www.anthropic.com/research/measuring-faithfulness-in-chain-of-thought-reasoning)).

- **Mechanistic hypothesis:** faithful examples route task evidence through intermediate reasoning
  states into the answer; unfaithful examples route the hint directly into an answer choice and may
  then generate a post-hoc rationale by working backward.
- **Graph unit:** unhinted/hinted prompt pairs, graphs throughout a teacher-forced rationale, and a
  contrastive answer graph. Estimate whether hint-to-answer influence is mediated by claimed CoT
  steps or bypasses them.
- **Pruning/relevance:** target the final answer margin while seeding relevance separately from the
  hint and task evidence. Add a mediation score: how much of the hint effect disappears when the
  claimed reasoning supernodes are patched or ablated.
- **Validation intervention:** patch task-evidence states across pairs, suppress hint pathways, or
  alter proposed reasoning supernodes and predict both answer and later rationale changes.
- **Metrics:** answer-flip rate, hint verbalization, CoT intervention sensitivity, causal mediation,
  held-out intervention prediction, and standard task accuracy.
- **Feasibility and risks:** instruction/reasoning capability and many output-position graphs are
  required. The attribution method does not trace why attention patterns form. Anthropic's own CoT
  case study says this missing QK mechanism prevented its graph from explaining why the model
  attended to a human answer hint
  ([Biology, CoT limitation](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)).
  This makes the task scientifically important but a poor first implementation target.

### 7. Backdoors and model organisms: ambitious model-diffing studies

The Sleeper Agents release provides code-backdoor training data, generation prompts, and random
samples for deceptive backdoor experiments, but not a turnkey set of open model checkpoints
([paper](https://arxiv.org/abs/2401.05566),
[official data repository](https://github.com/anthropics/sleeper-agents-paper)). Emergent
Misalignment releases training/evaluation code and data and studies open Qwen2.5-Coder base and
instruction models, including the finding that insecure-code fine-tuning can induce broader
misaligned behavior ([peer-reviewed paper](https://proceedings.mlr.press/v267/betley25a.html),
[official repository](https://github.com/emergent-misalignment/emergent-misalignment)).

- **Mechanistic hypothesis:** a trigger or narrow fine-tuning distribution activates a reusable
  policy/persona/goal representation that changes many downstream behaviors, rather than a separate
  memorized output circuit for every prompt.
- **Graph unit:** clean/triggered prompt pairs over many task contents and paraphrased triggers;
  aligned/misaligned checkpoint pairs; response trajectories; and a population circuit for the
  hypothesized policy.
- **Pruning/relevance:** target a sequence-level backdoor or misalignment score. Seed relevance on
  the trigger or fine-tuning-correlated context, and require cross-task recurrence. Separate trigger
  recognition, policy activation, and behavior execution.
- **Validation intervention:** ablate or transplant proposed goal supernodes, then test unseen
  prompts and trigger paraphrases. A successful intervention should change the target behavior
  without globally destroying instruction-following or fluency.
- **Metrics:** attack/backdoor success, clean performance, unseen-trigger and unseen-task transfer,
  intervention specificity, graph recurrence, and replacement-model faithfulness on both
  checkpoints.
- **Feasibility and risks:** current public model organisms do not align with the repository's
  qualified CLTs. Fine-tuning changes the model whose MLPs the transcoder approximates, so applying
  the original dictionary after training can create a fictitious model difference. Feature drift,
  subjective misalignment judges, persona-style surface features, and checkpoint scale make this a
  later project. The hidden-goal case in Anthropic's Biology paper is proof of relevance, not an
  unsolved novelty claim.

## Model feasibility: base, instruction, and fine-tuned checkpoints

Do not collapse these three questions:

1. **Can the software load the architecture?** The NNSight backend and tests cover several model
   families and chat-formatted input paths.
2. **Does a trained decomposition exist for the exact checkpoint?** Transcoders are tied to model
   weights. A base-model CLT is not automatically a decomposition of the instruction-tuned or
   safety-fine-tuned checkpoint.
3. **Does the checkpoint exhibit the benchmark behavior?** A 1B base model may run WMDP or BBQ but
   remain at chance; it cannot substitute for an assistant with learned refusal, honesty under
   pressure, or sycophancy.

Before choosing a benchmark-model pair, run a cheap qualification gate:

- behavioral performance above the task's floor and enough positive/negative examples for the
  intended contrast;
- transcoder reconstruction, completeness, and replacement scores on a stratified sample of the
  actual safety prompts, including chat scaffolding and adversarial inputs;
- stability across prompt template and option permutations;
- causal agreement for a small set of feature interventions in the underlying model;
- comparison to neurons or per-layer transcoders if a CLT does not exist.

The methods authors explicitly allow per-layer transcoders or neurons as alternatives, although
they found CLTs produce shorter and more parsimonious paths and note that training a CLT has a
significant up-front cost ([Attribution Graphs methods](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)). It is therefore reasonable to prototype SycoBench with the public
Gemma-3-1B-IT transcoders, but the result should be labeled as such and should not inherit CLT
faithfulness claims.

## Validation standard for a benchmark-scale mechanistic claim

Every result should have four layers of evidence:

1. **Behavioral validity:** reproduce the official benchmark metric and manually inspect a random,
   non-cherry-picked sample. For generative tasks, keep the benchmark's external classifier/judge
   separate from the differentiable graph target.
2. **Graph validity:** report unpruned and pruned completeness/replacement, retained task margin,
   error-node contribution, and signed path mass. The methods paper defines completeness and
   replacement precisely to expose computation passing through reconstruction errors
   ([source](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)).
3. **Mechanistic validity:** freeze the hypothesis before testing held-out items; intervene in the
   underlying model; predict direction and relative size; include random-feature, activation-
   magnitude, simple probe, and direct-attribution baselines. The original paper found graph
   influence more predictive of ablation effects than direct attribution or activation magnitude,
   but also found compounding perturbation discrepancies, so this must be re-established on the
   safety distribution ([source](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)).
4. **Specificity/generalization:** test benign or retain controls, unseen prompt variants,
   categories, and templates. A feature that merely controls the answer token, refusal phrase, or
   chat syntax is not the claimed safety mechanism.

Use a discovery/validation/test split at the **item or normalized-question level**, not at the graph
level. Graph labeling, supernode grouping, pruning thresholds, intervention strengths, and any LLM-
generated interpretations are discovery choices. Freeze them before the test split. Report failed
interventions; attribution graphs are hypothesis generators, not causal proof by themselves.

## Prioritized roadmap

### Phase 0: task-target and paired-graph infrastructure

- Add a benchmark record schema: benchmark, item ID, normalized pair ID, condition, model/checkpoint,
  tokenizer, prompt template, response prefix, output position, target definition, intervention
  split, and random seed.
- Make a first-class contrastive target for a signed logit margin and evaluate pruning by retained
  margin rather than only target-token probability.
- Add paired graph alignment by model feature identity plus semantic token role; represent shared,
  gained, lost, and sign-flipped paths.
- Add signed positive/suppressive pruning budgets and population support/effect statistics.

Exit criterion: on a controlled synthetic or existing analogy task, the paired graph predicts
held-out answer-margin intervention effects better than activation magnitude and direct attribution.

### Phase 1: immediate base-model pilots

Run the binary TruthfulQA task and a small, domain-stratified WMDP subset on the existing Gemma or
Llama CLT path. Optionally use BBQ to stress paired ambiguous/disambiguated aggregation. These are
methodology shakedowns, not the headline safety claim.

Exit criteria: model performance above chance, robust answer margins under option permutation,
qualified replacement metrics on task prompts, and at least one recurring circuit whose held-out
intervention is selective.

### Phase 2: first flagship on SycoBench-600

Black-box screen Gemma-3-1B-IT for baseline correctness, wrong-pressure flips, and correct updates.
If there are enough events, qualify the public per-layer transcoders on the benchmark and run the
baseline/pressure/correction graph triplets. If the model or decomposition fails the gate, move to
a more capable open instruction model and budget for training a suitable transcoder rather than
forcing a negative-quality setup.

Primary claim: a population-level graph distinction predicts which internal routes cause
capitulation versus valid correction, and targeted interventions improve correction selectivity on
held-out questions better than simple activation-difference or probe baselines.

### Phase 3: honesty with MASK

Reuse the paired infrastructure but maintain two outcomes: belief consistency and factual accuracy.
This is the cleanest test of whether the method can separate “what the model reports” from “what the
model appears to believe,” while remaining explicit that belief elicitation is a proxy.

### Phase 4: sequence targets and HarmBench

Implement teacher-forced multi-token target accumulation and trajectory aggregation. Then study
matched refusal/jailbreak/over-refusal conditions, with HarmBench behavior labels as external
validation and a simple refusal direction as the required mechanistic baseline.

### Phase 5: CoT faithfulness and model organisms

Only after trajectory graphs, attention-related limitations, and checkpoint-specific decomposition
are addressed should the project claim mechanisms for CoT mediation, backdoors, hidden goals, or
emergent misalignment. These are the highest-upside tasks and the easiest places to overinterpret a
local graph.

## Decision recommendation

Start by implementing **contrastive targets, paired delta graphs, signed pruning, and held-out
causal evaluation**. Use TruthfulQA/WMDP/BBQ only to debug that machinery. Make **SycoBench-600 the
first intended paper result**, subject to a model/decomposition qualification gate, then extend to
MASK. This produces a sharper contribution than “summarize attribution graphs on safety data”:

> Discover and causally validate population-level circuits that distinguish robust belief updating
> from unsafe social-pressure compliance.

That claim is task-specific, measurable, falsifiable, and directly forces both methodology changes
the project currently needs: relevance/pruning matched to a mechanistic estimand, and graph
summarization that moves beyond one prompt and one next token.
