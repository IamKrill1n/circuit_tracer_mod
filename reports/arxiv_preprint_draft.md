# Faithful Summarization of Attribution Graphs for Circuit Tracing

**Phạm Anh Tú**  
*Affiliation and contact information to be added before submission*

*Draft preprint — August 2026*

## Abstract

Circuit tracing can express a language model's computation for a particular prompt as a signed attribution graph over interpretable features. These graphs support mechanistic hypotheses, but their scale makes them difficult to inspect: a single graph may contain thousands of feature nodes and a dense set of causal edges. We present an automated pipeline that converts an attribution graph into a compact, labelled summary graph while explicitly tracking the faithfulness costs of compression. The pipeline first ranks nodes and edges by both their downstream influence on a selected output and their upstream relevance to selected input tokens. It then groups retained feature nodes with a binary integer linear program that balances similarity of signed causal roles against the causal edge mass hidden inside groups. Finally, it aggregates feature-level edges, removes cyclic directions, and labels the resulting supernodes for inspection. On 100 analogy prompts derived from BATS, relevance-aware pruning preserves replacement and completeness scores at matched graph sizes relative to influence-only pruning. At a matched summary size, the proposed clusterer retains more visible feature-to-feature causal structure than generic baselines, though K-means produces tighter role-vector clusters. On 6,765 measured summary edges, interventions show high agreement between predicted and observed edge direction, with sign consistency of 0.829 for feature targets and 0.762 for logit targets; edge magnitude also correlates with measured response magnitude. These results position graph summarization as a testable compression layer for circuit tracing rather than a purely visual post-processing step.

## 1. Introduction

Mechanistic interpretability aims to explain a model's behavior in terms of the internal computations that produce it, rather than only associating inputs with outputs. Circuit tracing is a concrete version of this goal: it identifies a prompt-specific subgraph of internal features and effects that support a chosen prediction [Olah et al., 2020; Ameisen et al., 2025]. Recent replacement-model methods make this object especially useful by representing MLP computations through sparse, interpretable transcoder features. The resulting attribution graph contains signed feature-to-feature effects that can be inspected and tested through interventions.

The obstacle is that attribution graphs are not yet human-scale explanations. Even after tracing one next-token prediction, the graph can contain thousands of nodes, many low-level or syntactic features, and a dense web of signed edges. Existing circuit-tracing workflows therefore depend on manual pruning, grouping, and naming of features. This makes the final explanation slow to produce and makes its compression decisions difficult to reproduce.

We study automated attribution-graph summarization: given one signed attribution graph and a target prediction, produce a small directed acyclic graph whose nodes are interpretable groups of original features. A useful summary must retain the input-to-output computation that motivated the graph, keep grouped features mechanistically coherent, expose rather than hide important causal relations, and be small enough to inspect. These objectives conflict. A more aggressive grouping reduces visual complexity but can conceal causal edges inside a supernode; preserving every edge, in contrast, leaves the graph unreadable.

Our method makes this trade-off explicit. It uses upstream relevance as well as downstream influence for pruning, then clusters retained feature nodes through a constrained integer program over signed causal-role similarity. The final graph is constructed deterministically from the retained partition and is validated by interventions on the underlying replacement model. The central contribution is therefore not merely graph compression, but a compression procedure with measurable structural and causal failure modes.

Our contributions are:

1. We formalize prompt-level attribution-graph summarization as producing a compact partition of a retained signed graph, subject to importance retention, causal-role coherence, visible causal structure, and acyclicity.
2. We introduce a three-stage pipeline: influence–relevance pruning, causal-role clustering with a binary integer linear program, and deterministic aggregation and labelling into a summary DAG.
3. We evaluate the pipeline on BATS-derived analogy prompts using graph-faithfulness, structural, and intervention metrics. The evaluation exposes a compression–coherence trade-off and shows that summary-edge signs and magnitudes remain predictive of measured intervention responses.

## 2. Background and Problem Definition

Sparse autoencoders and transcoders recover an overcomplete feature basis from model activations, addressing the fact that individual neurons can represent multiple unrelated features in superposition [Elhage et al., 2022; Cunningham et al., 2023; Dunefsky et al., 2024]. Circuit tracing extends this idea with cross-layer transcoders and constructs a local replacement model for a particular prompt. Its attribution graph is a signed directed acyclic graph: embedding nodes supply prompt information, feature nodes represent transcoder features, error nodes record residual reconstruction effects, and logit nodes represent selected outputs [Ameisen et al., 2025].

We take this graph as fixed input. The goal is not to train transcoders, trace attention patterns, or claim that a summary is a complete explanation of a model. Instead, we ask how to summarize a single attributed computation without making the structure that motivated it inaccessible. This focus complements work on automatic feature descriptions [Bills et al., 2023] and on automatically describing attribution graphs [Arora et al., 2026]. It also applies to neuron-basis circuit graphs, which can be sparse and faithful but still too large for direct inspection [Arora et al., 2026].

Formally, the input is a signed DAG G with nodes V, edges E, and signed edge weights W. The output is a summary DAG H. Each summary node is a nonempty supernode containing original non-error nodes; embedding and logit nodes remain singleton supernodes, whereas retained feature nodes may be grouped. A valid summary should satisfy four desiderata:

- **Importance retention:** retain nodes and edges on high-mass paths from selected inputs to selected outputs.
- **Atomicity:** group features with similar signed upstream and downstream causal roles.
- **Causal-information preservation:** avoid hiding high-mass feature-to-feature edges inside a supernode.
- **Interpretability:** produce a small, acyclic graph that a reader can follow as a forward causal story.

## 3. Method

### 3.1 Overview

The pipeline has three stages. Pruning selects a tractable subgraph; clustering partitions its retained feature nodes into supernodes; and interpretation constructs and labels the final DAG. The stages use computable proxies for a summary's eventual faithfulness and readability. We therefore evaluate the final object with interventions rather than treating any optimization objective as proof of faithfulness.

### 3.2 Influence–Relevance Pruning

Downstream influence measures the fraction of output-directed path mass that reaches each node. Upstream relevance analogously measures the fraction of input-directed path mass that reaches each node. Both quantities are computed by propagating through absolute, normalized edge weights on the attribution DAG. Influence alone identifies features that can affect the target logit, but it need not distinguish prompt-relevant pathways from generic output-side or syntactic effects.

We score each node and edge with a rank-normalized geometric combination of influence and relevance. The mixture parameter α controls the relative contribution of downstream influence; α = 1 recovers influence-only pruning. Nodes and edges are sorted by this score and retained until their respective importance-mass budgets are reached. We additionally remove feature nodes with extremely low or excessively high activation frequency, because these features are unlikely to yield useful prompt-specific summaries. The evaluation varies α and the node budget while retaining an edge-mass budget of 0.95.

### 3.3 Causal-Role Clustering

Pruning alone does not make a graph readable. We represent each retained feature by a signed causal-role vector consisting of two parts: its outgoing edge profile, weighted by the influence of destination nodes, and its incoming edge profile, weighted by the relevance of source nodes. The representation retains edge signs, so features with opposing effects on the same targets are not treated as equivalent.

The clusterer solves a binary integer linear program over allowed pairs of retained features. A pair receives a reward for merging when its role-vector cosine similarity exceeds an adaptive threshold and a penalty otherwise. Transitivity constraints ensure that pairwise merge decisions form a valid partition. Additional constraints limit the feature-edge mass that would become internal to a supernode, bound the number of supernodes, and optionally restrict the maximum layer span of a group. The causal-mass constraint is important: it makes causal preservation a hard feasibility requirement instead of a tunable term that can be traded away silently for compactness.

### 3.4 Summary Construction and Labelling

For every ordered pair of distinct supernodes, we sum the retained signed feature-level edges crossing between their members. This block sum defines a candidate summary edge. We remove antiparallel pairs by retaining only the direction with larger absolute net weight, then impose a stable depth order: embeddings are sources, logits are sinks, and feature supernodes are ordered by their member layers. Any edge that points backward in this order is removed, producing a DAG by construction.

For display, an LLM receives the prompt, predicted token, and compact feature evidence for each supernode, including activation contexts, positions, and top logits. It returns a short label, a one-sentence description, and one of four roles: Input, Abstract, Output, or Trash. These labels aid inspection but are not used as evidence of causal validity.

## 4. Experimental Setup

### 4.1 Data and implementation

The main quantitative evaluation uses 100 synthetic analogy prompts derived from the lexicography category of the Bigger Analogy Test Set (BATS) [Gladkova et al., 2016]. Prompts have the form “The saying goes: A is to B as C is to D.” They cover ten relation types: country–capital, country–language, UK city–county, name–nationality, name–occupation, animal–young, animal–sound, animal–shelter, object–color, and male–female. We retain prompts the model answers correctly with confidence above 20%, so each graph explains a behavior the model actually produces.

We also summarize 35 multi-hop reasoning prompts from the circuit-tracing evaluation suite as a qualitative stress test. These graphs are not used for the main numerical comparisons because the analogy set supports controlled, repeated evaluation.

Attribution graphs are generated for Google Gemma-2-2B with the `mntss/clt-gemma-2-2b-2.5M` transcoder set using the circuit-tracer library. The experiments use an RTX 4090 GPU with 24 GB VRAM and 128 GB host memory. The exact clustering program is solved through SciPy's mixed-integer linear-program interface and the HiGHS solver.

### 4.2 Evaluation questions and metrics

**Pruning faithfulness.** We compare relevance-aware pruning to influence-only pruning at matched graph sizes. Replacement score measures the retained fraction of end-to-end input-to-output influence after omitted mass is assigned to complement/error nodes. Graph completeness measures the fraction of influence-weighted incoming edge mass explained by retained nodes. Both metrics are evaluated across pruning budgets and normalizations.

**Clustering structure.** We compare the integer-program clusterer with K-means on role vectors, spectral clustering on adjacency or role vectors, and random same-size clusters. All methods are matched to the number of feature supernodes produced by our method. Role gap and signed upstream/downstream gaps measure within-cluster causal-role coherence relative to cross-cluster pairs. Internalized feature-edge mass measures the causal structure hidden inside clusters, and DAG loss measures crossing edge mass removed when enforcing the summary's forward order.

**Intervention faithfulness.** For each non-fixed supernode, we negate its member feature activations over the direct-effect layer window and measure responses in downstream supernodes and target logits. Sign consistency asks whether the signed summary edge predicts the observed direction of response; the reported value is weighted by edge magnitude. Spearman correlation tests whether larger summary edges produce larger measured response magnitudes. We additionally perform entity-swap interventions: suppress the source prompt's Output supernode and inject an Output supernode from a prompt with the same relation but a different answer.

## 5. Results

### 5.1 Relevance-aware pruning preserves graph-faithfulness curves

Across both softmax and entmax normalizations, adding upstream relevance to downstream influence does not substantially reduce replacement score or graph completeness at matched retained-node counts. The advantage is most apparent in the small-graph regime, where pruning must select a limited set of features likely to connect the prompt context to the target prediction. These curves support the narrower claim that upstream relevance is compatible with preserving the graph-level path measures used here; they do not yet establish that relevance-aware pruning improves human semantic judgments. A planned manual syntactic-feature count is therefore not reported as a result.

### 5.2 The ILP exposes a structural trade-off rather than dominating every clustering metric

On 100 pruned analogy graphs, K-means produces the strongest role-gap metrics, indicating especially compact groups in the chosen role-vector geometry. However, these groups also internalize more feature-edge mass and incur more loss when a DAG is formed. The ILP sacrifices some role-vector compactness to preserve visible causal structure. This is the intended behavior of its causal-mass and layer-span constraints.

| Method | Role gap ↑ | Signed up gap ↑ | Signed down gap ↑ | Internalized mass ↓ | DAG loss ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| ILP, no layer bound, causal budget 0.050 | 0.680 | 0.625 | 0.474 | 0.050 | 0.087 |
| ILP, layer span 7, causal budget 0.050 | 0.663 | 0.604 | 0.541 | **0.049** | 0.062 |
| ILP, layer span 7, causal budget 0.090 | 0.680 | 0.601 | 0.541 | 0.083 | **0.061** |
| K-means on role vectors | **0.696** | **0.631** | **0.593** | 0.086 | 0.090 |
| Random same-size clusters | −0.014 | −0.008 | −0.005 | 0.086 | 0.125 |
| Spectral clustering on adjacency | 0.125 | 0.186 | −0.030 | 0.159 | 0.115 |
| Spectral clustering on role vectors | 0.575 | 0.542 | 0.210 | 0.109 | 0.078 |

The role-similarity threshold controls the granularity of this trade-off. At the 65th percentile threshold, the ILP has a role gap of 0.6465, internalized mass of 0.0994, DAG loss of 0.0359, and 5.48 feature singleton supernodes per graph. Raising the threshold from the 50th to the 80th percentile improves role gaps and lowers hidden edge mass, but increases singleton count from 4.32 to 8.78. Thus, the method supports selecting a point on a compression–structural-preservation frontier rather than a single universally best partition.

### 5.3 Summary edges predict intervention responses

We evaluated 6,765 edge-result rows from 100 summaries constructed with α = 0.5 and a node importance-mass threshold of 0.02. Summary-edge signs agree with the measured response direction for most weighted edge mass. Edge magnitude also positively ranks response magnitude, providing an intervention-based check that the displayed graph retains information beyond its visual layout.

| Target type | Evaluated edges | Sign consistency ↑ | Spearman edge–effect correlation ↑ | Mean absolute effect |
| --- | ---: | ---: | ---: | ---: |
| Feature supernode | 5,612 | 0.8286 | 0.7609 | 0.5980 |
| Logit | 1,153 | 0.7624 | 0.6840 | 0.1324 |

Entity-swap steering gives a complementary whole-supernode test. For closed-class factual relations, injecting a donor Output supernode while suppressing the source Output supernode can transfer the donor answer. At a donor factor of 4, top-1 transfer reaches 0.44 for country–capital, 0.38 for country–language, and 0.38 for name–occupation. The effect is substantially weaker for looser semantic relations such as animal–young and animal–sound. The difference cautions against interpreting an Output label as proof that a supernode encodes a context-independent concept; transfer depends on how cleanly the task maps the query to a single answer token.

## 6. Discussion

The results support a practical view of attribution-graph summarization. Compression should not be judged only by how few nodes remain or how similar the members of a cluster appear. A summary is useful when its graph-level path measures remain high, its displayed edges retain important structure, and its directions predict interventions. The proposed pipeline converts these criteria into explicit design choices and exposes their trade-offs in evaluation.

Several limitations bound the current claim. First, the method inherits the approximation quality and scope of the underlying replacement model and attribution graph. A faithful summary of an imperfect graph is not a complete causal account of the original language model. Second, the evaluation focuses on prompt-level, single-token predictions from one model family and a controlled analogy benchmark. It does not establish generalization across models, long-form generation, or population-level circuits. Third, input relevance uses SHAP-derived token weights; this is a pragmatic boundary condition, not a validated measure of human-relevant evidence. Fourth, summary labels are generated for readability but are not independently evaluated for correctness. Finally, imposing a forward DAG necessarily removes some crossing edge mass, and the current clustering experiment does not include an end-to-end automated attribution-graph summarization baseline.

Future work should evaluate causal frontiers over graph size and intervention faithfulness across diverse models and tasks, validate labels against feature behavior, and extend summaries beyond one prompt and one target token. A particularly important direction is to define input relevance directly in the graph's causal formalism and to test summaries on held-out, task-specific causal contrasts.

## 7. Conclusion

We introduced an automated method for turning dense prompt-level attribution graphs into compact summary DAGs. The method combines relevance-aware pruning, constrained causal-role clustering, and deterministic aggregation with intervention-aware evaluation. On analogy circuits, the pipeline preserves graph-faithfulness measures during pruning, makes the causal-structure cost of clustering explicit, and yields signed summary edges that predict measured feature and logit responses. The result is a reproducible compression layer for circuit tracing: one that helps make graphs readable while keeping the resulting explanation tied to structural and causal tests.

## References

Arora, A., Wu, Z., Steinhardt, J., and Schwettmann, S. (2026). *Language Model Circuits Are Sparse in the Neuron Basis*. arXiv:2601.22594.

Arora, A., Wu, Z., Steinhardt, J., and Schwettmann, S. (2026). *ADAG: Automatically Describing Attribution Graphs*. arXiv:2604.07615.

Ameisen, E., Lindsey, J., Pearce, A., Gurnee, W., Turner, N. L., Chen, B., et al. (2025). *Circuit Tracing: Revealing Computational Graphs in Language Models*. Transformer Circuits Thread.

Bills, S., Cammarata, N., Mossing, D., Tillman, H., Gao, L., Goh, G., et al. (2023). *Language Models Can Explain Neurons in Language Models*. OpenAI.

Cunningham, H., Ewart, A., Riggs, L., Huben, R., and Sharkey, L. (2023). *Sparse Autoencoders Find Highly Interpretable Features in Language Models*. arXiv:2309.08600.

Dunefsky, J., Chlenski, P., and Nanda, N. (2024). *Transcoders Find Interpretable LLM Feature Circuits*. arXiv:2406.11944.

Elhage, N., Hume, T., Olsson, C., Schiefer, N., Henighan, T., Kravec, S., et al. (2022). *Toy Models of Superposition*. Transformer Circuits Thread.

Gladkova, A., Drozd, A., and Matsuoka, S. (2016). *Analogy-Based Detection of Morphological and Semantic Relations with Word Embeddings: What Works and What Doesn't*. Proceedings of NAACL-HLT.

Olah, C., Cammarata, N., Schubert, L., Goh, G., Petrov, M., and Carter, S. (2020). *Zoom In: An Introduction to Circuits*. Distill.
