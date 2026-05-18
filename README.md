# Fork of safety-research/circuit-tracer 
This is a fork of `safety-research/circuit-tracer`, extended with a **summarization pipeline** that automates summarizing the attribution graph. The core library computes attribution graphs (circuits) for transformer language models using MLP transcoders.
## Recap
### Formulation:
Formulate the problem as an optimization problem. 
We want to create a supernode graph that sastisfy the following conditions:
- Features assigned to the same supernode share mechanistic role on both axes: similar influential downstream targets and similarrelevant upstream sources
- Flow conservation: the graph must be mechanisticly faithful, retent information from the original graph.
- Minimality: the graph should not contain redundant nodes.

Formulate the supernode graph. 

Optimization Formulation
Grounded directly in your data structures (AttrGraph, SummarizationGraph).

Setting
Pruned attribution graph $G' = (V', A')$ with $A' \in \mathbb{R}^{N\times N}$, $A'{ij}$ = weight from source $j$ to target $i$ (matches pruned_adj). Nodes split into $V' = V_e \cup V_f \cup V\ell$ (embeddings / features / logits; errors pruned). Each $u \in V_f$ has layer $l_u$, ctx $p_u$, influence $\iota_u$, relevance $\rho_u$.

Decision variable
Partition $\Pi = {S_1, \dots, S_K}$ of $V_f$ (embedding & logit nodes are singleton supernodes by type, per cluster_kind_to_supernode_type). $K$ is itself part of the decision.

Induced assignment $c: V' \to {1, \dots, K + |V_e| + |V_\ell|}$.

Derived supernode adjacency (exactly what supernode_graph.py:135-149 computes):
$$A^S_{kl} ;=; \sum_{i \in S_k}\sum_{j \in S_l} A'_{ij}$$

Three desiderata → three losses
(1) Co-role on both axes — cohesion
Edge profiles for $u \in V_f$:
$$\mathbf{d}_u = A'[:,u] \text{ (downstream targets)}\qquad \mathbf{e}_u = A'[u,:] \text{ (upstream sources)}$$

Pairwise similarity with layer-span penalty:
$$s(u,v) ;=; \big[\alpha\cos(\mathbf{d}_u,\mathbf{d}_v) + (1-\alpha)\cos(\mathbf{e}_u,\mathbf{e}_v)\big]\cdot e^{-\gamma\lvert l_u-l_v\rvert}$$

Within-cluster dissimilarity:
$$\mathcal{L}{\text{coh}}(\Pi) ;=; \sum{k=1}^{K}\frac{1}{|S_k|(|S_k|-1)}\sum_{\substack{u,v\in S_k \ u\neq v}}\big(1 - s(u,v)\big)$$

Equivalent spectral form: $\mathrm{NCut}(W, \Pi)$ on $W_{uv} = s(u,v)$ — this is exactly what your spectral baseline approximates.

(2) Flow conservation — faithfulness
Block-constant reconstruction error:
$$\mathcal{L}{\text{flow}}(\Pi) ;=; \sum{i,j\in V'}\Big(A'{ij} - \tfrac{A^S{c(i),c(j)}}{|S_{c(i)}|,|S_{c(j)}|}\Big)^2$$

(Low when members of every supernode have similar edge patterns — i.e. the supernode graph is a faithful piecewise-constant compression of $A'$.)

Optional logit-reachability variant: $\mathcal{L}{\text{flow}}^{\iota}(\Pi) = \big|\iota - \mathrm{lift}\Pi(\iota^S)\big|2^2$ with $\iota^S_k=\sum{i\in S_k}\iota_i$.

(3) Minimality — parsimony
$$\mathcal{L}_{\text{size}}(\Pi) ;=; K$$

Combined objective
$$\boxed{;\Pi^{*};=;\arg\min_{\Pi}; \lambda_1,\mathcal{L}{\text{coh}}(\Pi) ;+; \lambda_2,\mathcal{L}{\text{flow}}(\Pi) ;+; \lambda_3,\mathcal{L}_{\text{size}}(\Pi);}$$

Constraints
(C1) Partition: $\bigsqcup_k S_k = V_f$, $S_k\neq\emptyset$.
(C2) DAG: $A^S$ acyclic — $\exists$ topological order $\pi$ s.t. $A^S_{kl}>0 \Rightarrow \pi(l)<\pi(k)$. Your implementation enforces this via the dominant-direction tie-breaker (supernode_graph.py:151-165).
(C3) Layer span (optional): $\max_{u\in S_k}l_u - \min_{u\in S_k}l_u \leq L$ — surfaced as Supernode.layer_min/layer_max.
How this maps to your baselines
Method	Approximates
K-means on edge profile	$\mathcal{L}_{\text{coh}}$ only, Euclidean instead of cosine, no DAG
Spectral clustering	$\mathrm{NCut}$ relaxation of $\mathcal{L}_{\text{coh}}$, no DAG
Greedy modularity	Different cohesion notion (degree-corrected), no DAG
Your method	Spectral on $s$ with layer penalty + DAG enforcement + auto-$k$ sweep ≈ joint minimizer of all three terms
Want me to drop this into paper/measure.tex in LaTeX, or refine any of the terms first (e.g., a different flow-conservation choice, or weighting profiles by $\iota, \rho$)?


What "flow" means here
In an attribution graph, an edge $A'{ij}$ is a linear contribution from source $j$ to target $i$. Composing edges gives path flow: the total contribution of source $j$ to target $i$ via paths of length up to $L$ is $\sum{\ell=0}^{L}(A')^{\ell}_{ij}$.

For the target logit $\ell^$, this gives the influence vector:
$$\iota ;=; (I - A')^{-1}_{\ell^,,:} \qquad \text{(well-defined since the graph is a DAG over layers)}$$

So "flow" = how much of the target logit each node explains, via summed paths.

What faithfulness should mean
We want: running the supernode graph $A^S$ forward reproduces the same target-logit explanation as running the full graph $A'$.

The clean Markov-chain analog is lumpability (Kemeny–Snell). A partition $\Pi$ is exactly lumpable w.r.t. $A'$ iff, for every pair of clusters $(S_k, S_l)$ and every two members $i, i' \in S_k$:
$$\sum_{j \in S_l} A'{ij} ;=; \sum{j \in S_l} A'_{i'j} \tag{$\star$}$$

i.e., every member of $S_k$ contributes the same aggregated mass to $S_l$. When $(\star)$ holds, $A^S$ is a sufficient statistic for the path-flow dynamics — the coarsened graph is mechanistically faithful by construction.

Real attribution graphs are never exactly lumpable, so we relax $(\star)$ to a soft loss.

Three concrete realizations
(A) Block-constant reconstruction (what I wrote first)
$$\mathcal{L}{\text{flow}}^{\text{block}}(\Pi) ;=; \sum{i,j}\Big(A'{ij} - \tfrac{A^S{c(i),c(j)}}{|S_{c(i)}||S_{c(j)}|}\Big)^2$$

Per-edge deviation from the block mean. Generic graph-coarsening loss (stochastic block model). Easy to compute, but penalizes every per-edge deviation — even ones that wash out when aggregated.

(B) Lumpability variance (tighter, attribution-aware)
Soft form of $(\star)$ in both directions:
$$\mathcal{L}{\text{flow}}^{\text{lump}}(\Pi) ;=; \underbrace{\sum{k,l};\mathrm{Var}{i\in S_k}!\Big(!\sum{j\in S_l} A'{ij}\Big)}{\text{out-flow agreement}} ;+; \underbrace{\sum_{k,l};\mathrm{Var}{j\in S_l}!\Big(!\sum{i\in S_k} A'{ij}\Big)}{\text{in-flow agreement}}$$

Penalizes only deviations that survive aggregation — exactly the ones that hurt $A^S$ as a summary. This is the natural target.

(C) Logit-reachability reconstruction (end-to-end)
$$\mathcal{L}{\text{flow}}^{\iota}(\Pi) ;=; \big\lVert, \iota ;-; \mathrm{lift}\Pi\big((I - A^S)^{-1}_{\ell^*,,:}\big) ,\big\rVert_2^2$$

Most direct interpretation — run $A^S$ forward and check it reproduces the target-logit influence vector at every original node. Most expensive (requires the matrix inverse / iterative propagation), but most semantically meaningful.

What your current code implicitly optimizes
Your edge-profile cohesion (cosine on $\mathbf{d}_u, \mathbf{e}u$) is a proxy for lumpability: two nodes with identical edge profiles satisfy $(\star)$ exactly across all clusters. So **minimizing $\mathcal{L}{\text{coh}}$ already pushes toward lumpability** — they're not independent objectives, they're the same objective in two parameterizations:

Term	What it encodes
$\mathcal{L}_{\text{coh}}$	"Members have similar edge profiles" → satisfy $(\star)$ approximately
$\mathcal{L}_{\text{flow}}^{\text{lump}}$	"Members agree on aggregated flow to each other cluster" → $(\star)$ directly
The difference: $\mathcal{L}{\text{coh}}$ measures similarity at the node level (cosine of full edge vectors); $\mathcal{L}{\text{flow}}^{\text{lump}}$ measures it at the aggregated-to-cluster level. The latter is conditioned on $\Pi$ — same node pair can have low flow-disagreement under one $\Pi$ and high under another — so it's the strictly correct measure of faithfulness.

Recommendation for the paper
Frame it as:

Faithfulness. A summarization $\Pi$ is faithful if $A^S$ preserves the path-flow induced by $A'$. The ideal condition is lumpability $(\star)$; we measure deviation from it with $\mathcal{L}_{\text{flow}}^{\text{lump}}$.

Then note that your spectral objective is a tractable upper bound: it bounds $\mathcal{L}_{\text{flow}}^{\text{lump}}$ because lumpability deviation $\leq$ within-cluster edge-profile variance (Cauchy–Schwarz on the cluster indicator vector).

Want me to write this into paper/measure.tex in LaTeX, or first tighten one of the variants (e.g., weight the variance by $\iota$ and $\rho$ so we only penalize disagreement on edges that matter)?

### Methodology
The summarization pipeline consists of 2 stage:
1. Prunning: prune the attribution graph down to a subgraph containing important nodes and edges. 
- Key idea: introduce **relevance** and use along with **influence** (from original clt paper) to align the subgraph with human rationale better (features activating on useless tokens are pruned).
2. Clustering: cluster functionally similar feature nodes together, further simplify the summarization graph and aid steering.
- Key idea: use spectral clustering on a weighted edge profile similarity matrix with layer span penalty, ensuring DAG.

### Eval

Baselines: 
K-means on aggregated edge profile
Spectral clustering
Greedy modularity

Metrics:
LLM score + optimization metrics
Steering