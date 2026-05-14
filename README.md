# Fork of safety-research/circuit-tracer 
This is a fork of `safety-research/circuit-tracer`, extended with a **summarization pipeline** that automates summarizing the attribution graph. The core library computes attribution graphs (circuits) for transformer language models using MLP transcoders.
## Recap
The summarization pipeline consists of 2 stage:
1. Prunning: prune the attribution graph down to a subgraph containing important nodes and edges. 
- Key idea: introduce **relevance** and use along with **influence** (from original clt paper) to align the subgraph with human rationale better (features activating on useless tokens are pruned).
2. Clustering: cluster functionally similar feature nodes together, further simplify the summarization graph and aid steering.
- Key idea: use agglomerative clustering on a weighted edge profile similarity matrix with layer span penalty, ensuring DAG.

## Evaluation Methodology

To rigorously assess the quality, interpretability, and structural soundness of our proposed supernode graph generation method, we evaluate it against both trivial and standard graph-theoretic baselines. 
Desiderata:
\textbf{(D1) DAG integrity.} The induced supernode graph must remain acyclic.
\textbf{(D2) Mechanism Preservation.} The supernode graph should preserve the causal flow of the original graph.
\textbf{(D3) Concept coherence.} Nodes in the same supernode should play similar structural roles in the circuit.
\textbf{(D4) Cluster balance.} The supernode graph should be balanced, i.e., the number of nodes in each supernode should be roughly equal.
\textbf{(D5) Minimality.} The supernode graph should be as small as possible.

### 1. Baselines

To contextualize the performance of our method, we compare it against three baselines representing naive or standard approaches to feature grouping:


*   **Standard Louvain Community Detection (Graph-Theoretic Baseline):** We apply the Louvain modularity optimization algorithm directly to the un-directed raw feature-to-feature adjacency matrix. While Louvain is a standard for maximizing community cohesion, it is agnostic to the sequential forward-pass of a transformer. It tends to cluster heavily connected sequential nodes together, thereby violating causal independence and generating massive structural cycles.

### 2. Scoring Metrics


### 3. Experimental Settings and Hyperparameters

To ensure a fair comparison across algorithms, we strictly control the resolution of the generated graphs. 

*   **Cluster Count Constraint:** For the Random, Louvain, and Proposed methods, the target number of supernodes $K$ is fixed to two structural resolutions: $K = \lfloor n/2 \rfloor$ and $K = \lfloor n/3 \rfloor$, where $n$ is the total number of individual features. The By-Layer baseline is naturally fixed to $K = L$, where $L$ is the number of layers in the network.
*   **Hyperparameter Sweep (Proposed Method):** To identify the optimal configuration for causal extraction, we perform a grid search over the hyperparameters defining our algorithmic constraints:
    *   **Mean Method:** For aggregating cross-layer cluster representations, we ablate *arithmetic*, *harmonic*, and *geometric* means to observe the impact of dampening extreme outlier weights.
    *   **Decay Rate ($\gamma$):** We sweep the temporal decay parameter governing layer-span distance penalties from $0.0$ to $1.0$ with a step size of $0.1$ ($\gamma \in \{0.0, 0.1, 0.2, \dots, 1.0\}$). This allows us to quantify the exact trade-off between localized DAG enforcement (high decay) and global conceptual grouping (low decay).

### 4. Ablation study
Suggestions:
