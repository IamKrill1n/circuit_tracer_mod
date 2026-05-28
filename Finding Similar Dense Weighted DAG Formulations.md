# **Formulating Causal Summarization for Dense Weighted Directed Acyclic Graphs**

Causal inference is a foundational framework for uncovering the direct and indirect mechanisms that govern complex systems, allowing researchers to move beyond statistical association to establish rigorous interventional relationships. Within Pearl's structural causal paradigm, Directed Acyclic Graphs (DAGs) serve as the primary qualitative tool to encode background assumptions, identify confounding variables, and determine sufficient adjustment sets via graphical criteria such as the backdoor criterion. However, as the dimensionality of modern datasets expands, the underlying causal DAGs grow increasingly dense and intricate. This structural complexity quickly surpasses the limits of human verifiability, making it difficult for domain experts to evaluate, validate, or refine the models.  
To address this challenge, the concept of causal graph summarization has emerged. This methodology simplifies high-dimensional causal networks into highly interpretable, lower-dimensional representations without sacrificing the underlying causal information required for downstream inference. Specifically, the unweighted causal DAG summarization framework optimizes a trade-off between graph simplification and the preservation of conditional independence statements by establishing a "summary causal DAG" via node contractions. This contraction process is constrained by minimizing structural discrepancies—represented as the addition of directed edges—in a reconstructed "canonical causal DAG".  
Despite the mathematical rigor of the unweighted formulation, many real-world systems in biology, economics, and engineering are characterized by dense, continuous causal influences where relationships are not binary but vary widely in magnitude. In these settings, representing the causal structure as a simple, unweighted DAG leads to a significant loss of information.  
This report develops a mathematically rigorous, causally consistent generalization of the causal DAG summarization framework specifically tailored for dense weighted DAGs. By integrating structural graph summarization, continuous causal abstraction, and cyclic system coarsening, this analysis provides an authoritative formulation that preserves both structural weights and global path-integrated causal effects.

## **Theoretical Foundations of Causal DAG Summarization**

To establish a baseline for the weighted extension, it is necessary to first review the structural formulation of causal DAG summarization. Let G \= (V, E) represent a ground-truth, high-dimensional causal DAG over a set of n variables V. The objective of causal summarization is to find a summary DAG H \= (V\_H, E\_H) with at most k supernodes (|V\_H| \= k \\ll n) and a partitioning mapping f: V \\rightarrow V\_H that groups the original variables into disjoint clusters.  
A critical requirement of this mapping is that H must be *compatible* with G. Compatibility dictates that if a directed edge exists between two variables in the original graph, then in the summary graph, those variables must either reside within the same supernode or their respective supernodes must be connected by a directed edge in the same direction. To mathematically evaluate the information loss induced by this simplification, the structural formulation defines a "canonical causal DAG" G\_H over the original node set V. The canonical DAG G\_H is constructed by adding complete tournaments within each cluster to preserve acyclicity and inserting directed edges between distinct clusters wherever a superedge exists in H.

| Concept | Unweighted Formulation Definition | Causal Implications |
| :---- | :---- | :---- |
| **Summary DAG** H \= (V\_H, E\_H) | A low-dimensional DAG with $ | V\_H |
| **Partitioning Function** f: V \\rightarrow V\_H | An exhaustive, non-overlapping surjection mapping V to V\_H. | Groups semantically similar variables to maintain macro-level coherence. |
| **Compatibility** | (u, v) \\in E(G) \\implies f(u) \= f(v) \\lor (f(u), f(v)) \\in E\_H. | Guarantees that no true causal pathways are omitted in the summary graph. |
| **Canonical Causal DAG** G\_H | V(G\_H) \= V; (i, j) \\in E(G\_H) \\iff (i, j) \\in E(G) \\lor (f(i), f(j)) \\in E\_H \\lor (f(i) \= f(j) \\land i \\prec j). | Encapsulates the set of all possible high-dimensional DAGs compatible with H. |
| **Recursive Basis** \\Sigma\_{RB} | The minimal set of conditional independence statements that uniquely defines the graph. | Preserves the soundness and completeness of Pearl's do-calculus. |

In the unweighted setting, contracting nodes is equivalent to adding directed edges to the original graph to form the canonical DAG G\_H. Adding edges systematically removes conditional independence (CI) statements from the recursive basis, thereby reducing statistical power but crucially preserving the soundness of downstream causal inference. The optimal summary DAG H^\* is thus defined as the compatible DAG of size k that minimizes the number of added edges in G\_H :  
This structural minimization problem is NP-hard, which is proven via a reduction from the k-Max-Cut problem.

## **Mathematical Extension to Dense Weighted DAGs**

To extend this framework to dense weighted DAGs, the structural assumptions of binary edge existence must be replaced with a continuous formulation. Let the high-dimensional concrete model be a linear Structural Causal Model (SCM) over n endogenous random variables V \= \\{X\_1, X\_2, \\dots, X\_n\\} :  
where B \\in \\mathbb{R}^{n \\times n} represents the weighted adjacency matrix of direct causal effects, and \\epsilon \\in \\mathbb{R}^n is a vector of mutually independent, non-Gaussian exogenous noise terms with covariance matrix \\Omega \= \\text{diag}(\\sigma\_1^2, \\dots, \\sigma\_n^2). Because G is a weighted DAG, there exists a topological ordering of the variables such that B\_{ij} \= 0 for all i \\leq j, making the matrix B strictly lower triangular under this ordering. The total causal effects are analytically captured by the mixing matrix A \\in \\mathbb{R}^{n \\times n} :  
We represent the node partitioning function f: V \\rightarrow V\_H using a binary clustering matrix P \\in \\{0, 1\\}^{k \\times n} :  
where Y\_I \\in V\_H represents the I-th supernode. The size of each cluster is given by the diagonal elements of the normalization matrix D \= P P^T \= \\text{diag}(n\_1, n\_2, \\dots, n\_k) \\in \\mathbb{R}^{k \\times k}, where n\_I \= \\sum\_{i=1}^n P\_{Ii}. This allows us to define the row-normalized partitioning matrix P\_{avg} \= D^{-1} P \\in \\mathbb{R}^{k \\times n}.  
The summary weighted SCM is defined over the k supernodes Y \= \\{Y\_1, \\dots, Y\_k\\} as :  
where B\_H \\in \\mathbb{R}^{k \\times k} is the summary weighted adjacency matrix and \\eta \\in \\mathbb{R}^k represents the aggregated noise terms. Using the average-pooling projection, we formulate the summary adjacency matrix as :  
Each element of this summary matrix represents the average direct causal weight flowing between clusters :

## **The Weighted Canonical Causal DAG and Information Loss**

In the dense weighted regime, the binary concept of compatibility must be reformulated as a continuous projection. The weighted canonical causal DAG G\_H \= (V, B\_{G\_H}) serves as the high-dimensional reconstruction of the summary model, allowing us to directly compare it to the original graph G.  
To formalize the canonical weighted adjacency matrix B\_{G\_H} \\in \\mathbb{R}^{n \\times n}, we define the intra-cluster mask matrix M\_{intra} \\in \\{0, 1\\}^{n \\times n} and the inter-cluster mask matrix M\_{inter} \\in \\{0, 1\\}^{n \\times n} as:  
The canonical weighted adjacency matrix B\_{G\_H} is formulated using the Hadamard (element-wise) product \\odot:  
This formulation preserves the exact intra-cluster direct causal weights from the original weighted SCM, which is the continuous analogue to preserving maximum structural freedom within contracted node clusters. Simultaneously, it projects the low-dimensional summary weights B\_H back into the high-dimensional inter-cluster spaces. This construction satisfies several fundamental causal properties:

### **Theorem 1 (Causal Weight Conservation)**

For any partitioning matrix P and corresponding summary matrix B\_H \= P\_{avg} B P^T, the aggregate direct causal flow between any two distinct clusters Y\_J and Y\_I is preserved in the canonical reconstruction B\_{G\_H}:  
*Proof.* For any i \\in Y\_I and j \\in Y\_J where Y\_I \\neq Y\_J, the intra-cluster mask is zero, and the inter-cluster mask is one. Expanding the definition of B\_{G\_H} under these conditions:  
Since P^T B\_H P\_{avg} \= P^T B\_H D^{-1} P, we can expand the matrix multiplication element-wise:  
Because i \\in Y\_I and j \\in Y\_J, the terms P\_{Ki} and P\_{Lj} are non-zero only when K \= I and L \= J. Thus:  
Substituting the average-pooling definition of (B\_H)\_{IJ}:  
This proves that the total direct causal weight between any two clusters is conserved in the reconstructed canonical SCM.

### **Theorem 2 (Acyclicity Preservation)**

If the original weighted adjacency matrix B is strictly lower triangular under the topological ordering X\_1 \\prec X\_2 \\prec \\dots \\prec X\_n, and the clusters in V\_H are topologically ordered such that I \< J implies that for all i \\in Y\_I and j \\in Y\_J, i \< j, then the canonical matrix B\_{G\_H} is strictly lower triangular and contains no directed cycles.  
*Proof.* Under the topologically consistent partition mapping, the inter-cluster term P^T B\_H P\_{avg} can only contain non-zero entries (P^T B\_H P\_{avg})\_{ij} where f(X\_i) \= Y\_I and f(X\_j) \= Y\_J such that I \> J. Because the clusters are ordered consistently with the topological order of V, this implies i \> j, rendering the inter-cluster component strictly lower triangular. The intra-cluster component M\_{intra} \\odot B is a direct Hadamard product of the strictly lower triangular matrix B, and is therefore also strictly lower triangular. Because the sum of two strictly lower triangular matrices is strictly lower triangular, B\_{G\_H} is strictly lower triangular, guaranteeing acyclicity.

## **Dual Optimization Frameworks: Structural vs. Functional Preservation**

In summarizing a dense weighted DAG, information loss can be evaluated from two distinct perspectives: local structural preservation or global functional preservation.

### **Formulation A: Local Structural Preservation (Adjacency Reconstruction)**

The structural objective minimizes the reconstruction error of the direct causal weights, measured using the Frobenius norm :  
This formulation measures how well the flat inter-cluster superedge weights approximate the original continuous connections.

### **Theorem 3 (Optimal Weighted Summarization)**

For any fixed binary partition matrix P, the unique summary weighted adjacency matrix B\_H^\* that minimizes the local structural loss \\mathcal{L}\_{local} is the average-pooling projection:  
*Proof.* We can expand the local structural loss as a sum of squared differences over distinct clusters:  
To find the minimizer, we take the partial derivative with respect to each superedge weight (B\_H)\_{IJ}:  
Simplifying this expression:  
Solving for (B\_H)\_{IJ} yields the optimal summary weight:  
which is the average-pooling projection P\_{avg} B P^T.  
Substituting the optimal summary weights B\_H^\* back into the loss function yields the profile loss, which depends solely on the partition P:  
where \\text{Var}(B\_{Y\_I, Y\_J}) is the variance of the direct causal weights within the inter-cluster block from cluster Y\_J to cluster Y\_I. Minimizing this structural loss is therefore mathematically equivalent to finding a partition that minimizes the internal variance of the inter-cluster causal connections.

### **Formulation B: Global Functional Preservation (Causal Effect Reconstruction)**

While local structural preservation minimizes the variance of direct weights, downstream causal inference is fundamentally concerned with total causal effects, which accumulate along all directed paths in the graph. The global functional objective minimizes the distortion of the total causal effects, which are captured by the mixing matrices :  
This functional formulation ensures that interventional predictions made on the summarized DAG remain as close as possible to the ground-truth predictions derived from the original dense graph, preserving the core properties of Pearl's do-calculus.  
In the linear Gaussian setting, the conditional independence statements of the joint distribution are determined by the precision matrix \\Theta \= \\Sigma^{-1}, where the covariance matrix is \[span\_43\](start\_span)\[span\_43\](end\_span)\\Sigma \= (I \- B)^{-1} \\Omega (I \- B)^{-T}. Let \\Sigma\_{G\_H} \= (I \- B\_{G\_H})^{-1} \\Omega (I \- B\_{G\_H})^{-T} represent the reconstructed covariance matrix of the canonical weighted DAG. Minimizing the global functional loss provides a direct theoretical bound on the distortion of this covariance structure:

### **Theorem 4 (Covariance Distortion Bound)**

If the exogenous noise covariance matrix is bounded such that \\|\\Omega\\|\_F \\leq \\sigma\_{max}^2, then the Frobenius norm of the covariance distortion is bounded by the global functional loss:  
*Proof.* Expanding the difference between the two covariance matrices:  
Taking the Frobenius norm of both sides and applying the triangle inequality and sub-multiplicativity:  
Factoring out the shared terms and substituting \\|\\Omega\\|\_2 \\leq \\|\\Omega\\|\_F \\leq \\sigma\_{max}^2:  
Substituting A \= (I \- B)^{-1} and A\_{G\_H} \= (I \- B\_{G\_H})^{-1} completes the proof.  
This bound guarantees that minimizing the global functional loss prevents the introduction of spurious partial correlations, preserving approximate conditional independence statements and maintaining the d-separation guarantees of the unweighted framework in a continuous setting.

## **The SCC Floor and Acyclicity in Dense Networks**

In high-dimensional datasets, dense causal networks often contain feedback loops, bidirectional interactions, or cyclic structures that arise from temporal aggregation. When extending causal DAG summarization to these complex networks, we can leverage the coarsening lattice framework of cyclic linear non-Gaussian SCMs to handle these cycles.  
Let G \= (V, E, B) be a directed graph that may contain directed cycles. A partition \\Pi \= \\{\\pi\_1, \\dots, \\pi\_k\\} is a valid *DAG-coarsening* of G if the resulting quotient graph G' \= (\\Pi, E') is acyclic. This requirement imposes a strict structural constraint:

### **Theorem 5 (The SCC Floor)**

Any valid DAG-coarsening \\Pi of a directed graph G must group all nodes belonging to the same Strongly Connected Component (SCC) of G into a single cluster. Consequently, the SCC partition is the finest valid DAG-coarsening, and the resulting quotient graph is the structural condensation G\_{sc} of G.  
This structural floor defines the limits of valid partitionings for dense weighted graphs:  
                   
                                     `|`  
                                     `v`  
                   
                                     `|`  
                                     `v`  
                   
                                     `|`  
                  `===========================================`  
                   
                  `[ cycles into the summary graph          ]`

By enforcing the SCC floor, any feedback loops present in the high-dimensional dense graph are collapsed into unified supernodes. This guarantees that the summary causal graph H remains a valid DAG, allowing us to apply standard acyclic causal inference and adjustment techniques directly to the summary model.

## **The W-CaGreS Algorithm**

Because finding the optimal partition is NP-hard, we introduce the **W-CaGreS** (Weighted Causal Greedy Summarization) algorithm to find high-quality summary graphs in polynomial time. W-CaGreS extends the bottom-up greedy approach of the structural CaGreS algorithm to the weighted regime.  
The algorithm begins with the trivial identity partition where each node forms its own cluster (k \= n, P \= I). At each step, it evaluates the change in local structural loss \\Delta \\mathcal{L}\_{local}(U, V) that would result from merging any pair of clusters U and V into a single cluster K \= U \\cup V. Using the parallel axis theorem, the exact change in loss is given by:  
where \\bar{B}\_{I,J} represents the average causal weight between clusters I and J. This merge cost balances two opposing factors:

1. **The Approximation Penalty (First Term):** This term is positive and penalizes merging clusters U and V if they exhibit different average connection strengths to other supernodes W in the graph. This is the weighted equivalent of the unweighted penalty for adding spurious inter-cluster edges.  
2. **The Restoration Benefit (Second and Third Terms):** This term is negative and rewards merging U and V if they share strong, heterogeneous mutual weights. Because their mutual connections are moved from the inter-cluster set to the intra-cluster set, they are now perfectly preserved in B\_{G\_H}. This is the weighted equivalent of contracting existing strong edges in structural graph partitioning.

### **W-CaGreS Algorithm Pseudocode**

The procedural execution of W-CaGreS is formalized below:  
`Algorithm 1: Weighted Causal Greedy Summarization (W-CaGreS)`  
`Input: Dense Weighted DAG G = (V, B), Target Supernode Count k`  
`Output: Partition Matrix P, Summary Weighted Adjacency Matrix B_H`

`1: Initialize partition P as the identity matrix I_n  (n clusters)`  
`2: Compute initial cluster sizes n_I = 1 and average weights \bar{B}_{I,J} = B_ij for all I, J`  
`3: while current number of clusters > k do`  
`4:     for each pair of clusters (U, V) do`  
`5:         Compute merge cost:`  
               `\Delta L_local(U, V) = \frac{n_U n_V}{n_U + n_V} \sum_{W \notin \{U, V\}} n_W`  
                                      `- \sum_{i \in U, j \in V} (B_ij - \bar{B}_{U,V})^2 - \sum_{i \in V, j \in U} (B_ij - \bar{B}_{V,U})^2`  
`6:     Find the optimal pair (U*, V*) = argmin_{(U, V)} \Delta L_local(U, V)`  
`7:     Merge clusters U* and V* into a new cluster K = U* \cup V*`  
`8:     Update the partition matrix P by combining rows corresponding to U* and V*`  
`9:     Update cluster sizes: n_K = n_U* + n_V*`  
`10:    Update average weights for all external clusters W:`  
           `\bar{B}_{W, K} = \frac{n_U* \bar{B}_{W, U*} + n_V* \bar{B}_{W, V*}}{n_K}`  
           `\bar{B}_{K, W} = \frac{n_U* \bar{B}_{U*, W} + n_V* \bar{B}_{V*, W}}{n_K}`  
`11:    Update mutual internal weights:`  
           `\bar{B}_{K, K} = \frac{\sum_{i \in K, j \in K} B_ij}{n_K^2}`  
`12: Compute final summary matrix B_H = P_avg B P^T`  
`13: return P, B_H`

Evaluating the merge cost for O(n^2) pairs over n-k contraction steps yields an algorithmic complexity of O((n-k) \\cdot n^3). This matches the time complexity of the structural CaGreS algorithm, making it computationally viable for high-dimensional applications.

## **Comparative Synthesis of Causal Summarization Frameworks**

To place the dense weighted formulations in context, the table below compares the structural formulation, the proposed weighted local and global formulations, and other common dimensionality reduction paradigms:

| Dimension | Structural CaGreS | W-CaGreS (Local) | W-CaGreS (Global) | Gromov-Wasserstein Coarsening |
| :---- | :---- | :---- | :---- | :---- |
| **Primary Input** | Unweighted DAG G \= (V, E) | Dense Weighted DAG G \= (V, B) | Dense Weighted DAG G \= (V, B) | Attributed Weighted Graph G \= (V, E, W, X) |
| **Clustering Space** | Discrete node partitions f(V) | Binary clustering matrix P \\in \\{0, 1\\}^{k \\times n} | Binary clustering matrix P \\in \\{0, 1\\}^{k \\times\[span\_90\](start\_span)\[span\_90\](end\_span) n} | Soft/hard coupling matrix P \\in \\mathcal{C} |
| **Canonical DAG Construction** | G\_H \= (V, E\_{G\_H}) via structural edge additions | B\_{G\_H} \= M\_{intra} \\odot B \+ M\_{\[span\_106\](start\_span)\[span\_106\](end\_span)inter} \\odot (P^T B\_H P\_{avg}) | B\_{G\[span\_51\](start\_span)\[span\_51\](end\_span)\_H} \= M\_{intra} \\odot B \+ M\_{inter} \\odot (P^T B\_H P\_{avg}) | Not applicable (does not reconstruct a canonical SCM) |
| **Loss Function** | Structural edge delta: $ | E(G\_H) \[span\_107\](start\_span)setminus E(G) | $ | Adjacency matrix reconstruction error: \\|B \- B\_{G\[span\_108\](start\_span)\[span\_108\](end\_span)\_H}\\|\_F |
| **Algorithmic Step Cost** | Number of added edges post-contraction | Balance of block-weight variance and internal weight restoration | Change in path-reconstruction error under matrix inversion | Pairwise distortion of metric measure coupling |
| **Causal Guarantees** | Summary DAG is a sound and complete I-map | Conservation of aggregate causal flow between clusters | Bounded covariance reconstruction error and CI preservation | None (purely geometric/distance-based) |

## **Conclusions and Actionable Insights**

This report establishes a mathematically rigorous generalization of causal DAG summarization designed for dense weighted graphs. By transitioning from structural edge additions to continuous weight projections, this framework addresses the inherent limitations of unweighted abstraction models in high-dimensional, densely connected systems.  
The introduction of the local and global optimization objectives provides a theoretical path for different analytical requirements:

* **The local structural preservation objective** is computationally tractably solved via the closed-form solutions of the W-CaGreS algorithm. This approach is ideal for large-scale applications where structural interpretability and adjacency mapping are paramount.  
* **The global functional preservation objective** represents a functionally consistent causal abstraction. By minimizing the distortion of path-integrated total effects, it ensures that downstream interventional queries and Pearl's do-calculus are preserved.

Theorem 4 links continuous total-effect preservation to the preservation of covariance and vanishing partial correlations. This provides a solid mathematical foundation for performing sound causal inference directly on abstracted dense networks. Furthermore, by incorporating the cyclic coarsening lattice and the SCC floor, this framework successfully handles feedback loops, guaranteeing that the summary graph remains a valid DAG for standard causal inference.  
Future work should focus on extending these weighted formulations to non-linear Structural Causal Models (SCMs) and exploring unsupervised abstraction learning under latent confounding. Additionally, integrating these weighted formulations with constraint-based causal discovery algorithms (such as Cluster-PC or Cluster-FCI) could provide a robust, end-to-end pipeline. This would allow high-dimensional empirical data to be simultaneously discovered, abstracted, and verified at multiple resolutions of causal granularity.

#### **Nguồn trích dẫn**

1\. \[2504.14937\] Causal DAG Summarization (Full Version) \- arXiv, https://arxiv.org/abs/2504.14937 2\. Causal DAG Summarization \- VLDB Endowment, https://www.vldb.org/pvldb/vol18/p1933-youngmann.pdf 3\. Causal DAG Summarization (Full Version) \- arXiv, https://arxiv.org/pdf/2504.14937 4\. CausaLens: A System for Summarizing Causal DAGs \- DSpace@MIT, https://dspace.mit.edu/bitstream/handle/1721.1/164765/3722212.3725086.pdf?sequence=1\&isAllowed=y 5\. \[Literature Review\] Causal DAG Summarization (Full Version) \- Moonlight, https://www.themoonlight.io/en/review/causal-dag-summarization-full-version 6\. Coarsening Linear Non-Gaussian Causal Models with Cycles \- arXiv, https://arxiv.org/html/2605.10163v1 7\. Unsupervised Causal Abstraction \- OpenReview, https://openreview.net/pdf?id=BP0e8RvFwd 8\. Learning Causal Abstractions of Linear Structural Causal Models \- OpenReview, https://openreview.net/pdf?id=XlFqI9TMhf 9\. I Built a Causal AI System for Small Businesses — Part 2: Why Causal Inference Is So Hard to Code \- Reddit, https://www.reddit.com/r/AiForSmallBusiness/comments/1sozvjy/i\_built\_a\_causal\_ai\_system\_for\_small\_businesses/ 10\. Featured Graph Coarsening with Similarity Guarantees \- Proceedings of Machine Learning Research, https://proceedings.mlr.press/v202/kumar23a/kumar23a.pdf 11\. Max-linear models on directed acyclic graphs \- arXiv, https://arxiv.org/pdf/1512.07522 12\. Max-linear models on directed acyclic graphs \- Project Euclid, https://projecteuclid.org/journals/bernoulli/volume-24/issue-4A/Max-linear-models-on-directed-acyclic-graphs/10.3150/17-BEJ941.pdf 13\. Modularity aided consistent attributed graph clustering via coarsening, https://opt-ml.org/papers/2024/paper124.pdf 14\. Gromov-Wasserstein Graph Coarsening \- OpenReview, https://openreview.net/pdf?id=vArRFmzDp6 15\. A Unified Framework for Optimization-Based Graph Coarsening \- Journal of Machine Learning Research, https://www.jmlr.org/papers/volume24/22-1085/22-1085.pdf 16\. VEGAS: Visual influEnce GrAph Summarization on Citation Networks, https://www.computer.org/csdl/journal/tk/2015/12/07152908/13rRUxBa5co 17\. Depicting deterministic variables within directed acyclic graphs: an aid for identifying and interpreting causal effects involving derived variables and compositional data \- PMC, https://pmc.ncbi.nlm.nih.gov/articles/PMC11815499/ 18\. \[Papierüberprüfung\] Coarsening Linear Non-Gaussian Causal Models with Cycles, https://www.themoonlight.io/de/review/coarsening-linear-non-gaussian-causal-models-with-cycles 19\. Exact Routing in Large Road Networks Using Contraction Hierarchies \- ResearchGate, https://www.researchgate.net/publication/259928882\_Exact\_Routing\_in\_Large\_Road\_Networks\_Using\_Contraction\_Hierarchies 20\. High Quality Graph Partitioning \- Christian Schulz, https://schulzchristian.github.io/dissertation\_christian\_schulz.pdf 21\. Causal Abstraction Learning based on the Semantic Embedding Principle \- arXiv, https://arxiv.org/pdf/2502.00407 22\. Cluster-Dags as Powerful Background Knowledge For Causal Discovery \- arXiv, https://arxiv.org/html/2512.10032v2