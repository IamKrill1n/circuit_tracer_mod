# Fork of safety-research/circuit-tracer 
This is a fork of `safety-research/circuit-tracer`, extended with a **summarization pipeline** that automates summarizing the attribution graph. The core library computes attribution graphs (circuits) for transformer language models using MLP transcoders.
## Recap
The summarization pipeline consists of 2 stage:
1. Prunning: prune the attribution graph down to a subgraph containing important nodes and edges. 
- Key idea: introduce **relevance** and use along with **influence** (from original clt paper) to align the subgraph with human rationale better (features activating on useless tokens are pruned).
2. Clustering: cluster functionally similar feature nodes together, further simplify the summarization graph and aid steering.
- Key idea: use agglomerative clustering on a weighted edge profile similarity matrix with layer span penalty, ensuring DAG.

## Eval 