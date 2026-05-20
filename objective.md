# Objectives

## Interpreting summarization graph as flow

Forward flow:
- Define outgoing weight normalized matrix: 
\begin{equation}
P^{\text{out}}_{uv} = \frac{|W_{uv}|}{\sum_{w : (u, w) \in E} |W_{uw}|},
\label{eq:Pout}
\end{equation}
with $\sum_v P^{\text{out}}_{uv} = 1$ whenever $u$ has at least one
out-edge. This redistributes any flow entering $u$ across its
out-edges in proportion to attribution mass.

Backward flow
- Define **relevance** as forward flow when we treat embeding nodes as sources, logit nodes as sinks. Guarantee to converge because the graph is a DAG so the matrix is nilpotent

- Define ingoing weight normalized matrix:
- Define **influence** as backward flow when we treat logit nodes as sources, embeding nodes as sinks. Converge...

Conservation lemma

## Loss function

- L_cons = prune loss + agg loss
    + prune loss = influence completeness + relevance completeness
    + agg loss = edge weight loss (same as before)

- L_coh (same as before)
- L_cplx: number of supernodes/number of nodes i

