# Automate summarization of attribution graphs for circuit tracing pipeline

A framework to summarize attribution graphs from circuit_tracer library.

We employ a 3 stage approach:
- Prune: reduce the attribution graph to important nodes and edges
- Cluster: group functionally similar nodes into supernodes.
- Interprete: visualize the graph as a DAG with supernodes' labels

