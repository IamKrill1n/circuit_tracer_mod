# Circuit Tracing Tutorial
#
# This script serves as a tutorial for the circuit tracing library. The library enables
# users to explain model behavior by generating Attribution Graphs.
#
# You can generate your own graphs at https://www.neuronpedia.org/gemma-2-2b/graph?slug=gemma-fact-dallas-austin&pruningThreshold=0.6&pinnedIds=27_22605_10%2C20_15589_10%2CE_26865_9%2C21_5943_10%2C...
# This demo dives into a couple of attribution graphs involving Gemma 2 (2B), based on the
# Multi-Step Reasoning and Multilingual Circuits examples in the original paper.

# --- Colab Setup Environment ---
try:
    import google.colab
    # The following code is intended for Colab environments. Commented out for .py compatibility.
    # !mkdir -p repository && cd repository && \
    #  git clone https://github.com/safety-research/circuit-tracer && \
    #  curl -LsSf https://astral.sh/uv/install.sh | sh && \
    #  uv pip install -e circuit-tracer/
    import sys
    from huggingface_hub import notebook_login
    sys.path.append("repository/circuit-tracer")
    sys.path.append("repository/circuit-tracer/demos")
    notebook_login(new_session=False)
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

from collections import namedtuple
from typing import List, Dict

import torch
from circuit_tracer import ReplacementModel
from circuit_tracer.utils.demo_utils import extract_supernode_features

backend = "transformerlens"  # change to "nnsight" for the nnsight backend!
model = ReplacementModel.from_pretrained("google/gemma-2-2b", "gemma", dtype=torch.bfloat16, backend=backend)

# --- Two-hop reasoning ---
#
# We'll start with the example Fact: The capital of the state containing Dallas is → Austin.
#
# The circuit is similar to that described in the original paper: it has a node corresponding
# to Texas, and shows both a direct path from Dallas to Austin as well as an indirect path
# going through Texas. We'll perform interventions on each of the supernodes shown.

dallas_austin_url = "https://www.neuronpedia.org/gemma-2-2b/graph?slug=gemma-fact-dallas-austin&clerps=%5B%5D&pruningThreshold=0.53&pinnedIds=27_22605_10%2C20_15589_10%2CE_26865_9%2C21_5943_10%2C..."
supernode_features = extract_supernode_features(dallas_austin_url)

# We'll then create a representation of the circuit being used to solve this task.
from graph_visualization import create_graph_visualization, Supernode, InterventionGraph, Feature

# Supernodes that upweight certain outputs
say_austin_node = Supernode(
    name="Say Austin", features=[Feature(layer=23, pos=10, feature_idx=12237)]
)
say_capital_node = Supernode(
    name="Say a capital",
    features=supernode_features["capital cities / say a capital city"],
    children=[say_austin_node],
)
texas_node = Supernode(
    name="Texas", features=supernode_features["Texas"], children=[say_austin_node]
)
state_node = Supernode(
    name="state", features=supernode_features["state"], children=[say_capital_node, texas_node]
)
capital_node = Supernode(
    name="capital", features=supernode_features["capital"], children=[say_capital_node]
)
# Embedding nodes
dallas_node = Supernode(name="Emb: Dallas", features=None, children=[texas_node])
state_emb_node = Supernode(name="Emb: state", features=None, children=[state_node])
capital_emb_node = Supernode(name="Emb: capital", features=None, children=[capital_node])

# Initialize the InterventionGraph
prompt = "Fact: the capital of the state containing Dallas is"
ordered_nodes = [
    [capital_emb_node, state_emb_node],
    [capital_node, state_node, dallas_node],
    [say_capital_node, texas_node],
    [say_austin_node],
]
dallas_austin_graph = InterventionGraph(ordered_nodes=ordered_nodes, prompt=prompt)
logits, dallas_activations = model.get_activations(prompt)

for node in [capital_node, state_node, dallas_node, say_capital_node, texas_node, say_austin_node]:
    dallas_austin_graph.initialize_node(node, dallas_activations)
dallas_austin_graph.set_node_activation_fractions(dallas_activations)

# Record the top-5 logits and visualize the graph
def get_top_outputs(logits: torch.Tensor, k: int = 5):
    top_probs, top_token_ids = logits.squeeze(0)[-1].softmax(-1).topk(k)
    top_tokens = [model.tokenizer.decode(token_id) for token_id in top_token_ids]
    top_outputs = list(zip(top_tokens, top_probs.tolist()))
    return top_outputs

top_outputs = get_top_outputs(logits)
create_graph_visualization(dallas_austin_graph, top_outputs)

# Interventions: verify each supernode's role by ablation.
Intervention = namedtuple("Intervention", ["supernode", "scaling_factor"])

def supernode_intervention(
    intervention_graph: InterventionGraph,
    interventions: List[Intervention],
    replacements: Dict[str, Supernode] = None,
):
    """
    Performs interventions on a set of supernodes, records the outputs, and draws the corresponding graph.
    """
    intervention_values = [
        (*feature, scaling_factor * default_act)
        for intervened_supernode, scaling_factor in interventions
        for feature, default_act in zip(
            intervened_supernode.features, intervened_supernode.default_activations
        )
    ]
    new_logits, new_activations = model.feature_intervention(
        intervention_graph.prompt, intervention_values
    )
    intervention_graph.set_node_activation_fractions(new_activations)
    top_outputs = get_top_outputs(new_logits)

    for intervened_supernode, scaling_factor in interventions:
        intervened_supernode.activation = None
        intervened_supernode.intervention = f"{scaling_factor}x"

    if replacements is not None:
        for target, replacement in replacements.items():
            intervention_graph.nodes[target].replacement_node = replacement

    return create_graph_visualization(intervention_graph, top_outputs)

# Example: Turning off the "Say a capital" supernode.
supernode_intervention(dallas_austin_graph, [Intervention(say_capital_node, -2)])

# Example: Turning off the "capital" supernode.
supernode_intervention(dallas_austin_graph, [Intervention(capital_node, -2)])

# Example: Turning off the Texas supernode.
supernode_intervention(dallas_austin_graph, [Intervention(texas_node, -2)])

# Example: Turning off the "state" supernode.
supernode_intervention(dallas_austin_graph, [Intervention(state_node, -2)])

# Example of cross-circuit intervention (injecting different nodes, e.g., for Oakland/Sacramento)
oakland_prompt = "Fact: the capital of the state containing Oakland is"
_, oakland_activations = model.get_activations(oakland_prompt)
oakland_url = "https://www.neuronpedia.org/gemma-2-2b/graph?slug=gemma-fact-oakland-sacramento&clerps=%5B%5D&pruningThreshold=0.5&pinnedIds=27_43939_10%2CE_49024_9%2C21_5943_10%2C19_9209_10%2C18..."
oakland_supernodes = extract_supernode_features(oakland_url)

say_sacramento_node = Supernode(
    "Say Sacramento", features=[Feature(layer=19, pos=10, feature_idx=9209)]
)
california_node = Supernode(
    "California",
    features=oakland_supernodes["California"] + oakland_supernodes["California (2)"],
    children=[say_sacramento_node],
)

for node in [say_sacramento_node, california_node]:
    dallas_austin_graph.initialize_node(node, oakland_activations)

oakland_interventions = [Intervention(texas_node, -2), Intervention(california_node, 2)]
supernode_intervention(
    dallas_austin_graph,
    oakland_interventions,
    {texas_node.name: california_node, say_austin_node.name: say_sacramento_node},
)