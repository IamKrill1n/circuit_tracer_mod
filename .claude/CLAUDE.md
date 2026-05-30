# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project Overview

Fork of `safety-research/circuit-tracer` with an added **summarization pipeline** that automates pruning, clustering, and visualization of attribution graphs. The core library computes circuits for transformer LMs using cross-layer transcoders (CLTs).

## Environment

Always activate the conda environment before running Python:
```bash
conda activate circuit
```

Credentials live in `.env` at the repo root (loaded via `config.py`):
- `NEURONPEDIA_API_KEY` — feature label lookups
- `HUGGINGFACE_API_KEY` — gated models
- `GEMINI_API_KEY` / `GENAI_API_KEY` — LLM auto-interpretation
- `OPENAI_API_KEY`

## Commands

```bash
# Install
pip install -e ".[dev]"
pip install -e ".[dev,viz]"   # also installs Streamlit extras

# Run app
streamlit run app.py

# Lint / format
ruff check
ruff format

# Type check
pyright

# Tests (skip disk-heavy)
pytest tests -m "not requires_disk"
pytest tests/test_prune.py                          # single file
pytest tests/test_prune.py::test_function_name      # single test
```

## Architecture

### Core library (`circuit_tracer/`)

Pipeline: prompt → `ReplacementModel` → `attribute()` → `Graph`

| Module | Purpose |
|---|---|
| `ReplacementModel` | Loads transformer + transcoder set. Backends: `nnsight`, `transformerlens`. Use `lazy_encoder=True` to defer weight loading. |
| `attribute()` | Main entry point. Routes to `attribute_nnsight.py` or `attribute_transformerlens.py`. |
| `Graph` | Sparse adjacency matrix over nodes. Saved as `.pt`. |
| `transcoder` | `CrossLayerTranscoder` and `SingleLayerTranscoder` + activation functions. |
| `frontend` | Local HTTP server + Pydantic models for the JS circuit viewer (`assets/`). |
| `utils.create_graph_files` | Converts `Graph` to JSON for the frontend viewer. |

### Summarization pipeline (`summarization/`)

Layered pipeline: `Graph` → `AttrGraph` → `PruneGraph` → `SummarizationGraph`

| Module | Purpose |
|---|---|
| `attr_graph.AttrGraph` | Node-level wrapper around `Graph` or frontend JSON. Nodes are `supernode_graph.Node` dataclasses. |
| `prune.prune_attr_graph()` | Prunes via combined influence + relevance scores. Optional SHAP token weights and logit-probability weighting. |
| `cluster` | `cluster_graph_spectral()`, `cluster_graph_agglomerative()`, `compute_phi_vectors()`. |
| `supernode_graph.SummarizationGraph` | Final summarized graph of `Supernode` objects and their adjacency. |
| `cluster_viz` | Plotly figure for the supernode graph. |
| `token_attribution` | SHAP-based token importance for embedding nodes. |
| `graph_utils` | Score combiners (geometric / arithmetic / harmonic), normalization, influence / relevance. |

### Evaluation (`eval/`)

Standalone scripts — not a library. Each is runnable end-to-end:
- `eval_faithfulness.py` — faithfulness metrics
- `eval_intervention.py` — intervention validation
- `eval_prune.py`, `prune_graphs.py` — pruning quality
- `eval_cluster.py` — clustering quality

### Streamlit app (`app.py`)

Five-stage pipeline UI: generate graph → view raw circuit → prune → cluster → display supernode graph.

### External API (`api.py`)

Neuronpedia REST wrapper. Key function: `get_feature(modelId, layer, index)`.

## Key Conventions

- **Adjacency**: `adj[row_i, col_j]` = edge weight from source `j` to target `i` (columns = sources, rows = targets).
- **Node ordering in `Graph`**: `[active_features..., error_nodes..., embed_nodes..., logit_nodes]`.
- **Type annotations**: `X | Y` not `Optional[X]`; built-in `list`/`dict`/`tuple` not `List`/`Dict`/`Tuple` (enforced by ruff).
- **Backend parity**: nnsight and transformerlens outputs must match. Tests in `tests/test_transformerlens_nnsight_same_*.py` verify this.
- **Seeds**: every stochastic operation must accept an explicit seed — never hardcode or silently default.
- **Output format**: plans and experiment summaries → self-contained HTML with inline CSS. Math in chat → Unicode (∑ θ α), never LaTeX.
