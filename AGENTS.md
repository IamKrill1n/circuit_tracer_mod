# AGENTS.md

Guidance for AI coding agents working in this repository.

## Project Overview

Fork of `safety-research/circuit-tracer` with an added **summarization pipeline** that automates pruning, clustering, and visualization of attribution graphs. The core library computes circuits for transformer LMs using cross-layer transcoders (CLTs).

## Setup

```bash
conda activate circuit
pip install -e ".[dev]"
pip install -e ".[dev,viz]"   # also installs Streamlit extras
```

Credentials live in `.env` at the repo root (loaded via `config.py`):
- `NEURONPEDIA_API_KEY` — feature label lookups
- `HUGGINGFACE_API_KEY` — gated models
- `GEMINI_API_KEY` / `GENAI_API_KEY` — LLM auto-interpretation
- `OPENAI_API_KEY`

## Commands

```bash
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

After any code change: run `ruff check` and `pytest tests -m "not requires_disk"` to verify.

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

Stages: `Graph` → `AttrGraph` → `PruneGraph` → clusters → `SummaryGraph`
(attribute → [token_attribution] → prune → [classify] → cluster → summarize). `pipeline.run_pipeline()` orchestrates end-to-end; `python -m summarization` is a thin CLI over it.

| Module | Purpose |
|---|---|
| `attr_graph.AttrGraph` | Node-level wrapper around `Graph` or frontend JSON. Nodes are `summarize.Node` dataclasses. |
| `prune.prune_attr_graph()` | `AttrGraph`/`Graph` → `PruneGraph` via combined influence + relevance scores. Optional SHAP token weights and logit-probability weighting. Pure tensor math, no Neuronpedia. |
| `summarize` | `Node`, `Supernode`, `SummaryGraph` types + `summarize()` (clusters → `SummaryGraph` with block-sum supernode adjacency). |
| `cluster` | `cluster()` (dispatch: spectral / agglomerative / ilp), `find_best_k()`, `cluster_graph_spectral()`, `cluster_graph_agglomerative()`, `compute_phi_vectors()`. |
| `ilp_cluster.cluster_graph_ilp()` | Stage-2 facility-location MILP (scipy HiGHS): minimizes atomicity + λ·causal loss under a complexity budget. |
| `scoring` | Stage-2 loss terms: `compute_L_atom` (signed-cosine correlation clustering), `compute_L` (atom + causal), silhouette. |
| `classify` | Neuronpedia feature labels + `filter_act_density()` activation-density filtering. |
| `group_llm` | LLM (Gemini) supernode labelling and scoring. |
| `cluster_viz.supernode_graph_figure()` | Plotly figure for the supernode graph. |
| `token_attribution` | SHAP-based token importance for embedding nodes. |
| `graph_utils` | Pure-math scoring: influence / relevance, score combiners (geometric / arithmetic / harmonic), normalization. |
| `utils` | Frontend-JSON node parsing and node-role helpers (`node_is_embedding`/`_logit`/`_fixed`, layer indexing). |

### Evaluation (`eval/`)

Standalone scripts — not a library. Each is runnable end-to-end:
- `eval_faithfulness.py` — causal faithfulness of pruning (zero dropped CLT features, measure P(target) ratio)
- `eval_intervention.py` — supernode causal validation via feature interventions
- `eval_prune.py`, `prune_graphs.py` — pruning sweep + quality metrics
- `analyze_prune.py` — compare pruning normalization schemes vs. baselines
- `eval_cluster.py` — clustering quality
- `eval_steering.py` — supernodes as steerable concepts on the BATS analogy dataset (ablate / steer)
- `pareto_curve.py` — Stage-2 atomicity-vs-causal Pareto front (sweep `lambda_causal` over `cluster_graph_ilp`)

### Streamlit app (`app.py`)

Seven-stage pipeline UI: input → original attribution graph → prune → cluster → supernode graph → upload to Neuronpedia → steering intervention.

### Paper & thesis

LaTeX sources for the write-up live in `paper/` and `SOICT_DATN_Research_ENG_Template/` (graduation thesis). LaTeX math is permitted only there — see `.claude/rules/math.md`.

### External API (`api.py`)

Neuronpedia REST wrapper. Key function: `get_feature(modelId, layer, index)`.

## Key Conventions

- **Adjacency**: `adj[row_i, col_j]` = edge weight from source `j` to target `i` (columns = sources, rows = targets).
- **Node ordering in `Graph`**: `[active_features..., error_nodes..., embed_nodes..., logit_nodes]`.
- **Type annotations**: `X | Y` not `Optional[X]`; built-in `list`/`dict`/`tuple` not `List`/`Dict`/`Tuple` (enforced by ruff).
- **Backend parity**: nnsight and transformerlens outputs must match. Tests in `tests/test_transformerlens_nnsight_same_*.py` verify this.
- **Seeds**: every stochastic operation must accept an explicit seed — never hardcode or silently default.
- **Eval scripts**: inline logic is preferred over abstractions; duplication across two scripts is acceptable.
- **No speculative robustness**: don't add fallbacks for inputs that can't occur. Validate only at repo boundaries (CLI args, files from disk).

## Code Style

- Name variables after their mathematical meaning (`phi_vectors`, `influence_scores`, `adj`), not their type.
- Add a `# (N, d)` shape comment on a tensor the first time it appears in a non-trivial computation.
- One concept per function. Flat over nested.
- Comments explain *why*, not what. Omit if the code is self-explanatory.
