Guidance for Agent when working in this repository.

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

**No LaTeX compiler** is installed here — do not try to compile `.tex` or verify exact page counts. State this limitation when delivering LaTeX edits to `paper/` or the thesis template.

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

## Testing

- Default to **targeted** tests scoped to the change: `pytest tests/test_prune.py -k <name> -x -q`.
- Do **not** run the full suite unless explicitly asked — it loads slow models and can take 20+ minutes. Use `-m "not requires_disk"` to skip disk-heavy tests; reserve a full run for an explicit final verification step.

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
| `group_llm` | LLM supernode labelling via a model registry (`llm_models.json`) + provider router (OpenAI / Gemini / OpenAI-compatible). Entry point: `label_supernodes(sng, model_name, settings=, scheme=)`. |
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
- **Output format**: plans and experiment summaries → self-contained HTML with inline CSS. Math in chat → Unicode (∑ θ α), never LaTeX.

## Data Pipeline & Model Identity

- The pipeline runs **local attribution computation**, not Neuronpedia frontend-JSON downloads. Neuronpedia is used only for the upload path (`save_subgraph`).
- Derive model identity (tokenizer) from the graph's `.pt` via `cfg.tokenizer_name` (the HF id). Do **not** assume `cfg.model_name` is HF-loadable — it's a TransformerLens alias.

## Code Organization

- `app.py` is **display-only**. Put evaluation/intervention logic in `eval/` (e.g. `eval_intervention.py`), not in `app.py`; the app should only render results.

## Claude Docs Directories

Per the global output-format rule, plans/reports are self-contained HTML (inline CSS). Save them to the matching top-level folder — do **not** drop HTML/report files in the repo root:

| Folder | Holds |
|---|---|
| `brainstorm/` | Early-stage idea exploration, scheme proposals, design comparisons |
| `plans/` | Implementation plans (step-by-step, with verify checks) |
| `debug/` | Diagnose / root-cause reports |
| `reports/` | Experiment summaries, code reviews, architecture decisions |

Name files descriptively (e.g. `debug/ilp_collapse_diagnosis.html`). `notes/` is pre-existing and out of scope for this scheme.

## Model & Tokenizer Notes

- For Qwen3 chat prompts, use the **non-thinking** prefix to avoid `<think>` tokens, and preserve special tokens.
- SHAP token attribution must align with the chat template. Known pitfalls: a **double-BOS off-by-one** position shift (silent steering no-op) and a **transposed SHAP values axis**.

## Working Style & Scope

- Before wiring a solution through many files, confirm the intended **design** — e.g. one fixed behavior vs. several selectable modes. Prefer minimal, approved scope.
- Don't expand a focused task into unrelated type/dependency/formatting fixes; surface those separately instead.
