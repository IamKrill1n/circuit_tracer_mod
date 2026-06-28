# Automate summarization of attribution graphs for circuit tracing pipeline

A framework to summarize attribution graphs from the `circuit_tracer` library.

We employ a 3-stage approach:

- **Prune** — reduce the attribution graph to important nodes and edges
- **Cluster** — group functionally similar nodes into supernodes
- **Visualize** — label supernodes and visualize the summary graph as a DAG

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA (attribution, SHAP token weights, and steering are GPU-bound)
- A conda environment named `circuit` (or your own env with the same dependencies)

## Installation

Clone the repository and install the package in editable mode:

```bash
git clone <repo-url> circuit_tracer_mod
cd circuit_tracer_mod

conda create -n circuit python=3.10 -y
conda activate circuit

pip install -e ".[dev,viz]"
```

The `[dev]` extra installs pytest, ruff, and pyright. The `[viz]` extra adds Plotly, scipy, scikit-learn, and networkx used by clustering visualization and the summary figure.

If you already have the `circuit` environment, activate it before any Python command:

```bash
conda activate circuit
```

## Configuration

Create a `.env` file at the repo root. Values are loaded by `config.py` at import time.

```bash
# Optional — Neuronpedia feature lookups and upload
NEURONPEDIA_API_KEY=

# Optional — gated Hugging Face models
HUGGINGFACE_API_KEY=

# Optional — LLM supernode labeling (Gemini / Google GenAI)
GEMINI_API_KEY=
# or
GENAI_API_KEY=

# Optional — OpenAI-compatible labeling models
OPENAI_API_KEY=
```

| Variable | Used for |
|---|---|
| `HUGGINGFACE_API_KEY` | Downloading gated models and transcoders |
| `NEURONPEDIA_API_KEY` | Activation-density filtering, feature evidence, Neuronpedia upload |
| `GEMINI_API_KEY` / `GENAI_API_KEY` | Supernode labeling via Google models in `summarization/llm_models.json` |
| `OPENAI_API_KEY` | Supernode labeling via OpenAI models |

Labeling and Neuronpedia upload are optional. Local attribution, pruning, and clustering work without API keys.

## Summarization pipeline (CLI)

The end-to-end pipeline lives in `summarization/`:

**Graph → token weights → prune → [activation-density filter] → cluster → SummaryGraph**

Run it with:

```bash
conda run -n circuit python -m summarization --help
```

### Example: attribute a prompt and summarize

```bash
conda run -n circuit python -m summarization \
  --prompt "The capital of Texas is" \
  --model google/gemma-2-2b \
  --transcoder mntss/clt-gemma-2-2b-2.5M \
  --graph-pt-out generated_graphs/capital_texas.pt \
  --prune-graph-out temp_graph_files/capital_texas_pruned.pt \
  --summary-graph-out summary_graphs/capital_texas.sng.pt \
  --figure-html-out reports/capital_texas_summary.html \
  --supernodes-out temp_graph_files/supernodes.json
```

### Example: start from an existing `.pt` graph

```bash
conda run -n circuit python -m summarization \
  --graph-pt generated_graphs/capital_texas.pt \
  --no-auto-token-weights \
  --summary-graph-out summary_graphs/capital_texas.sng.pt
```

Useful flags:

| Flag | Default | Purpose |
|---|---|---|
| `--node-threshold` | `0.02` | Pruning node score cutoff |
| `--edge-threshold` | `0.9` | Pruning edge score cutoff |
| `--method` | `ilp` | Clustering method (`ilp`, `spectral`, `agglomerative`) |
| `--target-k` | `7` | Number of supernodes (when not using `--auto-k`) |
| `--filter-act-density` | on | Drop features outside the interpretability band |
| `--features-dir` | — | Local Neuronpedia feature mirror for density filtering |
| `--figure-html-out` | — | Write a Plotly HTML summary figure |

Upload the result to Neuronpedia with `--upload --slug ... --display-name ...` (requires `NEURONPEDIA_API_KEY`).

## Circuit tracer CLI

The upstream `circuit-tracer` entry point handles raw attribution and the stock JS graph viewer:

```bash
conda run -n circuit circuit-tracer --help
```

### Run attribution and save a graph

```bash
conda run -n circuit circuit-tracer attribute \
  -p "The capital of Texas is" \
  -t mntss/clt-gemma-2-2b-2.5M \
  -o generated_graphs/capital_texas.pt \
  --slug capital-texas \
  --graph_file_dir graph_files/custom/capital-texas \
  --node_threshold 0.8 \
  --edge_threshold 0.98
```

This writes a `.pt` graph and JSON files for the frontend viewer under `graph_files/`.

## Visualization app (FastAPI)

The main dashboard for browsing graphs, generating new attributions, running the full summary pipeline (including LLM labeling), and steering interventions is a FastAPI app under `visualization_app/`.

Start it with:

```bash
conda run -n circuit uvicorn visualization_app.server:app --reload --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`. From the UI you can:

- Browse graphs under `graph_files/` (`analogies`, `multihop`, `custom`)
- Upload a `.pt` graph or generate one from a prompt
- Run **Generate Summary** (prune → cluster → label → save)
- Open the raw attribution graph or the labeled summary in the embedded circuit_tracer viewer
- Run steering interventions on stored supernodes

Data directories (relative to the repo root):

| Path | Contents |
|---|---|
| `graph_files/` | Frontend JSON graphs for the viewer |
| `summary/` | Labeled summary artifacts per dataset |
| `summary_graphs/` | Saved `SummaryGraph` `.sng.pt` files |
| `generated_graphs/` | Raw attribution `.pt` files from generation/upload |

### Import a dataset into the app

Register precomputed dataset graphs and summaries:

```bash
conda run -n circuit python import_dataset.py \
  --dataset analogies \
  --graphs-root dataset/analogies \
  --summary-dir labeled_summary/entmax/alpha_0.50/node_0.02
```

Use `--dataset multihop` for the multihop benchmark set. Pass `--source-set mntss/clt-gemma-2-2b-426k` when graphs live under a transcoder namespace.

## Development

```bash
conda activate circuit

# Format and lint
python -m ruff format .
python -m ruff check .

# Type check
python -m pyright

# Tests (skip disk-heavy cases)
python -m pytest tests -m "not requires_disk"

# Targeted test while developing
python -m pytest tests/test_prune.py -x -q
```

See `AGENTS.md` for project layout, conventions, and module map.

## Project layout

| Path | Role |
|---|---|
| `circuit_tracer/` | Core attribution library, transcoders, frontend assets |
| `summarization/` | Prune → cluster → visualize pipeline and LLM labeling |
| `visualization_app/` | FastAPI dashboard and job orchestration |
| `eval/` | Standalone evaluation and ablation scripts |
| `tests/` | Pytest suite |
| `dataset/` | Benchmark prompts and precomputed graphs |
| `notebook/` | Reproducible experiment notebooks |
