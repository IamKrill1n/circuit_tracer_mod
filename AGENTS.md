# AUTOMATIC ATTRIBUTION GRAPHS SUMMARIZATION FOR CIRCUIT TRACING PIPELINE

Extends `circuit-tracer` with a summarization pipeline: attribute → prune → cluster → summarize into supernode graphs.

## Structure

- `circuit_tracer/` — upstream attribution library (graph, replacement model, transcoder). Don't fork logic here; import it.
- `summarization/` — the pipeline. Entry: `summarization/pipeline.py::run_pipeline`, CLI: `python -m summarization`. Stages: `prune.py` → `cluster.py` (`ilp_cluster.py`) → `summarize.py`. `label.py` / `group_llm.py` handle LLM supernode labeling via registry in `summarization/llm_models.json`; prompts in `summarization/prompts/`.
- `eval/` — experiment scripts (`eval_prune.py`, `eval_cluster.py`, `eval_faithfulness.py`, `eval_intervention.py`, `eval_steering.py`) + `legacy_cluster_baselines.py` (spectral/agglomerative baselines; `--method spectral|agglomerative` routes here, `ilp` is canonical).
- `visualization_app/` — Streamlit app (`server.py`, `services.py`). Root `api.py`, `config.py`, `attribute_utils.py`, `import_dataset.py` support it.
- `tests/` mirror package layout. `CONTEXT.md` defines domain terms (supernode, role vocabulary Input/Abstract/Output/Trash, summary graph).

## Commands

Install: `pip install -e ".[dev]"` (Python >=3.10). CI order — run in this order:
`python -m ruff format --check` → `python -m ruff check` → `python -m pyright` → `python -m pytest tests -m "not requires_disk"`

- Full pipeline: `python -m summarization --prompt "..." --model google/gemma-2-2b --transcoder mntss/clt-gemma-2-2b-2.5M` (needs GPU/CUDA; `--graph-pt` loads a saved graph instead, `--graph-pt-out` saves one).
- Focused test: `python -m pytest tests/test_prune.py` (same for `test_group_llm.py`, `test_pipeline_cli.py`, `test_ilp_cluster.py`).
- Skip GPU/disk-heavy suites with `-m "not requires_disk"`; attribution tests (`test_attributions_*`, `test_transformerlens_*`) need model weights + VRAM.

## Conventions

- Ruff: line-length 100, `E501` ignored; `TID` rules ban `typing.List/Dict/Tuple/Union/Optional` — use `list`, `dict`, `A | B`, `T | None`. Pyright `basic`, excludes `demos/`.
- Research code: flat explicit functions, no speculative frameworks. Eval scripts may duplicate; abstract only at ≥3 concrete cases. No broad `try/except`, no fallback branches for impossible cases.
- Numeric code: named intermediate tensors + shape comment `# (N, d)` on first non-trivial appearance.
- Every stochastic op takes an explicit seed param (`random_state`, never hardcoded); `tests/conftest.py` seeds `torch.manual_seed(42)`.
- Validate early at boundaries (CLI args, config, `.pt` files, LLM responses); don't re-check shapes mid-pipeline.
- Test invariants, not helpers: pruning retains target logit, supernodes are non-overlapping DAG, use `@pytest.mark.requires_disk` for storage-heavy cases.
- Secrets via `.env` loaded by `config.py` (`OPENAI_API_KEY`, `GEMINI_API_KEY`/`GENAI_API_KEY`, `HUGGINGFACE_API_KEY`, `NEURONPEDIA_API_KEY`). Never commit `.env`, keys, or `temp_graph_files/` artifacts. New provider → update `summarization/llm_models.json` + document env var in PR.
- Commits: short imperative (`add pruning eval`, `fix visualization`). PRs list verification commands + skipped heavy tests.
