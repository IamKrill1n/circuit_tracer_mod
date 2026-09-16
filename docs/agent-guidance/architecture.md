# Architecture

- `circuit_tracer/` is the upstream attribution library (graphs, replacement models, transcoders). Import its logic instead of duplicating it in the summarization pipeline.
- `summarization/` implements the pipeline. Entry: `summarization/pipeline.py::run_pipeline`; CLI: `python -m summarization`. Stages: `prune.py` → `cluster.py` (`ilp_cluster.py`) → `summarize.py`.
- `summarization/label.py` and `group_llm.py` handle LLM supernode labeling. Model registry: `summarization/llm_models.json`; prompts: `summarization/prompts/`.
- `eval/` contains pruning, clustering, faithfulness, intervention, and steering experiments. `legacy_cluster_baselines.py` handles `--method spectral|agglomerative`; `ilp` is canonical.
- `visualization_app/` contains the Streamlit app (`server.py`, `services.py`). Root `api.py`, `config.py`, `attribute_utils.py`, and `import_dataset.py` support it.
- Tests live in `tests/`. Read [CONTEXT.md](../../CONTEXT.md) when working with domain terms, including supernodes, Input/Abstract/Output/Trash roles, and summary graphs.
