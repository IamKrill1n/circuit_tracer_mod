# Architecture

- `circuit_tracer/` is the upstream attribution library (graphs, replacement models, transcoders). Import its logic instead of duplicating it in the summarization pipeline.
- `summarization/` implements the pipeline. Entry: `summarization/pipeline.py::run_pipeline`; CLI: `python -m summarization`. Stages: `prune.py` → `cluster.py` (`ilp_cluster.py`) → `summarize.py`.
- Optional pruning substages run after influence pruning: `prune.py::filter_act_density` (activation-density band) and `jev_relevance.py::filter_by_jev_relevance` (query-conditioned TypeSafe Jev Noul judgments; needs `TYPESAFE_API_KEY`).
- Claim-conditioned token seeds come from `semantic_seeds.py::select_semantic_seeds` (selector-LLM spans, default) or `jev_seeds.py::select_jev_semantic_seeds` (`--seed-selector jev`, per-token Jev Nouls; arithmetic/alpha=1 influence-only fallback when no token clears the threshold). Shared TypeSafe batching lives in `typesafe_batch.py`.
- `summarization/label.py` and `group_llm.py` handle LLM supernode labeling. Model registry: `summarization/llm_models.json`; prompts: `summarization/prompts/`.
- `eval/` contains pruning, clustering, faithfulness, intervention, and steering experiments. `legacy_cluster_baselines.py` handles `--method spectral|agglomerative`; `ilp` is canonical.
- `visualization_app/` contains the Streamlit app (`server.py`, `services.py`). Root `api.py`, `config.py`, `attribute_utils.py`, and `import_dataset.py` support it.
- Tests live in `tests/`. Read [CONTEXT.md](../../CONTEXT.md) when working with domain terms, including supernodes, Input/Abstract/Output/Trash roles, and summary graphs.
