# Repository Guidelines

## Project Structure & Module Organization

This repository extends `circuit-tracer` for attribution-graph analysis and summarization. Core library code lives in `circuit_tracer/`, including `attribution/`, `replacement_model/`, `transcoder/`, `utils/`, and browser visualization support in `frontend/`. The graph summarization pipeline is in `summarization/`; prompts and model routing data are under `summarization/prompts/` and `summarization/llm_models.json`. Evaluation and pruning scripts are in `eval/`. Tests mirror the package layout in `tests/`. Research writing and generated reports live in `paper/`, `SOICT_DATN_Research_ENG_Template/`, `docs/adr/`, `reports/`, and `figures/`.

## Build, Test, and Development Commands

- `pip install -e ".[dev]"`: install the package in editable mode with pytest, Ruff, Pyright, and IPython.
- `python -m pytest tests -m "not requires_disk"`: run the CI test set, excluding disk-heavy tests.
- `python -m pytest tests/test_prune.py`: run a targeted test file while developing.
- `python -m ruff format .`: format Python files.
- `python -m ruff check .`: lint for Pyflakes, pycodestyle, and tidy-import issues.
- `python -m pyright`: run basic type checking.
- `python -m summarization ...` or `circuit-tracer ...`: use package entry points.

Always run commands that involve Python, pytest, pip, or project scripts in the `circuit` conda environment. Prefix commands with `conda run -n circuit`, for example:

```bash
conda run -n circuit pytest tests/test_prune.py
conda run -n circuit python -m summarization --help
```

Do not assume the environment is already active because shell state does not persist between tool calls.

## Coding Style & Naming Conventions

Use Python 3.10+ syntax. Ruff uses a 100-column line length, with `E501` ignored, and prefers modern types: `list`, `dict`, `tuple`, `A | B`, and `T | None` instead of legacy `typing` aliases. Keep module and test filenames lowercase with underscores, for example `test_group_llm.py`. Prefer small, explicit functions and keep experiment scripts separate from reusable package modules.

This is a research codebase. Optimize for readability, traceability of assumptions, and fast iteration. Prioritize readability, reproducibility, maintainability, then correctness on real inputs. Prefer explicit, flat code over abstractions, and avoid speculative robustness such as fallback branches, broad `try`/`except` blocks, or type coercions for cases that cannot happen in the current workflow.

Name variables after their mathematical meaning, such as `phi_vectors`, `influence_scores`, or `adj`, rather than their container type. Use short comments only for non-obvious invariants, tensor shapes, adjacency conventions, threshold semantics, numerical pitfalls, or deliberate deviations from a paper. Validate early at research boundaries such as CLI args, config values, files, and API responses; do not re-check types or shapes mid-pipeline when callers already guarantee them.

For numeric code, prefer named intermediate tensors over long chained expressions, and add a shape comment like `# (N, d)` the first time a tensor appears in a non-trivial computation. Every stochastic operation must accept an explicit seed parameter. Do not hardcode or silently default to a random seed.

Treat eval scripts as experiments, not frameworks. Do not create base classes, registries, or plugin systems speculatively. Duplication across a small number of eval scripts is acceptable; introduce abstractions only when at least three concrete cases already exist.

Never use LaTeX math syntax in chat or terminal output. Use Unicode math symbols such as `∑`, `∫`, `√`, `θ`, `α`, `→`, `∞`, `x²`, and `xᵢ`. LaTeX is permitted only inside `paper/`.

## Testing Guidelines

Tests use `pytest`. Name tests `test_*.py` and place package-specific tests near the matching area under `tests/`. Mark storage-heavy cases with `@pytest.mark.requires_disk` so CI can skip them. Add focused tests for new graph logic, pruning behavior, model routing, prompts, or frontend serialization changes.

Write tests that verify concrete research invariants, such as pruning retaining the target logit or clustering producing non-overlapping supernodes. Skip tests for internal helpers unless they encode a subtle invariant worth pinning down.

## Commit & Pull Request Guidelines

Recent commits use short, imperative summaries such as `add pruning eval`, `fix visualization`, and `add labeling prompt`. Keep subjects concise and behavior-focused. PRs should describe the change, list verification commands, note skipped heavy tests or notebook checks, and link related issues or design notes. Include screenshots or report paths for graph visualization, Streamlit UI, or HTML report changes.

## Security & Configuration Tips

Do not commit `.env`, API keys, model tokens, large generated graph artifacts, or cache directories. Keep provider details in `config.py` or `summarization/llm_models.json`, and document new required environment variables in the PR.
