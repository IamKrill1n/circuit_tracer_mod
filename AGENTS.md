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

## Coding Style & Naming Conventions

Use Python 3.10+ syntax. Ruff uses a 100-column line length, with `E501` ignored, and prefers modern types: `list`, `dict`, `tuple`, `A | B`, and `T | None` instead of legacy `typing` aliases. Keep module and test filenames lowercase with underscores, for example `test_group_llm.py`. Prefer small, explicit functions and keep experiment scripts separate from reusable package modules.

## Testing Guidelines

Tests use `pytest`. Name tests `test_*.py` and place package-specific tests near the matching area under `tests/`. Mark storage-heavy cases with `@pytest.mark.requires_disk` so CI can skip them. Add focused tests for new graph logic, pruning behavior, model routing, prompts, or frontend serialization changes.

## Commit & Pull Request Guidelines

Recent commits use short, imperative summaries such as `add pruning eval`, `fix visualization`, and `add labeling prompt`. Keep subjects concise and behavior-focused. PRs should describe the change, list verification commands, note skipped heavy tests or notebook checks, and link related issues or design notes. Include screenshots or report paths for graph visualization, Streamlit UI, or HTML report changes.

## Security & Configuration Tips

Do not commit `.env`, API keys, model tokens, large generated graph artifacts, or cache directories. Keep provider details in `config.py` or `summarization/llm_models.json`, and document new required environment variables in the PR.
