# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

Python library for mechanistic interpretability of transformer language models (circuit tracing). Single Python package with a Streamlit web UI. See `README.md` and `CONTRIBUTING.md` for full details.

### Development commands

All commands use `python3` (not `python`) in this environment.

| Task | Command |
|------|---------|
| Install deps | `pip install -e ".[dev,viz]" && pip install -r requirements.txt` |
| Lint | `python3 -m ruff check` |
| Format check | `python3 -m ruff format --check` |
| Type check | `python3 -m pyright` |
| Tests (CI-match) | `python3 -m pytest tests -m "not requires_disk"` |
| Run Streamlit app | `python3 -m streamlit run app.py --server.headless true --server.port 8501` |

### Caveats

- **No GPU in Cloud Agent VMs**: 84 tests are skipped because they require CUDA GPU (many need ≥32GB VRAM). The 144 CPU-only tests all pass and match CI behavior.
- **`requirements.txt` supplements `pyproject.toml`**: Packages like `shap`, `google-genai`, `entmax`, and `python-dotenv` are only in `requirements.txt`, not in `pyproject.toml`. Both must be installed.
- **Pre-existing lint/type issues**: The repo has pre-existing ruff and pyright errors (not caused by setup). `ruff format --check` and `ruff check` exit non-zero; `pyright` reports ~864 errors. This is the repo's current state.
- **API keys**: Full E2E features (Neuronpedia upload, LLM labeling) require `.env` with `NEURONPEDIA_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `HUGGINGFACE_API_KEY`. These are optional for development and testing.
- **Streamlit frontend server**: The app auto-starts a local HTTP server on port 8032 for graph visualization when graphs are loaded.
