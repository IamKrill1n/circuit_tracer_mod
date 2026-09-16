# Git and provider configuration

- Write short imperative commit messages, such as `add pruning eval` or `fix visualization`.
- PR descriptions list verification commands and any skipped heavy tests.
- Secrets come from `.env`, loaded by `config.py`: `OPENAI_API_KEY`, `GEMINI_API_KEY`/`GENAI_API_KEY`, `HUGGINGFACE_API_KEY`, and `NEURONPEDIA_API_KEY`.
- Never commit `.env`, API keys, or `temp_graph_files/` artifacts.
- When adding a provider, update `summarization/llm_models.json` and document its environment variable in the PR.
