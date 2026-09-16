# Research code

- Prefer flat, explicit functions. Eval scripts may duplicate code; introduce shared abstractions only after at least three concrete uses exist.
- Catch specific expected exceptions; avoid broad `try/except` blocks and fallback branches for impossible cases.
- Give intermediate tensors descriptive names and add a shape comment such as `# (N, d)` at their first non-trivial appearance.
- Validate CLI arguments, configuration, `.pt` files, and LLM responses at their input boundaries. Avoid repeating shape checks inside the pipeline.
- [pyproject.toml](../../pyproject.toml) is authoritative for Python compatibility, Ruff formatting and import rules, and Pyright settings.

## Seed policy

- Production stochastic operations take an explicit seed parameter, such as `random_state`; do not hardcode production seed values.
- Fixed seeds are allowed in tests. `tests/conftest.py` sets `torch.manual_seed(42)`.
