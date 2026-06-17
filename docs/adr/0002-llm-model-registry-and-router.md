# 2. LLM access goes through a model registry + provider router

Status: accepted

## Context

The labeling code must work with an unknown, changing set of models: official OpenAI, Google,
and self-hosted / third-party OpenAI-compatible endpoints. The previous code guessed the
backend from the model-name prefix (`gpt-`/`o*` → OpenAI, else Gemini). That heuristic cannot
disambiguate OpenAI-compatible endpoints, whose model names are arbitrary, and it scattered
credentials and endpoint logic through the call path.

## Decision

A committed JSON **model registry** (`summarization/llm_models.json`, overridable via
`LLM_MODELS_PATH`) maps each model name to its `provider`, `base_url`, `api_key_env` (the *name*
of the env var, resolved via `config.get_env` — never the literal secret), an optional wire
`model` id, and a `defaults` block (`temperature`, `thinking_effort`).

A router resolves the entry by model name and dispatches to one of three providers
(`openai`, `google`, `openai_compat`), owning retry/backoff internally. Generation settings
resolve as: call-site `settings` → registry `defaults` → hardcoded fallback. An unknown model
name raises.

## Consequences

- Adding or repointing a model is a JSON edit, not a code change.
- `thinking_effort` is a unified enum (`low`/`medium`/`high`/`None`) translated per provider
  (OpenAI `reasoning_effort`; Google Gemini 2.5 `thinking_budget` token count; Google Gemma 4
  `thinking_level="high"` only when `thinking_effort` is `high`).
- Secrets stay in `.env`; only env-var names live in the committed registry.
- Callers must register a model before using it — there is no silent prefix-based fallback.
