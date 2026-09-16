# Research workflow

- Treat source datasets and input graph artifacts as read-only. Write derived graphs and evaluation results separately.
- Use each evaluation command's documented output options and existing formats, including CSV and JSON. Keep its directory layout rather than imposing a new shared output hierarchy.
- For stochastic code and experiments, follow the [seed policy](research-code.md#seed-policy).
- Before launching a long-running or GPU-bound experiment, read the [experiment skill](../../.agents/skills/experiment/SKILL.md). It specifies tracked jobs, unique logs under `runs/`, and output verification.
- When recording a new run, include its parameters, seed, input references, and Git commit hash in the existing manifest or an accompanying JSON record in the run's output directory. Record whether the working tree has uncommitted changes. Check the artifacts: do not assume every script already emits this metadata.
- Report evaluation results only from verified, non-empty outputs; identify failed or incomplete runs explicitly.
