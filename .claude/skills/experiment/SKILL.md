---
name: experiment
description: Launch a long-running experiment (clustering / steering / intervention / pruning sweep over graphs) as a tracked background job with GPU checks, logging, and output verification. Use when asked to run an experiment or eval that is slow or GPU-bound.
---

# Run a long experiment safely

Experiments over the graph set (clustering, steering, intervention, pruning
sweeps) are long and GPU-bound. Launch them so they survive, log, and can be
inspected — never fire-and-forget. Background jobs here have been killed or
produced no output before, causing redundant re-runs.

## Before launching

1. Check for a free GPU:
   ```bash
   nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
   ```
   If every GPU is busy with other users' processes, say so and stop — don't contend.
2. Check whether result artifacts for this config already exist. If so, ask
   before re-running rather than silently overwriting.

## Launch

Run as a tracked background job, unbuffered, writing to a unique log, and record
the PID:
```bash
mkdir -p runs
ts=$(date +%s)
CUDA_VISIBLE_DEVICES=<free_gpu> nohup conda run --no-capture-output -n circuit \
  python -u <script> <args> > runs/<exp>_$ts.log 2>&1 &
echo $! > runs/<exp>_$ts.pid
```
`--no-capture-output` + `python -u` keep the log live so you can watch progress.

## Verify it started

Tail the log and confirm real progress (model loaded, first iteration logged)
before moving on. Don't report success until you've seen actual output, not just
a PID.

## On completion

Confirm the run produced non-empty results (inspect the artifact and the log
tail). If the job was killed or the output is empty, report it as failed — never
present partial or missing numbers as results.
