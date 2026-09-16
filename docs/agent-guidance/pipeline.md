# Running the pipeline

Full attribution requires a GPU with CUDA:

```bash
python -m summarization --prompt "..." --model google/gemma-2-2b --transcoder mntss/clt-gemma-2-2b-2.5M
```

Use `--graph-pt` to load a saved graph instead of generating attribution, and `--graph-pt-out` to save a graph.
