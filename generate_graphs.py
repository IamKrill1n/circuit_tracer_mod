"""
Generate attribution graphs for each prompt in a prompt file.

Each line in the prompt file: "<prefix> <target_word>"
For each prompt, runs attribute() on the prefix and saves the graph to <output_dir>/<idx>.pt.
If --hf-repo is given, each .pt is uploaded to `graphs/<idx>.pt` in that dataset repo.

Usage:
  conda run -n circuit python generate_graphs.py prompts.txt
  conda run -n circuit python generate_graphs.py prompts.txt --output-dir graphs/run1
  conda run -n circuit python generate_graphs.py prompts.txt --hf-repo user/dataset-name
"""

import argparse
import os
from pathlib import Path

from circuit_tracer.utils.create_graph_files import create_graph_files
import torch
from huggingface_hub import HfApi

from circuit_tracer import ReplacementModel, attribute
from circuit_tracer.utils.demo_utils import cleanup_cuda
from config import HUGGINGFACE_API_KEY

MODEL_NAME = "google/gemma-2-2b"
TRANSCODER_NAME = "mntss/clt-gemma-2-2b-2.5M"


def load_prompts(path: Path) -> list[tuple[str, str]]:
    pairs = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        prefix, _, target = line.rpartition(" ")
        pairs.append((prefix, target))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("graphs"))
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--transcoder", default=TRANSCODER_NAME)
    parser.add_argument("--backend", default="transformerlens")
    parser.add_argument("--max-n-logits", type=int, default=15)
    parser.add_argument("--desired-logit-prob", type=float, default=0.99)
    parser.add_argument("--hf-repo", default=None, help="HF dataset repo id; if set, uploads each .pt to graphs/<idx>.pt")
    args = parser.parse_args()

    api = None
    if args.hf_repo:
        api = HfApi(token=HUGGINGFACE_API_KEY or None)
        api.create_repo(args.hf_repo, repo_type="dataset", exist_ok=True)

    model = ReplacementModel.from_pretrained(
        args.model, args.transcoder, dtype=torch.bfloat16, lazy_encoder=True, backend=args.backend
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_prompts(args.prompt_file)

    for idx, (prefix, target) in enumerate(pairs):
        print(f"[{idx}] {prefix!r} → {target!r}")

        graph = attribute(
            prompt=prefix,
            model=model,
            max_n_logits=args.max_n_logits,
            desired_logit_prob=args.desired_logit_prob,
            batch_size=256,
            max_feature_nodes=8192,
            offload="cpu",
            verbose=False,
        )

        out = args.output_dir / f"{idx:03d}.pt"
        graph.to_pt(out)
        print(f"  saved → {out}")

        if api is not None:
            api.upload_file(
                path_or_fileobj=str(out),
                path_in_repo=f"graphs/{idx:03d}.pt",
                repo_id=args.hf_repo,
                repo_type="dataset",
            )
            print(f"  pushed → {args.hf_repo}:graphs/{idx:03d}.pt")

        cleanup_cuda()

    print(f"\n{len(pairs)} graphs generated → {args.output_dir}")


if __name__ == "__main__":
    main()
