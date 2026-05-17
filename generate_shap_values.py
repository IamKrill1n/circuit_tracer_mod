"""
Generate SHAP token attributions for each prompt in a prompt file.

Each line in the prompt file: "<prefix> <target_word>"
For each prompt, runs SHAP token attribution on the prefix and saves to JSON.
If --hf-repo is given, the JSON is also uploaded to that dataset repo.

Usage:
  conda run -n circuit python generate_shap_values.py prompts.txt
  conda run -n circuit python generate_shap_values.py prompts.txt --output-file shap_values.json
  conda run -n circuit python generate_shap_values.py prompts.txt --hf-repo user/dataset-name
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import torch
from huggingface_hub import HfApi

from summarization.token_attribution import get_token_attribution
from config import HUGGINGFACE_API_KEY

DEFAULT_MODEL_NAME = "google/gemma-2-2b"


def _resolve_device(flag: str) -> str:
    if flag == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if flag == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested --device cuda, but CUDA is not available.")
    return flag


def _load_prompts(path: Path) -> list[tuple[str, str]]:
    pairs = []
    for line in path.read_text().splitlines():
        text = line.strip()
        if not text:
            continue
        prefix, _, target = text.rpartition(" ")
        if prefix and target:
            pairs.append((prefix, target))
    return pairs


@lru_cache(maxsize=8)
def _tokenizer(model_name: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


def _prompt_tokens(prompt: str, model_name: str) -> list[str]:
    tok = _tokenizer(model_name)
    token_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    return [str(t) for t in tok.convert_ids_to_tokens(token_ids)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--output-file", type=Path, default=Path("demos") / "shap_values.json")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--masker-keep-prefix", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--hf-repo", default=None, help="HF dataset repo id; if set, uploads the output JSON")
    args = parser.parse_args()

    device = _resolve_device(args.device)
    pairs = _load_prompts(args.prompt_file)
    if args.limit:
        pairs = pairs[: args.limit]
    if not pairs:
        raise ValueError(f"No prompts found in {args.prompt_file}")

    results: list[dict] = []
    for idx, (prefix, _target) in enumerate(pairs, start=1):
        prompt_tokens = _prompt_tokens(prefix, args.model_name)
        raw, _normalized = get_token_attribution(
            prompt=prefix,
            prompt_tokens=prompt_tokens,
            model_name=args.model_name,
            device=device,
            masker_keep_prefix=args.masker_keep_prefix,
        )
        results.append(
            {
                "index": idx,
                "prompt": prefix,
                "prompt_tokens": prompt_tokens,
                "raw_shap": raw.detach().cpu().to(torch.float32).tolist(),
            }
        )
        print(f"[{idx}/{len(pairs)}] computed shap for prompt")

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": args.model_name,
        "device": device,
        "masker_keep_prefix": args.masker_keep_prefix,
        "n_prompts": len(results),
        "results": results,
    }
    args.output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\n[done] wrote {len(results)} prompt attributions to {args.output_file}")

    if args.hf_repo:
        api = HfApi(token=HUGGINGFACE_API_KEY or None)
        api.create_repo(args.hf_repo, repo_type="dataset", exist_ok=True)
        api.upload_file(
            path_or_fileobj=str(args.output_file),
            path_in_repo=args.output_file.name,
            repo_id=args.hf_repo,
            repo_type="dataset",
        )
        print(f"[done] pushed → {args.hf_repo}:{args.output_file.name}")


if __name__ == "__main__":
    main()
