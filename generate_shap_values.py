"""
Validate prompts and generate SHAP token attributions for those that pass.

Each line in the prompt file: "<prefix> <target_word>"
Keeps prompts where model top-1 == target and confidence is in (LOWER, UPPER).
For each kept prompt, runs SHAP token attribution on the prefix and saves to JSON.

Usage:
  conda run -n circuit python generate_shap_values.py prompts.txt
  conda run -n circuit python generate_shap_values.py prompts.txt --output-file shap_values.json
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import torch

from summarization.token_attribution import get_token_attribution

DEFAULT_MODEL_NAME = "google/gemma-2-2b"
LOWER, UPPER = 0.2, 1


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


def _first_token_id(tokenizer, word: str) -> int:
    # Leading space encodes mid-sentence word boundary.
    ids = tokenizer.encode(" " + word, add_special_tokens=False)
    return ids[0]


def _validate(
    pairs: list[tuple[str, str]], model_name: str
) -> list[tuple[str, str]]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    kept: list[tuple[str, str]] = []
    for prefix, target in pairs:
        target_tid = _first_token_id(tokenizer, target)
        inputs = tokenizer(prefix, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            logits = model(**inputs).logits[0, -1]  # (vocab,)
        probs = logits.softmax(-1)
        top1_id = int(logits.argmax())
        p = probs[target_tid].item()

        top1_matches = top1_id == target_tid
        in_range = LOWER < p < UPPER

        if top1_matches and in_range:
            print(f"[KEEP] {prefix!r} → {target!r}  [p={p:.3f}]")
            kept.append((prefix, target))
        else:
            reasons = []
            if not top1_matches:
                reasons.append(f"top1={tokenizer.decode([top1_id]).strip()!r}")
            if not in_range:
                reasons.append(f"p={p:.3f}")
            print(f"[DROP] {prefix!r} → {target!r}  [{'; '.join(reasons)}]")

    del model  # free GPU memory before SHAP
    return kept


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
    args = parser.parse_args()

    device = _resolve_device(args.device)
    pairs = _load_prompts(args.prompt_file)
    if args.limit:
        pairs = pairs[: args.limit]
    if not pairs:
        raise ValueError(f"No prompts found in {args.prompt_file}")

    kept = _validate(pairs, args.model_name)
    print(f"\n{len(kept)}/{len(pairs)} prompts passed validation\n")

    results: list[dict] = []
    for idx, (prefix, _target) in enumerate(kept, start=1):
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
        print(f"[{idx}/{len(kept)}] computed shap for prompt")

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


if __name__ == "__main__":
    main()
