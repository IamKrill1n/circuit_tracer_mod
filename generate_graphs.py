"""
Validate prompts and generate attribution graphs for those that pass.

Each line in the prompt file: "<prefix> <target_word>"
Keeps prompts where model top-1 == target and confidence is in (LOWER, UPPER).
For each kept prompt, runs attribute() and saves the graph to <output_dir>/<idx>.pt.

Usage:
  conda run -n circuit python generate_graphs.py prompts.txt
  conda run -n circuit python generate_graphs.py prompts.txt --output-dir graphs/run1
"""

import argparse
from pathlib import Path

from circuit_tracer.utils.create_graph_files import create_graph_files
import torch

from circuit_tracer import ReplacementModel, attribute
from circuit_tracer.utils.demo_utils import cleanup_cuda

MODEL_NAME = "google/gemma-2-2b"
TRANSCODER_NAME = "mntss/clt-gemma-2-2b-2.5M"
LOWER, UPPER = 0.2, 1


def load_prompts(path: Path) -> list[tuple[str, str]]:
    pairs = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        prefix, _, target = line.rpartition(" ")
        pairs.append((prefix, target))
    return pairs


def first_token_id(tokenizer, word: str) -> int:
    # Encode with a leading space so the token matches mid-sentence usage.
    ids = tokenizer.encode(" " + word, add_special_tokens=False)
    return ids[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt_file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("graphs"))
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--transcoder", default=TRANSCODER_NAME)
    parser.add_argument("--backend", default="transformerlens")
    args = parser.parse_args()

    model = ReplacementModel.from_pretrained(
        args.model, args.transcoder, dtype=torch.bfloat16, lazy_encoder=True, backend=args.backend
    )
    tokenizer = model.tokenizer
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_prompts(args.prompt_file)
    kept = 0

    for idx, (prefix, target) in enumerate(pairs):
        target_tid = first_token_id(tokenizer, target)
        input_ids = model.ensure_tokenized(prefix)

        with torch.no_grad():
            logits, _ = model.get_activations(input_ids)

        # logits: [batch, seq, vocab] or [seq, vocab] — take last token
        last_logits = logits.reshape(-1, logits.shape[-1])[-1]
        probs = last_logits.softmax(-1)
        top1_id = int(last_logits.argmax())
        p = probs[target_tid].item()

        top1_matches = top1_id == target_tid
        in_range = LOWER < p < UPPER

        if not (top1_matches and in_range):
            reasons = []
            if not top1_matches:
                reasons.append(f"top1={tokenizer.decode([top1_id]).strip()!r}")
            if not in_range:
                reasons.append(f"p={p:.3f}")
            print(f"[DROP] {prefix!r} → {target!r}  [{'; '.join(reasons)}]")
            continue

        print(f"[KEEP] {prefix!r} → {target!r}  [p={p:.3f}]")

        graph = attribute(
            prompt=prefix,
            model=model,
            max_n_logits=1,
            desired_logit_prob=0.99,
            batch_size=256,
            max_feature_nodes=8192,
            offload="cpu",
            verbose=True,
        )

        out = args.output_dir / f"{idx:03d}.pt"
        graph.to_pt(out)
        print(f"  saved → {out}")
        kept += 1
        cleanup_cuda()

    print(f"\n{kept}/{len(pairs)} graphs generated → {args.output_dir}")


if __name__ == "__main__":
    main()
