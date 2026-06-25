"""Generate a validated BATS analogy dataset: 10 prompts per relation, 100 total.

Each sentence: "The saying goes: A1 is to B1 as A2 is to B2", validated against
gemma-2-2b (top-1 prediction == target, p > 0.2). Uses disjoint pairs within a
relation to avoid entity reuse.
"""
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LOWER, UPPER = 0.2, 1.0
MODEL_NAME = "google/gemma-2-2b"
PER_RELATION = 10

BATS_DIR = Path("BATS_3.0")
FILES = [
    "E01 [country - capital].txt",
    "E02 [country - language].txt",
    "E03 [UK_city - county].txt",
    "E04 [name - nationality].txt",
    "E05 [name - occupation].txt",
    "E06 [animal - young].txt",
    "E07 [animal - sound].txt",
    "E08 [animal - shelter].txt",
    "E09 [things - color].txt",
    "E10 [male - female].txt",
]


def load_pairs(path: Path) -> list[tuple[str, str]]:
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        a, b = line.split("\t")
        pairs.append((a, b.split("/")[0]))  # first alternative
    return pairs


def first_token_id(tokenizer, word: str) -> int:
    return tokenizer.encode(" " + word, add_special_tokens=False)[0]


@torch.inference_mode()
def is_valid(model, tokenizer, prefix: str, target: str) -> bool:
    target_tid = first_token_id(tokenizer, target)
    inputs = tokenizer(prefix, return_tensors="pt").to(model.device)
    logits = model(**inputs).logits[0, -1]
    probs = logits.softmax(-1)
    top1_id = logits.argmax().item()
    p = probs[target_tid].item()
    return top1_id == target_tid and LOWER < p < UPPER


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    all_sentences = []

    for fname in FILES:
        relation = fname[:3]
        pairs = load_pairs(BATS_DIR / fname)
        kept = []
        used = set()  # indices of pairs already in a kept analogy

        def try_candidate(i: int, j: int) -> bool:
            a1, b1 = pairs[i]
            a2, b2 = pairs[j]
            prefix = f"The saying goes: {a1} is to {b1} as {a2} is to"
            full = f"{prefix} {b2}"
            ok = is_valid(model, tokenizer, prefix, b2)
            print(f"[{'KEEP' if ok else 'DROP'}] {relation}: {full}")
            if ok:
                kept.append(full)
                used.add(i)
                used.add(j)
            return ok

        # Pass 1: disjoint consecutive pairing (0,1), (2,3), ...
        for i in range(0, len(pairs) - 1, 2):
            if len(kept) >= PER_RELATION:
                break
            try_candidate(i, i + 1)

        # Pass 2: fallback — try (i, j) for any unused i,j to fill remaining slots
        if len(kept) < PER_RELATION:
            for i in range(len(pairs)):
                if len(kept) >= PER_RELATION:
                    break
                if i in used:
                    continue
                for j in range(i + 1, len(pairs)):
                    if len(kept) >= PER_RELATION:
                        break
                    if j in used:
                        continue
                    if try_candidate(i, j):
                        break  # i now used, move on

        print(f"  → {relation}: {len(kept)}/{PER_RELATION} kept\n")
        all_sentences.extend(kept)

    Path("bats_analogies.txt").write_text("\n".join(all_sentences) + "\n")
    print(f"\nTotal: {len(all_sentences)} prompts → bats_analogies.txt")


if __name__ == "__main__":
    main()
