"""Compare SHAP vs integrated-gradients token attribution on the last N analogies.

For each line ``The saying goes: X is to Y as Z is to W`` we feed the model
``<bos>The saying goes: X is to Y as Z is to`` (final answer word W dropped, matching
the ``dataset/analogies`` graph convention) and force the target to W's token, then save
a per-prompt stacked bar chart comparing SHAP and IG.

    python compare_ig_shap_analogies.py            # last 10, gemma-2-2b, cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

from plot_token_attribution import compute_attributions, plot_attributions
from summarization.token_attribution import _cached_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analogies-file", type=Path, default=Path("analogies.txt"))
    parser.add_argument("--model", default="google/gemma-2-2b")
    parser.add_argument("--n", type=int, default=10, help="Number of trailing prompts.")
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--shap-normalize", default="softmax")
    parser.add_argument("--ig-steps", type=int, default=32)
    parser.add_argument("--out-dir", type=Path, default=Path("figures") / "analogies_ig_shap")
    args = parser.parse_args()

    tokenizer = _cached_tokenizer(args.model)
    # Our prompts carry a literal "<bos>"; disable auto-BOS so gemma doesn't double-BOS
    # (also keeps SHAP's internal teacher-forced target a single token).
    tokenizer.add_bos_token = False
    lines = args.analogies_file.read_text(encoding="utf-8").strip().splitlines()
    selected = list(enumerate(lines))[-args.n :]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for idx, line in selected:
        body, answer = line.rsplit(" ", 1)  # drop final answer word W
        prompt = f"<bos>{body}"
        # Force the target to W's first token (all answers are single tokens here).
        target_id = tokenizer(" " + answer, add_special_tokens=False)["input_ids"][0]

        tokens, target_id, results = compute_attributions(
            prompt,
            args.model,
            device=args.device,
            target_token_id=target_id,
            include_shap=True,
            shap_normalize=args.shap_normalize,  # type: ignore[arg-type]
            ig_steps=args.ig_steps,
        )
        target_text = tokenizer.decode([target_id])
        print(f"\n[{idx:02d}] {line!r}  target={target_text!r}")
        for method, weights in results.items():
            order = sorted(range(len(tokens)), key=lambda i: -float(weights[i]))[:5]
            top = ", ".join(f"{tokens[i]!r}={float(weights[i]):.3f}" for i in order)
            print(f"     {method:18s} {top}")

        out_path = args.out_dir / f"{idx:03d}.png"
        plot_attributions(tokens, results, target_text, out_path, show=False)


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    main()
