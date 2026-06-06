"""Plot and compare token-attribution methods for a single prompt + model.

Computes SHAP (forced-target log-odds) and the ``integrated_gradients`` seed from
``summarization.token_attribution`` for one prompt, then saves a stacked bar chart
(one row per method) so the per-token weights can be compared side by side. Every
series is the ``normalized`` weight vector (each sums to 1), and all methods share the
same target token, so the rows are directly comparable as distributions over input tokens.

Examples
--------
    python plot_token_attribution.py --model Qwen/Qwen3-4B \
        --question "The capital of the country containing the Great Wall is the city of?"

    python plot_token_attribution.py --model Qwen/Qwen3-4B --no-shap \
        --prompt "<|im_start|>user\nThe capital of France is<|im_end|>\n<|im_start|>assistant\n"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import torch

from summarization.token_attribution import (
    NormalizeMethod,
    _cached_model,
    _cached_tokenizer,
    get_integrated_gradients_token_attribution,
    get_token_attribution,
)


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _prompt_tokens(tokenizer, prompt: str) -> list[str]:
    """Per-token surface strings aligned 1:1 with ``tokenizer(prompt)`` (special tokens kept)."""
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    return [tokenizer.decode([tid]) for tid in ids]


def _greedy_target(model, tokenizer, prompt: str) -> int:
    ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
    ids = ids.to(model.device)
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0, -1]
    return int(logits.argmax().item())


def compute_attributions(
    prompt: str,
    model_name: str,
    *,
    device: str,
    target_token_id: int | None,
    include_shap: bool,
    shap_normalize: NormalizeMethod,
    ig_steps: int,
) -> tuple[list[str], int, dict[str, torch.Tensor]]:
    """Run every method on the same prompt and target; returns (tokens, target_id, weights)."""
    tokenizer = _cached_tokenizer(model_name)
    tokens = _prompt_tokens(tokenizer, prompt)
    if target_token_id is None:
        target_token_id = _greedy_target(_cached_model(model_name, device), tokenizer, prompt)

    results: dict[str, torch.Tensor] = {}
    if include_shap:
        _raw, norm = get_token_attribution(
            prompt=prompt,
            prompt_tokens=tokens,
            model_name=model_name,
            normalize_method=shap_normalize,
            device=device,
            pin_special_tokens=True,
            target_token_id=target_token_id,
        )
        results[f"shap ({shap_normalize})"] = norm
    _raw, norm = get_integrated_gradients_token_attribution(
        prompt=prompt,
        prompt_tokens=tokens,
        model_name=model_name,
        target_token_id=target_token_id,
        device=device,
        ig_steps=ig_steps,
    )
    results["integrated_gradients"] = norm
    return tokens, target_token_id, results


def plot_attributions(
    tokens: list[str],
    results: dict[str, torch.Tensor],
    target_text: str,
    out_path: Path,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt

    methods = list(results)
    x = list(range(len(tokens)))
    labels = [f"{i}:{tok}".replace("\n", "\\n") for i, tok in enumerate(tokens)]

    fig, axes = plt.subplots(
        len(methods), 1, figsize=(max(10, 0.42 * len(tokens)), 2.0 * len(methods) + 1), sharex=True
    )
    axes = axes if len(methods) > 1 else [axes]
    for ax, method in zip(axes, methods):
        ax.bar(x, results[method].numpy())
        ax.set_ylabel(method, fontsize=9)
        ax.margins(x=0.005)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    fig.suptitle(f"Token attribution (normalized weights) — target {target_text!r}", fontsize=11)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved {out_path}")
    if show:
        plt.show()


def _print_top(tokens: list[str], results: dict[str, torch.Tensor], k: int = 6) -> None:
    for method, weights in results.items():
        order = sorted(range(len(tokens)), key=lambda i: -float(weights[i]))[:k]
        top = ", ".join(f"{i}:{tokens[i]!r}={float(weights[i]):.3f}" for i in order)
        print(f"  {method:22s} {top}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot all token-attribution methods for a prompt.")
    parser.add_argument("--model", default="Qwen/Qwen3-4B", help="HF model id (tokenizer + LM).")
    parser.add_argument("--prompt", help="Literal prompt string fed to the model.")
    parser.add_argument(
        "--question", help="User message; wrapped with the Qwen chat template (non-thinking)."
    )
    parser.add_argument(
        "--system", default="Answer in one word and no more", help="System message for --question."
    )
    parser.add_argument(
        "--target-token-id",
        type=int,
        default=None,
        help="Target token id; default = greedy argmax.",
    )
    parser.add_argument("--no-shap", action="store_true", help="Skip the (slow) SHAP method.")
    parser.add_argument("--shap-normalize", default="softmax", help="SHAP normalize method.")
    parser.add_argument("--ig-steps", type=int, default=32, help="Integrated-gradients steps.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output", type=Path, default=Path("figures") / "token_attribution.png")
    parser.add_argument("--show", action="store_true", help="Display the figure interactively.")
    args = parser.parse_args()

    if args.question:
        from attribute_utils import format_qwen

        prompt = format_qwen(
            [
                {"role": "system", "content": args.system},
                {"role": "user", "content": args.question},
            ],
            add_generation_prompt=True,
            enable_thinking=False,
        )
    elif args.prompt:
        prompt = args.prompt
    else:
        raise SystemExit("Provide --prompt or --question.")

    device = _resolve_device(args.device)
    tokens, target_id, results = compute_attributions(
        prompt,
        args.model,
        device=device,
        target_token_id=args.target_token_id,
        include_shap=not args.no_shap,
        shap_normalize=args.shap_normalize,  # type: ignore[arg-type]
        ig_steps=args.ig_steps,
    )

    target_text = _cached_tokenizer(args.model).decode([target_id])
    print(f"target token: {target_id} = {target_text!r}")
    print("top tokens per method:")
    _print_top(tokens, results)

    if not args.show:
        matplotlib.use("Agg")
    plot_attributions(tokens, results, target_text, args.output, args.show)


if __name__ == "__main__":
    main()
