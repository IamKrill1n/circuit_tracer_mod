from transformers import AutoTokenizer


def format_qwen(
    messages: list[dict],
    add_generation_prompt: bool = True,
    enable_thinking: bool = False,
) -> str:
    parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in messages]
    s = "\n".join(parts)
    if add_generation_prompt:
        s += "\n<|im_start|>assistant\n"
        if not enable_thinking:
            s += "<think>\n\n</think>\n\n"  # forces Qwen3 to skip the thinking trace
    return s


def format_qwen_with_tokenizer(
    messages: list[dict],
    model_name: str = "Qwen/Qwen3-4B",
    add_generation_prompt: bool = True,
    enable_thinking: bool = False,
) -> str:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )
