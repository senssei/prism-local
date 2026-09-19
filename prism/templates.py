"""
prism.templates: Chat prompt templates and per-model template detection.
Kept free of heavy imports so the catalog can use it without loading ONNX Runtime.
"""

from typing import Dict, List, Optional, Tuple

DEFAULT_TEMPLATE = "phi4"


def detect_template(name: str = "", model_type: str = "") -> str:
    """Picks a chat template from the model name and genai_config `model.type`."""
    hint = f"{name} {model_type}".lower()
    if "phi" in hint:
        return "phi4"
    if "deepseek" in hint:
        return "deepseek"
    if "llama" in hint:
        return "llama3"
    # Qwen and most other instruct models are ChatML.
    return "chatml"


def format_prompt(messages: List[Dict[str, str]], template: Optional[str] = None) -> str:
    """Formats OpenAI-style messages into the prompt format of the given template."""
    template = template or DEFAULT_TEMPLATE
    system_parts: List[str] = []
    turns: List[Tuple[str, str]] = []

    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        if role == "system":
            system_parts.append(content)
        else:
            turns.append((role, content))
    system_msg = "\n\n".join(system_parts)

    if template in ("phi4", "phi3"):
        out = ""
        if system_msg:
            out += f"<|system|>\n{system_msg}<|end|>\n"
        for role, content in turns:
            out += f"<|{role}|>\n{content}<|end|>\n"
        out += "<|assistant|>\n"
        return out
    if template == "chatml":
        out = ""
        if system_msg:
            out += f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        for role, content in turns:
            out += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        out += "<|im_start|>assistant\n"
        return out
    if template == "llama3":
        # No <|begin_of_text|>: the tokenizer adds BOS itself.
        out = ""
        if system_msg:
            out += f"<|start_header_id|>system<|end_header_id|>\n\n{system_msg}<|eot_id|>"
        for role, content in turns:
            out += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        out += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return out
    if template == "deepseek":
        out = system_msg
        for role, content in turns:
            if role == "assistant":
                out += f"<｜Assistant｜>{content}<｜end▁of▁sentence｜>"
            else:
                out += f"<｜User｜>{content}"
        out += "<｜Assistant｜>"
        return out

    out = f"System: {system_msg}\n\n" if system_msg else ""
    for role, content in turns:
        out += f"{role.capitalize()}: {content}\n"
    out += "Assistant: "
    return out
