"""
prism.templates: Chat prompt templates and per-model template detection.
Kept free of heavy imports so the catalog can use it without loading ONNX Runtime.
"""

import functools
import json
import os
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_TEMPLATE = "phi4"

# Where a Hugging Face style model folder keeps its Jinja chat template, most specific first.
_TEMPLATE_FILES = ("chat_template.jinja", "chat_template.json", "tokenizer_config.json")
_MAX_TEMPLATE_FILE_BYTES = 4 * 1024 * 1024


def detect_template(name: str = "", model_type: str = "") -> str:
    """Picks a chat template from the model name and genai_config `model.type`."""
    hint = f"{name} {model_type}".lower()
    if any(tag in hint for tag in ("phi-4", "phi4", "phi_4")):
        return "phi4_mini" if "mini" in hint else "phi4_im"
    if "phi" in hint:
        return "phi4"
    if "deepseek" in hint:
        return "deepseek"
    if "llama" in hint:
        return "llama3"
    # Qwen and most other instruct models are ChatML.
    return "chatml"


@functools.lru_cache(maxsize=256)
def _read_template_file(path: str, mtime_ns: int, size: int) -> str:
    """Reads one file's chat template. `mtime_ns` and `size` only key the cache, so an edited file is read again."""
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except (OSError, UnicodeDecodeError):
        return ""
    if path.endswith(".jinja"):
        return text
    try:
        value = json.loads(text).get("chat_template")
    except (ValueError, AttributeError):
        return ""
    if isinstance(value, list):  # several named templates: the default one is what plain chat uses
        named = {t.get("name"): t.get("template") for t in value if isinstance(t, dict)}
        value = named.get("default") or next(iter(named.values()), "")
    return value if isinstance(value, str) else ""


def read_chat_template(model_dir: str) -> str:
    """The Jinja chat template shipped with a model (`chat_template.jinja`, `chat_template.json`, or the `chat_template` key of
    `tokenizer_config.json`), or "" when it has none. Never raises."""
    for filename in _TEMPLATE_FILES:
        path = os.path.join(model_dir, filename)
        try:
            st = os.stat(path)
        except OSError:
            continue
        if st.st_size > _MAX_TEMPLATE_FILE_BYTES:
            continue
        text = _read_template_file(path, st.st_mtime_ns, st.st_size)
        if text:
            return text
    return ""


def classify_chat_template(chat_template: str) -> Optional[str]:
    """Which of Prism's templates a Jinja chat template produces, judged by the special tokens it writes; None when it is
    empty or none match. The template is what the model was trained with, so it outranks the model's name."""
    t = chat_template or ""
    if "<｜Assistant｜>" in t or "<｜User｜>" in t:
        return "deepseek"
    if "<|start_header_id|>" in t:
        return "llama3"
    if "<|im_sep|>" in t:  # Phi-4: <|im_start|>role<|im_sep|>..., which is not ChatML
        return "phi4_im"
    if "<|im_start|>" in t:
        return "chatml"
    if "<|end|>" in t:
        # Phi-4-mini builds every tag from the role and writes no newlines; Phi-3 and Phi-3.5 spell them out with newlines.
        return "phi4_mini" if "message['role'] + '|>'" in t else "phi4"
    return None


def resolve_template(name: str, model_type: str, model_dir: str) -> Tuple[str, str]:
    """(template, source): the model's own chat template when it has one Prism knows, else a guess from its name."""
    found = classify_chat_template(read_chat_template(model_dir))
    if found:
        return found, "chat_template"
    return detect_template(name, model_type), "name"


def flatten_content(content: Any) -> str:
    """Message `content` as text. OpenAI clients may send a list of parts (`{"type": "text", "text": ...}`); non-text parts
    (images, ...) are dropped, since these models are text-only. `None` (e.g. an assistant turn with tool calls) is empty."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, str):
                texts.append(part)
            elif isinstance(part, dict) and part.get("type", "text") == "text" and isinstance(part.get("text"), str):
                texts.append(part["text"])
        return "\n".join(texts)
    return str(content)


def format_prompt(messages: List[Dict[str, str]], template: Optional[str] = None) -> str:
    """Formats OpenAI-style messages into the prompt format of the given template."""
    template = template or DEFAULT_TEMPLATE
    system_parts: List[str] = []
    turns: List[Tuple[str, str]] = []

    for m in messages:
        role = m.get("role", "user")
        content = flatten_content(m.get("content", ""))
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
    if template == "phi4_mini":
        out = "".join(f"<|{role}|>{content}<|end|>" for role, content in (("system", system_msg),) if content)
        for role, content in turns:
            out += f"<|{role}|>{content}<|end|>"
        return out + "<|assistant|>"
    if template == "phi4_im":
        # Phi-4: <|im_start|>role<|im_sep|>content<|im_end|>, with no newlines.
        out = f"<|im_start|>system<|im_sep|>{system_msg}<|im_end|>" if system_msg else ""
        for role, content in turns:
            out += f"<|im_start|>{role}<|im_sep|>{content}<|im_end|>"
        out += "<|im_start|>assistant<|im_sep|>"
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
