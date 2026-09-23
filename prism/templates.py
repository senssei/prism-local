"""
prism.templates: Chat prompt templates and per-model template detection.
Kept free of heavy imports so the catalog can use it without loading ONNX Runtime.
"""

import datetime
import functools
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("prism.templates")

DEFAULT_TEMPLATE = "phi4"

# Where a Hugging Face style model folder keeps its Jinja chat template, most specific first.
_TEMPLATE_FILES = ("chat_template.jinja", "chat_template.json", "tokenizer_config.json")
_MAX_TEMPLATE_FILE_BYTES = 4 * 1024 * 1024


def detect_template(name: str = "", model_type: str = "") -> str:
    """Picks a chat template from the model name and genai_config `model.type`."""
    hint = f"{name} {model_type}".lower()
    if "gemma" in hint:
        return "gemma"
    if "mistral" in hint or "mixtral" in hint:
        return "mistral_v02" if re.search(r"v0[._-]?[12]\b", hint) else "mistral"
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
    if "<start_of_turn>" in t:
        return "gemma"
    if "[INST]" in t and "<<SYS>>" not in t:  # not Llama 2, which has its own <<SYS>> block
        # v0.1/v0.2 put a space before [/INST] and none after it; v0.3 and later the reverse
        return "mistral_v02" if " [/INST]" in t else "mistral"
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


# --------------------------------------------------------------------------- rendering the model's own Jinja template

TEMPLATE_MODES = ("auto", "jinja", "builtin")


def template_mode() -> str:
    """$PRISM_TEMPLATE: `auto` (render the model's Jinja chat template when jinja2 is installed), `jinja` (same, but warn when
    it cannot) or `builtin` (only Prism's own formats). Default `auto`."""
    value = os.environ.get("PRISM_TEMPLATE", "auto").strip().lower() or "auto"
    if value not in TEMPLATE_MODES:
        raise ValueError(f"PRISM_TEMPLATE must be one of {', '.join(TEMPLATE_MODES)} (got '{value}')")
    return value


def jinja_available() -> bool:
    try:
        import jinja2  # noqa: F401
    except ImportError:
        return False
    return True


class TemplateError(ValueError):
    """A chat template rejected the conversation (its own `raise_exception`, or a construct the sandbox forbids)."""


class TemplateToolRenderError(TemplateError):
    """A chat template raised while being rendered with a non-empty `tools` argument. Prism does not silently
    fall back to the built-in format in this case, because that would drop the caller's tool definitions and
    hand the model a prompt without them — a silent change in behaviour. The HTTP layer surfaces this as
    `400 template_render_failed`."""


@functools.lru_cache(maxsize=1)
def _jinja_env():
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise TemplateError(message)

    # Chat templates come from downloaded files, so they run in the sandbox: no attribute tricks, no mutation of the inputs.
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: datetime.datetime.now().strftime(fmt)
    env.filters["tojson"] = lambda value, ensure_ascii=False, indent=None, **_: json.dumps(
        value, ensure_ascii=ensure_ascii, indent=indent)  # as Hugging Face does: not Jinja's HTML-escaping tojson
    return env


@functools.lru_cache(maxsize=32)
def _compile_template(text: str):
    return _jinja_env().from_string(text)


@functools.lru_cache(maxsize=256)
def _read_tokenizer_tokens(path: str, mtime_ns: int, size: int) -> Tuple[str, str, bool]:
    """(bos_token, eos_token, add_bos_token) from tokenizer_config.json. `mtime_ns` and `size` only key the cache."""
    try:
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        return "", "", False
    if not isinstance(cfg, dict):
        return "", "", False

    def token(value):  # a string, or {"content": "<s>", ...}
        if isinstance(value, dict):
            value = value.get("content")
        return value if isinstance(value, str) else ""

    return token(cfg.get("bos_token")), token(cfg.get("eos_token")), cfg.get("add_bos_token") is True


def read_tokenizer_tokens(model_dir: str) -> Tuple[str, str, bool]:
    path = os.path.join(model_dir, "tokenizer_config.json")
    try:
        st = os.stat(path)
    except OSError:
        return "", "", False
    return _read_tokenizer_tokens(path, st.st_mtime_ns, st.st_size)


def render_chat_template(text: str, messages: List[Dict[str, Any]], *, bos_token: str = "", eos_token: str = "",
                         tools: Optional[List[Dict[str, Any]]] = None, add_generation_prompt: bool = True) -> str:
    """Renders a Hugging Face style Jinja chat template. Raises TemplateError (or a jinja2 error) when the template cannot take the messages."""
    return _compile_template(text).render(messages=messages, tools=tools, add_generation_prompt=add_generation_prompt,
                                          bos_token=bos_token, eos_token=eos_token)


def render_prompt(resolved: Dict[str, Any], messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> str:
    """The prompt for `messages` on the model `resolved` (a catalog entry): its own Jinja chat template when it has one and jinja2
    is installed, else Prism's built-in format for the detected family.

    When `tools is None`, a template that fails (its own `raise_exception`, a forbidden sandbox construct, etc.)
    is logged and replaced by the built-in format — invariant I7. The built-in format renders the same
    conversation without the template's peculiarities (it folds a system prompt into a user turn where the
    template refuses one).

    When `tools is not None`, a template that fails raises `TemplateToolRenderError` instead of falling back.
    The built-in format has no `tools` argument and would silently drop the caller's tool definitions, which is
    a silent change in what the model sees. The HTTP layer converts the exception to `400 template_render_failed`
    (spec.md P11)."""
    builtin = lambda: format_prompt(messages, resolved.get("template"))  # noqa: E731
    mode = template_mode()
    path = resolved.get("path")
    if mode == "builtin" or not path:
        return builtin()
    text = read_chat_template(path)
    if not text:
        return builtin()
    if not jinja_available():
        if mode == "jinja":
            logger.warning("PRISM_TEMPLATE=jinja needs jinja2 (pip install 'prism-local[jinja]'); using the built-in template")
        return builtin()
    bos, eos, add_bos = read_tokenizer_tokens(path)
    try:
        out = render_chat_template(text, [{**m, "content": flatten_content(m.get("content"))} for m in messages],
                                   bos_token=bos, eos_token=eos, tools=tools)
    except Exception as ex:
        # When the caller passed tools, do NOT fall back to the built-in format — it has no `tools` argument and
        # would silently drop the tool definitions from the prompt. The server surfaces this as 400
        # `template_render_failed` (spec.md P11). The no-tools case keeps the silent fallback (I7 carve-out: the
        # built-in format renders the same conversation, so the caller has no idea anything changed).
        if tools:
            raise TemplateToolRenderError(
                f"chat template of {resolved.get('id')!r} failed to render with tools: {ex}") from ex
        logger.warning("chat template of %s failed (%s); using the built-in '%s' format", resolved.get("id"), ex,
                       resolved.get("template"))
        return builtin()
    if add_bos and bos and out.startswith(bos):
        out = out[len(bos):]  # the tokenizer adds BOS itself; a second one would confuse the model
    return out


def supports_tools(resolved: Dict[str, Any]) -> bool:
    """Whether Prism can put tool definitions into this model's prompts: it renders the model's own Jinja template (the `jinja` extra),
    and that template uses `tools`. The check is structural on the template source — `tools` must appear inside a Jinja
    expression (`{{ ... tools ... }}`) or inside a Jinja block (`{% ... tools ... %}`, including `{% if tools ... %}` and
    `{% for t in tools ... %}`). Prose mentions in `{# ... #}` Jinja comments or in non-Jinja text do not count, so a template
    that documents `tools` without rendering them no longer falsely reports support (Phase 10 / P14)."""
    if template_mode() == "builtin" or not jinja_available():
        return False
    text = read_chat_template(resolved.get("path") or "")
    # Structural Jinja usage: `tools` must appear inside a Jinja expression or block, with a word boundary so
    # `tool_registry`, `tools_dict`, etc. don't match. Comment blocks (`{# ... #}`) are excluded because their
    # delimiters are `{#` and `#}`, not `{%` / `%}`.
    return bool(text) and re.search(
        r"\{\{[^}]*\btools\b[^}]*\}\}"
        r"|\{\%[^%]*\btools\b[^%]*%\}",
        text,
    ) is not None


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
    if template == "gemma":
        # Gemma has no system role: the system prompt leads the first user turn. BOS is left to the tokenizer.
        out, pending = "", system_msg
        for role, content in turns:
            role = "model" if role == "assistant" else "user"
            content = content.strip()
            if pending and role == "user":
                content, pending = f"{pending}\n\n{content}", ""
            out += f"<start_of_turn>{role}\n{content}<end_of_turn>\n"
        return out + "<start_of_turn>model\n"
    if template in ("mistral", "mistral_v02"):
        # No system role either: it leads the last user message. BOS (<s>) is left to the tokenizer.
        v02 = template == "mistral_v02"
        last_user = max((i for i, (role, _) in enumerate(turns) if role != "assistant"), default=-1)
        out = ""
        for i, (role, content) in enumerate(turns):
            if role == "assistant":
                out += f"{content}</s>" if v02 else f" {content}</s>"
            else:
                if system_msg and i == last_user:
                    content = f"{system_msg}\n\n{content}"
                out += f"[INST] {content} [/INST]" if v02 else f"[INST] {content}[/INST]"
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
