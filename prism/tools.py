"""
prism.tools: OpenAI-style tool calling on top of models that write their calls as text.

The model's chat template puts the tool definitions in the prompt; what comes back is text in the model family's own convention.
`parse_tool_calls` recognises the conventions of the common families from the output itself, so it does not need to know which
model produced it:

    <tool_call>{"name": ..., "arguments": {...}}</tool_call>        Qwen, Hermes and many fine-tunes
    <|tool_call|>[{"name": ..., "arguments": {...}}]<|/tool_call|>   Phi-4-mini
    [TOOL_CALLS] [{"name": ..., "arguments": {...}}]                 Mistral
    <|python_tag|>{"name": ..., "parameters": {...}}                 Llama 3.1 (also a bare JSON object as the whole reply)
"""

import json
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

# The special tokens that mark a tool call. A tokenizer may decode them to nothing (ONNX Runtime GenAI's does), which the engine undoes for these.
TOOL_MARKERS = frozenset({"<tool_call>", "</tool_call>", "<|tool_call|>", "<|/tool_call|>", "[TOOL_CALLS]", "<|python_tag|>"})

_HERMES = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.S)
# (marker that starts the calls, marker that ends them or None for "to the end of the text")
_MARKERS = (("<|tool_call|>", "<|/tool_call|>"), ("[TOOL_CALLS]", None), ("<|python_tag|>", None))


def _loads_prefix(text: str) -> Any:
    """The first JSON value in `text` (anything after it is ignored), or None."""
    try:
        return json.JSONDecoder().raw_decode(text.lstrip())[0]
    except ValueError:
        return None


def _normalise(obj: Any) -> Optional[List[Dict[str, Any]]]:
    """[{"name": str, "arguments": dict}] from one call or a list of calls in any of the spellings above, or None if anything does not look like a call."""
    items = obj if isinstance(obj, list) else [obj]
    calls = []
    for item in items:
        if isinstance(item, dict) and isinstance(item.get("function"), dict):  # OpenAI's own spelling
            item = item["function"]
        if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not item["name"]:
            return None
        args = item.get("arguments", item.get("parameters", {}))
        if isinstance(args, str):
            args = _loads_prefix(args) if args.strip() else {}
        if args is None:
            args = {}
        if not isinstance(args, dict):
            return None
        calls.append({"name": item["name"], "arguments": args})
    return calls or None


def parse_tool_calls(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Splits a model reply into (text, calls). `calls` is a list of {"name", "arguments" (a dict)}; when the reply holds no call that
    parses, it is returned unchanged with an empty list, so an ordinary answer that mentions a marker is never lost."""
    hermes = list(_HERMES.finditer(text))
    if hermes:
        calls: List[Dict[str, Any]] = []
        for m in hermes:
            parsed = _normalise(_loads_prefix(m.group(1)))
            if not parsed:
                return text, []
            calls.extend(parsed)
        return text[: hermes[0].start()].strip(), calls

    for start, end in _MARKERS:
        at = text.find(start)
        if at < 0:
            continue
        payload = text[at + len(start):]
        if end and end in payload:
            payload = payload[: payload.index(end)]
        parsed = _normalise(_loads_prefix(payload))
        if parsed:
            return text[:at].strip(), parsed
        return text, []

    stripped = text.strip()  # Llama 3.1 sometimes answers with the bare JSON call
    if stripped[:1] in ("{", "["):
        parsed = _normalise(_loads_prefix(stripped))
        if parsed and json.JSONDecoder().raw_decode(stripped)[1] == len(stripped):
            return "", parsed
    return text, []


def new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:16]}"


def to_openai_tool_calls(calls: List[Dict[str, Any]], with_index: bool = False) -> List[Dict[str, Any]]:
    """OpenAI's `tool_calls`: each with an id, and `arguments` as a JSON *string*. `with_index` adds the `index` streaming deltas carry."""
    out = []
    for i, call in enumerate(calls):
        entry: Dict[str, Any] = {"id": new_call_id(), "type": "function",
                                 "function": {"name": call["name"], "arguments": json.dumps(call.get("arguments") or {})}}
        if with_index:
            entry = {"index": i, **entry}
        out.append(entry)
    return out


def arguments_as_objects(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """A copy of `messages` in which each assistant `tool_calls[].function.arguments` is a dict, not the JSON string OpenAI sends:
    chat templates (and Ollama) expect the object. Arguments that are not valid JSON are left as they are."""
    out = []
    for m in messages:
        calls = m.get("tool_calls")
        if not isinstance(calls, list):
            out.append(m)
            continue
        fixed = []
        for call in calls:
            fn = call.get("function") if isinstance(call, dict) else None
            if isinstance(fn, dict) and isinstance(fn.get("arguments"), str):
                parsed = _loads_prefix(fn["arguments"]) if fn["arguments"].strip() else {}
                call = {**call, "function": {**fn, "arguments": parsed if isinstance(parsed, dict) else fn["arguments"]}}
            fixed.append(call)
        out.append({**m, "tool_calls": fixed})
    return out
