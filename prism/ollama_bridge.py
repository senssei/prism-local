"""
prism.ollama_bridge: Integrated Ollama bridge for GGUF model execution.
Enables prism to discover, run, and serve Ollama models alongside ONNX models.
"""

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Iterator, List, Optional, Tuple

OLLAMA_BASE_URL = "http://localhost:11434"


def ollama_base_url() -> str:
    """The Ollama daemon URL: $OLLAMA_HOST (Ollama's own variable; `host`, `host:port` or a full URL), else localhost:11434."""
    raw = os.environ.get("OLLAMA_HOST", "").strip()
    if not raw:
        return OLLAMA_BASE_URL
    if "://" in raw:  # a full URL: like Ollama itself, a missing port means the scheme's default
        return raw.rstrip("/")
    parts = urllib.parse.urlsplit(f"//{raw}")
    if parts.port is None and parts.hostname:
        host = f"[{parts.hostname}]" if ":" in parts.hostname else parts.hostname
        return f"http://{host}:11434"
    return f"http://{raw}"


def is_ollama_running(base_url: Optional[str] = None) -> bool:
    """Checks whether the local Ollama daemon is reachable."""
    base_url = base_url or ollama_base_url()
    try:
        req = urllib.request.Request(f"{base_url}/api/version", method="GET")
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            return resp.status == 200
    except Exception:
        return False

def list_ollama_models(base_url: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieves list of installed Ollama models (empty if the daemon is unreachable)."""
    base_url = base_url or ollama_base_url()
    try:
        req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = []
            for m in data.get("models", []):
                name = m.get("name")
                size = m.get("size", 0)
                details = m.get("details", {})
                models.append({
                    "id": f"ollama:{name}",
                    "name": name,
                    "engine": "Ollama (llama.cpp)",
                    "type": "GGUF",
                    "size_mb": round(size / (1024 * 1024), 1),
                    "family": details.get("family", "unknown"),
                    "quantization": details.get("quantization_level", "unknown"),
                    "backend": "ollama",
                })
            return models
    except Exception:
        return []

def stream_ollama_chat(
    model: str,
    messages: List[Dict[str, str]],
    options: Optional[Dict[str, Any]] = None,
    base_url: Optional[str] = None,
    stats: Optional[Dict[str, Any]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Iterator[str]:
    """Streams chat completions from Ollama.

    When `stats` is given it is filled from the final chunk once the stream ends: `prompt_tokens`, `completion_tokens`
    and `finish_reason` ("stop" or "length"), whichever the daemon reported, and `tool_calls` ([{"name", "arguments"}]) when the model
    called `tools`.
    """
    base_url = base_url or ollama_base_url()
    clean_model = model.replace("ollama:", "")
    payload = {
        "model": clean_model,
        "messages": messages,
        "stream": True,
        "options": options or {},
    }
    if tools:
        payload["tools"] = tools
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        for line in resp:
            if line:
                try:
                    chunk = json.loads(line.decode("utf-8"))
                except Exception:
                    continue
                if stats is not None and chunk.get("done"):
                    if "prompt_eval_count" in chunk:
                        stats["prompt_tokens"] = chunk["prompt_eval_count"]
                    if "eval_count" in chunk:
                        stats["completion_tokens"] = chunk["eval_count"]
                    stats["finish_reason"] = "length" if chunk.get("done_reason") == "length" else "stop"
                message = chunk.get("message") or {}
                if stats is not None:
                    for call in message.get("tool_calls") or []:
                        fn = call.get("function") or {}
                        if fn.get("name"):
                            args = fn.get("arguments")
                            stats.setdefault("tool_calls", []).append(
                                {"name": fn["name"], "arguments": args if isinstance(args, dict) else {}})
                content = message.get("content", "")
                if content:
                    yield content


def embed_ollama(model: str, inputs: List[str], base_url: Optional[str] = None) -> Tuple[List[List[float]], int]:
    """Embeddings of `inputs` from Ollama's /api/embed: (one vector per input, prompt tokens the daemon counted)."""
    base_url = base_url or ollama_base_url()
    payload = {"model": model.replace("ollama:", "", 1), "input": inputs}
    req = urllib.request.Request(
        f"{base_url}/api/embed",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    vectors = data.get("embeddings")
    if not isinstance(vectors, list) or len(vectors) != len(inputs):
        raise ValueError("Ollama returned no embeddings for this model (is it an embedding model?)")
    return vectors, int(data.get("prompt_eval_count") or 0)


def pull_ollama_model(model_name: str, base_url: Optional[str] = None) -> bool:
    """Pulls an Ollama model using the streaming /api/pull endpoint with progress display."""
    base_url = base_url or ollama_base_url()
    clean_model = model_name.replace("ollama:", "")
    if not is_ollama_running(base_url):
        print(f"❌ Cannot pull Ollama model '{clean_model}': Ollama daemon is not running at {base_url}.")
        return False

    print(f"🦙 Pulling Ollama model '{clean_model}' from registry...")
    payload = {"name": clean_model, "stream": True}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/api/pull",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    last_status = ""
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            for line in resp:
                if not line:
                    continue
                try:
                    chunk = json.loads(line.decode("utf-8"))
                    status = chunk.get("status", "")
                    total = chunk.get("total", 0)
                    completed = chunk.get("completed", 0)

                    if total > 0 and completed > 0:
                        pct = (completed / total) * 100
                        mb_done = completed / (1024 * 1024)
                        mb_total = total / (1024 * 1024)
                        msg = f"\r   📥 {status}: {mb_done:.1f} MB / {mb_total:.1f} MB ({pct:.1f}%)"
                        print(msg, end="", flush=True)
                    elif status != last_status:
                        if last_status and "\r" in last_status:
                            print()
                        print(f"   ℹ️  {status}")
                        last_status = status
                except Exception:
                    continue
        print(f"\n✅ Ollama model '{clean_model}' pulled successfully.")
        return True
    except Exception as ex:
        print(f"\n❌ Failed to pull Ollama model '{clean_model}': {ex}")
        return False

