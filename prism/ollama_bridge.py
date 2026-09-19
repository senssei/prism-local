"""
prism.ollama_bridge: Integrated Ollama bridge for GGUF model execution.
Enables prism to discover, run, and serve Ollama models alongside ONNX models.
"""

import json
import urllib.error
import urllib.request
from typing import Any, Dict, Iterator, List, Optional

OLLAMA_BASE_URL = "http://localhost:11434"

def is_ollama_running(base_url: str = OLLAMA_BASE_URL) -> bool:
    """Checks whether the local Ollama daemon is reachable."""
    try:
        req = urllib.request.Request(f"{base_url}/api/version", method="GET")
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            return resp.status == 200
    except Exception:
        return False

def list_ollama_models(base_url: str = OLLAMA_BASE_URL) -> List[Dict[str, Any]]:
    """Retrieves list of installed Ollama models."""
    if not is_ollama_running(base_url):
        return []
    try:
        req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3.0) as resp:
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
    base_url: str = OLLAMA_BASE_URL,
) -> Iterator[str]:
    """Streams chat completions from Ollama."""
    clean_model = model.replace("ollama:", "")
    payload = {
        "model": clean_model,
        "messages": messages,
        "stream": True,
        "options": options or {},
    }
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
                    msg = chunk.get("message", {})
                    content = msg.get("content", "")
                    if content:
                        yield content
                except Exception:
                    continue
