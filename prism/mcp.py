#!/usr/bin/env python3
"""
prism.mcp: Lightweight Stdio JSON-RPC 2.0 Model Context Protocol (MCP) Server.
Exposes Prism's multi-engine inference (ONNX Runtime GenAI on CUDA & Ollama GGUF),
code generation, code review, model listing, and GPU telemetry to AI agents.
"""

import atexit
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from prism.catalog import ModelCatalog
from prism.telemetry import get_gpu_info

PRISM_DEFAULT_URL = os.environ.get("PRISM_BASE_URL", "http://localhost:5272/v1")

# With no server to talk to, tool calls run the model inside this process. It stays loaded between calls (loading takes seconds),
# and is released after this much idle time so it does not hold VRAM against a `prism serve` started later.
IDLE_UNLOAD_SEC = 120.0

_catalog_instance: Optional[ModelCatalog] = None
_manager = None
_idle_timer: Optional[threading.Timer] = None
_state_lock = threading.Lock()


def _catalog() -> ModelCatalog:
    """One catalog per process, so its 5-second discovery cache works across tool calls."""
    global _catalog_instance
    with _state_lock:
        if _catalog_instance is None:
            _catalog_instance = ModelCatalog()
        return _catalog_instance


def _engine_manager():
    """The in-process engine manager (imported lazily: importing the engine loads the CUDA libraries)."""
    global _manager
    with _state_lock:
        if _manager is None:
            from prism.server import ActiveEngineManager
            _manager = ActiveEngineManager(catalog=_catalog())
            atexit.register(_manager.unload)
        return _manager


def _schedule_idle_unload() -> None:
    global _idle_timer
    with _state_lock:
        if _idle_timer is not None:
            _idle_timer.cancel()
        _idle_timer = threading.Timer(IDLE_UNLOAD_SEC, _manager.unload)
        _idle_timer.daemon = True
        _idle_timer.start()


def _auth_headers(headers: Dict[str, str]) -> Dict[str, str]:
    key = os.environ.get("PRISM_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def call_prism_server(
    prompt: str,
    model: Optional[str] = None,
    system: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
    base_url: str = PRISM_DEFAULT_URL,
) -> str:
    """Sends a chat completion request to the Prism OpenAI-compatible server."""
    catalog = _catalog()
    all_models = catalog.list_all_models(include_ollama=True)

    if not model:
        # Prefer CUDA ONNX model if available, else first available
        cuda_models = [m["id"] for m in all_models if "cuda" in m.get("device", "").lower()]
        model = cuda_models[0] if cuda_models else (all_models[0]["id"] if all_models else "Phi-4-mini-instruct-cuda-gpu")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    opts = options or {}
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "max_tokens": opts.get("max_tokens", 1024),
        "temperature": opts.get("temperature", 0.1),
    }

    t0 = time.perf_counter()
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/chat/completions",
            data=data,
            headers=_auth_headers({"Content-Type": "application/json"}),
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            elapsed = time.perf_counter() - t0
            res = json.loads(resp.read().decode("utf-8"))
            choices = res.get("choices", [])
            if not choices:
                return "Error: Prism server returned empty choices."
            reply = choices[0].get("message", {}).get("content", "")

            # Append telemetry footnote
            usage = res.get("usage", {})
            tok_count = usage.get("completion_tokens", 0)
            tok_per_sec = (tok_count / elapsed) if elapsed > 0 else 0.0
            telemetry = (
                f"\n\n[💎 Prism: {model} @ {tok_per_sec:.1f} tok/s in {elapsed:.2f}s | "
                f"{tok_count} tokens generated | Zero Cloud Cost]"
            )
            return f"{reply}{telemetry}"
    except urllib.error.HTTPError as http_ex:
        detail = http_ex.read().decode("utf-8", "replace")[:300]
        hint = ""
        if "insufficient_resources" in detail:
            hint = "\nHint: Model load was refused due to resource limits. Override with PRISM_RESOURCE_CHECK=off or adjust memory reserves."
        return f"Error: Prism server responded {http_ex.code}: {detail}{hint}"
    except urllib.error.URLError:
        # Fallback to direct engine generation if server is offline
        try:
            resolved = catalog.resolve_model(model)
        except ValueError as ex:  # AmbiguousModelError
            return f"Error: {ex}"
        if resolved and resolved.get("backend") == "onnx":
            try:
                from prism.templates import render_prompt
                formatted = render_prompt(resolved, messages)
                manager = _engine_manager()
                with manager.use_engine(resolved) as engine:
                    gen_res = engine.generate(formatted, max_tokens=opts.get("max_tokens", 1024))
                _schedule_idle_unload()
                telemetry = (
                    f"\n\n[💎 Prism (Direct CUDA Engine): {model} @ {gen_res['decode_tok_per_sec']:.1f} tok/s | "
                    f"TTFT: {gen_res['ttft_sec']*1000:.1f}ms | Zero Cloud Cost]"
                )
                return f"{gen_res['text']}{telemetry}"
            except Exception as engine_ex:
                hint = ""
                from prism.resources import InsufficientResourcesError
                if isinstance(engine_ex, InsufficientResourcesError) or "insufficient_resources" in str(engine_ex).lower():
                    hint = "\nHint: Model load refused due to resource limits. Override with PRISM_RESOURCE_CHECK=off."
                return f"Error executing direct ONNX engine for model '{model}': {engine_ex}{hint}"

        return (
            f"Error: Could not connect to Prism inference server at {base_url}.\n"
            f"Start the server with: 'prism serve --port 5272'"
        )
    except Exception as ex:
        return f"Error executing Prism completion: {ex}"


def handle_list_tools() -> List[Dict[str, Any]]:
    """Defines MCP tools exposed by Prism."""
    return [
        {
            "name": "prism_ask_coder",
            "description": (
                "Author clean code, solve algorithmic problems, debug errors, or write unit tests "
                "using Prism's local multi-engine AI (ONNX Runtime GenAI on NVIDIA CUDA or Ollama GGUF). "
                "Runs with zero cloud token cost and ultra-low latency."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The specific coding task, bug fix, or instruction.",
                    },
                    "context_code": {
                        "type": "string",
                        "description": "Optional code context, existing files, or compiler errors.",
                    },
                    "model": {
                        "type": "string",
                        "description": "Model ID to run (e.g. 'Phi-4-mini-instruct-cuda-gpu' or 'ollama:qwen2.5-coder:7b').",
                    },
                },
                "required": ["task"],
            },
        },
        {
            "name": "prism_code_review",
            "description": (
                "Perform an in-depth security, race condition, memory safety, and edge-case code review "
                "using local AI models via Prism."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Source code snippet, diff, or function to review.",
                    },
                    "focus": {
                        "type": "string",
                        "description": "Review focus area (e.g. 'security, concurrency, memory leaks, performance').",
                        "default": "security and edge cases",
                    },
                    "model": {
                        "type": "string",
                        "description": "Model ID to use for the review.",
                    },
                },
                "required": ["code"],
            },
        },
        {
            "name": "prism_list_models",
            "description": "List all local ONNX and Ollama models discovered by Prism with hardware targets and sizes.",
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "prism_get_status",
            "description": "Query GPU hardware telemetry (NVML VRAM, compute capability, RTX model) and Prism daemon status.",
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "prism_benchmark",
            "description": "Run an automated micro-benchmark (Time to First Token, throughput in tok/s, VRAM) on any local model.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "Model name or alias to benchmark (e.g. 'Phi-4-mini-instruct-cuda-gpu').",
                    },
                },
                "required": ["model"],
            },
        },
    ]


def handle_tool_call(name: str, args: Dict[str, Any]) -> str:
    """Dispatches MCP tool call requests."""
    catalog = _catalog()

    if name == "prism_ask_coder":
        task = args.get("task", "")
        context = args.get("context_code", "")
        model = args.get("model")
        prompt = f"Context Code:\n```\n{context}\n```\n\nTask: {task}" if context else task
        system = (
            "You are an expert systems software engineer. Provide high-quality, concise, "
            "bug-free code. Wrap code blocks in markdown ```language tags."
        )
        return call_prism_server(prompt=prompt, model=model, system=system)

    elif name == "prism_code_review":
        code = args.get("code", "")
        focus = args.get("focus", "security and edge cases")
        model = args.get("model")
        prompt = (
            f"Review the following code with focus on: {focus}.\n"
            f"Identify potential bugs, race conditions, memory leaks, and edge cases. "
            f"Provide concrete, actionable code fixes.\n\n```\n{code}\n```"
        )
        system = "You are a senior principal engineer performing a rigorous code review. Be concise, precise, and actionable."
        return call_prism_server(prompt=prompt, model=model, system=system)

    elif name == "prism_list_models":
        models = catalog.list_all_models(include_ollama=True)
        lines = [f"Discovered {len(models)} local models in Prism:"]
        lines.append(f"{'NAME / ID':<40} {'ENGINE':<22} {'SIZE':<10} {'DEVICE'}")
        lines.append("-" * 78)
        for m in models:
            dev = m.get("device", "CPU/GPU")
            size_str = f"{m.get('size_mb', 0)} MB"
            lines.append(f"{m['id']:<40} {m.get('engine', 'Unknown'):<22} {size_str:<10} {dev}")
        return "\n".join(lines)

    elif name == "prism_get_status":
        gpu = get_gpu_info()
        lines = ["🖥️  PRISM SYSTEM & HARDWARE TELEMETRY:"]
        if gpu.get("available"):
            for dev in gpu.get("devices", []):
                lines.append(f"• GPU #{dev['index']}: {dev['name']}")
                lines.append(f"  Compute Capability: {dev['compute_capability']}")
                lines.append(f"  VRAM: {dev['vram_used_mb']:.1f} MB used / {dev['vram_total_mb']:.1f} MB total ({dev['vram_free_mb']:.1f} MB free)")
        else:
            lines.append(f"• GPU: {gpu.get('error', 'Not detected via NVML')}")

        # Check server reachability
        try:
            req = urllib.request.Request(f"{PRISM_DEFAULT_URL.rstrip('/')}/models", method="GET")
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                lines.append(f"• Prism REST Server: Active at {PRISM_DEFAULT_URL}")
        except Exception:
            lines.append(f"• Prism REST Server: Offline (will use direct CUDA engine fallback)")

        return "\n".join(lines)

    elif name == "prism_benchmark":
        from prism.benchmark import run_benchmark
        model = args.get("model", "")
        if not model:
            return "Error: model parameter is required for benchmarking."
        res = run_benchmark(model)
        return json.dumps(res, indent=2)

    return f"Unknown tool: {name}"


def run_mcp_server():
    """Main stdio JSON-RPC 2.0 loop."""
    sys.stderr.write("💎 Prism MCP Server starting on stdio...\n")
    sys.stderr.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue

        req_id = req.get("id")
        method = req.get("method")
        params = req.get("params", {})

        if method == "initialize":
            resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name": "prism-mcp",
                        "version": "1.0.0",
                    },
                },
            }
        elif method == "notifications/initialized":
            continue
        elif method == "ping":
            resp = {"jsonrpc": "2.0", "id": req_id, "result": {}}
        elif method == "tools/list":
            resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": handle_list_tools()},
            }
        elif method == "tools/call":
            tool_name = params.get("name", "")
            tool_args = params.get("arguments", {})
            output_text = handle_tool_call(tool_name, tool_args)
            resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": output_text}],
                },
            }
        else:
            if req_id is not None:
                resp = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}",
                    },
                }
            else:
                continue

        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    run_mcp_server()
