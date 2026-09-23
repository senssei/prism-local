#!/usr/bin/env python3
"""
prism.acp: Stdio JSON-RPC server speaking the Agent Client Protocol (ACP).

First milestone (spec.md P16): plain-text prompts and streamed responses only — no file or
process access. `session/prompt` can run for a long time, so unlike `prism/mcp.py`'s fully
synchronous loop, each prompt runs on its own daemon thread while the main stdin loop keeps
reading (so a `session/cancel` notification can arrive and interrupt it). `_stdout_lock` keeps
concurrent writers (a prompt's worker thread and the main loop) from interleaving JSON lines.
"""

import dataclasses
import json
import os
import sys
import threading
import urllib.error
import urllib.request
import uuid
from typing import Any, Dict, Iterator, List, Optional

from prism.catalog import ModelCatalog, pick_default_model, DEFAULT_FALLBACK_MODEL_ID

PRISM_DEFAULT_URL = os.environ.get("PRISM_BASE_URL", "http://localhost:5272/v1")
ACP_PROTOCOL_VERSION = 1

_stdout_lock = threading.Lock()
_catalog_instance: Optional[ModelCatalog] = None
_manager = None
# RLock, not Lock: `_engine_manager()` calls `_catalog()` while already holding this lock, so a
# plain non-reentrant Lock would self-deadlock the first time it runs (review finding — acp.py is
# the one file in this codebase that actually calls these lazily-initialized singletons from more
# than one thread, since `session/prompt` runs on its own daemon thread).
_state_lock = threading.RLock()
_sessions: Dict[str, "_Session"] = {}
_sessions_lock = threading.Lock()


@dataclasses.dataclass
class _Session:
    session_id: str
    cwd: str
    model: str
    messages: List[Dict[str, str]] = dataclasses.field(default_factory=list)
    cancel_event: threading.Event = dataclasses.field(default_factory=threading.Event)
    busy: bool = False


class AcpRequestError(Exception):
    """A request whose incoming shape was explicitly validated and found malformed — always
    -32602 Invalid params. Never raised for an unrelated internal fault (see `_dispatch`)."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _catalog() -> ModelCatalog:
    global _catalog_instance
    with _state_lock:
        if _catalog_instance is None:
            _catalog_instance = ModelCatalog()
        return _catalog_instance


def _engine_manager():
    """The in-process engine manager used only when `prism serve` is unreachable (mirrors `mcp.py`).

    Two sessions can independently hit the direct-engine fallback on their own daemon threads at
    the same time; without a lock here, both could pass the `is None` check before either finishes
    constructing, producing two `ActiveEngineManager`s (and so two engine locks) — exactly the
    concurrent-model-load scenario AGENTS.md and invariant I2 forbid.
    """
    global _manager
    with _state_lock:
        if _manager is None:
            from prism.server import ActiveEngineManager
            _manager = ActiveEngineManager(catalog=_catalog())
        return _manager


def _auth_headers(headers: Dict[str, str]) -> Dict[str, str]:
    key = os.environ.get("PRISM_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _write(obj: Dict[str, Any]) -> None:
    """Writes one JSON-RPC line. A broken stdout (the editor closed the pipe) is logged and
    swallowed rather than raised: raising here would escape every caller — including the error
    handler in `_dispatch` and the `finally` cleanup in `_run_prompt` — and kill the whole stdio
    loop over a peer that is simply gone (mirrors `prism/server.py`'s `_safe_write`, spec P13)."""
    line = json.dumps(obj)
    with _stdout_lock:
        try:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
        except (BrokenPipeError, OSError, ValueError) as ex:
            try:
                sys.stderr.write(f"prism acp: dropped a response, stdout is gone ({ex}).\n")
            except Exception:
                pass  # stderr is gone too; nothing left to report to, but still must not raise


def _send_result(req_id: Any, result: Dict[str, Any]) -> None:
    _write({"jsonrpc": "2.0", "id": req_id, "result": result})


def _send_error(req_id: Any, code: int, message: str) -> None:
    _write({"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}})


def _send_notification(method: str, params: Dict[str, Any]) -> None:
    _write({"jsonrpc": "2.0", "method": method, "params": params})


def _default_model() -> str:
    """`$PRISM_ACP_MODEL` overrides; otherwise prefer a CUDA ONNX model, else the first available model."""
    override = os.environ.get("PRISM_ACP_MODEL", "").strip()
    if override:
        return override
    all_models = _catalog().list_all_models(include_ollama=True)
    return pick_default_model(all_models) or DEFAULT_FALLBACK_MODEL_ID


def handle_initialize(params: Dict[str, Any]) -> Dict[str, Any]:
    requested = params.get("protocolVersion", ACP_PROTOCOL_VERSION)
    is_valid_version = isinstance(requested, int) and not isinstance(requested, bool) and requested >= 1
    version = min(requested, ACP_PROTOCOL_VERSION) if is_valid_version else ACP_PROTOCOL_VERSION
    return {
        "protocolVersion": version,
        "agentCapabilities": {
            "loadSession": False,
            "promptCapabilities": {"image": False, "audio": False, "embeddedContext": False},
        },
        "authMethods": [],
    }


def handle_session_new(params: Dict[str, Any]) -> Dict[str, Any]:
    session_id = uuid.uuid4().hex
    session = _Session(session_id=session_id, cwd=params.get("cwd", ""), model=_default_model())
    with _sessions_lock:
        _sessions[session_id] = session
    return {"sessionId": session_id}


def _require_session_id(params: Dict[str, Any]) -> str:
    session_id = params.get("sessionId")
    if not isinstance(session_id, str):
        raise AcpRequestError(-32602, "sessionId must be a string")
    return session_id


def _extract_prompt_text(prompt: Any) -> str:
    if not isinstance(prompt, list):
        raise AcpRequestError(-32602, "prompt must be a list of content blocks")
    parts = []
    for block in prompt:
        if not isinstance(block, dict):
            raise AcpRequestError(-32602, "each prompt content block must be an object")
        block_type = block.get("type")
        if block_type != "text":
            raise AcpRequestError(-32602, f"Unsupported prompt content block type: {block_type!r}")
        parts.append(block.get("text", ""))
    return "".join(parts)


def _server_delta_stream(messages: List[Dict[str, str]], model: str,
                          base_url: str = PRISM_DEFAULT_URL) -> Iterator[Dict[str, str]]:
    """Opens a streaming chat completion against `prism serve` and returns a generator of deltas.

    Raises `urllib.error.HTTPError` / `urllib.error.URLError` immediately (before any generator is
    returned), so callers can tell "server reachable but errored" from "server unreachable" without
    having to start iterating first.
    """
    payload = {"model": model, "messages": messages, "stream": True, "max_tokens": 4096}
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=_auth_headers({"Content-Type": "application/json"}),
        method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=300)
    return _iter_sse_response(resp)


def _iter_sse_response(resp) -> Iterator[Dict[str, str]]:
    try:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                return
            chunk = json.loads(data)
            choices = chunk.get("choices") or [{}]
            delta = choices[0].get("delta", {})
            if delta.get("content"):
                yield {"content": delta["content"]}
            if delta.get("reasoning_content"):
                yield {"reasoning_content": delta["reasoning_content"]}
    finally:
        resp.close()


def _direct_engine_delta_stream(session: "_Session", messages: List[Dict[str, str]]) -> Iterator[Dict[str, str]]:
    """Falls back to the in-process engine when `prism serve` is unreachable (mirrors `mcp.py`)."""
    from prism.templates import render_prompt
    resolved = _catalog().resolve_model(session.model)
    formatted = render_prompt(resolved, messages)
    manager = _engine_manager()
    with manager.use_engine(resolved) as engine:
        for piece, _is_first, _speed in engine.stream_generate(formatted):
            yield {"content": piece}


def handle_session_prompt(params: Dict[str, Any], req_id: Any) -> Optional[threading.Thread]:
    try:
        session_id = _require_session_id(params)
        with _sessions_lock:
            session = _sessions.get(session_id)
        if session is None:
            raise AcpRequestError(-32602, f"Unknown sessionId: {session_id!r}")
        if session.busy:
            raise AcpRequestError(-32602, "a prompt is already in progress for this session")
        prompt_text = _extract_prompt_text(params.get("prompt", []))
    except AcpRequestError as ex:
        _send_error(req_id, ex.code, ex.message)
        return None

    session.busy = True
    session.cancel_event.clear()
    thread = threading.Thread(target=_run_prompt, args=(session, prompt_text, req_id), daemon=True)
    thread.start()
    return thread


def _run_prompt(session: "_Session", prompt_text: str, req_id: Any) -> None:
    session.messages.append({"role": "user", "content": prompt_text})
    answered = False
    try:
        try:
            deltas = _server_delta_stream(session.messages, session.model)
        except urllib.error.HTTPError:
            raise
        except urllib.error.URLError:
            deltas = _direct_engine_delta_stream(session, session.messages)

        answer: List[str] = []
        stop_reason = "end_turn"
        for delta in deltas:
            if session.cancel_event.is_set():
                stop_reason = "cancelled"
                break
            if "content" in delta:
                answer.append(delta["content"])
                _send_notification("session/update", {
                    "sessionId": session.session_id,
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": delta["content"]},
                    },
                })
            elif "reasoning_content" in delta:
                _send_notification("session/update", {
                    "sessionId": session.session_id,
                    "update": {
                        "sessionUpdate": "agent_thought_chunk",
                        "content": {"type": "text", "text": delta["reasoning_content"]},
                    },
                })
        session.messages.append({"role": "assistant", "content": "".join(answer)})
        answered = True
        _send_result(req_id, {"stopReason": stop_reason})
    except urllib.error.HTTPError as ex:
        # Only drop the dangling user turn if the assistant never got a chance to answer it —
        # once `answered` is True the failure happened while reporting a real result (e.g. the
        # send itself failed), and popping would silently discard a completed answer instead
        # (review finding: a naive "always pop" reintroduces the exact bug it was meant to fix).
        if not answered:
            session.messages.pop()
        detail = ex.read().decode("utf-8", "replace")[:500]
        _send_error(req_id, -32000, f"Prism server responded {ex.code}: {detail}")
    except Exception as ex:
        if not answered:
            session.messages.pop()
        _send_error(req_id, -32000, str(ex))
    finally:
        session.busy = False


def handle_session_cancel(params: Dict[str, Any]) -> None:
    session_id = params.get("sessionId")
    if not isinstance(session_id, str):
        return  # malformed notification; nothing to cancel, and no response is expected anyway
    with _sessions_lock:
        session = _sessions.get(session_id)
    if session is not None:
        session.cancel_event.set()


def _dispatch(req: Dict[str, Any]) -> None:
    """Routes one JSON-RPC message.

    Two exception tiers, on purpose: `AcpRequestError` is raised only where the incoming shape has
    been explicitly validated (non-dict `params`, a `sessionId` that isn't a string, a `prompt`
    that isn't a list of objects, ...) and always means -32602 Invalid params. Anything else —
    including an `AttributeError`/`TypeError` from a genuine internal bug elsewhere in the call
    chain (e.g. a catalog entry with an unexpected field type) — is reported as -32603 Internal
    error instead. An earlier revision classified by exception *type* rather than by this explicit
    contract; that is unsound, because an internal bug can raise the exact same `AttributeError` /
    `TypeError` a malformed request raises, and would have been mislabeled -32602 (review finding).
    """
    req_id = req.get("id") if isinstance(req, dict) else None
    try:
        method = req.get("method")
        params = req.get("params")
        params = params if params is not None else {}
        if method in ("initialize", "session/new", "session/prompt", "session/cancel") \
                and not isinstance(params, dict):
            raise AcpRequestError(-32602, f"params must be an object for method {method!r}")

        if method == "initialize":
            _send_result(req_id, handle_initialize(params))
        elif method == "session/new":
            _send_result(req_id, handle_session_new(params))
        elif method == "session/prompt":
            handle_session_prompt(params, req_id)
        elif method == "session/cancel":
            handle_session_cancel(params)
        elif method in ("notifications/initialized", "initialized"):
            return
        else:
            if req_id is not None:
                _send_error(req_id, -32601, f"Method not found: {method}")
    except AcpRequestError as ex:
        if req_id is not None:
            _send_error(req_id, ex.code, ex.message)
    except Exception as ex:
        if req_id is not None:
            _send_error(req_id, -32603, f"Internal error: {ex}")


def _handle_line(line: str) -> None:
    """Parses and dispatches one stdio line. Must never raise: `json.loads` on a syntactically
    invalid line raises `json.JSONDecodeError`, but on a pathologically deep structure (e.g.
    thousands of nested `[`) it raises `RecursionError` instead — a third, JSON-valid-shaped crash
    vector `_dispatch`'s own hardening cannot see because the exception happens before `_dispatch`
    is ever called (review finding)."""
    try:
        req = json.loads(line)
        _dispatch(req)
    except Exception as ex:
        sys.stderr.write(f"prism acp: dropped one malformed request ({ex}).\n")


def run_acp_server() -> None:
    """Main stdio JSON-RPC 2.0 loop."""
    sys.stderr.write("💎 Prism ACP Server starting on stdio...\n")
    sys.stderr.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        _handle_line(line)


if __name__ == "__main__":
    run_acp_server()
