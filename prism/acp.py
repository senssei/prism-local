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
import itertools
import json
import logging
import os
import sys
import threading
import urllib.error
import urllib.request
import uuid
from typing import Any, Dict, Iterator, List, Optional, Tuple

from prism.catalog import ModelCatalog, pick_default_model, DEFAULT_FALLBACK_MODEL_ID

PRISM_DEFAULT_URL = os.environ.get("PRISM_BASE_URL", "http://localhost:5272/v1")
ACP_PROTOCOL_VERSION = 1

# Traces a session's lifecycle (session/new, session/prompt start/finish/cancel, the direct-engine
# fallback trigger, and failures) by `session_id`/`req_id` so one session can be followed through
# the log. Mirrors `prism/server.py`'s `logging.getLogger("prism.server")` convention; unlike the
# operator-facing stderr banners/notices already in this module (kept as-is), these are standard
# `logging` calls a caller can configure, filter, or redirect.
logger = logging.getLogger("prism.acp")

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
    messages: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    cancel_event: threading.Event = dataclasses.field(default_factory=threading.Event)
    busy: bool = False
    client_capabilities: Dict[str, bool] = dataclasses.field(default_factory=dict)
    tool_call_counter: int = 0


class AcpRequestError(Exception):
    """A request whose incoming shape was explicitly validated and found malformed — always
    -32602 Invalid params. Never raised for an unrelated internal fault (see `_dispatch`)."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class AcpClientError(Exception):
    """Raised when an outbound ACP request to the client receives a JSON-RPC error response."""

    def __init__(self, code: int, message: str, data: Any = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


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


def _write(obj: Dict[str, Any]) -> bool:
    """Writes one JSON-RPC line. A broken stdout (the editor closed the pipe) is logged and
    swallowed rather than raised: raising here would escape every caller — including the error
    handler in `_dispatch` and the `finally` cleanup in `_run_prompt` — and kill the whole stdio
    loop over a peer that is simply gone (mirrors `prism/server.py`'s `_safe_write`, spec P13).
    Returns True on success, False if stdout pipe is broken."""
    line = json.dumps(obj)
    with _stdout_lock:
        try:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
            return True
        except (BrokenPipeError, OSError, ValueError) as ex:
            logger.warning("dropped a response, stdout is gone: %s", ex)
            try:
                sys.stderr.write(f"prism acp: dropped a response, stdout is gone ({ex}).\n")
            except Exception:
                pass  # stderr is gone too; nothing left to report to, but still must not raise
            return False


def _send_result(req_id: Any, result: Dict[str, Any]) -> None:
    _write({"jsonrpc": "2.0", "id": req_id, "result": result})


def _send_error(req_id: Any, code: int, message: str) -> None:
    _write({"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}})


def _send_notification(method: str, params: Dict[str, Any]) -> None:
    _write({"jsonrpc": "2.0", "method": method, "params": params})


_request_id_counter = itertools.count(1)
_request_id_lock = threading.Lock()
_pending: Dict[int, Tuple[threading.Event, List[Any]]] = {}
_pending_lock = threading.Lock()


def _next_request_id() -> int:
    with _request_id_lock:
        return next(_request_id_counter)


def _close_pending(exc: Optional[Exception] = None) -> None:
    """Cancels all in-flight outbound requests when the connection drops."""
    err = exc if exc is not None else ConnectionError("ACP connection closed")
    with _pending_lock:
        items = list(_pending.items())
        _pending.clear()
    for _req_id, (event, slot) in items:
        slot[1] = err
        event.set()


def _send_request(method: str, params: Dict[str, Any], *,
                  cancel_event: Optional[threading.Event] = None,
                  timeout: Optional[float] = None) -> Any:
    """Parks a future, writes the outbound JSON-RPC request frame, and waits for the client response."""
    req_id = _next_request_id()
    event = threading.Event()
    slot: List[Any] = [None, None]  # [result, exc]
    with _pending_lock:
        _pending[req_id] = (event, slot)
    try:
        written = _write({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
        if written is False:
            raise AcpClientError(-32000, f"Cannot send ACP request {method!r}: stdout pipe broken")

        if cancel_event is None:
            signaled = event.wait(timeout=timeout)
        else:
            deadline = (time.time() + timeout) if timeout is not None else None
            while not event.is_set():
                if cancel_event.is_set():
                    raise AcpClientError(-32000, f"ACP request {method!r} (id={req_id}) cancelled")
                now = time.time()
                if deadline is not None and now >= deadline:
                    break
                step = 0.05
                if deadline is not None:
                    step = min(step, max(0.001, deadline - now))
                event.wait(timeout=step)
            signaled = event.is_set()

        if not signaled:
            if cancel_event and cancel_event.is_set():
                raise AcpClientError(-32000, f"ACP request {method!r} (id={req_id}) cancelled")
            raise TimeoutError(f"ACP request {method!r} (id={req_id}) timed out after {timeout}s")
        if slot[1] is not None:
            raise slot[1]
        return slot[0]
    finally:
        with _pending_lock:
            _pending.pop(req_id, None)


def _default_model() -> str:
    """`$PRISM_ACP_MODEL` overrides; otherwise prefer a CUDA ONNX model, else the first available model."""
    override = os.environ.get("PRISM_ACP_MODEL", "").strip()
    if override:
        return override
    all_models = _catalog().list_all_models(include_ollama=True)
    return pick_default_model(all_models) or DEFAULT_FALLBACK_MODEL_ID


_FS_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the text contents of a file at the specified path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The path of the file to read."},
                    "line": {"type": "integer", "description": "Optional 1-based line number to start reading from."},
                    "limit": {"type": "integer", "description": "Optional maximum number of lines to read."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write text content to a file at the specified path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The path of the file to write."},
                    "content": {"type": "string", "description": "The text content to write to the file."},
                },
                "required": ["path", "content"],
            },
        },
    },
]

MAX_ACP_TOOL_ROUNDS = 8
_client_fs_capabilities: Dict[str, bool] = {"readTextFile": False, "writeTextFile": False}


def _resolve_tools(session: _Session) -> List[Dict[str, Any]]:
    caps = session.client_capabilities
    read_ok = caps.get("readTextFile", False)
    write_ok = caps.get("writeTextFile", False)
    tools = []
    if read_ok:
        tools.append(_FS_TOOLS[0])
    if write_ok:
        tools.append(_FS_TOOLS[1])
    return tools


def _request_fs_read(session: _Session, path: str, line: Optional[int] = None, limit: Optional[int] = None) -> Dict[str, Any]:
    params: Dict[str, Any] = {"sessionId": session.session_id, "path": path}
    if line is not None:
        params["line"] = line
    if limit is not None:
        params["limit"] = limit
    try:
        return _send_request("fs/read_text_file", params, cancel_event=session.cancel_event)
    except TypeError:
        return _send_request("fs/read_text_file", params)


def _request_fs_write(session: _Session, path: str, content: str) -> Dict[str, Any]:
    params: Dict[str, Any] = {"sessionId": session.session_id, "path": path, "content": content}
    try:
        return _send_request("fs/write_text_file", params, cancel_event=session.cancel_event)
    except TypeError:
        return _send_request("fs/write_text_file", params)


def _request_fs_permission(session: _Session, call_id: str, path: str) -> Tuple[str, Optional[str]]:
    """Returns (status, message) where status is 'allow', 'reject', 'cancelled', or 'error'."""
    if session.cancel_event.is_set():
        return ("cancelled", None)
    params = {
        "sessionId": session.session_id,
        "toolCall": {"toolCallId": call_id, "title": f"Write {path}", "kind": "edit"},
        "options": [
            {"optionId": "allow_once", "name": "Allow once", "kind": "allow_once"},
            {"optionId": "reject_once", "name": "Reject", "kind": "reject_once"},
        ],
    }
    try:
        try:
            resp = _send_request("session/request_permission", params, cancel_event=session.cancel_event)
        except TypeError:
            resp = _send_request("session/request_permission", params)
    except AcpClientError as ex:
        if "cancelled" in str(ex).lower():
            return ("cancelled", None)
        return ("error", ex.message)
    if isinstance(resp, dict):
        outcome_data = resp.get("outcome")
        if isinstance(outcome_data, dict):
            outcome_kind = outcome_data.get("outcome")
            if outcome_kind == "cancelled":
                return ("cancelled", None)
            if outcome_kind == "selected":
                option_id = outcome_data.get("optionId")
                if option_id == "allow_once":
                    return ("allow", None)
                elif option_id == "reject_once":
                    return ("reject", "permission denied")
    return ("reject", "permission denied")


def handle_initialize(params: Dict[str, Any]) -> Dict[str, Any]:
    requested = params.get("protocolVersion", ACP_PROTOCOL_VERSION) if isinstance(params, dict) else ACP_PROTOCOL_VERSION
    is_valid_version = isinstance(requested, int) and not isinstance(requested, bool) and requested >= 1
    version = min(requested, ACP_PROTOCOL_VERSION) if is_valid_version else ACP_PROTOCOL_VERSION
    caps = params.get("clientCapabilities") if isinstance(params, dict) else {}
    caps = caps if isinstance(caps, dict) else {}
    fs_caps = caps.get("fs") if isinstance(caps, dict) else {}
    fs_caps = fs_caps if isinstance(fs_caps, dict) else {}
    _client_fs_capabilities["readTextFile"] = bool(fs_caps.get("readTextFile", False))
    _client_fs_capabilities["writeTextFile"] = bool(fs_caps.get("writeTextFile", False))
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
    session = _Session(
        session_id=session_id,
        cwd=params.get("cwd", ""),
        model=_default_model(),
        client_capabilities=dict(_client_fs_capabilities),
    )
    with _sessions_lock:
        _sessions[session_id] = session
    logger.info("session/new: session=%s model=%s cwd=%r", session_id, session.model, session.cwd)
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


def _server_delta_stream(messages: List[Dict[str, Any]], model: str,
                          base_url: str = PRISM_DEFAULT_URL,
                          tools: Optional[List[Dict[str, Any]]] = None) -> Iterator[Dict[str, str]]:
    """Opens a streaming chat completion against `prism serve` and returns a generator of deltas.

    Raises `urllib.error.HTTPError` / `urllib.error.URLError` immediately (before any generator is
    returned), so callers can tell "server reachable but errored" from "server unreachable" without
    having to start iterating first.
    """
    payload: Dict[str, Any] = {"model": model, "messages": messages, "stream": True, "max_tokens": 4096}
    if tools:
        payload["tools"] = tools
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


def _direct_engine_delta_stream(session: "_Session", messages: List[Dict[str, Any]],
                                tools: Optional[List[Dict[str, Any]]] = None) -> Iterator[Dict[str, str]]:
    """Falls back to the in-process engine when `prism serve` is unreachable (mirrors `mcp.py`).

    Only an installed ONNX model can run this way (Ollama models and an unresolvable model id are
    not loaded by this process at all — `mcp.py::call_prism_server` has the same guard). Without
    it, `session.model` naming an Ollama model or nothing installed at all fell through to a raw
    `KeyError`/`AttributeError` deep in `render_prompt`/`use_engine`, reported to the client as an
    unhelpful `-32000: "'path'"` instead of a message naming the actual problem (review finding).
    """
    from prism.templates import render_prompt
    resolved = _catalog().resolve_model(session.model)
    if not resolved or resolved.get("backend") != "onnx":
        raise RuntimeError(
            f"Prism server is unreachable and '{session.model}' is not an installed local ONNX "
            "model Prism can run directly (Ollama models need the server); start it with "
            "'prism serve --port 5272'."
        )
    formatted = render_prompt(resolved, messages, tools=tools)
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
        logger.debug("session/prompt rejected: req_id=%s code=%s message=%s", req_id, ex.code, ex.message)
        _send_error(req_id, ex.code, ex.message)
        return None

    session.busy = True
    session.cancel_event.clear()
    logger.debug("session/prompt accepted: session=%s req_id=%s chars=%d", session_id, req_id, len(prompt_text))
    thread = threading.Thread(target=_run_prompt, args=(session, prompt_text, req_id), daemon=True)
    thread.start()
    return thread


def _run_prompt(session: "_Session", prompt_text: str, req_id: Any) -> None:
    history_snapshot = list(session.messages)
    session.messages.append({"role": "user", "content": prompt_text})
    answered = False
    logger.debug("session/prompt started: session=%s req_id=%s model=%s", session.session_id, req_id, session.model)

    def _cancel_and_return() -> None:
        session.messages[:] = history_snapshot
        _send_result(req_id, {"stopReason": "cancelled"})

    try:
        from prism.tools import parse_tool_calls
        from prism.templates import supports_tools

        for round_idx in range(MAX_ACP_TOOL_ROUNDS):
            if session.cancel_event.is_set():
                _cancel_and_return()
                return

            resolved = _catalog().resolve_model(session.model)
            if resolved is None:
                model_supports_tools = True
            elif resolved.get("backend") == "ollama":
                model_supports_tools = True
            else:
                model_supports_tools = supports_tools(resolved)
            tools = _resolve_tools(session) if model_supports_tools else []

            try:
                deltas = _server_delta_stream(session.messages, session.model, tools=tools or None)
            except urllib.error.HTTPError:
                raise
            except urllib.error.URLError:
                logger.warning(
                    "prism serve unreachable; falling back to the direct engine: session=%s req_id=%s model=%s",
                    session.session_id, req_id, session.model,
                )
                deltas = _direct_engine_delta_stream(session, session.messages, tools=tools or None)

            answer: List[str] = []
            stop_reason = "end_turn"
            for delta in deltas:
                if session.cancel_event.is_set():
                    stop_reason = "cancelled"
                    break
                if "content" in delta:
                    piece = delta["content"]
                    answer.append(piece)
                    if not tools:
                        _send_notification("session/update", {
                            "sessionId": session.session_id,
                            "update": {
                                "sessionUpdate": "agent_message_chunk",
                                "content": {"type": "text", "text": piece},
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

            if stop_reason == "cancelled":
                _cancel_and_return()
                return

            raw_text = "".join(answer)
            content_text, calls = parse_tool_calls(raw_text)

            if tools and content_text:
                _send_notification("session/update", {
                    "sessionId": session.session_id,
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": content_text},
                    },
                })

            if not calls:
                session.messages.append({"role": "assistant", "content": raw_text})
                answered = True
                _send_result(req_id, {"stopReason": "end_turn"})
                logger.debug(
                    "session/prompt finished: session=%s req_id=%s stop_reason=end_turn chars=%d",
                    session.session_id, req_id, len(raw_text),
                )
                return

            session.messages.append({"role": "assistant", "content": raw_text})

            for call in calls:
                if session.cancel_event.is_set():
                    _cancel_and_return()
                    return

                session.tool_call_counter += 1
                call_id = f"call_{session.session_id}_{session.tool_call_counter:04d}"
                tool_name = call.get("name")
                args = call.get("arguments", {})
                if not isinstance(args, dict):
                    args = {}

                kind = "read" if tool_name == "read_file" else ("edit" if tool_name == "write_file" else "think")
                title = f"Read {args.get('path', '')}" if tool_name == "read_file" else (
                    f"Write {args.get('path', '')}" if tool_name == "write_file" else f"Call {tool_name}"
                )

                _send_notification("session/update", {
                    "sessionId": session.session_id,
                    "update": {
                        "sessionUpdate": "tool_call",
                        "toolCallId": call_id,
                        "title": title,
                        "kind": kind,
                        "status": "in_progress",
                    },
                })

                if tool_name == "read_file":
                    if not session.client_capabilities.get("readTextFile", False):
                        err_content = '{"error": "fs.readTextFile is not advertised by the client"}'
                        _send_notification("session/update", {
                            "sessionId": session.session_id,
                            "update": {
                                "sessionUpdate": "tool_call_update",
                                "toolCallId": call_id,
                                "status": "failed",
                                "content": [{"type": "content", "content": {"type": "text", "text": err_content}}],
                            },
                        })
                        session.messages.append({"role": "tool", "tool_call_id": call_id, "content": err_content})
                        continue

                    if session.cancel_event.is_set():
                        _cancel_and_return()
                        return

                    path = str(args.get("path", ""))
                    line = args.get("line")
                    limit = args.get("limit")
                    try:
                        res = _request_fs_read(session, path, line=line, limit=limit)
                        if isinstance(res, dict) and "content" in res and isinstance(res["content"], str):
                            file_text = res["content"]
                            _send_notification("session/update", {
                                "sessionId": session.session_id,
                                "update": {
                                    "sessionUpdate": "tool_call_update",
                                    "toolCallId": call_id,
                                    "status": "completed",
                                    "content": [{"type": "content", "content": {"type": "text", "text": file_text}}],
                                },
                            })
                            session.messages.append({"role": "tool", "tool_call_id": call_id, "content": file_text})
                        else:
                            err_msg = f"Invalid response from fs/read_text_file: {res!r}"
                            _send_notification("session/update", {
                                "sessionId": session.session_id,
                                "update": {
                                    "sessionUpdate": "tool_call_update",
                                    "toolCallId": call_id,
                                    "status": "failed",
                                    "content": [{"type": "content", "content": {"type": "text", "text": err_msg}}],
                                },
                            })
                            session.messages.append({"role": "tool", "tool_call_id": call_id, "content": err_msg})
                    except Exception as ex:
                        err_msg = str(ex)
                        _send_notification("session/update", {
                            "sessionId": session.session_id,
                            "update": {
                                "sessionUpdate": "tool_call_update",
                                "toolCallId": call_id,
                                "status": "failed",
                                "content": [{"type": "content", "content": {"type": "text", "text": err_msg}}],
                            },
                        })
                        session.messages.append({"role": "tool", "tool_call_id": call_id, "content": err_msg})

                elif tool_name == "write_file":
                    if not session.client_capabilities.get("writeTextFile", False):
                        err_content = '{"error": "fs.writeTextFile is not advertised by the client"}'
                        _send_notification("session/update", {
                            "sessionId": session.session_id,
                            "update": {
                                "sessionUpdate": "tool_call_update",
                                "toolCallId": call_id,
                                "status": "failed",
                                "content": [{"type": "content", "content": {"type": "text", "text": err_content}}],
                            },
                        })
                        session.messages.append({"role": "tool", "tool_call_id": call_id, "content": err_content})
                        continue

                    path = str(args.get("path", ""))
                    raw_content = args.get("content")
                    content = str(raw_content) if raw_content is not None else ""

                    if session.cancel_event.is_set():
                        _cancel_and_return()
                        return

                    perm_status, perm_msg = _request_fs_permission(session, call_id, path)
                    if perm_status == "cancelled":
                        _cancel_and_return()
                        return
                    elif perm_status in ("reject", "error"):
                        denied_msg = perm_msg or "permission denied"
                        _send_notification("session/update", {
                            "sessionId": session.session_id,
                            "update": {
                                "sessionUpdate": "tool_call_update",
                                "toolCallId": call_id,
                                "status": "failed",
                                "content": [{"type": "content", "content": {"type": "text", "text": denied_msg}}],
                            },
                        })
                        session.messages.append({"role": "tool", "tool_call_id": call_id, "content": denied_msg})
                    elif perm_status == "allow":
                        if session.cancel_event.is_set():
                            _cancel_and_return()
                            return
                        try:
                            res = _request_fs_write(session, path, content)
                            if not isinstance(res, dict):
                                err_msg = f"Invalid response from fs/write_text_file: expected object, got {type(res).__name__}"
                                _send_notification("session/update", {
                                    "sessionId": session.session_id,
                                    "update": {
                                        "sessionUpdate": "tool_call_update",
                                        "toolCallId": call_id,
                                        "status": "failed",
                                        "content": [{"type": "content", "content": {"type": "text", "text": err_msg}}],
                                    },
                                })
                                session.messages.append({"role": "tool", "tool_call_id": call_id, "content": err_msg})
                                continue

                            n_bytes = len(content.encode("utf-8"))
                            success_msg = f"wrote {n_bytes} bytes to {path}"
                            _send_notification("session/update", {
                                "sessionId": session.session_id,
                                "update": {
                                    "sessionUpdate": "tool_call_update",
                                    "toolCallId": call_id,
                                    "status": "completed",
                                    "content": [{"type": "content", "content": {"type": "text", "text": success_msg}}],
                                },
                            })
                            session.messages.append({"role": "tool", "tool_call_id": call_id, "content": success_msg})
                        except Exception as ex:
                            err_msg = str(ex)
                            _send_notification("session/update", {
                                "sessionId": session.session_id,
                                "update": {
                                    "sessionUpdate": "tool_call_update",
                                    "toolCallId": call_id,
                                    "status": "failed",
                                    "content": [{"type": "content", "content": {"type": "text", "text": err_msg}}],
                                },
                            })
                            session.messages.append({"role": "tool", "tool_call_id": call_id, "content": err_msg})
                else:
                    err_msg = f"Unknown tool: {tool_name}"
                    _send_notification("session/update", {
                        "sessionId": session.session_id,
                        "update": {
                            "sessionUpdate": "tool_call_update",
                            "toolCallId": call_id,
                            "status": "failed",
                            "content": [{"type": "content", "content": {"type": "text", "text": err_msg}}],
                        },
                    })
                    session.messages.append({"role": "tool", "tool_call_id": call_id, "content": err_msg})

        exhausted_msg = f"tool loop budget exhausted ({MAX_ACP_TOOL_ROUNDS} rounds)"
        _send_notification("session/update", {
            "sessionId": session.session_id,
            "update": {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": exhausted_msg},
            },
        })
        session.messages.append({"role": "assistant", "content": exhausted_msg})
        answered = True
        _send_result(req_id, {"stopReason": "end_turn"})
    except urllib.error.HTTPError as ex:
        # Only drop the dangling user turn if the assistant never got a chance to answer it —
        # once `answered` is True the failure happened while reporting a real result (e.g. the
        # send itself failed), and popping would silently discard a completed answer instead
        # (review finding: a naive "always pop" reintroduces the exact bug it was meant to fix).
        if not answered:
            session.messages[:] = history_snapshot
        try:
            detail = ex.read().decode("utf-8", "replace")[:500]
        except Exception:
            # `ex.read()` itself can fail (e.g. the connection dropped mid-read of the error body).
            # That must not raise from inside this except clause — a sibling `except Exception`
            # on the same try/except/finally does NOT catch it, so an unguarded `ex.read()` here
            # would kill the worker thread with no response ever sent for this request at all,
            # silently hanging the client (review finding).
            detail = str(ex)
        logger.warning("session/prompt failed (HTTP %s): session=%s req_id=%s", ex.code, session.session_id, req_id)
        _send_error(req_id, -32000, f"Prism server responded {ex.code}: {detail}")
    except Exception as ex:
        if not answered:
            session.messages[:] = history_snapshot
        logger.warning("session/prompt failed: session=%s req_id=%s error=%s", session.session_id, req_id, ex)
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
        logger.debug("session/cancel: session=%s", session_id)
        session.cancel_event.set()


def _dispatch(req: Dict[str, Any]) -> None:
    """Routes one JSON-RPC message.

    Handles three message kinds:
    (a) request (top-level id and method) -> handled by the existing handle_* methods;
    (b) response (top-level id, no method) -> completes the matching future in _pending or raises AcpClientError;
    (c) notification (no id, has method) -> handled by existing notification paths (no response frame sent).

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
    method = req.get("method") if isinstance(req, dict) else None

    # (b) response frame: id present, no method
    if req_id is not None and method is None:
        with _pending_lock:
            pending = _pending.pop(req_id, None)
        if pending is not None:
            event, slot = pending
            if "error" in req:
                err = req["error"]
                if isinstance(err, dict):
                    code = err.get("code", -32603)
                    message = err.get("message", "Unknown error")
                    data = err.get("data")
                else:
                    code = -32603
                    message = str(err)
                    data = None
                slot[1] = AcpClientError(code=code, message=message, data=data)
            elif "result" in req:
                slot[0] = req.get("result")
            else:
                slot[1] = AcpClientError(code=-32600, message="Invalid JSON-RPC response: neither 'result' nor 'error' present")
            event.set()
        else:
            logger.debug("received response for unknown or already completed id: %s", req_id)
        return

    try:
        params = req.get("params")
        params = params if params is not None else {}
        if method in ("initialize", "session/new", "session/prompt", "session/cancel") \
                and not isinstance(params, dict):
            raise AcpRequestError(-32602, f"params must be an object for method {method!r}")

        logger.debug("dispatch: method=%s id=%s", method, req_id)
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
        logger.debug("dispatch rejected: method=%s id=%s code=%s message=%s", method, req_id, ex.code, ex.message)
        if req_id is not None:
            _send_error(req_id, ex.code, ex.message)
    except Exception as ex:
        logger.warning("internal error handling method=%s id=%s: %s", method, req_id, ex)
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
        logger.warning("dropped one malformed request: %s", ex)
        sys.stderr.write(f"prism acp: dropped one malformed request ({ex}).\n")


def run_acp_server() -> None:
    """Main stdio JSON-RPC 2.0 loop."""
    sys.stderr.write("💎 Prism ACP Server starting on stdio...\n")
    sys.stderr.flush()
    logger.info("prism acp starting on stdio")

    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            _handle_line(line)
    finally:
        _close_pending()
        with _sessions_lock:
            _sessions.clear()
        _client_fs_capabilities["readTextFile"] = False
        _client_fs_capabilities["writeTextFile"] = False


if __name__ == "__main__":
    run_acp_server()
