"""
prism.server: Standalone OpenAI-Compatible REST Server.
Provides predictable static port serving (default: 5272), SSE streaming,
and multi-engine routing between ONNX Runtime GenAI and Ollama.
"""

import base64
import contextlib
import hmac
import http.server
import json
import logging
import os
import socketserver
import struct
import threading
import time
import urllib.error
import uuid
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

from prism import PRISM_BANNER
from prism.catalog import AmbiguousModelError, ModelCatalog, planned_device
from prism.engine import Engine, OnnxGenAiEngine
from prism.ollama_bridge import embed_ollama, stream_ollama_chat
from prism.telemetry import get_gpu_info
from prism.templates import flatten_content, render_prompt, supports_tools
from prism.tools import arguments_as_objects, parse_tool_calls, to_openai_tool_calls

logger = logging.getLogger("prism.server")

MAX_BODY_BYTES = 10 * 1024 * 1024
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


DEFAULT_QUEUE_TIMEOUT_SEC = 300.0


def default_queue_timeout() -> Optional[float]:
    """Seconds a request may wait for the engine: $PRISM_QUEUE_TIMEOUT (`0` = wait forever), default 300."""
    raw = os.environ.get("PRISM_QUEUE_TIMEOUT", "").strip()
    if not raw:
        return DEFAULT_QUEUE_TIMEOUT_SEC
    try:
        value = float(raw)
    except ValueError:
        value = -1.0
    if value < 0:
        raise ValueError(f"PRISM_QUEUE_TIMEOUT must be a number of seconds >= 0 (got '{raw}')")
    return value or None


class EngineBusyError(RuntimeError):
    """The engine stayed busy with other requests for longer than the queue timeout."""


class ActiveEngineManager:
    """
    Owns the single active ONNX engine. A lock serializes model swaps and generation:
    ORT-GenAI engines are not safe to share across threads, and unloading a model
    while another request is mid-generation would crash. A request that cannot get the lock
    within `queue_timeout` seconds (None: wait as long as it takes) fails with EngineBusyError.
    """

    def __init__(
        self,
        catalog: Optional[ModelCatalog] = None,
        engine_factory: Callable[[str], Engine] = OnnxGenAiEngine,
        queue_timeout: Optional[float] = None,
    ):
        self.catalog = catalog or ModelCatalog()
        self.engine_factory = engine_factory
        self.queue_timeout = queue_timeout
        self.current_model_id: Optional[str] = None
        self.engine: Optional[Engine] = None
        self.lock = threading.Lock()

    @contextlib.contextmanager
    def use_engine(self, resolved: Dict[str, Any]) -> Iterator[Engine]:
        """Holds the lock for the duration of the block and yields a loaded engine."""
        if not self.lock.acquire(timeout=self.queue_timeout if self.queue_timeout else -1):
            raise EngineBusyError(f"the model has been busy with other requests for more than {self.queue_timeout:g} s")
        try:
            if self.engine is None or self.current_model_id != resolved["id"]:
                if self.engine is not None:
                    self.engine.unload()
                    self.engine = None
                    self.current_model_id = None
                self.engine = self.engine_factory(resolved["path"])
                self.current_model_id = resolved["id"]
            yield self.engine
        finally:
            self.lock.release()

    def unload(self) -> None:
        with self.lock:
            if self.engine is not None:
                self.engine.unload()
                self.engine = None
                self.current_model_id = None


ENGINE_MANAGER = ActiveEngineManager()


class ApiError(Exception):
    def __init__(self, status: int, message: str, err_type: str = "invalid_request_error", code: Optional[str] = None,
                 headers: Optional[Dict[str, str]] = None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.err_type = err_type
        self.code = code
        self.headers = headers or {}


class PrismHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, addr, handler, manager: ActiveEngineManager, api_key: Optional[str], cors_origins: Sequence[str]):
        super().__init__(addr, handler)
        self.manager = manager
        self.api_key = api_key
        self.cors_origins = list(cors_origins)
        self.bound_host = addr[0]


def _include_usage(req: Dict[str, Any]) -> bool:
    """Whether the request asked for token usage in the last streamed chunk (`stream_options.include_usage`)."""
    options = req.get("stream_options")
    return isinstance(options, dict) and bool(options.get("include_usage"))


def _int_param(req: Dict[str, Any], keys: Sequence[str], default: int) -> int:
    for k in keys:
        v = req.get(k)
        if v is not None:
            try:
                n = int(v)
            except (TypeError, ValueError):
                raise ApiError(400, f"'{k}' must be an integer")
            if n < 1:
                raise ApiError(400, f"'{k}' must be >= 1")
            return n
    return default


def _float_param(req: Dict[str, Any], key: str, default: float) -> float:
    v = req.get(key)
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        raise ApiError(400, f"'{key}' must be a number")


def _sampling_params(req: Dict[str, Any], resolved: Dict[str, Any]) -> Dict[str, Any]:
    """The sampling parameters beyond temperature/top_p, in the form the backend takes: keyword arguments for the ONNX engine
    (`top_k`, `repetition_penalty`), Ollama options otherwise. ONNX Runtime GenAI has no frequency/presence penalty, so a non-zero one is
    refused there instead of being silently ignored."""
    top_k = req.get("top_k")
    if top_k is not None:
        top_k = _int_param(req, ("top_k",), 0)
    repetition = req.get("repetition_penalty")
    if repetition is not None:
        repetition = _float_param(req, "repetition_penalty", 1.0)
        if repetition <= 0:
            raise ApiError(400, "'repetition_penalty' must be > 0")
    penalties = {}
    for key in ("frequency_penalty", "presence_penalty"):
        value = _float_param(req, key, 0.0)
        if not -2.0 <= value <= 2.0:
            raise ApiError(400, f"'{key}' must be between -2 and 2")
        penalties[key] = value
    if resolved.get("backend") == "ollama":
        options: Dict[str, Any] = {k: v for k, v in penalties.items() if v}
        if top_k is not None:
            options["top_k"] = top_k
        if repetition is not None:
            options["repeat_penalty"] = repetition
        return options
    unsupported = [k for k, v in penalties.items() if v]
    if unsupported:
        raise ApiError(400, f"'{unsupported[0]}' is not supported by ONNX Runtime GenAI models; use 'repetition_penalty' "
                            "(a multiplier, 1.0 = off; values above about 1.05 can degrade the output) or an Ollama model",
                       code="unsupported_parameter")
    kwargs: Dict[str, Any] = {}
    if top_k is not None:
        kwargs["top_k"] = top_k
    if repetition is not None:
        kwargs["repetition_penalty"] = repetition
    return kwargs


MAX_STOP_SEQUENCES = 4  # OpenAI's limit
MAX_EMBEDDING_INPUTS = 256


def _stop_param(req: Dict[str, Any]) -> List[str]:
    """`stop`: a string or a list of up to four non-empty strings."""
    v = req.get("stop")
    if v is None:
        return []
    stops = [v] if isinstance(v, str) else v
    if (not isinstance(stops, list) or len(stops) > MAX_STOP_SEQUENCES
            or not all(isinstance(x, str) and x for x in stops)):
        raise ApiError(400, f"'stop' must be a non-empty string or a list of up to {MAX_STOP_SEQUENCES} non-empty strings")
    return stops


def _tools_param(req: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """`tools` as OpenAI sends them (`[{"type": "function", "function": {"name", ...}}]`), or None when absent or when `tool_choice` is "none".
    Other `tool_choice` values are treated as "auto": Prism cannot force a call."""
    tools = req.get("tools")
    if tools is None or tools == []:
        return None
    ok = isinstance(tools, list) and all(
        isinstance(t, dict) and t.get("type") == "function" and isinstance(t.get("function"), dict)
        and isinstance(t["function"].get("name"), str) and t["function"]["name"] for t in tools)
    if not ok:
        raise ApiError(400, "'tools' must be a list of {\"type\": \"function\", \"function\": {\"name\": ...}} objects")
    return None if req.get("tool_choice") == "none" else tools


def _ollama_error_detail(err: "urllib.error.HTTPError") -> str:
    """Ollama's own `{"error": "..."}` message from an HTTP error reply, else the status line."""
    try:
        message = json.loads(err.read().decode("utf-8", "replace")).get("error")
    except Exception:
        message = None
    return message if isinstance(message, str) and message else str(err)


def _ollama_message(m: Dict[str, Any]) -> Dict[str, Any]:
    """An OpenAI message as Ollama wants it: plain-text content, tool calls with object arguments."""
    out: Dict[str, Any] = {"role": m.get("role", "user"), "content": flatten_content(m.get("content"))}
    calls = m.get("tool_calls")
    if isinstance(calls, list):
        out["tool_calls"] = [{"function": {"name": c["function"]["name"], "arguments": c["function"].get("arguments") or {}}}
                             for c in calls if isinstance(c, dict) and isinstance(c.get("function"), dict) and c["function"].get("name")]
    if out["role"] == "tool" and isinstance(m.get("name"), str):
        out["tool_name"] = m["name"]
    return out


class StopMatcher:
    """Cuts generated text at the first stop sequence.

    Text that might still turn out to be the start of a stop sequence is held back until the next piece settles it, so a stop
    split across tokens never leaks into the output. With no stop sequences it passes everything straight through.
    """

    def __init__(self, stops: Sequence[str]):
        self.stops = list(stops)
        self.held = ""
        self.hit = False

    def feed(self, piece: str) -> str:
        """Returns the part of the text that is now safe to send."""
        if self.hit:
            return ""
        self.held += piece
        cuts = [i for i in (self.held.find(s) for s in self.stops) if i >= 0]
        if cuts:
            out, self.held, self.hit = self.held[:min(cuts)], "", True
            return out
        keep = 0
        for s in self.stops:
            for n in range(min(len(s) - 1, len(self.held)), keep, -1):
                if self.held.endswith(s[:n]):
                    keep = n
                    break
        out, self.held = self.held[:len(self.held) - keep], self.held[len(self.held) - keep:]
        return out

    def flush(self) -> str:
        """The held-back tail, once generation ended without a stop sequence."""
        out, self.held = self.held, ""
        return out


def _stopped_pieces(gen: Iterator[Any], stop: Sequence[str], stats: Dict[str, Any]) -> Iterator[str]:
    """Text pieces of an engine stream with stop sequences applied. Fills `stats` (tokens, ttft, rate, stopped) as it goes;
    when a stop sequence matches it ends early and leaves closing `gen` (which aborts generation) to the caller."""
    matcher = StopMatcher(stop)
    started = time.perf_counter()
    for token_text, _, tok_per_sec in gen:
        if stats["ttft"] is None:
            stats["ttft"] = time.perf_counter() - started
        stats["tokens"] += 1
        stats["rate"] = tok_per_sec
        out = matcher.feed(token_text)
        if out:
            yield out
        if matcher.hit:
            stats["stopped"] = True
            return
    tail = matcher.flush()
    if tail:
        yield tail


def _new_stats() -> Dict[str, Any]:
    return {"tokens": 0, "ttft": None, "rate": 0.0, "stopped": False}


class OpenAIApiHandler(http.server.BaseHTTPRequestHandler):
    server: PrismHTTPServer

    # ------------------------------------------------------------------ plumbing

    @property
    def manager(self) -> ActiveEngineManager:
        return self.server.manager

    def _cors_allowed_origin(self) -> Optional[str]:
        origin = self.headers.get("Origin")
        allowed = self.server.cors_origins
        if origin and ("*" in allowed or origin in allowed):
            return "*" if "*" in allowed else origin
        return None

    def _send_cors_headers(self):
        origin = self._cors_allowed_origin()
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.send_header("Vary", "Origin")

    def _send_json(self, status: int, obj: Any, headers: Optional[Dict[str, str]] = None):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self._send_cors_headers()
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_api_error(self, err: ApiError):
        self._send_json(err.status, {
            "error": {"message": err.message, "type": err.err_type, "param": None, "code": err.code}
        }, err.headers)

    def _check_host(self):
        """Rejects foreign Host headers on loopback binds (DNS-rebinding defence)."""
        if self.server.bound_host not in LOOPBACK_HOSTS:
            return
        host = (self.headers.get("Host") or "").strip()
        hostname = host.rsplit(":", 1)[0].strip("[]") if not host.startswith("[") else host.split("]")[0].strip("[")
        if hostname not in LOOPBACK_HOSTS:
            raise ApiError(403, f"Host '{host}' is not allowed", "forbidden", "host_not_allowed")

    def _check_auth(self):
        key = self.server.api_key
        if not key:
            return
        header = self.headers.get("Authorization", "")
        supplied = header[7:] if header.lower().startswith("bearer ") else ""
        if not hmac.compare_digest(supplied.encode("utf-8"), key.encode("utf-8")):
            raise ApiError(401, "Invalid or missing API key", "authentication_error", "invalid_api_key")

    def _read_json_body(self) -> Dict[str, Any]:
        raw_len = self.headers.get("Content-Length")
        try:
            length = int(raw_len) if raw_len is not None else 0
        except ValueError:
            raise ApiError(400, "Invalid Content-Length")
        if length <= 0:
            raise ApiError(400, "Request body is required")
        if length > MAX_BODY_BYTES:
            raise ApiError(413, f"Request body exceeds {MAX_BODY_BYTES} bytes", code="body_too_large")
        try:
            req = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            raise ApiError(400, "Invalid JSON payload")
        if not isinstance(req, dict):
            raise ApiError(400, "JSON body must be an object")
        return req

    def _dispatch(self, routes: Dict[str, Callable[[], None]], public: Sequence[str] = ()):
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        try:
            self._check_host()
            handler = routes.get(path)
            if handler is None:
                raise ApiError(404, f"Unknown route {path}", "not_found_error", "not_found")
            if path not in public:
                self._check_auth()
            handler()
        except ApiError as err:
            self._send_api_error(err)
        except (BrokenPipeError, ConnectionResetError):
            logger.debug("client disconnected")
        except Exception as ex:  # last resort: never drop the connection without a reply
            logger.exception("unhandled error serving %s", path)
            try:
                self._send_api_error(ApiError(500, f"Internal error: {ex}", "server_error"))
            except Exception:
                pass

    # ------------------------------------------------------------------ routes

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        self._dispatch(
            {
                "/v1/models": self._handle_list_models,
                "/health": self._handle_health,
                "/v1/health": self._handle_health,
                "/v1/status": self._handle_health,
            },
            public=("/health", "/v1/health", "/v1/status"),
        )

    def do_POST(self):
        self._dispatch({
            "/v1/chat/completions": self._handle_chat_completions,
            "/v1/completions": self._handle_completions,
            "/v1/embeddings": self._handle_embeddings,
        })

    def _handle_list_models(self):
        models = self.manager.catalog.list_all_models(include_ollama=True)
        data = [
            {
                "id": m["id"],
                "object": "model",
                "created": int(time.time()),
                "owned_by": m.get("engine", "prism"),
                "size_mb": m.get("size_mb", 0),
                "device": planned_device(m),  # where it will run, not what the files were exported for
                **({"exported_for": m.get("device")} if m.get("backend") == "onnx" else {}),
            }
            for m in models
        ]
        self._send_json(200, {"object": "list", "data": data})

    def _handle_health(self):
        self._send_json(200, {
            "status": "ok",
            "active_model": self.manager.current_model_id,
            "active_device": getattr(self.manager.engine, "device", None),
            "hardware": get_gpu_info(),
        })

    # ------------------------------------------------------------------ inference

    def _resolve(self, model_id: Any) -> Dict[str, Any]:
        if not isinstance(model_id, str) or not model_id:
            raise ApiError(400, "'model' is required")
        try:
            resolved = self.manager.catalog.resolve_model(model_id)
        except AmbiguousModelError as ex:
            raise ApiError(400, str(ex), code="ambiguous_model")
        if not resolved:
            raise ApiError(404, f"Model '{model_id}' not found", "not_found_error", "model_not_found")
        return resolved

    def _handle_chat_completions(self):
        req = self._read_json_body()
        messages = req.get("messages")
        if not isinstance(messages, list) or not messages or not all(isinstance(m, dict) for m in messages):
            raise ApiError(400, "'messages' must be a non-empty list of message objects")
        model_id = req.get("model")
        tools = _tools_param(req)
        resolved = self._resolve(model_id)
        if tools and resolved.get("backend") == "onnx" and not supports_tools(resolved):
            raise ApiError(400, f"Model '{model_id}' cannot take tools: Prism needs the model's own chat template with tool support "
                                "and the 'jinja' extra (pip install 'prism-local[jinja]'). Ollama models handle tools themselves.",
                           code="tools_not_supported")
        self._generate(
            resolved=resolved,
            model_id=model_id,
            messages=arguments_as_objects(messages),
            stream=bool(req.get("stream", False)),
            max_tokens=_int_param(req, ("max_completion_tokens", "max_tokens"), 512),
            temperature=_float_param(req, "temperature", 0.1),
            top_p=_float_param(req, "top_p", 0.9),
            chat=True,
            include_usage=_include_usage(req),
            stop=_stop_param(req),
            tools=tools,
            sampling=_sampling_params(req, resolved),
        )

    def _handle_embeddings(self):
        """OpenAI's /v1/embeddings, served by Ollama: ONNX Runtime GenAI does not produce embeddings."""
        req = self._read_json_body()
        inputs = req.get("input")
        if isinstance(inputs, str):
            inputs = [inputs]
        if (not isinstance(inputs, list) or not inputs or len(inputs) > MAX_EMBEDDING_INPUTS
                or not all(isinstance(x, str) and x for x in inputs)):
            raise ApiError(400, f"'input' must be a non-empty string or a list of 1 to {MAX_EMBEDDING_INPUTS} non-empty strings "
                                "(token arrays are not supported)")
        encoding = req.get("encoding_format", "float")
        if encoding not in ("float", "base64"):
            raise ApiError(400, "'encoding_format' must be 'float' or 'base64'")
        if req.get("dimensions") is not None:
            raise ApiError(400, "'dimensions' is not supported", code="dimensions_not_supported")
        model_id = req.get("model")
        resolved = self._resolve(model_id)
        if resolved.get("backend") != "ollama":
            raise ApiError(400, f"Model '{model_id}' cannot produce embeddings: ONNX Runtime GenAI does not support them. "
                                "Use an Ollama embedding model, e.g. 'ollama:nomic-embed-text'.", code="embeddings_not_supported")
        try:
            vectors, prompt_tokens = embed_ollama(resolved["id"], inputs)
        except urllib.error.HTTPError as ex:
            detail = _ollama_error_detail(ex)
            if ex.code == 404:
                raise ApiError(404, f"Model '{model_id}' not found in Ollama", "not_found_error", "model_not_found")
            if ex.code in (400, 501):  # the daemon refuses: the model is not an embedding model
                raise ApiError(400, f"Ollama cannot produce embeddings with '{model_id}': {detail}. "
                                    "Use an embedding model such as 'ollama:nomic-embed-text'.", code="embeddings_not_supported")
            raise ApiError(502, f"Ollama backend error: {detail}", "server_error", "backend_unavailable")
        except Exception as ex:
            raise ApiError(502, f"Ollama backend error: {ex}", "server_error", "backend_unavailable")
        data = [{"object": "embedding", "index": i,
                 "embedding": base64.b64encode(struct.pack(f"<{len(v)}f", *v)).decode("ascii") if encoding == "base64" else v}
                for i, v in enumerate(vectors)]
        self._send_json(200, {"object": "list", "data": data, "model": model_id,
                              "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens}})

    def _handle_completions(self):
        req = self._read_json_body()
        prompt = req.get("prompt", "")
        if isinstance(prompt, list):
            prompt = prompt[0] if prompt else ""
        if not isinstance(prompt, str) or not prompt:
            raise ApiError(400, "'prompt' must be a non-empty string")
        model_id = req.get("model")
        resolved = self._resolve(model_id)
        self._generate(
            resolved=resolved,
            model_id=model_id,
            messages=[{"role": "user", "content": prompt}],
            stream=bool(req.get("stream", False)),
            max_tokens=_int_param(req, ("max_tokens",), 512),
            temperature=_float_param(req, "temperature", 0.1),
            top_p=_float_param(req, "top_p", 0.9),
            chat=False,
            raw_prompt=prompt,
            include_usage=_include_usage(req),
            stop=_stop_param(req),
            sampling=_sampling_params(req, resolved),
        )

    def _generate(self, resolved, model_id, messages, stream, max_tokens, temperature, top_p, chat, raw_prompt=None,
                  include_usage=False, stop=(), tools=None, sampling=None):
        sampling = sampling or {}
        obj = "chat.completion" if chat else "text_completion"
        cmpl_id = f"{'chatcmpl' if chat else 'cmpl'}-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        def chunk(delta_text: Optional[str], finish: Optional[str], role: bool = False,
                  tool_calls: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
            if chat:
                delta: Dict[str, Any] = {}
                if role:
                    delta["role"] = "assistant"
                if delta_text is not None:
                    delta["content"] = delta_text
                if tool_calls:
                    delta["tool_calls"] = tool_calls
                choice = {"index": 0, "delta": delta, "finish_reason": finish}
            else:
                choice = {"index": 0, "text": delta_text or "", "finish_reason": finish}
            return {"id": cmpl_id, "object": f"{obj}.chunk" if chat else obj, "created": created,
                    "model": model_id, "choices": [choice]}

        def final(text: str, finish: str, prompt_tokens: Optional[int], completion_tokens: Optional[int],
                  telemetry: Optional[Dict[str, Any]] = None,
                  tool_calls: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
            if chat and tool_calls:
                message = {"role": "assistant", "content": text or None, "tool_calls": tool_calls}
                choice = {"index": 0, "message": message, "finish_reason": finish}
            elif chat:
                choice = {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}
            else:
                choice = {"index": 0, "text": text, "finish_reason": finish}
            body: Dict[str, Any] = {"id": cmpl_id, "object": obj, "created": created, "model": model_id,
                                    "choices": [choice]}
            if prompt_tokens is not None and completion_tokens is not None:
                body["usage"] = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                                 "total_tokens": prompt_tokens + completion_tokens}
            if telemetry:
                body["telemetry"] = telemetry
            return body

        def usage_chunk(prompt_tokens: int, completion_tokens: int,
                        telemetry: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            # OpenAI's stream_options.include_usage: one last chunk with no choices, carrying the token counts. The
            # `telemetry` block (device, timings) is the same one non-streaming responses carry.
            body: Dict[str, Any] = {
                "id": cmpl_id, "object": f"{obj}.chunk" if chat else obj, "created": created, "model": model_id,
                "choices": [], "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                                         "total_tokens": prompt_tokens + completion_tokens}}
            if telemetry:
                body["telemetry"] = telemetry
            return body

        if resolved.get("backend") == "ollama":
            self._generate_ollama(resolved, messages, stream, max_tokens, temperature, top_p, stop, include_usage,
                                  chunk, final, usage_chunk, tools, sampling)
            return

        def resolve_tools(text: str, finish: str):
            """(text, finish_reason, tool_calls): when the model called a tool, the calls are split out of the text."""
            if not tools:
                return text, finish, None
            content, calls = parse_tool_calls(text)
            return (content, "tool_calls", to_openai_tool_calls(calls)) if calls else (text, finish, None)

        prompt = raw_prompt if raw_prompt is not None else render_prompt(resolved, messages, tools)
        stack = contextlib.ExitStack()
        try:
            engine = stack.enter_context(self.manager.use_engine(resolved))
        except EngineBusyError as ex:
            raise ApiError(503, f"Server busy: {ex}", "server_error", "server_busy", {"Retry-After": "30"})
        except Exception as ex:
            logger.exception("failed to load model %s", resolved.get("id"))
            raise ApiError(500, f"Failed to load model '{model_id}': {ex}", "server_error", "model_load_failed")

        with stack:
            prompt_tokens = engine.count_tokens(prompt)
            window = resolved.get("context_length")
            if window:
                if prompt_tokens >= window:
                    raise ApiError(400, f"The prompt is {prompt_tokens} tokens; the model's context window is {window}",
                                   code="context_length_exceeded")
                max_tokens = min(max_tokens, window - prompt_tokens)  # the room left; hitting it reports finish_reason "length"
            if not stream and not stop:
                result = engine.generate(prompt=prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, **sampling)
                text, finish, calls = resolve_tools(result["text"], result.get("finish_reason", "stop"))
                self._send_json(200, final(
                    text, finish, prompt_tokens, result["tokens_generated"],
                    {"ttft_sec": result["ttft_sec"], "decode_tok_per_sec": result["decode_tok_per_sec"],
                     "device": result.get("device")}, calls,
                ))
                return

            gen = engine.stream_generate(prompt=prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, **sampling)
            stats = _new_stats()
            pieces = _stopped_pieces(gen, stop, stats)

            def telemetry() -> Dict[str, Any]:
                return {"ttft_sec": round(stats["ttft"] or 0.0, 4), "decode_tok_per_sec": round(stats["rate"], 1),
                        "device": getattr(engine, "device", None)}

            if not stream:
                try:
                    text = "".join(pieces)
                finally:
                    gen.close()
                text, finish, calls = resolve_tools(text, "stop" if stats["stopped"] else engine.last_finish_reason)
                self._send_json(200, final(text, finish, prompt_tokens, stats["tokens"], telemetry(), calls))
                return

            self._begin_sse()
            try:
                if chat:
                    self._sse(chunk(None, None, role=True))
                if tools:
                    # A call can only be recognised in the finished text, so with tools the reply is not streamed token by token.
                    text = "".join(pieces)
                    text, finish, calls = resolve_tools(text, "stop" if stats["stopped"] else engine.last_finish_reason)
                    if text:
                        self._sse(chunk(text, None))
                    if calls:
                        self._sse(chunk(None, None, tool_calls=[{"index": i, **c} for i, c in enumerate(calls)]))
                    self._sse(chunk(None, finish))
                else:
                    for piece in pieces:
                        self._sse(chunk(piece, None))
                    self._sse(chunk(None, "stop" if stats["stopped"] else engine.last_finish_reason))
                if include_usage:
                    self._sse(usage_chunk(prompt_tokens, stats["tokens"], telemetry()))
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                logger.debug("client disconnected mid-stream; aborting generation")
            except Exception as ex:
                logger.exception("generation failed")
                self._sse_error(f"Generation failed: {ex}")
            finally:
                gen.close()

    def _generate_ollama(self, resolved, messages, stream, max_tokens, temperature, top_p, stop, include_usage,
                         chunk, final, usage_chunk, tools=None, sampling=None):
        options: Dict[str, Any] = {"num_predict": max_tokens, "temperature": temperature, "top_p": top_p, **(sampling or {})}
        if stop:
            options["stop"] = list(stop)  # Ollama applies stop sequences itself
        text_messages = [_ollama_message(m) for m in messages]  # Ollama wants plain-text content and object arguments
        stats: Dict[str, Any] = {}
        it = stream_ollama_chat(resolved["id"], text_messages, options, stats=stats, tools=tools)
        try:
            first = next(it, None)  # surface connection errors before any headers are sent
        except Exception as ex:
            raise ApiError(502, f"Ollama backend error: {ex}", "server_error", "backend_unavailable")

        def chunks() -> Iterator[str]:
            if first is not None:
                yield first
            yield from it

        if not stream:
            try:
                text = "".join(chunks())
            except Exception as ex:
                raise ApiError(502, f"Ollama backend error: {ex}", "server_error", "backend_unavailable")
            calls = to_openai_tool_calls(stats["tool_calls"]) if stats.get("tool_calls") else None
            self._send_json(200, final(text, "tool_calls" if calls else stats.get("finish_reason", "stop"),
                                       stats.get("prompt_tokens"), stats.get("completion_tokens"), None, calls))
            return

        self._begin_sse()
        try:
            self._sse(chunk(None, None, role=True))
            for piece in chunks():
                self._sse(chunk(piece, None))
            calls = to_openai_tool_calls(stats["tool_calls"], with_index=True) if stats.get("tool_calls") else None
            if calls:
                self._sse(chunk(None, None, tool_calls=calls))
            self._sse(chunk(None, "tool_calls" if calls else stats.get("finish_reason", "stop")))
            if include_usage and "prompt_tokens" in stats and "completion_tokens" in stats:
                self._sse(usage_chunk(stats["prompt_tokens"], stats["completion_tokens"]))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            logger.debug("client disconnected mid-stream")
        except Exception as ex:
            self._sse_error(f"Ollama backend error: {ex}")
        finally:
            close = getattr(it, "close", None)
            if close:
                close()

    def _begin_sse(self):
        self.send_response(200)
        self._send_cors_headers()
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def _sse(self, payload: Dict[str, Any]):
        self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode("utf-8"))
        self.wfile.flush()

    def _sse_error(self, message: str):
        try:
            self._sse({"error": {"message": message, "type": "server_error", "code": None}})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, format, *args):
        logger.debug("%s - %s", self.address_string(), format % args)


def create_server(
    port: int = 5272,
    host: str = "127.0.0.1",
    api_key: Optional[str] = None,
    cors_origins: Sequence[str] = (),
    manager: Optional[ActiveEngineManager] = None,
) -> PrismHTTPServer:
    """Builds (but does not start) the HTTP server. Use port=0 for an ephemeral port."""
    return PrismHTTPServer((host, port), OpenAIApiHandler, manager or ENGINE_MANAGER, api_key, cors_origins)


def start_server(
    port: int = 5272,
    host: str = "127.0.0.1",
    api_key: Optional[str] = None,
    cors_origins: Sequence[str] = (),
    queue_timeout: Optional[float] = None,
):
    """Launches the multi-threaded OpenAI REST server. `queue_timeout`: seconds a request may wait for the engine (None: forever)."""
    server = create_server(port=port, host=host, api_key=api_key, cors_origins=cors_origins)
    server.manager.queue_timeout = queue_timeout
    with server:
        print(f"\n{PRISM_BANNER}\n")
        print(f"🚀 prism OpenAI Server active at http://{host}:{server.server_address[1]}/v1")
        print("   Listening for chat completions and models list.")
        if host not in LOOPBACK_HOSTS and not api_key:
            print("   ⚠️  Bound to a non-loopback interface without --api-key: anyone on the network can use this server.")
        print("   Press Ctrl+C to terminate.")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
        finally:
            server.manager.unload()
