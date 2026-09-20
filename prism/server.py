"""
prism.server: Standalone OpenAI-Compatible REST Server.
Provides predictable static port serving (default: 5272), SSE streaming,
and multi-engine routing between ONNX Runtime GenAI and Ollama.
"""

import contextlib
import hmac
import http.server
import json
import logging
import socketserver
import threading
import time
import uuid
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

from prism import PRISM_BANNER
from prism.catalog import AmbiguousModelError, ModelCatalog, planned_device
from prism.engine import Engine, OnnxGenAiEngine, format_prompt
from prism.ollama_bridge import stream_ollama_chat
from prism.telemetry import get_gpu_info

logger = logging.getLogger("prism.server")

MAX_BODY_BYTES = 10 * 1024 * 1024
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


class ActiveEngineManager:
    """
    Owns the single active ONNX engine. A lock serializes model swaps and generation:
    ORT-GenAI engines are not safe to share across threads, and unloading a model
    while another request is mid-generation would crash.
    """

    def __init__(
        self,
        catalog: Optional[ModelCatalog] = None,
        engine_factory: Callable[[str], Engine] = OnnxGenAiEngine,
    ):
        self.catalog = catalog or ModelCatalog()
        self.engine_factory = engine_factory
        self.current_model_id: Optional[str] = None
        self.engine: Optional[Engine] = None
        self.lock = threading.Lock()

    @contextlib.contextmanager
    def use_engine(self, resolved: Dict[str, Any]) -> Iterator[Engine]:
        """Holds the lock for the duration of the block and yields a loaded engine."""
        with self.lock:
            if self.engine is None or self.current_model_id != resolved["id"]:
                if self.engine is not None:
                    self.engine.unload()
                    self.engine = None
                    self.current_model_id = None
                self.engine = self.engine_factory(resolved["path"])
                self.current_model_id = resolved["id"]
            yield self.engine

    def unload(self) -> None:
        with self.lock:
            if self.engine is not None:
                self.engine.unload()
                self.engine = None
                self.current_model_id = None


ENGINE_MANAGER = ActiveEngineManager()


class ApiError(Exception):
    def __init__(self, status: int, message: str, err_type: str = "invalid_request_error", code: Optional[str] = None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.err_type = err_type
        self.code = code


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

    def _send_json(self, status: int, obj: Any):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_api_error(self, err: ApiError):
        self._send_json(err.status, {
            "error": {"message": err.message, "type": err.err_type, "param": None, "code": err.code}
        })

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
        resolved = self._resolve(model_id)
        self._generate(
            resolved=resolved,
            model_id=model_id,
            messages=messages,
            stream=bool(req.get("stream", False)),
            max_tokens=_int_param(req, ("max_completion_tokens", "max_tokens"), 512),
            temperature=_float_param(req, "temperature", 0.1),
            top_p=_float_param(req, "top_p", 0.9),
            chat=True,
            include_usage=_include_usage(req),
        )

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
        )

    def _generate(self, resolved, model_id, messages, stream, max_tokens, temperature, top_p, chat, raw_prompt=None,
                  include_usage=False):
        obj = "chat.completion" if chat else "text_completion"
        cmpl_id = f"{'chatcmpl' if chat else 'cmpl'}-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        def chunk(delta_text: Optional[str], finish: Optional[str], role: bool = False) -> Dict[str, Any]:
            if chat:
                delta: Dict[str, Any] = {}
                if role:
                    delta["role"] = "assistant"
                if delta_text is not None:
                    delta["content"] = delta_text
                choice = {"index": 0, "delta": delta, "finish_reason": finish}
            else:
                choice = {"index": 0, "text": delta_text or "", "finish_reason": finish}
            return {"id": cmpl_id, "object": f"{obj}.chunk" if chat else obj, "created": created,
                    "model": model_id, "choices": [choice]}

        def final(text: str, finish: str, prompt_tokens: Optional[int], completion_tokens: Optional[int],
                  telemetry: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            if chat:
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

        if resolved.get("backend") == "ollama":
            self._generate_ollama(resolved, messages, stream, max_tokens, temperature, top_p, chunk, final)
            return

        prompt = raw_prompt if raw_prompt is not None else format_prompt(messages, resolved.get("template"))
        stack = contextlib.ExitStack()
        try:
            engine = stack.enter_context(self.manager.use_engine(resolved))
        except Exception as ex:
            logger.exception("failed to load model %s", resolved.get("id"))
            raise ApiError(500, f"Failed to load model '{model_id}': {ex}", "server_error", "model_load_failed")

        with stack:
            prompt_tokens = engine.count_tokens(prompt)
            if not stream:
                result = engine.generate(prompt=prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
                self._send_json(200, final(
                    result["text"], result.get("finish_reason", "stop"), prompt_tokens, result["tokens_generated"],
                    {"ttft_sec": result["ttft_sec"], "decode_tok_per_sec": result["decode_tok_per_sec"],
                     "device": result.get("device")},
                ))
                return

            gen = engine.stream_generate(prompt=prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
            self._begin_sse()
            try:
                if chat:
                    self._sse(chunk(None, None, role=True))
                started = time.perf_counter()
                tokens, ttft, rate = 0, None, 0.0
                for token_chunk, _, tok_per_sec in gen:
                    if ttft is None:
                        ttft = time.perf_counter() - started
                    tokens += 1
                    rate = tok_per_sec
                    self._sse(chunk(token_chunk, None))
                self._sse(chunk(None, engine.last_finish_reason))
                if include_usage:
                    # OpenAI's stream_options.include_usage: one last chunk with no choices, carrying the token counts. The
                    # `telemetry` block (device, timings) is the same one non-streaming responses carry.
                    self._sse({
                        "id": cmpl_id, "object": f"{obj}.chunk" if chat else obj, "created": created,
                        "model": model_id, "choices": [],
                        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": tokens,
                                  "total_tokens": prompt_tokens + tokens},
                        "telemetry": {"ttft_sec": round(ttft or 0.0, 4), "decode_tok_per_sec": round(rate, 1),
                                      "device": getattr(engine, "device", None)},
                    })
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                logger.debug("client disconnected mid-stream; aborting generation")
            except Exception as ex:
                logger.exception("generation failed")
                self._sse_error(f"Generation failed: {ex}")
            finally:
                gen.close()

    def _generate_ollama(self, resolved, messages, stream, max_tokens, temperature, top_p, chunk, final):
        options = {"num_predict": max_tokens, "temperature": temperature, "top_p": top_p}
        it = stream_ollama_chat(resolved["id"], messages, options)
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
            self._send_json(200, final(text, "stop", None, None))
            return

        self._begin_sse()
        try:
            self._sse(chunk(None, None, role=True))
            for piece in chunks():
                self._sse(chunk(piece, None))
            self._sse(chunk(None, "stop"))
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
):
    """Launches the multi-threaded OpenAI REST server."""
    server = create_server(port=port, host=host, api_key=api_key, cors_origins=cors_origins)
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
