"""
prism.server: Standalone OpenAI-Compatible REST Server.
Provides predictable static port serving (default: 5272), SSE streaming,
and multi-engine routing between ONNX Runtime GenAI and Ollama.
"""

import http.server
import json
import logging
import socketserver
import time
import uuid
from typing import Any, Dict, Optional

from prism.catalog import ModelCatalog
from prism.engine import OnnxGenAiEngine, format_prompt
from prism.ollama_bridge import stream_ollama_chat
from prism.telemetry import get_gpu_info

logger = logging.getLogger("prism.server")


class ActiveEngineManager:
    """Manages currently active model engine instances."""
    def __init__(self):
        self.catalog = ModelCatalog()
        self.current_model_id: Optional[str] = None
        self.engine: Optional[OnnxGenAiEngine] = None

    def get_or_load_engine(self, model_id: str) -> Optional[OnnxGenAiEngine]:
        resolved = self.catalog.resolve_model(model_id)
        if not resolved or resolved.get("backend") == "ollama":
            return None

        model_path = resolved["path"]
        if self.current_model_id == resolved["id"] and self.engine is not None:
            return self.engine

        # Unload previous model
        if self.engine is not None:
            self.engine.unload()
            self.engine = None

        self.engine = OnnxGenAiEngine(model_path)
        self.current_model_id = resolved["id"]
        return self.engine


ENGINE_MANAGER = ActiveEngineManager()


class OpenAIApiHandler(http.server.BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        if self.path in ["/v1/models", "/v1/models/"]:
            self._handle_list_models()
        elif self.path in ["/health", "/v1/status", "/v1/health"]:
            self._handle_health()
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        if self.path in ["/v1/chat/completions", "/v1/chat/completions/"]:
            self._handle_chat_completions()
        elif self.path in ["/v1/completions", "/v1/completions/"]:
            self._handle_completions()
        else:
            self.send_error(404, "Not Found")

    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def _handle_list_models(self):
        models = ENGINE_MANAGER.catalog.list_all_models(include_ollama=True)
        data = [
            {
                "id": m["id"],
                "object": "model",
                "created": int(time.time()),
                "owned_by": m.get("engine", "prism"),
                "size_mb": m.get("size_mb", 0),
                "device": m.get("device", "GPU"),
            }
            for m in models
        ]
        payload = json.dumps({"object": "list", "data": data}).encode("utf-8")
        self.send_response(200)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _handle_health(self):
        gpu_telemetry = get_gpu_info()
        payload = json.dumps({
            "status": "ok",
            "active_model": ENGINE_MANAGER.current_model_id,
            "hardware": gpu_telemetry,
        }, indent=2).encode("utf-8")
        self.send_response(200)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _handle_chat_completions(self):
        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length).decode("utf-8")
        try:
            req = json.loads(raw_body)
        except Exception:
            self.send_error(400, "Invalid JSON payload")
            return

        model_id = req.get("model", "")
        messages = req.get("messages", [])
        stream = req.get("stream", False)
        max_tokens = req.get("max_tokens", 512)
        temperature = req.get("temperature", 0.1)
        top_p = req.get("top_p", 0.9)

        # Route to Ollama if specified or prefixed
        if model_id.startswith("ollama:"):
            self._stream_or_send_ollama(model_id, messages, stream, max_tokens, temperature, top_p)
            return

        # Route to ONNX Engine
        engine = ENGINE_MANAGER.get_or_load_engine(model_id)
        if not engine:
            self.send_error(404, f"Model '{model_id}' not found or unsupported")
            return

        prompt = format_prompt(messages)

        if stream:
            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            created = int(time.time())

            for token_chunk, _, _ in engine.stream_generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            ):
                chunk_payload = {
                    "id": cmpl_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": token_chunk},
                            "finish_reason": None,
                        }
                    ],
                }
                self.wfile.write(f"data: {json.dumps(chunk_payload)}\n\n".encode("utf-8"))
                self.wfile.flush()

            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            result = engine.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            resp_payload = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": result["text"]},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": result["tokens_generated"],
                    "total_tokens": len(prompt.split()) + result["tokens_generated"],
                },
                "telemetry": {
                    "ttft_sec": result["ttft_sec"],
                    "decode_tok_per_sec": result["decode_tok_per_sec"],
                }
            }
            body = json.dumps(resp_payload).encode("utf-8")
            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def _stream_or_send_ollama(self, model_id, messages, stream, max_tokens, temperature, top_p):
        options = {"num_predict": max_tokens, "temperature": temperature, "top_p": top_p}
        if stream:
            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            created = int(time.time())

            for chunk in stream_ollama_chat(model_id, messages, options):
                chunk_payload = {
                    "id": cmpl_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
                }
                self.wfile.write(f"data: {json.dumps(chunk_payload)}\n\n".encode("utf-8"))
                self.wfile.flush()

            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            chunks = list(stream_ollama_chat(model_id, messages, options))
            full_text = "".join(chunks)
            resp_payload = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_id,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": full_text}, "finish_reason": "stop"}],
            }
            body = json.dumps(resp_payload).encode("utf-8")
            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def _handle_completions(self):
        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length).decode("utf-8")
        try:
            req = json.loads(raw_body)
        except Exception:
            self.send_error(400, "Invalid JSON")
            return
        prompt = req.get("prompt", "")
        model_id = req.get("model", "")
        engine = ENGINE_MANAGER.get_or_load_engine(model_id)
        if not engine:
            self.send_error(404, f"Model '{model_id}' not found")
            return
        res = engine.generate(prompt)
        payload = json.dumps({
            "id": f"cmpl-{uuid.uuid4().hex[:12]}",
            "object": "text_completion",
            "choices": [{"text": res["text"], "index": 0, "finish_reason": "stop"}],
        }).encode("utf-8")
        self.send_response(200)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass


def start_server(port: int = 5272, host: str = "127.0.0.1"):
    """Launches the multi-threaded OpenAI REST server."""
    class ThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
        allow_reuse_address = True
        daemon_threads = True

    with ThreadingTCPServer((host, port), OpenAIApiHandler) as server:
        print(f"🚀 prism OpenAI Server active at http://{host}:{port}/v1")
        print(f"   Listening for chat completions and models list.")
        print(f"   Press Ctrl+C to terminate.")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
        finally:
            if ENGINE_MANAGER.engine:
                ENGINE_MANAGER.engine.unload()
