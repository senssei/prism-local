import http.client
import json
import os
import shutil
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from prism.catalog import ModelCatalog
from prism.templates import jinja_available
from prism.server import ActiveEngineManager, StopMatcher, create_server
from tests.fakes import FakeEngine, make_model


class ServerTestBase(unittest.TestCase):
    api_key = None
    cors_origins = ()

    def setUp(self):
        FakeEngine.reset()
        self.ollama_models = []
        self.tmp = tempfile.mkdtemp()
        make_model(self.tmp, "alpha-phi-cuda-gpu", "phi3")
        make_model(self.tmp, "beta-phi-cuda-gpu", "phi3")
        make_model(self.tmp, "qwen-coder-gpu", "qwen2")
        patcher = patch("prism.catalog.list_ollama_models", side_effect=lambda: self.ollama_models)
        patcher.start()
        self.addCleanup(patcher.stop)
        catalog = ModelCatalog(search_paths=[self.tmp])
        self.manager = ActiveEngineManager(catalog=catalog, engine_factory=FakeEngine)
        self.server = create_server(port=0, host="127.0.0.1", api_key=self.api_key,
                                    cors_origins=self.cors_origins, manager=self.manager)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def request(self, method, path, body=None, headers=None, raw=False):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        hdrs = dict(headers or {})
        payload = None
        if body is not None:
            payload = body if isinstance(body, (bytes, str)) else json.dumps(body)
            hdrs.setdefault("Content-Type", "application/json")
        conn.request(method, path, body=payload, headers=hdrs)
        resp = conn.getresponse()
        data = resp.read()
        conn.close()
        return resp, (data if raw else (json.loads(data) if data else None))

    def chat(self, model="qwen-coder-gpu", **extra):
        body = {"model": model, "messages": [{"role": "user", "content": "hi"}], **extra}
        return self.request("POST", "/v1/chat/completions", body)

    def chat_body(self, **extra):
        return {"model": "qwen-coder-gpu", "stream": True, "messages": [{"role": "user", "content": "hi there"}], **extra}

    def stream(self, path, body):
        resp, raw = self.request("POST", path, body, raw=True)
        self.assertEqual(resp.status, 200)
        events = [e[len("data: "):] for e in raw.decode().strip().split("\n\n")]
        self.assertEqual(events[-1], "[DONE]")
        return [json.loads(e) for e in events[:-1]]


class TestChatCompletions(ServerTestBase):
    def test_non_stream_response_shape_and_usage(self):
        resp, data = self.chat()
        self.assertEqual(resp.status, 200)
        self.assertEqual(data["choices"][0]["message"]["content"], "Hello world")
        self.assertEqual(data["choices"][0]["finish_reason"], "stop")
        self.assertGreater(data["usage"]["prompt_tokens"], 0)
        self.assertEqual(data["usage"]["total_tokens"],
                         data["usage"]["prompt_tokens"] + data["usage"]["completion_tokens"])

    def test_finish_reason_length(self):
        _, data = self.chat(max_tokens=2)
        self.assertEqual(data["choices"][0]["finish_reason"], "length")

    def test_template_follows_model_family(self):
        self.chat(model="qwen-coder-gpu")
        self.assertIn("<|im_start|>user", FakeEngine.prompts[-1])
        self.chat(model="alpha-phi-cuda-gpu")
        self.assertIn("<|user|>", FakeEngine.prompts[-1])

    def test_stream_has_role_delta_finish_and_done(self):
        resp, raw = self.request("POST", "/v1/chat/completions",
                                 {"model": "qwen-coder-gpu", "stream": True,
                                  "messages": [{"role": "user", "content": "hi"}]}, raw=True)
        self.assertEqual(resp.status, 200)
        events = [e[len("data: "):] for e in raw.decode().strip().split("\n\n")]
        self.assertEqual(events[-1], "[DONE]")
        chunks = [json.loads(e) for e in events[:-1]]
        self.assertEqual(chunks[0]["choices"][0]["delta"], {"role": "assistant"})
        text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
        self.assertEqual(text, "Hello world")
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")
        self.assertTrue(all(c["choices"][0]["finish_reason"] is None for c in chunks[:-1]))

    def test_legacy_completions(self):
        resp, data = self.request("POST", "/v1/completions", {"model": "qwen-coder-gpu", "prompt": "hello"})
        self.assertEqual(resp.status, 200)
        self.assertEqual(data["choices"][0]["text"], "Hello world")
        self.assertEqual(FakeEngine.prompts[-1], "hello")  # raw prompt, no template

    def test_chat_completion_with_reasoning_non_streaming(self):
        FakeEngine.pieces = ["<think>\n", "Step by step thinking\n", "</think>\n", "42"]
        resp, data = self.chat()
        self.assertEqual(resp.status, 200)
        message = data["choices"][0]["message"]
        self.assertEqual(message.get("reasoning_content"), "Step by step thinking")
        self.assertEqual(message.get("content"), "42")

    def test_chat_completion_with_reasoning_streaming(self):
        FakeEngine.pieces = ["<think>\n", "thinking step 1\n", "</think>\n", "answer"]
        chunks = self.stream("/v1/chat/completions", self.chat_body())
        reasoning = "".join(c["choices"][0]["delta"].get("reasoning_content", "") for c in chunks if "delta" in c["choices"][0])
        content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if "delta" in c["choices"][0])
        self.assertIn("thinking step 1", reasoning)
        self.assertEqual(content, "answer")
        self.assertNotIn("<think>", reasoning)
        self.assertNotIn("</think>", reasoning)

    def test_chat_completion_unclosed_reasoning_non_streaming(self):
        FakeEngine.pieces = ["<think>\n", "cut off thinking"]
        resp, data = self.chat()
        self.assertEqual(resp.status, 200)
        message = data["choices"][0]["message"]
        self.assertEqual(message.get("reasoning_content"), "cut off thinking")
        self.assertEqual(message.get("content"), "")

    def test_chat_completion_unclosed_reasoning_streaming(self):
        # The model emitted <think> but `max_tokens` ended the stream before the closing tag. Per spec P7 the
        # whole remaining buffer must surface as reasoning_content and content must stay empty; no `<think>`
        # substring may leak into any delta.
        FakeEngine.pieces = ["<think>", "still ", "thinking"]
        chunks = self.stream("/v1/chat/completions", self.chat_body(max_tokens=3))
        reasoning = "".join(
            c["choices"][0]["delta"].get("reasoning_content", "")
            for c in chunks if "delta" in c["choices"][0]
        )
        content = "".join(
            c["choices"][0]["delta"].get("content", "")
            for c in chunks if "delta" in c["choices"][0]
        )
        self.assertEqual(reasoning, "still thinking")
        self.assertEqual(content, "")
        for c in chunks:
            for delta in (c["choices"][0].get("delta") or {},):
                self.assertNotIn("<think>", delta.get("reasoning_content", ""))
                self.assertNotIn("<think>", delta.get("content", ""))
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "length")

    def test_chat_completion_empty_think_omits_reasoning_field(self):
        # An empty `<think></think>` block must not surface as `reasoning_content: ""`; the key is omitted so the
        # client cannot confuse "no reasoning" with "empty reasoning". Content must remain the answer text.
        FakeEngine.pieces = ["<think></think>", "answer"]
        resp, data = self.chat()
        self.assertEqual(resp.status, 200)
        message = data["choices"][0]["message"]
        self.assertNotIn("reasoning_content", message)
        self.assertEqual(message.get("content"), "answer")

    def test_chat_completion_empty_think_omits_reasoning_field_streaming(self):
        # Same wire shape for the streaming path: an empty think block does not produce any
        # `delta: reasoning_content` chunk.
        FakeEngine.pieces = ["<think></think>", "answer"]
        chunks = self.stream("/v1/chat/completions", self.chat_body())
        for c in chunks:
            delta = c["choices"][0].get("delta") or {}
            self.assertNotIn("reasoning_content", delta)
        content = "".join(
            c["choices"][0]["delta"].get("content", "")
            for c in chunks if "delta" in c["choices"][0]
        )
        self.assertEqual(content, "answer")

    def test_onnx_legacy_completions_leave_think_tags_intact(self):
        # The reasoning extractor only runs on /v1/chat/completions (chat=True). On /v1/completions the
        # caller owns the raw prompt and the raw text; any <think>…</think> in the output must arrive
        # verbatim so legacy clients can parse it themselves.
        FakeEngine.pieces = ["<think>foo</think>bar"]
        # Non-streaming
        resp, data = self.request("POST", "/v1/completions", {"model": "qwen-coder-gpu", "prompt": "hi"})
        self.assertEqual(resp.status, 200)
        self.assertEqual(data["choices"][0]["text"], "<think>foo</think>bar")

        # Streaming
        chunks = self.stream("/v1/completions", {"model": "qwen-coder-gpu", "prompt": "hi", "stream": True})
        full_text = "".join(c["choices"][0]["text"] for c in chunks if "text" in c["choices"][0])
        self.assertEqual(full_text, "<think>foo</think>bar")


class TestErrors(ServerTestBase):
    def assertError(self, resp, data, status, code):
        self.assertEqual(resp.status, status)
        self.assertEqual(data["error"]["code"], code)

    def test_unknown_model(self):
        resp, data = self.chat(model="nope")
        self.assertError(resp, data, 404, "model_not_found")

    def test_ambiguous_model(self):
        resp, data = self.chat(model="phi")
        self.assertError(resp, data, 400, "ambiguous_model")

    def test_invalid_json_and_messages(self):
        resp, data = self.request("POST", "/v1/chat/completions", "{not json")
        self.assertEqual(resp.status, 400)
        self.assertIn("error", data)
        resp, data = self.request("POST", "/v1/chat/completions", {"model": "qwen-coder-gpu", "messages": []})
        self.assertEqual(resp.status, 400)

    def test_bad_numeric_param(self):
        resp, data = self.chat(max_tokens="abc")
        self.assertEqual(resp.status, 400)

    def test_model_load_failure_is_json_500(self):
        FakeEngine.fail_load = True
        resp, data = self.chat()
        self.assertError(resp, data, 500, "model_load_failed")

    def test_insufficient_resources_is_json_503(self):
        from prism.resources import InsufficientResourcesError
        with patch.object(self.manager, "use_engine", side_effect=InsufficientResourcesError("RAM shortage")):
            resp, data = self.chat()
            self.assertError(resp, data, 503, "insufficient_resources")
            self.assertEqual(resp.getheader("Retry-After"), "30")

    def test_unknown_route_is_json_404(self):
        resp, data = self.request("GET", "/nope")
        self.assertError(resp, data, 404, "not_found")

    def test_body_too_large(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.putrequest("POST", "/v1/chat/completions")
        conn.putheader("Content-Length", str(50 * 1024 * 1024))
        conn.endheaders()
        resp = conn.getresponse()
        self.assertEqual(resp.status, 413)
        conn.close()

    def test_foreign_host_header_rejected(self):
        resp, data = self.request("GET", "/v1/models", headers={"Host": "evil.example.com"})
        self.assertError(resp, data, 403, "host_not_allowed")


class TestConcurrency(ServerTestBase):
    def test_requests_for_different_models_are_serialized(self):
        results = []

        def call(model):
            resp, data = self.chat(model=model)
            results.append((resp.status, data))

        threads = [threading.Thread(target=call, args=(m,))
                   for m in ["alpha-phi-cuda-gpu", "qwen-coder-gpu", "beta-phi-cuda-gpu", "qwen-coder-gpu"]]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=20)
        self.assertEqual(len(results), 4)
        self.assertTrue(all(status == 200 for status, _ in results), results)
        self.assertEqual(FakeEngine.max_active, 1)


class TestHealthAndModels(ServerTestBase):
    def test_health_reports_active_model_after_a_request(self):
        _, data = self.request("GET", "/health")
        self.assertEqual((data["status"], data["active_model"], data["active_device"]), ("ok", None, None))
        self.assertIn("hardware", data)
        self.chat()
        _, data = self.request("GET", "/health")
        self.assertEqual(data["active_model"], "qwen-coder-gpu")
        self.assertEqual(data["active_device"], "cpu")  # FakeEngine.device

    def test_models_lists_onnx_and_ollama(self):
        self.ollama_models = [{"id": "ollama:tiny:1b", "name": "tiny:1b", "engine": "Ollama (llama.cpp)",
                               "backend": "ollama", "size_mb": 100}]
        _, data = self.request("GET", "/v1/models")
        ids = {m["id"] for m in data["data"]}
        self.assertEqual(ids, {"alpha-phi-cuda-gpu", "beta-phi-cuda-gpu", "qwen-coder-gpu", "ollama:tiny:1b"})


    def test_models_report_where_they_will_run_and_what_the_files_were_exported_for(self):
        with patch("prism.catalog.get_gpu_info", return_value={"available": True}), \
                patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PRISM_DEVICE", None)
            _, data = self.request("GET", "/v1/models")
        onnx = [m for m in data["data"] if m["id"] != "ollama:tiny:1b"]
        self.assertTrue(onnx)
        for model in onnx:
            self.assertEqual(model["device"], "CUDA (GPU)")
            self.assertIn("exported_for", model)
        with patch("prism.catalog.get_gpu_info", return_value={"available": True}), \
                patch.dict(os.environ, {"PRISM_DEVICE": "cpu"}):
            _, data = self.request("GET", "/v1/models")
        self.assertTrue(all(m["device"] == "CPU" for m in data["data"] if m["id"] != "ollama:tiny:1b"))


class TestStreamUsage(ServerTestBase):
    """OpenAI's stream_options.include_usage: token counts (and Prism's device telemetry) in the last streamed chunk."""

    def test_usage_and_telemetry_arrive_in_a_last_chunk_without_choices(self):
        chunks = self.stream("/v1/chat/completions", self.chat_body(stream_options={"include_usage": True}))
        last = chunks[-1]
        self.assertEqual(last["choices"], [])
        self.assertEqual(last["usage"]["completion_tokens"], 3)  # FakeEngine streams three pieces
        self.assertGreater(last["usage"]["prompt_tokens"], 0)
        self.assertEqual(last["usage"]["total_tokens"], last["usage"]["prompt_tokens"] + 3)
        self.assertEqual(last["telemetry"]["device"], "cpu")  # FakeEngine.device
        self.assertGreater(last["telemetry"]["ttft_sec"], 0)
        # the finish chunk still comes before it, so clients that read choices see the same stream as before
        self.assertEqual(chunks[-2]["choices"][0]["finish_reason"], "stop")

    def test_no_usage_chunk_unless_asked(self):
        chunks = self.stream("/v1/chat/completions", self.chat_body())
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")
        self.assertFalse(any("usage" in c for c in chunks))
        chunks = self.stream("/v1/chat/completions", self.chat_body(stream_options={"include_usage": False}))
        self.assertFalse(any("usage" in c for c in chunks))
        chunks = self.stream("/v1/chat/completions", self.chat_body(stream_options="yes"))  # malformed: ignored
        self.assertFalse(any("usage" in c for c in chunks))

    def test_legacy_completions_support_it_too(self):
        chunks = self.stream("/v1/completions", {"model": "qwen-coder-gpu", "prompt": "hello", "stream": True,
                                                 "stream_options": {"include_usage": True}})
        self.assertEqual(chunks[-1]["choices"], [])
        self.assertEqual(chunks[-1]["usage"]["completion_tokens"], 3)

    def test_length_cap_is_reflected_in_the_count(self):
        chunks = self.stream("/v1/chat/completions", self.chat_body(max_tokens=2, stream_options={"include_usage": True}))
        self.assertEqual(chunks[-1]["usage"]["completion_tokens"], 2)
        self.assertEqual(chunks[-2]["choices"][0]["finish_reason"], "length")


class TestOllamaRouting(ServerTestBase):
    def setUp(self):
        super().setUp()
        self.ollama_models = [{"id": "ollama:tiny:1b", "name": "tiny:1b", "engine": "Ollama (llama.cpp)",
                               "backend": "ollama"}]
        self.calls = []

        def fake_stream(model, messages, options=None, **kw):
            self.calls.append((model, messages, options))
            self.call_kwargs = kw
            if getattr(self, "backend_error", None):
                raise self.backend_error
            yield "Hi"
            yield " there"
            if kw.get("stats") is not None:
                kw["stats"].update(getattr(self, "ollama_stats", {}))

        patcher = patch("prism.server.stream_ollama_chat", side_effect=fake_stream)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_prefixed_and_unprefixed_names_route_to_ollama(self):
        for name in ("ollama:tiny:1b", "tiny:1b"):
            resp, data = self.chat(model=name, max_tokens=7)
            self.assertEqual(resp.status, 200)
            self.assertEqual(data["choices"][0]["message"]["content"], "Hi there")
        model, messages, options = self.calls[-1]
        self.assertEqual(model, "ollama:tiny:1b")
        self.assertEqual(options["num_predict"], 7)
        self.assertEqual(FakeEngine.prompts, [])  # never touched the ONNX engine

    def test_stream_shape(self):
        resp, raw = self.request("POST", "/v1/chat/completions",
                                 {"model": "tiny:1b", "stream": True,
                                  "messages": [{"role": "user", "content": "x"}]}, raw=True)
        events = [e[len("data: "):] for e in raw.decode().strip().split("\n\n")]
        chunks = [json.loads(e) for e in events[:-1]]
        self.assertEqual(events[-1], "[DONE]")
        self.assertEqual(chunks[0]["choices"][0]["delta"], {"role": "assistant"})
        self.assertEqual("".join(c["choices"][0]["delta"].get("content", "") for c in chunks), "Hi there")
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")

    def test_backend_down_is_json_502_for_both_modes(self):
        import urllib.error
        self.backend_error = urllib.error.URLError("connection refused")
        for stream in (False, True):
            resp, data = self.chat(model="tiny:1b", stream=stream)
            self.assertEqual((resp.status, data["error"]["code"]), (502, "backend_unavailable"))

    def test_they_become_ollama_options(self):
        self.chat(model="tiny:1b", top_k=30, repetition_penalty=1.1, frequency_penalty=0.4, presence_penalty=0.2)
        options = self.calls[-1][2]
        self.assertEqual((options["top_k"], options["repeat_penalty"], options["frequency_penalty"], options["presence_penalty"]),
                         (30, 1.1, 0.4, 0.2))

    def test_nothing_extra_is_sent_when_absent(self):
        self.chat(model="tiny:1b", frequency_penalty=0)
        self.assertEqual(set(self.calls[-1][2]), {"num_predict", "temperature", "top_p"})

    def test_legacy_completions_route_to_ollama(self):
        resp, data = self.request("POST", "/v1/completions", {"model": "tiny:1b", "prompt": "hello"})
        self.assertEqual(data["choices"][0]["text"], "Hi there")
        self.assertEqual(self.calls[-1][1], [{"role": "user", "content": "hello"}])

    def test_usage_and_finish_reason_come_from_the_daemon(self):
        self.ollama_stats = {"prompt_tokens": 5, "completion_tokens": 2, "finish_reason": "length"}
        _, data = self.chat(model="tiny:1b")
        self.assertEqual(data["choices"][0]["finish_reason"], "length")
        self.assertEqual(data["usage"], {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7})

    def test_stream_usage_chunk_only_when_asked_and_known(self):
        self.ollama_stats = {"prompt_tokens": 5, "completion_tokens": 2, "finish_reason": "stop"}

        def events(**extra):
            _, raw = self.request("POST", "/v1/chat/completions",
                                  {"model": "tiny:1b", "stream": True, "messages": [{"role": "user", "content": "x"}],
                                   **extra}, raw=True)
            return [json.loads(e[len("data: "):]) for e in raw.decode().strip().split("\n\n")[:-1]]

        last = events(stream_options={"include_usage": True})[-1]
        self.assertEqual((last["choices"], last["usage"]["total_tokens"]), ([], 7))
        self.assertFalse(any("usage" in c for c in events()))
        self.ollama_stats = {}  # the daemon reported no counts: no chunk rather than made-up numbers
        self.assertFalse(any("usage" in c for c in events(stream_options={"include_usage": True})))

    def test_stop_is_passed_on_and_content_parts_are_flattened(self):
        self.chat(model="tiny:1b", stop="###", messages=[
            {"role": "user", "content": [{"type": "text", "text": "hello"}, {"type": "image_url", "image_url": {}}]}])
        _, messages, options = self.calls[-1]
        self.assertEqual(options["stop"], ["###"])
        self.assertEqual(messages, [{"role": "user", "content": "hello"}])


    def test_tools_go_to_the_daemon_and_its_calls_come_back_in_openai_shape(self):
        tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
        self.ollama_stats = {"tool_calls": [{"name": "get_weather", "arguments": {"city": "Paris"}}], "finish_reason": "stop"}
        history = [{"role": "user", "content": "weather?"},
                   {"role": "assistant", "content": None, "tool_calls": [
                       {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Rome"}}}]},
                   {"role": "tool", "name": "get_weather", "content": "sunny", "tool_call_id": "c1"}]
        _, data = self.chat(model="tiny:1b", tools=tools, messages=history)
        self.assertEqual(self.call_kwargs["tools"], tools)
        _, sent, _ = self.calls[-1]
        self.assertEqual(sent[1]["tool_calls"], [{"function": {"name": "get_weather", "arguments": {"city": "Rome"}}}])
        self.assertEqual((sent[2]["role"], sent[2]["tool_name"]), ("tool", "get_weather"))
        choice = data["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        call = choice["message"]["tool_calls"][0]
        self.assertEqual((call["function"]["name"], json.loads(call["function"]["arguments"])), ("get_weather", {"city": "Paris"}))

    def test_streamed_tool_calls_arrive_as_an_indexed_delta(self):
        self.ollama_stats = {"tool_calls": [{"name": "f", "arguments": {}}], "finish_reason": "stop"}
        _, raw = self.request("POST", "/v1/chat/completions",
                              {"model": "tiny:1b", "stream": True, "messages": [{"role": "user", "content": "x"}],
                               "tools": [{"type": "function", "function": {"name": "f"}}]}, raw=True)
        chunks = [json.loads(e[len("data: "):]) for e in raw.decode().strip().split("\n\n")[:-1]]
        deltas = [c["choices"][0]["delta"] for c in chunks]
        call = next(d["tool_calls"][0] for d in deltas if "tool_calls" in d)
        self.assertEqual((call["index"], call["function"]["name"]), (0, "f"))
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "tool_calls")

    def test_ollama_reasoning_streaming_and_non_streaming(self):
        self.custom_chunks = ["<think>\n", "ollama thinking\n", "</think>\n", "ollama response"]

        def fake_stream(model, messages, options=None, **kw):
            yield from getattr(self, "custom_chunks", ["Hi", " there"])
            if kw.get("stats") is not None:
                kw["stats"].update(getattr(self, "ollama_stats", {}))

        with patch("prism.server.stream_ollama_chat", side_effect=fake_stream):
            # Non-streaming
            resp, data = self.chat(model="tiny:1b")
            self.assertEqual(resp.status, 200)
            msg = data["choices"][0]["message"]
            self.assertEqual(msg.get("reasoning_content"), "ollama thinking")
            self.assertEqual(msg.get("content"), "ollama response")

            # Streaming
            _, raw = self.request("POST", "/v1/chat/completions",
                                  {"model": "tiny:1b", "stream": True, "messages": [{"role": "user", "content": "x"}]},
                                  raw=True)
            chunks = [json.loads(e[len("data: "):]) for e in raw.decode().strip().split("\n\n") if e.startswith("data: ") and e != "data: [DONE]"]
            reasoning = "".join(c["choices"][0]["delta"].get("reasoning_content", "") for c in chunks if "delta" in c["choices"][0])
            content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if "delta" in c["choices"][0])
            self.assertIn("ollama thinking", reasoning)
            self.assertEqual(content, "ollama response")

    def test_ollama_legacy_completions_leave_think_tags_intact(self):
        self.custom_chunks = ["<think>thinking</think>answer"]

        def fake_stream(model, messages, options=None, **kw):
            yield from getattr(self, "custom_chunks", ["Hi", " there"])

        with patch("prism.server.stream_ollama_chat", side_effect=fake_stream):
            # Non-streaming
            resp, data = self.request("POST", "/v1/completions", {"model": "tiny:1b", "prompt": "hello"})
            self.assertEqual(resp.status, 200)
            self.assertEqual(data["choices"][0]["text"], "<think>thinking</think>answer")

            # Streaming
            _, raw = self.request("POST", "/v1/completions", {"model": "tiny:1b", "prompt": "hello", "stream": True}, raw=True)
            chunks = [json.loads(e[len("data: "):]) for e in raw.decode().strip().split("\n\n") if e.startswith("data: ") and e != "data: [DONE]"]
            full_text = "".join(c["choices"][0]["text"] for c in chunks if "text" in c["choices"][0])
            self.assertEqual(full_text, "<think>thinking</think>answer")

    def test_ollama_tool_calling_with_reasoning(self):
        self.custom_chunks = ["<think>ollama plan</think>"]
        self.ollama_stats = {"tool_calls": [{"name": "get_weather", "arguments": {"city": "Paris"}}], "finish_reason": "tool_calls"}

        def fake_stream(model, messages, options=None, **kw):
            yield from getattr(self, "custom_chunks", ["Hi", " there"])
            if kw.get("stats") is not None:
                kw["stats"].update(getattr(self, "ollama_stats", {}))

        with patch("prism.server.stream_ollama_chat", side_effect=fake_stream):
            # Non-streaming
            resp, data = self.chat(model="tiny:1b", tools=[WEATHER_TOOL])
            self.assertEqual(resp.status, 200)
            msg = data["choices"][0]["message"]
            self.assertEqual(msg.get("reasoning_content"), "ollama plan")
            self.assertEqual(msg["tool_calls"][0]["function"]["name"], "get_weather")


class TestStopMatcher(unittest.TestCase):
    def run_matcher(self, stops, pieces):
        m = StopMatcher(stops)
        out = "".join(m.feed(p) for p in pieces)
        return out if m.hit else out + m.flush(), m.hit

    def test_no_stops_passes_everything_through(self):
        self.assertEqual(self.run_matcher([], ["a", "b"]), ("ab", False))

    def test_cuts_at_the_stop_and_drops_the_rest(self):
        self.assertEqual(self.run_matcher(["END"], ["Hello ", "wor", "ld END", " tail"]), ("Hello world ", True))

    def test_stop_split_across_pieces_never_leaks(self):
        self.assertEqual(self.run_matcher(["cd"], ["ab", "c", "d", "e"]), ("ab", True))

    def test_held_back_prefix_is_released_when_it_turns_out_not_to_be_a_stop(self):
        self.assertEqual(self.run_matcher(["cd"], ["a", "c", "x"]), ("acx", False))
        self.assertEqual(self.run_matcher(["cd"], ["a", "c"]), ("ac", False))  # flushed at the end

    def test_earliest_of_several_stops_wins(self):
        self.assertEqual(self.run_matcher(["ZZ", "b"], ["aabZZ"]), ("aa", True))


class TestStopSequences(ServerTestBase):
    def setUp(self):
        super().setUp()
        FakeEngine.pieces = ["Hello", " wor", "ld END", " tail"]

    def test_non_stream_cuts_text_and_reports_stop(self):
        resp, data = self.chat(stop=["END"])
        self.assertEqual(data["choices"][0]["message"]["content"], "Hello world ")
        self.assertEqual(data["choices"][0]["finish_reason"], "stop")
        self.assertEqual(data["usage"]["completion_tokens"], 3)  # generation aborted at the third token
        self.assertEqual(FakeEngine.produced, 3)

    def test_stream_cuts_text_and_aborts_generation(self):
        _, raw = self.request("POST", "/v1/chat/completions",
                              {"model": "qwen-coder-gpu", "stream": True, "stop": "END",
                               "messages": [{"role": "user", "content": "hi"}]}, raw=True)
        chunks = [json.loads(e[len("data: "):]) for e in raw.decode().strip().split("\n\n")[:-1]]
        self.assertEqual("".join(c["choices"][0]["delta"].get("content", "") for c in chunks), "Hello world ")
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(FakeEngine.produced, 3)

    def test_without_a_match_the_full_text_arrives(self):
        _, data = self.chat(stop="nope")
        self.assertEqual(data["choices"][0]["message"]["content"], "Hello world END tail")

    def test_invalid_stop_is_a_400(self):
        for bad in (5, ["a", 1], [""], ["a", "b", "c", "d", "e"], {"x": 1}):
            resp, data = self.chat(stop=bad)
            self.assertEqual(resp.status, 400, bad)
            self.assertIn("'stop'", data["error"]["message"])


class TestContextWindow(ServerTestBase):
    def setUp(self):
        super().setUp()
        # FakeEngine counts whitespace-separated words; this prompt is 3 of them
        make_model(self.tmp, "tiny-ctx", "phi3", context_length=4)
        make_model(self.tmp, "no-room", "phi3", context_length=3)

    def test_max_tokens_is_clamped_to_the_room_left(self):
        _, data = self.chat(model="tiny-ctx", max_tokens=1000)
        self.assertEqual(data["usage"]["completion_tokens"], 1)
        self.assertEqual(data["choices"][0]["finish_reason"], "length")

    def test_prompt_that_fills_the_window_is_a_400(self):
        resp, data = self.chat(model="no-room")
        self.assertEqual((resp.status, data["error"]["code"]), (400, "context_length_exceeded"))
        resp, _ = self.chat(model="no-room", stream=True)  # an error before any header, so still JSON
        self.assertEqual(resp.status, 400)
        resp, _ = self.chat(model="qwen-coder-gpu")  # the lock was released
        self.assertEqual(resp.status, 200)

    def test_models_without_a_known_window_are_not_limited(self):
        _, data = self.chat(model="qwen-coder-gpu", max_tokens=1000)
        self.assertEqual(data["choices"][0]["message"]["content"], "Hello world")
        self.assertEqual(data["choices"][0]["finish_reason"], "stop")


class TestContentParts(ServerTestBase):
    def test_text_parts_reach_the_prompt_and_other_parts_are_dropped(self):
        resp, _ = self.chat(messages=[{"role": "user", "content": [
            {"type": "text", "text": "hello"}, {"type": "image_url", "image_url": {"url": "x"}}]}])
        self.assertEqual(resp.status, 200)
        self.assertIn("hello", FakeEngine.prompts[-1])
        self.assertNotIn("image_url", FakeEngine.prompts[-1])


class TestQueueTimeout(ServerTestBase):
    def setUp(self):
        super().setUp()
        self.manager.queue_timeout = 0.3
        FakeEngine.delay = 0.3  # a generation of three pieces takes about a second

    def test_a_request_that_waits_too_long_gets_503_and_the_first_one_finishes(self):
        first = {}
        t = threading.Thread(target=lambda: first.update(zip(("resp", "data"), self.chat())))
        t.start()
        deadline = time.time() + 5
        while time.time() < deadline and not FakeEngine.active:
            time.sleep(0.01)
        resp, data = self.chat()
        self.assertEqual((resp.status, data["error"]["code"]), (503, "server_busy"))
        self.assertEqual(resp.getheader("Retry-After"), "30")
        t.join(10)
        self.assertEqual(first["resp"].status, 200)
        self.assertEqual(first["data"]["choices"][0]["message"]["content"], "Hello world")
        FakeEngine.delay = 0.01
        resp, _ = self.chat()  # the lock is free again
        self.assertEqual(resp.status, 200)

    def test_no_limit_waits_for_its_turn(self):
        self.manager.queue_timeout = None
        FakeEngine.delay = 0.05
        results = []
        threads = [threading.Thread(target=lambda: results.append(self.chat()[0].status)) for _ in range(3)]
        for th in threads:
            th.start()
        for th in threads:
            th.join(10)
        self.assertEqual(results, [200, 200, 200])
        self.assertEqual(FakeEngine.max_active, 1)

    def test_max_queue_overflow_gets_503_immediately(self):
        self.manager.max_queue = 1
        self.manager.queue_timeout = 5.0
        FakeEngine.delay = 0.3
        t1 = threading.Thread(target=lambda: self.chat())
        t1.start()
        deadline = time.time() + 5
        while time.time() < deadline and not FakeEngine.active:
            time.sleep(0.01)

        t2 = threading.Thread(target=lambda: self.chat())
        t2.start()
        time.sleep(0.05)

        t0 = time.monotonic()
        resp, data = self.chat()
        elapsed = time.monotonic() - t0
        self.assertEqual((resp.status, data["error"]["code"]), (503, "server_busy"))
        self.assertLess(elapsed, 1.5)
        t1.join(5)
        t2.join(5)


TOOL_TEMPLATE = ("{% if tools %}TOOLS:{% for t in tools %}{{ t.function.name }};{% endfor %}\n{% endif %}"
                 "{% for m in messages %}{{ m.role }}:{{ m.content }}"
                 "{% if m.tool_calls %} CALL:{{ m.tool_calls[0].function.arguments.city }}{% endif %}\n{% endfor %}")
CALL_TEXT = ['<tool_call>{"name": "get_weather", ', '"arguments": {"city": "Paris"}}</tool_call>']
WEATHER_TOOL = {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}


@unittest.skipUnless(jinja_available(), "tool calling on ONNX models needs the optional jinja2")
class TestToolCalling(ServerTestBase):
    def setUp(self):
        super().setUp()
        d = make_model(self.tmp, "tool-model", "qwen2")
        with open(os.path.join(d, "tokenizer_config.json"), "w") as f:
            json.dump({"chat_template": TOOL_TEMPLATE}, f)
        FakeEngine.pieces = CALL_TEXT

    def ask(self, **extra):
        return self.chat(model="tool-model", tools=[WEATHER_TOOL], **extra)

    def test_tools_reach_the_prompt_and_the_call_comes_back_in_openai_shape(self):
        resp, data = self.ask()
        self.assertEqual(resp.status, 200)
        self.assertIn("TOOLS:get_weather;", FakeEngine.prompts[-1])
        choice = data["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertIsNone(choice["message"]["content"])
        call = choice["message"]["tool_calls"][0]
        self.assertTrue(call["id"].startswith("call_"))
        self.assertEqual((call["type"], call["function"]["name"]), ("function", "get_weather"))
        self.assertEqual(json.loads(call["function"]["arguments"]), {"city": "Paris"})
        self.assertGreater(data["usage"]["prompt_tokens"], 0)

    def test_streaming_is_buffered_and_ends_with_the_tool_calls_finish(self):
        _, raw = self.request("POST", "/v1/chat/completions",
                              {"model": "tool-model", "stream": True, "tools": [WEATHER_TOOL],
                               "stream_options": {"include_usage": True},
                               "messages": [{"role": "user", "content": "weather?"}]}, raw=True)
        events = raw.decode().strip().split("\n\n")
        self.assertEqual(events[-1], "data: [DONE]")
        chunks = [json.loads(e[len("data: "):]) for e in events[:-1]]
        deltas = [c["choices"][0]["delta"] for c in chunks if c["choices"]]
        self.assertEqual(deltas[0], {"role": "assistant"})
        call = next(d["tool_calls"][0] for d in deltas if "tool_calls" in d)
        self.assertEqual((call["index"], call["function"]["name"]), (0, "get_weather"))
        self.assertEqual(json.loads(call["function"]["arguments"]), {"city": "Paris"})
        self.assertEqual([c for c in chunks if c["choices"]][-1]["choices"][0]["finish_reason"], "tool_calls")
        self.assertIn("usage", chunks[-1])

    def test_an_ordinary_answer_is_still_an_ordinary_answer(self):
        FakeEngine.pieces = ["It is ", "sunny."]
        _, data = self.ask()
        self.assertEqual(data["choices"][0]["message"]["content"], "It is sunny.")
        self.assertEqual(data["choices"][0]["finish_reason"], "stop")
        self.assertNotIn("tool_calls", data["choices"][0]["message"])

    def test_tool_choice_none_hides_the_tools(self):
        _, data = self.ask(tool_choice="none")
        self.assertNotIn("TOOLS:", FakeEngine.prompts[-1])
        self.assertIn("<tool_call>", data["choices"][0]["message"]["content"])  # returned as text, not parsed

    def test_a_tool_result_conversation_reaches_the_template_with_object_arguments(self):
        history = [{"role": "user", "content": "weather?"},
                   {"role": "assistant", "content": None, "tool_calls": [
                       {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Rome"}'}}]},
                   {"role": "tool", "tool_call_id": "c1", "content": "sunny"}]
        FakeEngine.pieces = ["Sunny in Rome."]
        resp, data = self.ask(messages=history)
        self.assertEqual(resp.status, 200)
        self.assertIn("assistant: CALL:Rome", FakeEngine.prompts[-1])
        self.assertIn("tool:sunny", FakeEngine.prompts[-1])

    def test_stop_sequences_still_apply_with_tools(self):
        FakeEngine.pieces = ["Hello END", " tail"]
        _, data = self.ask(stop="END")
        self.assertEqual(data["choices"][0]["message"]["content"], "Hello ")

    def test_tool_calling_with_reasoning_non_streaming(self):
        FakeEngine.pieces = ["<think>need current weather</think>", '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>']
        resp, data = self.ask()
        self.assertEqual(resp.status, 200)
        choice = data["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual(choice["message"]["reasoning_content"], "need current weather")
        self.assertEqual(choice["message"]["tool_calls"][0]["function"]["name"], "get_weather")

    def test_tool_calling_with_reasoning_streaming(self):
        FakeEngine.pieces = ["<think>need current weather</think>", '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>']
        _, raw = self.request("POST", "/v1/chat/completions",
                              {"model": "tool-model", "stream": True, "tools": [WEATHER_TOOL],
                               "messages": [{"role": "user", "content": "weather?"}]}, raw=True)
        chunks = [json.loads(e[len("data: "):]) for e in raw.decode().strip().split("\n\n") if e.startswith("data: ") and e != "data: [DONE]"]
        reasoning = "".join(c["choices"][0]["delta"].get("reasoning_content", "") for c in chunks if "delta" in c["choices"][0])
        self.assertEqual(reasoning, "need current weather")
        call = next(d["tool_calls"][0] for d in [c["choices"][0]["delta"] for c in chunks if "delta" in c["choices"][0]] if "tool_calls" in d)
        self.assertEqual(call["function"]["name"], "get_weather")


class TestToolsRejected(ServerTestBase):
    def test_a_model_whose_template_ignores_tools_is_a_400(self):
        resp, data = self.chat(model="qwen-coder-gpu", tools=[WEATHER_TOOL])  # no chat_template at all
        self.assertEqual((resp.status, data["error"]["code"]), (400, "tools_not_supported"))
        self.assertEqual(FakeEngine.prompts, [])

    def test_without_jinja_no_onnx_model_can_take_tools(self):
        d = make_model(self.tmp, "tool-model", "qwen2")
        with open(os.path.join(d, "tokenizer_config.json"), "w") as f:
            json.dump({"chat_template": TOOL_TEMPLATE}, f)
        with patch("prism.templates.jinja_available", return_value=False):
            resp, data = self.chat(model="tool-model", tools=[WEATHER_TOOL])
        self.assertEqual((resp.status, data["error"]["code"]), (400, "tools_not_supported"))

    def test_tool_choice_none_needs_no_tool_support(self):
        resp, _ = self.chat(model="qwen-coder-gpu", tools=[WEATHER_TOOL], tool_choice="none")
        self.assertEqual(resp.status, 200)

    def test_malformed_tools_are_a_400(self):
        for bad in ("get_weather", [{"type": "function"}], [{"type": "retrieval", "function": {"name": "x"}}],
                    [{"type": "function", "function": {"name": ""}}], {"a": 1}):
            resp, data = self.chat(tools=bad)
            self.assertEqual(resp.status, 400, bad)
            self.assertIn("'tools'", data["error"]["message"])

    def test_empty_tools_are_ignored(self):
        resp, _ = self.chat(tools=[])
        self.assertEqual(resp.status, 200)


class TestEmbeddings(ServerTestBase):
    def setUp(self):
        super().setUp()
        self.ollama_models = [{"id": "ollama:embed-model", "name": "embed-model", "engine": "Ollama (llama.cpp)",
                               "backend": "ollama"}]
        self.embed_calls = []

        def fake_embed(model, inputs, **kw):
            self.embed_calls.append((model, inputs))
            if getattr(self, "embed_error", None):
                raise self.embed_error
            return [[float(i), 0.5, -1.25] for i, _ in enumerate(inputs)], 7

        patcher = patch("prism.server.embed_ollama", side_effect=fake_embed)
        patcher.start()
        self.addCleanup(patcher.stop)

    def embed(self, **body):
        return self.request("POST", "/v1/embeddings", {"model": "ollama:embed-model", **body})

    def test_openai_shape_for_one_string_and_for_a_list(self):
        resp, data = self.embed(input="hello")
        self.assertEqual(resp.status, 200)
        self.assertEqual((data["object"], data["model"]), ("list", "ollama:embed-model"))
        self.assertEqual(data["data"], [{"object": "embedding", "index": 0, "embedding": [0.0, 0.5, -1.25]}])
        self.assertEqual(data["usage"], {"prompt_tokens": 7, "total_tokens": 7})
        _, data = self.embed(input=["a", "b"])
        self.assertEqual([d["index"] for d in data["data"]], [0, 1])
        self.assertEqual(self.embed_calls[-1], ("ollama:embed-model", ["a", "b"]))

    def test_unprefixed_installed_names_route_to_ollama(self):
        resp, _ = self.request("POST", "/v1/embeddings", {"model": "embed-model", "input": "x"})
        self.assertEqual(resp.status, 200)

    def test_base64_is_little_endian_float32(self):  # the official Python client asks for this by default
        import base64, struct
        _, data = self.embed(input="hello", encoding_format="base64")
        raw = base64.b64decode(data["data"][0]["embedding"])
        self.assertEqual(struct.unpack("<3f", raw), (0.0, 0.5, -1.25))

    def test_onnx_models_cannot_embed(self):
        resp, data = self.request("POST", "/v1/embeddings", {"model": "qwen-coder-gpu", "input": "x"})
        self.assertEqual((resp.status, data["error"]["code"]), (400, "embeddings_not_supported"))
        self.assertEqual(self.embed_calls, [])

    def test_bad_requests_are_400(self):
        for body in ({"input": ""}, {"input": []}, {"input": [1, 2]}, {"input": ["a", ""]}, {"input": 5},
                     {"input": ["x"] * 257}, {"input": "x", "encoding_format": "hex"}, {}):
            resp, _ = self.embed(**body)
            self.assertEqual(resp.status, 400, body)
        resp, data = self.embed(input="x", dimensions=256)
        self.assertEqual((resp.status, data["error"]["code"]), (400, "dimensions_not_supported"))
        resp, data = self.request("POST", "/v1/embeddings", {"input": "x"})  # no model
        self.assertEqual(resp.status, 400)
        resp, data = self.request("POST", "/v1/embeddings", {"model": "nope", "input": "x"})
        self.assertEqual((resp.status, data["error"]["code"]), (404, "model_not_found"))

    def test_a_model_that_is_not_an_embedding_model_gets_the_daemons_reason(self):
        import io
        import urllib.error
        body = io.BytesIO(b'{"error":"This server does not support embeddings."}')
        self.embed_error = urllib.error.HTTPError("http://x", 501, "Not Implemented", {}, body)
        resp, data = self.embed(input="x")
        self.assertEqual((resp.status, data["error"]["code"]), (400, "embeddings_not_supported"))
        self.assertIn("This server does not support embeddings.", data["error"]["message"])
        self.assertIn("nomic-embed-text", data["error"]["message"])

    def test_daemon_down_is_502_and_unknown_model_is_404(self):
        import urllib.error
        self.embed_error = urllib.error.URLError("connection refused")
        resp, data = self.embed(input="x")
        self.assertEqual((resp.status, data["error"]["code"]), (502, "backend_unavailable"))
        self.embed_error = urllib.error.HTTPError("http://x", 404, "Not Found", {}, None)
        resp, data = self.embed(input="x")
        self.assertEqual((resp.status, data["error"]["code"]), (404, "model_not_found"))
        self.embed_error = ValueError("Ollama returned no embeddings for this model")
        resp, _ = self.embed(input="x")
        self.assertEqual(resp.status, 502)


class TestClientDisconnect(ServerTestBase):
    def test_generation_stops_when_client_goes_away(self):
        FakeEngine.pieces = [f"w{i} " for i in range(400)]
        FakeEngine.delay = 0.01
        body = json.dumps({"model": "qwen-coder-gpu", "stream": True,
                           "messages": [{"role": "user", "content": "go"}]})
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request("POST", "/v1/chat/completions", body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        resp.fp.readline()  # first SSE event, then hang up
        resp.close()  # the response holds its own reference to the socket; close both or nothing disconnects
        conn.close()
        deadline = time.time() + 5
        while time.time() < deadline and FakeEngine.active:
            time.sleep(0.05)
        self.assertEqual(FakeEngine.active, 0, "generation kept running after the client disconnected")
        self.assertLess(FakeEngine.produced, 400)
        resp, data = self.chat()  # lock was released; the server still serves
        self.assertEqual(resp.status, 200)


class TestAuthAndCors(ServerTestBase):
    api_key = "s3cret"
    cors_origins = ("http://localhost:3000",)

    def test_missing_or_wrong_key_rejected(self):
        resp, data = self.request("GET", "/v1/models")
        self.assertEqual((resp.status, data["error"]["code"]), (401, "invalid_api_key"))
        resp, _ = self.request("GET", "/v1/models", headers={"Authorization": "Bearer wrong"})
        self.assertEqual(resp.status, 401)
        resp, _ = self.chat()
        self.assertEqual(resp.status, 401)

    def test_valid_key_accepted_and_health_public(self):
        resp, data = self.request("GET", "/v1/models", headers={"Authorization": "Bearer s3cret"})
        self.assertEqual(resp.status, 200)
        self.assertEqual(len(data["data"]), 3)
        resp, _ = self.request("GET", "/health")
        self.assertEqual(resp.status, 200)

    def test_cors_only_for_allowlisted_origin(self):
        auth = {"Authorization": "Bearer s3cret"}
        resp, _ = self.request("GET", "/v1/models", headers={**auth, "Origin": "http://localhost:3000"})
        self.assertEqual(resp.getheader("Access-Control-Allow-Origin"), "http://localhost:3000")
        resp, _ = self.request("GET", "/v1/models", headers={**auth, "Origin": "http://evil.example"})
        self.assertIsNone(resp.getheader("Access-Control-Allow-Origin"))


class TestDefaultNoCors(ServerTestBase):
    def test_no_cors_headers_by_default(self):
        resp, _ = self.request("GET", "/v1/models", headers={"Origin": "http://evil.example"})
        self.assertEqual(resp.status, 200)
        self.assertIsNone(resp.getheader("Access-Control-Allow-Origin"))


if __name__ == "__main__":
    unittest.main()


class TestSamplingParameters(ServerTestBase):
    def test_top_k_and_repetition_penalty_reach_the_engine(self):
        resp, _ = self.chat(top_k=30, repetition_penalty=1.05, temperature=0.6)
        self.assertEqual(resp.status, 200)
        self.assertEqual(FakeEngine.sampling[-1], {"temperature": 0.6, "top_p": 0.9, "top_k": 30, "repetition_penalty": 1.05})

    def test_they_are_left_to_the_engine_when_absent(self):
        self.chat()
        self.assertEqual((FakeEngine.sampling[-1]["top_k"], FakeEngine.sampling[-1]["repetition_penalty"]), (None, None))

    def test_completions_take_them_too(self):
        self.request("POST", "/v1/completions", {"model": "qwen-coder-gpu", "prompt": "hi", "top_k": 12})
        self.assertEqual(FakeEngine.sampling[-1]["top_k"], 12)

    def test_zero_penalties_are_accepted_because_clients_send_them_by_default(self):
        resp, _ = self.chat(frequency_penalty=0, presence_penalty=0.0)
        self.assertEqual(resp.status, 200)

    def test_frequency_and_presence_penalties_are_refused_not_ignored(self):
        for key in ("frequency_penalty", "presence_penalty"):
            with self.subTest(key=key):
                resp, data = self.chat(**{key: 0.5})
                self.assertEqual((resp.status, data["error"]["code"]), (400, "unsupported_parameter"))
                self.assertIn(key, data["error"]["message"])

    def test_bad_values_are_400(self):
        for extra in ({"top_k": 0}, {"top_k": "many"}, {"repetition_penalty": 0}, {"repetition_penalty": "x"},
                      {"frequency_penalty": 3}, {"presence_penalty": "x"}):
            with self.subTest(extra=extra):
                resp, _ = self.chat(**extra)
                self.assertEqual(resp.status, 400)

