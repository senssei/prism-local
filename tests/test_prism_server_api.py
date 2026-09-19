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
from prism.server import ActiveEngineManager, create_server
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


class TestOllamaRouting(ServerTestBase):
    def setUp(self):
        super().setUp()
        self.ollama_models = [{"id": "ollama:tiny:1b", "name": "tiny:1b", "engine": "Ollama (llama.cpp)",
                               "backend": "ollama"}]
        self.calls = []

        def fake_stream(model, messages, options=None, **kw):
            self.calls.append((model, messages, options))
            if getattr(self, "backend_error", None):
                raise self.backend_error
            yield "Hi"
            yield " there"

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

    def test_legacy_completions_route_to_ollama(self):
        resp, data = self.request("POST", "/v1/completions", {"model": "tiny:1b", "prompt": "hello"})
        self.assertEqual(data["choices"][0]["text"], "Hi there")
        self.assertEqual(self.calls[-1][1], [{"role": "user", "content": "hello"}])


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
