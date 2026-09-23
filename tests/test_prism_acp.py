import builtins
import io
import json
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
import urllib.error
from unittest.mock import patch, MagicMock

import prism.acp as acp
from prism.catalog import ModelCatalog
from prism.server import ActiveEngineManager
from tests.fakes import FakeEngine, make_model


class AcpTestCase(unittest.TestCase):
    def setUp(self):
        acp._sessions.clear()

    def _new_session(self, cwd="/tmp"):
        return acp.handle_session_new({"cwd": cwd})["sessionId"]


class TestHandshake(AcpTestCase):
    def test_initialize_returns_capabilities(self):
        result = acp.handle_initialize({"protocolVersion": 1, "clientCapabilities": {}})
        self.assertEqual(result["protocolVersion"], 1)
        self.assertFalse(result["agentCapabilities"]["loadSession"])
        self.assertEqual(result["authMethods"], [])

    def test_session_new_returns_session_id_and_resolves_model(self):
        class StubCatalog:
            def list_all_models(self, include_ollama=True):
                return [{"id": "qwen-coder", "device": "CUDA (GPU)"}]

        with patch("prism.acp._catalog", return_value=StubCatalog()):
            result = acp.handle_session_new({"cwd": "/tmp"})
        session_id = result["sessionId"]
        self.assertIn(session_id, acp._sessions)
        self.assertEqual(acp._sessions[session_id].model, "qwen-coder")


class TestSessionPrompt(AcpTestCase):
    def test_session_prompt_streams_agent_message_chunks_then_end_turn(self):
        captured = []

        def source(*_a, **_k):
            return iter([{"content": "Hel"}, {"content": "lo"}, {"reasoning_content": "thinking"}])

        with patch("prism.acp._server_delta_stream", side_effect=source), \
             patch("prism.acp._write", side_effect=captured.append):
            session_id = self._new_session()
            thread = acp.handle_session_prompt(
                {"sessionId": session_id, "prompt": [{"type": "text", "text": "hi"}]}, 7)
            self.assertIsNotNone(thread)
            thread.join(2.0)

        updates = [m for m in captured if m.get("method") == "session/update"]
        self.assertEqual(
            [u["params"]["update"]["sessionUpdate"] for u in updates],
            ["agent_message_chunk", "agent_message_chunk", "agent_thought_chunk"],
        )
        message_text = "".join(
            u["params"]["update"]["content"]["text"] for u in updates
            if u["params"]["update"]["sessionUpdate"] == "agent_message_chunk"
        )
        self.assertEqual(message_text, "Hello")
        response = next(m for m in captured if m.get("id") == 7)
        self.assertEqual(response["result"]["stopReason"], "end_turn")
        self.assertEqual(acp._sessions[session_id].messages[-1], {"role": "assistant", "content": "Hello"})

    def test_session_prompt_rejects_non_text_content_block(self):
        captured = []
        with patch("prism.acp._write", side_effect=captured.append):
            session_id = self._new_session()
            thread = acp.handle_session_prompt(
                {"sessionId": session_id, "prompt": [{"type": "image", "data": "..."}]}, 5)
        self.assertIsNone(thread)
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["error"]["code"], -32602)
        self.assertIn("image", captured[0]["error"]["message"])

    def test_session_prompt_rejects_concurrent_prompt_on_same_session(self):
        started = threading.Event()
        proceed = threading.Event()

        def source(*_a, **_k):
            started.set()
            proceed.wait(2.0)
            return iter([{"content": "hi"}])

        captured = []
        with patch("prism.acp._server_delta_stream", side_effect=source), \
             patch("prism.acp._write", side_effect=captured.append):
            session_id = self._new_session()
            t1 = acp.handle_session_prompt(
                {"sessionId": session_id, "prompt": [{"type": "text", "text": "hi"}]}, 1)
            self.assertTrue(started.wait(2.0))
            t2 = acp.handle_session_prompt(
                {"sessionId": session_id, "prompt": [{"type": "text", "text": "again"}]}, 2)
            proceed.set()
            t1.join(2.0)

        self.assertIsNone(t2)
        errors = [m for m in captured if m.get("id") == 2]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["error"]["code"], -32602)
        self.assertIn("already in progress", errors[0]["error"]["message"])

    def test_generation_error_resolves_prompt_with_json_rpc_error(self):
        err_body = io.BytesIO(b'{"error":{"message":"boom","code":"model_load_failed"}}')
        http_err = urllib.error.HTTPError(
            "http://localhost:5272/v1/chat/completions", 500, "Server Error", {}, err_body)
        captured = []
        with patch("prism.acp._server_delta_stream", side_effect=http_err), \
             patch("prism.acp._write", side_effect=captured.append):
            session_id = self._new_session()
            thread = acp.handle_session_prompt(
                {"sessionId": session_id, "prompt": [{"type": "text", "text": "hi"}]}, 9)
            thread.join(2.0)

        response = next(m for m in captured if m.get("id") == 9)
        self.assertEqual(response["error"]["code"], -32000)
        self.assertIn("500", response["error"]["message"])
        self.assertIn("boom", response["error"]["message"])


class TestSessionCancel(AcpTestCase):
    def test_session_cancel_stops_in_flight_prompt_and_reports_cancelled(self):
        started = threading.Event()
        proceed = threading.Event()

        def source(*_a, **_k):
            def gen():
                yield {"content": "first"}
                started.set()
                proceed.wait(2.0)
                yield {"content": "second"}
                yield {"content": "third"}
            return gen()

        captured = []
        with patch("prism.acp._server_delta_stream", side_effect=source), \
             patch("prism.acp._write", side_effect=captured.append):
            session_id = self._new_session()
            thread = acp.handle_session_prompt(
                {"sessionId": session_id, "prompt": [{"type": "text", "text": "hi"}]}, 1)
            self.assertTrue(started.wait(2.0))
            acp.handle_session_cancel({"sessionId": session_id})
            proceed.set()
            thread.join(2.0)

        updates = [m for m in captured if m.get("method") == "session/update"]
        self.assertEqual(len(updates), 1)
        self.assertEqual(updates[0]["params"]["update"]["content"]["text"], "first")
        response = next(m for m in captured if m.get("id") == 1)
        self.assertEqual(response["result"]["stopReason"], "cancelled")

    def test_session_cancel_after_natural_completion_is_a_noop(self):
        captured = []
        with patch("prism.acp._server_delta_stream", side_effect=lambda *a, **k: iter([{"content": "hi"}])), \
             patch("prism.acp._write", side_effect=captured.append):
            session_id = self._new_session()
            thread = acp.handle_session_prompt(
                {"sessionId": session_id, "prompt": [{"type": "text", "text": "hi"}]}, 1)
            thread.join(2.0)
            response = next(m for m in captured if m.get("id") == 1)
            self.assertEqual(response["result"]["stopReason"], "end_turn")
            acp.handle_session_cancel({"sessionId": session_id})  # must not raise, must not change anything


class TestDispatch(AcpTestCase):
    def test_unknown_method_with_id_returns_method_not_found(self):
        captured = []
        with patch("prism.acp._write", side_effect=captured.append):
            acp._dispatch({"jsonrpc": "2.0", "id": 3, "method": "session/load"})
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["error"]["code"], -32601)
        self.assertIn("session/load", captured[0]["error"]["message"])

    def test_unknown_method_without_id_is_dropped(self):
        captured = []
        with patch("prism.acp._write", side_effect=captured.append):
            acp._dispatch({"jsonrpc": "2.0", "method": "notifications/whatever"})
        self.assertEqual(captured, [])


class TestMalformedInput(AcpTestCase):
    """Review finding #1: a malformed-but-valid JSON-RPC message must not crash the stdio loop."""

    def test_non_dict_params_on_synchronous_methods_returns_error_not_crash(self):
        captured = []
        with patch("prism.acp._write", side_effect=captured.append):
            acp._dispatch({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": ["x"]})
            acp._dispatch({"jsonrpc": "2.0", "id": 2, "method": "session/new", "params": "nope"})
            acp._dispatch({"jsonrpc": "2.0", "id": 3, "method": "session/cancel", "params": 5})
        ids = {m.get("id") for m in captured}
        self.assertEqual(ids, {1, 2, 3})
        for m in captured:
            self.assertIn("error", m)

    def test_prompt_null_returns_error_not_crash(self):
        captured = []
        with patch("prism.acp._write", side_effect=captured.append):
            session_id = self._new_session()
            acp._dispatch({"jsonrpc": "2.0", "id": 4, "method": "session/prompt",
                           "params": {"sessionId": session_id, "prompt": None}})
        response = next(m for m in captured if m.get("id") == 4)
        self.assertIn("error", response)

    def test_prompt_as_a_string_returns_error_not_crash(self):
        captured = []
        with patch("prism.acp._write", side_effect=captured.append):
            session_id = self._new_session()
            acp._dispatch({"jsonrpc": "2.0", "id": 5, "method": "session/prompt",
                           "params": {"sessionId": session_id, "prompt": "hi"}})
        response = next(m for m in captured if m.get("id") == 5)
        self.assertIn("error", response)

    def test_prompt_list_of_non_dict_blocks_returns_error_not_crash(self):
        captured = []
        with patch("prism.acp._write", side_effect=captured.append):
            session_id = self._new_session()
            acp._dispatch({"jsonrpc": "2.0", "id": 6, "method": "session/prompt",
                           "params": {"sessionId": session_id, "prompt": [42, None]}})
        response = next(m for m in captured if m.get("id") == 6)
        self.assertIn("error", response)

    def test_non_dict_request_does_not_crash(self):
        with patch("prism.acp._write"):
            acp._dispatch([1, 2, 3])  # must not raise


class TestNoDanglingUserTurnOnError(AcpTestCase):
    """Review finding #3: a failed generation must not leave an unanswered user turn behind."""

    def test_generation_error_leaves_no_dangling_user_turn(self):
        err_body = io.BytesIO(b'{"error":{"message":"boom","code":"model_load_failed"}}')
        http_err = urllib.error.HTTPError(
            "http://localhost:5272/v1/chat/completions", 500, "Server Error", {}, err_body)
        captured = []
        with patch("prism.acp._server_delta_stream", side_effect=http_err), \
             patch("prism.acp._write", side_effect=captured.append):
            session_id = self._new_session()
            thread = acp.handle_session_prompt(
                {"sessionId": session_id, "prompt": [{"type": "text", "text": "hi"}]}, 1)
            thread.join(2.0)
        self.assertEqual(acp._sessions[session_id].messages, [])

        def source(*_a, **_k):
            return iter([{"content": "ok"}])

        with patch("prism.acp._server_delta_stream", side_effect=source), \
             patch("prism.acp._write", side_effect=captured.append):
            thread2 = acp.handle_session_prompt(
                {"sessionId": session_id, "prompt": [{"type": "text", "text": "again"}]}, 2)
            thread2.join(2.0)
        response2 = next(m for m in captured if m.get("id") == 2)
        self.assertEqual(response2["result"]["stopReason"], "end_turn")
        self.assertEqual(
            acp._sessions[session_id].messages,
            [{"role": "user", "content": "again"}, {"role": "assistant", "content": "ok"}],
        )


class TestInitializeProtocolVersion(AcpTestCase):
    """Review finding #5: protocolVersion must reject bool and negative values."""

    def test_boolean_protocol_version_does_not_round_trip_as_a_bool(self):
        result = acp.handle_initialize({"protocolVersion": True})
        self.assertNotIsInstance(result["protocolVersion"], bool)
        self.assertEqual(result["protocolVersion"], acp.ACP_PROTOCOL_VERSION)

    def test_negative_protocol_version_is_clamped_to_the_supported_version(self):
        result = acp.handle_initialize({"protocolVersion": -5})
        self.assertEqual(result["protocolVersion"], acp.ACP_PROTOCOL_VERSION)


class TestEmptyCatalogDefaultModel(AcpTestCase):
    """Review finding #4: locks in the documented hardcoded-fallback behavior when no models exist."""

    def test_session_new_falls_back_to_hardcoded_default_when_catalog_is_empty(self):
        class EmptyCatalog:
            def list_all_models(self, include_ollama=True):
                return []

        with patch("prism.acp._catalog", return_value=EmptyCatalog()):
            result = acp.handle_session_new({"cwd": "/tmp"})
        session_id = result["sessionId"]
        from prism.catalog import DEFAULT_FALLBACK_MODEL_ID
        self.assertEqual(acp._sessions[session_id].model, DEFAULT_FALLBACK_MODEL_ID)


class TestStdoutWriteFailure(AcpTestCase):
    """Second-review finding #1: a broken stdout pipe must not kill the stdio loop."""

    def test_broken_pipe_while_sending_an_error_logs_to_stderr_and_does_not_raise(self):
        with patch.object(acp.sys.stdout, "write", side_effect=BrokenPipeError("broken")), \
             patch.object(acp.sys.stderr, "write") as stderr_write:
            acp._dispatch({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": ["x"]})
        self.assertTrue(any("stdout is gone" in call.args[0] for call in stderr_write.call_args_list))

    def test_broken_pipe_while_sending_a_normal_response_logs_to_stderr_and_does_not_raise(self):
        with patch.object(acp.sys.stdout, "write", side_effect=BrokenPipeError("broken")), \
             patch.object(acp.sys.stderr, "write") as stderr_write:
            acp._dispatch({"jsonrpc": "2.0", "id": 2, "method": "initialize", "params": {}})
        self.assertTrue(any("stdout is gone" in call.args[0] for call in stderr_write.call_args_list))

    def test_both_stdout_and_stderr_broken_does_not_raise(self):
        """Third-review finding #3: `_write`'s own stderr fallback must itself be exception-safe."""
        with patch.object(acp.sys.stdout, "write", side_effect=BrokenPipeError("stdout gone")), \
             patch.object(acp.sys.stderr, "write", side_effect=BrokenPipeError("stderr gone too")):
            acp._write({"jsonrpc": "2.0", "id": 1, "result": {}})


class TestLazySingletonThreadSafety(unittest.TestCase):
    """Third-review finding #1: `_catalog()`/`_engine_manager()` must be safe under concurrent
    daemon threads (one per `session/prompt`), and `_engine_manager()` calling `_catalog()` while
    holding the same lock must not self-deadlock."""

    def setUp(self):
        acp._manager = None
        acp._catalog_instance = None

    def tearDown(self):
        acp._manager = None
        acp._catalog_instance = None

    def test_engine_manager_does_not_deadlock_when_it_calls_catalog(self):
        result = {}

        def call():
            result["manager"] = acp._engine_manager()

        t = threading.Thread(target=call, daemon=True)
        t.start()
        t.join(3.0)
        self.assertFalse(t.is_alive(), "acp._engine_manager() appears to have deadlocked")
        self.assertIn("manager", result)

    def test_concurrent_calls_construct_the_engine_manager_only_once(self):
        import prism.server as server_module

        construction_started = threading.Event()
        release = threading.Event()
        call_count = {"n": 0}
        real_init = server_module.ActiveEngineManager.__init__

        def slow_init(self, *a, **kw):
            call_count["n"] += 1
            construction_started.set()
            release.wait(2.0)
            real_init(self, *a, **kw)

        results = []

        def worker():
            results.append(acp._engine_manager())

        with patch.object(server_module.ActiveEngineManager, "__init__", slow_init):
            t1 = threading.Thread(target=worker)
            t1.start()
            self.assertTrue(construction_started.wait(2.0))
            t2 = threading.Thread(target=worker)
            t2.start()
            release.set()
            t1.join(2.0)
            t2.join(2.0)

        self.assertEqual(call_count["n"], 1)
        self.assertEqual(len(results), 2)
        self.assertIs(results[0], results[1])


class TestCompletedAnswerSurvivesLateFailure(AcpTestCase):
    """Second-review finding #2: don't pop the assistant turn once it was actually appended."""

    def test_send_result_failure_after_a_completed_answer_does_not_discard_it(self):
        def source(*_a, **_k):
            return iter([{"content": "done"}])

        def flaky_send_result(_req_id, _result):
            raise RuntimeError("stdout gone")

        with patch("prism.acp._server_delta_stream", side_effect=source), \
             patch("prism.acp._send_result", side_effect=flaky_send_result), \
             patch("prism.acp._write"):
            session_id = self._new_session()
            thread = acp.handle_session_prompt(
                {"sessionId": session_id, "prompt": [{"type": "text", "text": "hi"}]}, 1)
            thread.join(2.0)

        self.assertEqual(
            acp._sessions[session_id].messages,
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "done"}],
        )


class TestErrorCodeLabeling(AcpTestCase):
    """Second-review finding #3: an internal fault must not be mislabeled as a bad request."""

    def test_internal_fault_in_session_new_is_labeled_internal_error(self):
        class BrokenCatalog:
            def list_all_models(self, include_ollama=True):
                raise OSError("disk gone")

        captured = []
        with patch("prism.acp._catalog", return_value=BrokenCatalog()), \
             patch("prism.acp._write", side_effect=captured.append):
            acp._dispatch({"jsonrpc": "2.0", "id": 1, "method": "session/new", "params": {"cwd": "/tmp"}})
        response = next(m for m in captured if m.get("id") == 1)
        self.assertEqual(response["error"]["code"], -32603)

    def test_malformed_params_is_still_labeled_invalid_params(self):
        captured = []
        with patch("prism.acp._write", side_effect=captured.append):
            acp._dispatch({"jsonrpc": "2.0", "id": 2, "method": "initialize", "params": ["x"]})
        response = next(m for m in captured if m.get("id") == 2)
        self.assertEqual(response["error"]["code"], -32602)

    def test_internal_bug_raising_attribute_error_is_still_labeled_internal_error(self):
        """Third-review finding #4: classifying by exception *type* is unsound — an internal bug
        (here, a catalog entry with a non-string `device`) raises the exact same `AttributeError` a
        malformed request would, and must not be mislabeled -32602 just because of that."""
        class BadCatalog:
            def list_all_models(self, include_ollama=True):
                return [{"id": "x", "device": None}]  # pick_default_model does device.lower()

        captured = []
        with patch("prism.acp._catalog", return_value=BadCatalog()), \
             patch("prism.acp._write", side_effect=captured.append):
            acp._dispatch({"jsonrpc": "2.0", "id": 3, "method": "session/new", "params": {"cwd": "/tmp"}})
        response = next(m for m in captured if m.get("id") == 3)
        self.assertEqual(response["error"]["code"], -32603)

    def test_non_string_session_id_is_invalid_params_not_a_crash(self):
        captured = []
        with patch("prism.acp._write", side_effect=captured.append):
            acp._dispatch({"jsonrpc": "2.0", "id": 4, "method": "session/prompt",
                           "params": {"sessionId": ["not", "a", "string"], "prompt": []}})
        response = next(m for m in captured if m.get("id") == 4)
        self.assertEqual(response["error"]["code"], -32602)


class TestPathologicalStdinLine(unittest.TestCase):
    """Third-review finding #2: a deeply-nested-but-JSON-valid line raises `RecursionError`, not
    `JSONDecodeError` — a third crash vector distinct from malformed shape or write failure."""

    def test_deeply_nested_json_does_not_crash(self):
        pathological = "[" * 100000 + "]" * 100000
        with patch.object(acp.sys, "stderr"):
            acp._handle_line(pathological)  # must not raise

    def test_plain_invalid_json_is_silently_dropped(self):
        with patch.object(acp.sys, "stderr"):
            acp._handle_line("not json at all")  # must not raise


class TestDirectEngineFallback(unittest.TestCase):
    """With `prism serve` unreachable, `session/prompt` runs the model in this process (mirrors `mcp.py`)."""

    def setUp(self):
        FakeEngine.reset()
        acp._sessions.clear()
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        make_model(self.tmp, "qwen-coder-gpu", "qwen2")
        catalog = ModelCatalog(search_paths=[self.tmp])
        manager = ActiveEngineManager(catalog=catalog, engine_factory=FakeEngine)
        for target, value in (("_catalog_instance", catalog), ("_manager", manager)):
            p = patch.object(acp, target, value)
            p.start()
            self.addCleanup(p.stop)
        p = patch("prism.catalog.list_ollama_models", return_value=[])
        p.start()
        self.addCleanup(p.stop)

    def test_falls_back_to_direct_engine_when_server_unreachable(self):
        captured = []
        with patch("prism.acp._server_delta_stream", side_effect=urllib.error.URLError("refused")), \
             patch("prism.acp._write", side_effect=captured.append):
            session_id = acp.handle_session_new({"cwd": "/tmp"})["sessionId"]
            self.assertEqual(acp._sessions[session_id].model, "qwen-coder-gpu")
            thread = acp.handle_session_prompt(
                {"sessionId": session_id, "prompt": [{"type": "text", "text": "hi"}]}, 3)
            thread.join(2.0)

        updates = [m for m in captured if m.get("method") == "session/update"]
        text = "".join(u["params"]["update"]["content"]["text"] for u in updates)
        self.assertEqual(text, "Hello world")
        response = next(m for m in captured if m.get("id") == 3)
        self.assertEqual(response["result"]["stopReason"], "end_turn")


class TestDirectEngineFallbackGuards(AcpTestCase):
    """Fourth-review finding #1: the direct-engine fallback must give a clear error for a model it
    cannot actually run in-process (an Ollama model, or a model that resolves to nothing), instead
    of a raw KeyError/AttributeError bubbling up as an unhelpful -32000 message."""

    def test_ollama_model_gives_a_clear_error_not_a_keyerror(self):
        class OllamaOnlyCatalog:
            def list_all_models(self, include_ollama=True):
                return [{"id": "ollama:llama3", "name": "llama3", "backend": "ollama"}]

            def resolve_model(self, model):
                return {"id": "ollama:llama3", "name": "llama3", "backend": "ollama"}

        captured = []
        with patch("prism.acp._catalog", return_value=OllamaOnlyCatalog()), \
             patch("prism.acp._server_delta_stream", side_effect=urllib.error.URLError("refused")), \
             patch("prism.acp._write", side_effect=captured.append):
            session_id = self._new_session()
            self.assertEqual(acp._sessions[session_id].model, "ollama:llama3")
            thread = acp.handle_session_prompt(
                {"sessionId": session_id, "prompt": [{"type": "text", "text": "hi"}]}, 1)
            thread.join(2.0)

        response = next(m for m in captured if m.get("id") == 1)
        self.assertEqual(response["error"]["code"], -32000)
        self.assertIn("prism serve", response["error"]["message"])
        self.assertNotIn("'path'", response["error"]["message"])

    def test_unresolvable_model_gives_a_clear_error_not_a_crash(self):
        class EmptyCatalog:
            def list_all_models(self, include_ollama=True):
                return []

            def resolve_model(self, model):
                return None

        captured = []
        with patch("prism.acp._catalog", return_value=EmptyCatalog()), \
             patch("prism.acp._server_delta_stream", side_effect=urllib.error.URLError("refused")), \
             patch("prism.acp._write", side_effect=captured.append):
            session_id = self._new_session()
            thread = acp.handle_session_prompt(
                {"sessionId": session_id, "prompt": [{"type": "text", "text": "hi"}]}, 1)
            thread.join(2.0)

        response = next(m for m in captured if m.get("id") == 1)
        self.assertEqual(response["error"]["code"], -32000)
        self.assertIn("prism serve", response["error"]["message"])


class TestHttpErrorDetailReadFailure(AcpTestCase):
    """Fourth-review finding #2: if reading the HTTPError body itself fails, a response must still
    be sent — not silently dropped by an exception escaping the `except HTTPError` clause."""

    def test_http_error_whose_body_read_fails_still_sends_a_response(self):
        class BrokenBody:
            def read(self):
                raise OSError("connection reset while reading body")

            def close(self):
                pass

        http_err = urllib.error.HTTPError(
            "http://localhost:5272/v1/chat/completions", 500, "Server Error", {}, BrokenBody())
        captured = []
        with patch("prism.acp._server_delta_stream", side_effect=http_err), \
             patch("prism.acp._write", side_effect=captured.append):
            session_id = self._new_session()
            thread = acp.handle_session_prompt(
                {"sessionId": session_id, "prompt": [{"type": "text", "text": "hi"}]}, 1)
            thread.join(2.0)

        self.assertFalse(thread.is_alive(), "worker thread died without sending a response")
        response = next((m for m in captured if m.get("id") == 1), None)
        self.assertIsNotNone(response, "no response was ever sent for this request")
        self.assertEqual(response["error"]["code"], -32000)


class TestLogging(AcpTestCase):
    """A session's lifecycle can be traced through `logging.getLogger("prism.acp")` by
    `session_id`/`req_id`: session creation, prompt accept/start/finish, the direct-engine
    fallback trigger, failures, and cancellation."""

    def test_session_new_logs_the_session_id_and_model(self):
        with self.assertLogs("prism.acp", level="INFO") as cm:
            result = acp.handle_session_new({"cwd": "/tmp"})
        joined = "\n".join(cm.output)
        self.assertIn("session/new", joined)
        self.assertIn(result["sessionId"], joined)

    def test_session_prompt_success_traces_accept_start_and_finish(self):
        def source(*_a, **_k):
            return iter([{"content": "hi"}])

        with patch("prism.acp._server_delta_stream", side_effect=source), \
             patch("prism.acp._write"):
            session_id = self._new_session()
            with self.assertLogs("prism.acp", level="DEBUG") as cm:
                thread = acp.handle_session_prompt(
                    {"sessionId": session_id, "prompt": [{"type": "text", "text": "hi"}]}, 1)
                thread.join(2.0)
        joined = "\n".join(cm.output)
        self.assertIn("session/prompt accepted", joined)
        self.assertIn("session/prompt started", joined)
        self.assertIn("session/prompt finished", joined)
        self.assertIn(session_id, joined)

    def test_rejected_prompt_logs_at_debug_with_the_reason(self):
        with patch("prism.acp._write"):
            session_id = self._new_session()
            with self.assertLogs("prism.acp", level="DEBUG") as cm:
                acp.handle_session_prompt(
                    {"sessionId": session_id, "prompt": [{"type": "image", "data": "x"}]}, 1)
        joined = "\n".join(cm.output)
        self.assertIn("session/prompt rejected", joined)

    def test_generation_failure_logs_a_warning(self):
        http_err = urllib.error.HTTPError(
            "http://localhost:5272/v1/chat/completions", 500, "Server Error", {},
            io.BytesIO(b'{"error":{"message":"boom"}}'))
        with patch("prism.acp._server_delta_stream", side_effect=http_err), \
             patch("prism.acp._write"):
            session_id = self._new_session()
            with self.assertLogs("prism.acp", level="WARNING") as cm:
                thread = acp.handle_session_prompt(
                    {"sessionId": session_id, "prompt": [{"type": "text", "text": "hi"}]}, 1)
                thread.join(2.0)
        joined = "\n".join(cm.output)
        self.assertIn("session/prompt failed", joined)

    def test_session_cancel_logs_at_debug(self):
        with patch("prism.acp._write"):
            session_id = self._new_session()
            with self.assertLogs("prism.acp", level="DEBUG") as cm:
                acp.handle_session_cancel({"sessionId": session_id})
        joined = "\n".join(cm.output)
        self.assertIn("session/cancel", joined)
        self.assertIn(session_id, joined)

    def test_dispatch_logs_the_method_and_id(self):
        with patch("prism.acp._write"):
            with self.assertLogs("prism.acp", level="DEBUG") as cm:
                acp._dispatch({"jsonrpc": "2.0", "id": 42, "method": "initialize", "params": {}})
        joined = "\n".join(cm.output)
        self.assertIn("dispatch", joined)
        self.assertIn("initialize", joined)
        self.assertIn("42", joined)

    def test_broken_stdout_write_logs_a_warning(self):
        with patch.object(acp.sys.stdout, "write", side_effect=BrokenPipeError("broken")), \
             patch.object(acp.sys.stderr, "write"):
            with self.assertLogs("prism.acp", level="WARNING") as cm:
                acp._write({"jsonrpc": "2.0", "id": 1, "result": {}})
        self.assertIn("stdout is gone", "\n".join(cm.output))

    def test_malformed_stdin_line_logs_a_warning(self):
        with patch.object(acp.sys, "stderr"):
            with self.assertLogs("prism.acp", level="WARNING") as cm:
                acp._handle_line("not json at all")
        self.assertIn("dropped one malformed request", "\n".join(cm.output))


class TestDirectEngineFallbackFallbackLogging(unittest.TestCase):
    """The direct-engine fallback trigger itself is logged at WARNING (it means `prism serve` is
    unreachable, which is operationally significant even though the request still succeeds)."""

    def setUp(self):
        FakeEngine.reset()
        acp._sessions.clear()
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        make_model(self.tmp, "qwen-coder-gpu", "qwen2")
        catalog = ModelCatalog(search_paths=[self.tmp])
        manager = ActiveEngineManager(catalog=catalog, engine_factory=FakeEngine)
        for target, value in (("_catalog_instance", catalog), ("_manager", manager)):
            p = patch.object(acp, target, value)
            p.start()
            self.addCleanup(p.stop)
        p = patch("prism.catalog.list_ollama_models", return_value=[])
        p.start()
        self.addCleanup(p.stop)

    def test_fallback_trigger_logs_a_warning(self):
        with patch("prism.acp._server_delta_stream", side_effect=urllib.error.URLError("refused")), \
             patch("prism.acp._write"):
            session_id = acp.handle_session_new({"cwd": "/tmp"})["sessionId"]
            with self.assertLogs("prism.acp", level="WARNING") as cm:
                thread = acp.handle_session_prompt(
                    {"sessionId": session_id, "prompt": [{"type": "text", "text": "hi"}]}, 1)
                thread.join(2.0)
        joined = "\n".join(cm.output)
        self.assertIn("falling back to the direct engine", joined)


class TestAcpRpc(unittest.TestCase):
    def setUp(self):
        if hasattr(acp, "_pending"):
            acp._pending.clear()
        if hasattr(acp, "_request_id_counter"):
            import itertools
            acp._request_id_counter = itertools.count(1)

    def test_send_request_round_trip_writes_request_and_resolves_on_response(self):
        captured = []
        result_holder = []
        exc_holder = []

        def worker():
            try:
                res = acp._send_request("_test_echo", {"x": 1})
                result_holder.append(res)
            except Exception as ex:
                exc_holder.append(ex)

        with patch("prism.acp._write", side_effect=captured.append):
            t = threading.Thread(target=worker)
            t.start()
            for _ in range(50):
                if captured or exc_holder:
                    break
                time.sleep(0.01)
            if exc_holder:
                raise exc_holder[0]
            self.assertEqual(len(captured), 1)
            req = captured[0]
            self.assertEqual(req.get("jsonrpc"), "2.0")
            self.assertEqual(req.get("id"), 1)
            self.assertEqual(req.get("method"), "_test_echo")
            self.assertEqual(req.get("params"), {"x": 1})

            acp._handle_line(json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"echo": 1}}))
            t.join(2.0)
            self.assertFalse(t.is_alive())
            self.assertEqual(exc_holder, [])
            self.assertEqual(result_holder, [{"echo": 1}])

    def test_send_request_raises_AcpClientError_on_jsonrpc_error_response(self):
        captured = []
        exc_holder = []

        def worker():
            try:
                acp._send_request("_test_fail", {})
            except Exception as ex:
                exc_holder.append(ex)

        with patch("prism.acp._write", side_effect=captured.append):
            t = threading.Thread(target=worker)
            t.start()
            for _ in range(50):
                if captured:
                    break
                time.sleep(0.01)
            self.assertEqual(len(captured), 1)
            req_id = captured[0]["id"]
            acp._handle_line(json.dumps({
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32600, "message": "bad"}
            }))
            t.join(2.0)
            self.assertFalse(t.is_alive())
            self.assertEqual(len(exc_holder), 1)
            err = exc_holder[0]
            self.assertIsInstance(err, acp.AcpClientError)
            self.assertEqual(err.code, -32600)
            self.assertEqual(err.message, "bad")

    def test_pending_futures_are_cancelled_when_connection_drops(self):
        captured = []
        exc1, exc2 = [], []

        def worker1():
            try:
                acp._send_request("req1", {})
            except Exception as ex:
                exc1.append(ex)

        def worker2():
            try:
                acp._send_request("req2", {})
            except Exception as ex:
                exc2.append(ex)

        with patch("prism.acp._write", side_effect=captured.append):
            t1 = threading.Thread(target=worker1)
            t2 = threading.Thread(target=worker2)
            t1.start()
            t2.start()
            for _ in range(50):
                if len(captured) >= 2:
                    break
                time.sleep(0.01)
            self.assertEqual(len(captured), 2)
            acp._close_pending()
            t1.join(2.0)
            t2.join(2.0)
            self.assertFalse(t1.is_alive())
            self.assertFalse(t2.is_alive())
            self.assertEqual(len(exc1), 1)
            self.assertEqual(len(exc2), 1)
            self.assertIsInstance(exc1[0], ConnectionError)
            self.assertIsInstance(exc2[0], ConnectionError)
            self.assertEqual(len(acp._pending), 0)

    def test_send_request_id_is_monotonic_across_calls(self):
        ids = [acp._next_request_id() for _ in range(5)]
        self.assertEqual(ids, [1, 2, 3, 4, 5])


class TestAcpToolLoop(AcpTestCase):
    def setUp(self):
        super().setUp()
        if hasattr(acp, "_pending"):
            acp._pending.clear()
        if hasattr(acp, "_request_id_counter"):
            import itertools
            acp._request_id_counter = itertools.count(1)
        if hasattr(acp, "_client_fs_capabilities"):
            acp._client_fs_capabilities = {"readTextFile": True, "writeTextFile": True}

    def test_initialize_captures_fs_capabilities_into_session(self):
        acp.handle_initialize({
            "protocolVersion": 1,
            "clientCapabilities": {"fs": {"readTextFile": True, "writeTextFile": False}},
        })
        session_id = self._new_session()
        tools = acp._resolve_tools(acp._sessions[session_id])
        self.assertEqual([t["function"]["name"] for t in tools], ["read_file"])

    def test_initialize_without_fs_capabilities_filters_tools(self):
        acp.handle_initialize({"protocolVersion": 1, "clientCapabilities": {}})
        session_id = self._new_session()
        tools = acp._resolve_tools(acp._sessions[session_id])
        self.assertEqual(tools, [])

    def test_session_prompt_with_read_file_tool_call_executes_via_fs_read_text_file(self):
        turn = 0
        def fake_stream(*_a, **_k):
            nonlocal turn
            turn += 1
            if turn == 1:
                return iter([{"content": '<tool_call>{"name":"read_file","arguments":{"path":"/tmp/x"}}</tool_call>'}])
            return iter([{"content": "Here is the content: hello"}])

        captured = []
        fs_requests = []

        def fake_send_request(method, params, *, timeout=None):
            fs_requests.append((method, params))
            if method == "fs/read_text_file":
                return {"content": "hello"}
            return {}

        with patch("prism.acp._server_delta_stream", side_effect=fake_stream), \
             patch("prism.acp._write", side_effect=captured.append), \
             patch("prism.acp._send_request", side_effect=fake_send_request):
            session_id = self._new_session()
            acp._sessions[session_id].client_capabilities = {"readTextFile": True, "writeTextFile": True}
            thread = acp.handle_session_prompt(
                {"sessionId": session_id, "prompt": [{"type": "text", "text": "read /tmp/x"}]}, 1)
            self.assertIsNotNone(thread)
            thread.join(2.0)
            self.assertFalse(thread.is_alive())

        tool_calls = [m for m in captured if m.get("method") == "session/update" and m.get("params", {}).get("update", {}).get("sessionUpdate") == "tool_call"]
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["params"]["update"]["kind"], "read")
        self.assertEqual(tool_calls[0]["params"]["update"]["status"], "in_progress")

        tool_updates = [m for m in captured if m.get("method") == "session/update" and m.get("params", {}).get("update", {}).get("sessionUpdate") == "tool_call_update"]
        self.assertEqual(len(tool_updates), 1)
        self.assertEqual(tool_updates[0]["params"]["update"]["status"], "completed")
        self.assertEqual(tool_updates[0]["params"]["update"]["content"][0]["content"]["text"], "hello")

        tool_msgs = [m for m in acp._sessions[session_id].messages if m.get("role") == "tool"]
        self.assertTrue(tool_msgs)
        self.assertEqual(tool_msgs[-1]["content"], "hello")

        resps = [m for m in captured if m.get("id") == 1]
        self.assertTrue(resps)
        self.assertEqual(resps[0]["result"]["stopReason"], "end_turn")

        write_requests = [r for r in fs_requests if r[0] == "fs/write_text_file"]
        self.assertEqual(write_requests, [])

    def test_session_prompt_with_write_file_requests_permission_then_writes(self):
        turn = 0
        def fake_stream(*_a, **_k):
            nonlocal turn
            turn += 1
            if turn == 1:
                return iter([{"content": '<tool_call>{"name":"write_file","arguments":{"path":"/tmp/x","content":"hello world"}}</tool_call>'}])
            return iter([{"content": "Done."}])

        captured = []
        calls_order = []

        def fake_send_request(method, params, *, timeout=None):
            calls_order.append(method)
            if method == "session/request_permission":
                return {"outcome": {"outcome": "selected", "optionId": "allow_once"}}
            elif method == "fs/write_text_file":
                return {}
            return {}

        with patch("prism.acp._server_delta_stream", side_effect=fake_stream), \
             patch("prism.acp._write", side_effect=captured.append), \
             patch("prism.acp._send_request", side_effect=fake_send_request):
            session_id = self._new_session()
            acp._sessions[session_id].client_capabilities = {"readTextFile": True, "writeTextFile": True}
            thread = acp.handle_session_prompt(
                {"sessionId": session_id, "prompt": [{"type": "text", "text": "write /tmp/x"}]}, 1)
            self.assertIsNotNone(thread)
            thread.join(2.0)
            self.assertFalse(thread.is_alive())

        self.assertEqual(calls_order, ["session/request_permission", "fs/write_text_file"])
        tool_updates = [m for m in captured if m.get("method") == "session/update" and m.get("params", {}).get("update", {}).get("sessionUpdate") == "tool_call_update"]
        self.assertEqual(len(tool_updates), 1)
        self.assertEqual(tool_updates[0]["params"]["update"]["status"], "completed")

        tool_msgs = [m for m in acp._sessions[session_id].messages if m.get("role") == "tool"]
        self.assertTrue(tool_msgs)
        self.assertEqual(tool_msgs[-1]["content"], "wrote 11 bytes to /tmp/x")

    def test_session_prompt_with_write_file_rejected_skips_fs_write(self):
        turn = 0
        def fake_stream(*_a, **_k):
            nonlocal turn
            turn += 1
            if turn == 1:
                return iter([{"content": '<tool_call>{"name":"write_file","arguments":{"path":"/tmp/x","content":"hello"}}</tool_call>'}])
            return iter([{"content": "Denied."}])

        captured = []
        calls = []

        def fake_send_request(method, params, *, timeout=None):
            calls.append(method)
            if method == "session/request_permission":
                return {"outcome": {"outcome": "selected", "optionId": "reject_once"}}
            return {}

        with patch("prism.acp._server_delta_stream", side_effect=fake_stream), \
             patch("prism.acp._write", side_effect=captured.append), \
             patch("prism.acp._send_request", side_effect=fake_send_request):
            session_id = self._new_session()
            acp._sessions[session_id].client_capabilities = {"readTextFile": True, "writeTextFile": True}
            thread = acp.handle_session_prompt(
                {"sessionId": session_id, "prompt": [{"type": "text", "text": "write"}]}, 1)
            thread.join(2.0)
            self.assertFalse(thread.is_alive())

        self.assertNotIn("fs/write_text_file", calls)
        tool_updates = [m for m in captured if m.get("method") == "session/update" and m.get("params", {}).get("update", {}).get("sessionUpdate") == "tool_call_update"]
        self.assertEqual(tool_updates[0]["params"]["update"]["status"], "failed")
        self.assertEqual(tool_updates[0]["params"]["update"]["content"][0]["content"]["text"], "permission denied")

        tool_msgs = [m for m in acp._sessions[session_id].messages if m.get("role") == "tool"]
        self.assertEqual(tool_msgs[-1]["content"], "permission denied")

    def test_session_prompt_with_write_file_cancelled_resolves_session_prompt_with_cancelled(self):
        turn = 0
        def fake_stream(*_a, **_k):
            nonlocal turn
            turn += 1
            if turn == 1:
                return iter([{"content": '<tool_call>{"name":"write_file","arguments":{"path":"/tmp/x","content":"hello"}}</tool_call>'}])
            return iter([{"content": "Never"}])

        captured = []

        def fake_send_request(method, params, *, timeout=None):
            if method == "session/request_permission":
                return {"outcome": {"outcome": "cancelled"}}
            return {}

        with patch("prism.acp._server_delta_stream", side_effect=fake_stream), \
             patch("prism.acp._write", side_effect=captured.append), \
             patch("prism.acp._send_request", side_effect=fake_send_request):
            session_id = self._new_session()
            acp._sessions[session_id].client_capabilities = {"readTextFile": True, "writeTextFile": True}
            thread = acp.handle_session_prompt(
                {"sessionId": session_id, "prompt": [{"type": "text", "text": "write"}]}, 1)
            thread.join(2.0)
            self.assertFalse(thread.is_alive())

        resps = [m for m in captured if m.get("id") == 1]
        self.assertEqual(resps[0]["result"]["stopReason"], "cancelled")

    def test_session_prompt_fs_read_text_file_error_returns_error_as_tool_content(self):
        turn = 0
        def fake_stream(*_a, **_k):
            nonlocal turn
            turn += 1
            if turn == 1:
                return iter([{"content": '<tool_call>{"name":"read_file","arguments":{"path":"/missing"}}</tool_call>'}])
            return iter([{"content": "File missing."}])

        captured = []

        def fake_send_request(method, params, *, timeout=None):
            if method == "fs/read_text_file":
                raise acp.AcpClientError(-32600, "File not found")
            return {}

        with patch("prism.acp._server_delta_stream", side_effect=fake_stream), \
             patch("prism.acp._write", side_effect=captured.append), \
             patch("prism.acp._send_request", side_effect=fake_send_request):
            session_id = self._new_session()
            acp._sessions[session_id].client_capabilities = {"readTextFile": True, "writeTextFile": True}
            thread = acp.handle_session_prompt(
                {"sessionId": session_id, "prompt": [{"type": "text", "text": "read"}]}, 1)
            thread.join(2.0)
            self.assertFalse(thread.is_alive())

        tool_updates = [m for m in captured if m.get("method") == "session/update" and m.get("params", {}).get("update", {}).get("sessionUpdate") == "tool_call_update"]
        self.assertEqual(tool_updates[0]["params"]["update"]["status"], "failed")
        tool_msgs = [m for m in acp._sessions[session_id].messages if m.get("role") == "tool"]
        self.assertIn("File not found", tool_msgs[-1]["content"])

    def test_session_prompt_tool_loop_respects_max_iterations(self):
        def fake_stream(*_a, **_k):
            return iter([{"content": '<tool_call>{"name":"read_file","arguments":{"path":"/loop"}}</tool_call>'}])

        captured = []

        def fake_send_request(method, params, *, timeout=None):
            return {"content": "loop"}

        with patch("prism.acp._server_delta_stream", side_effect=fake_stream), \
             patch("prism.acp._write", side_effect=captured.append), \
             patch("prism.acp._send_request", side_effect=fake_send_request):
            session_id = self._new_session()
            acp._sessions[session_id].client_capabilities = {"readTextFile": True, "writeTextFile": True}
            thread = acp.handle_session_prompt(
                {"sessionId": session_id, "prompt": [{"type": "text", "text": "loop"}]}, 1)
            thread.join(5.0)
            self.assertFalse(thread.is_alive())

        resps = [m for m in captured if m.get("id") == 1]
        self.assertEqual(resps[0]["result"]["stopReason"], "end_turn")
        chunks = [m["params"]["update"]["content"]["text"] for m in captured
                  if m.get("method") == "session/update" and m.get("params", {}).get("update", {}).get("sessionUpdate") == "agent_message_chunk"]
        self.assertTrue(any("tool loop budget exhausted" in c for c in chunks))

    def test_session_prompt_cancel_during_tool_loop_stops_cleanly(self):
        turn = 0
        captured = []
        calls = []

        def fake_stream(*_a, **_k):
            nonlocal turn
            turn += 1
            return iter([{"content": '<tool_call>{"name":"read_file","arguments":{"path":"/x"}}</tool_call>'}])

        def fake_send_request(method, params, *, timeout=None):
            calls.append(method)
            acp._sessions[session_id].cancel_event.set()
            return {"content": "x"}

        with patch("prism.acp._server_delta_stream", side_effect=fake_stream), \
             patch("prism.acp._write", side_effect=captured.append), \
             patch("prism.acp._send_request", side_effect=fake_send_request):
            session_id = self._new_session()
            acp._sessions[session_id].client_capabilities = {"readTextFile": True, "writeTextFile": True}
            thread = acp.handle_session_prompt(
                {"sessionId": session_id, "prompt": [{"type": "text", "text": "hi"}]}, 1)
            thread.join(2.0)
            self.assertFalse(thread.is_alive())

        resps = [m for m in captured if m.get("id") == 1]
        self.assertEqual(resps[0]["result"]["stopReason"], "cancelled")
        self.assertEqual(len(calls), 1)

    def test_tool_call_id_is_unique_within_session(self):
        turn = 0
        def fake_stream(*_a, **_k):
            nonlocal turn
            turn += 1
            if turn == 1:
                return iter([{"content": '<tool_call>{"name":"read_file","arguments":{"path":"/a"}}</tool_call><tool_call>{"name":"read_file","arguments":{"path":"/b"}}</tool_call>'}])
            return iter([{"content": "done"}])

        captured = []
        with patch("prism.acp._server_delta_stream", side_effect=fake_stream), \
             patch("prism.acp._write", side_effect=captured.append), \
             patch("prism.acp._send_request", return_value={"content": "file"}):
            session_id = self._new_session()
            acp._sessions[session_id].client_capabilities = {"readTextFile": True, "writeTextFile": True}
            thread = acp.handle_session_prompt(
                {"sessionId": session_id, "prompt": [{"type": "text", "text": "both"}]}, 1)
            thread.join(2.0)
            self.assertFalse(thread.is_alive())

        tool_calls = [m for m in captured if m.get("method") == "session/update" and m.get("params", {}).get("update", {}).get("sessionUpdate") == "tool_call"]
        self.assertEqual(len(tool_calls), 2)
        id1 = tool_calls[0]["params"]["update"]["toolCallId"]
        id2 = tool_calls[1]["params"]["update"]["toolCallId"]
        self.assertNotEqual(id1, id2)

    def test_tool_call_notifications_carry_correct_kind(self):
        turn = 0
        def fake_stream(*_a, **_k):
            nonlocal turn
            turn += 1
            if turn == 1:
                return iter([{"content": '<tool_call>{"name":"read_file","arguments":{"path":"/a"}}</tool_call><tool_call>{"name":"write_file","arguments":{"path":"/b","content":"c"}}</tool_call>'}])
            return iter([{"content": "done"}])

        captured = []
        def fake_send_request(method, params, *, timeout=None):
            if method == "session/request_permission":
                return {"outcome": {"outcome": "selected", "optionId": "allow_once"}}
            return {"content": "ok"}

        with patch("prism.acp._server_delta_stream", side_effect=fake_stream), \
             patch("prism.acp._write", side_effect=captured.append), \
             patch("prism.acp._send_request", side_effect=fake_send_request):
            session_id = self._new_session()
            acp._sessions[session_id].client_capabilities = {"readTextFile": True, "writeTextFile": True}
            thread = acp.handle_session_prompt(
                {"sessionId": session_id, "prompt": [{"type": "text", "text": "both"}]}, 1)
            thread.join(2.0)
            self.assertFalse(thread.is_alive())

        tool_calls = [m for m in captured if m.get("method") == "session/update" and m.get("params", {}).get("update", {}).get("sessionUpdate") == "tool_call"]
        self.assertEqual(len(tool_calls), 2)
        self.assertEqual(tool_calls[0]["params"]["update"]["kind"], "read")
        self.assertEqual(tool_calls[1]["params"]["update"]["kind"], "edit")

    def test_tool_call_for_non_advertised_capability_returns_error_content(self):
        turn = 0
        def fake_stream(*_a, **_k):
            nonlocal turn
            turn += 1
            if turn == 1:
                return iter([{"content": '<tool_call>{"name":"read_file","arguments":{"path":"/tmp/x"}}</tool_call>'}])
            return iter([{"content": "Understood."}])

        captured = []
        fs_calls = []
        with patch("prism.acp._server_delta_stream", side_effect=fake_stream), \
             patch("prism.acp._write", side_effect=captured.append), \
             patch("prism.acp._send_request", side_effect=lambda m, p, **k: fs_calls.append(m)):
            session_id = self._new_session()
            acp._sessions[session_id].client_capabilities = {"readTextFile": False, "writeTextFile": False}
            thread = acp.handle_session_prompt(
                {"sessionId": session_id, "prompt": [{"type": "text", "text": "read"}]}, 1)
            thread.join(2.0)
            self.assertFalse(thread.is_alive())

        self.assertEqual(fs_calls, [])
        tool_updates = [m for m in captured if m.get("method") == "session/update" and m.get("params", {}).get("update", {}).get("sessionUpdate") == "tool_call_update"]
        self.assertEqual(len(tool_updates), 1)
        self.assertEqual(tool_updates[0]["params"]["update"]["status"], "failed")
        tool_msgs = [m for m in acp._sessions[session_id].messages if m.get("role") == "tool"]
        self.assertTrue(tool_msgs)
        self.assertIn("fs.readTextFile is not advertised by the client", tool_msgs[-1]["content"])


class TestAcpAgentNeverTouchesFiles(AcpTestCase):
    def test_agent_never_opens_file_or_runs_subprocess(self):
        turn = 0
        def fake_stream(*_a, **_k):
            nonlocal turn
            turn += 1
            if turn == 1:
                return iter([{"content": '<tool_call>{"name":"read_file","arguments":{"path":"/secret"}}</tool_call><tool_call>{"name":"write_file","arguments":{"path":"/out","content":"data"}}</tool_call>'}])
            return iter([{"content": "done"}])

        def fake_send_request(method, params, *, timeout=None):
            if method == "session/request_permission":
                return {"outcome": {"outcome": "selected", "optionId": "allow_once"}}
            return {"content": "file contents"}

        mock_open = MagicMock()
        mock_popen = MagicMock()
        mock_run = MagicMock()

        session_id = self._new_session()
        acp._sessions[session_id].client_capabilities = {"readTextFile": True, "writeTextFile": True}

        with patch("builtins.open", mock_open), \
             patch("subprocess.Popen", mock_popen), \
             patch("subprocess.run", mock_run), \
             patch("prism.acp._server_delta_stream", side_effect=fake_stream), \
             patch("prism.acp._write"), \
             patch("prism.acp._send_request", side_effect=fake_send_request):
            thread = acp.handle_session_prompt(
                {"sessionId": session_id, "prompt": [{"type": "text", "text": "process"}]}, 1)
            thread.join(2.0)
            self.assertFalse(thread.is_alive())

        tool_msgs = [m for m in acp._sessions[session_id].messages if m.get("role") == "tool"]
        self.assertEqual(len(tool_msgs), 2)
        mock_open.assert_not_called()
        mock_popen.assert_not_called()
        mock_run.assert_not_called()



class TestPhase13ReviewFindings(AcpTestCase):
    def test_send_request_raises_client_error_when_write_fails_without_deadlock(self):
        with patch("prism.acp._write", return_value=False):
            t0 = time.time()
            with self.assertRaises(acp.AcpClientError) as cm:
                acp._send_request("fs/read_text_file", {"sessionId": "s", "path": "/f"}, timeout=0.1)
            self.assertLess(time.time() - t0, 0.5)
            self.assertIn("pipe broken", str(cm.exception).lower())

    def test_send_request_aborts_when_cancel_event_set_without_deadlock(self):
        cancel_evt = threading.Event()
        def slow_wait(*args, **kwargs):
            cancel_evt.set()
            return True
        with patch("prism.acp._write", side_effect=slow_wait):
            t0 = time.time()
            with self.assertRaises(acp.AcpClientError) as cm:
                acp._send_request("session/request_permission", {}, cancel_event=cancel_evt)
            self.assertLess(time.time() - t0, 1.0)
            self.assertIn("cancelled", str(cm.exception).lower())

    def test_prompt_cancel_restores_history_without_dangling_user_or_tool_turns(self):
        turn = 0
        def fake_stream(*_a, **_k):
            nonlocal turn
            turn += 1
            if turn == 1:
                return iter([{"content": '<tool_call>{"name":"write_file","arguments":{"path":"/x","content":"y"}}</tool_call>'}])
            return iter([{"content": "done"}])

        def fake_send_request(method, params, **kwargs):
            if method == "session/request_permission":
                return {"outcome": {"outcome": "cancelled"}}
            return {}

        captured = []
        with patch("prism.acp._server_delta_stream", side_effect=fake_stream), \
             patch("prism.acp._write", side_effect=captured.append), \
             patch("prism.acp._send_request", side_effect=fake_send_request):
            session_id = self._new_session()
            session = acp._sessions[session_id]
            session.client_capabilities = {"writeTextFile": True}
            thread = acp.handle_session_prompt(
                {"sessionId": session_id, "prompt": [{"type": "text", "text": "do write"}]}, 1)
            thread.join(2.0)
            self.assertFalse(thread.is_alive())

        resps = [m for m in captured if m.get("id") == 1]
        self.assertEqual(resps[0]["result"]["stopReason"], "cancelled")
        self.assertEqual(session.messages, [])

    def test_ollama_model_supports_tools_in_acp(self):
        turn = 0
        passed_tools = None
        def fake_stream(messages, model, tools=None):
            nonlocal turn, passed_tools
            turn += 1
            passed_tools = tools
            return iter([{"content": "Understood."}])

        class StubCatalog:
            def list_all_models(self, include_ollama=True):
                return [{"id": "ollama:qwen2.5-coder:7b", "backend": "ollama", "name": "qwen2.5-coder"}]
            def resolve_model(self, model_id):
                return {"id": "ollama:qwen2.5-coder:7b", "backend": "ollama", "name": "qwen2.5-coder"}

        captured = []
        with patch("prism.acp._catalog", return_value=StubCatalog()), \
             patch("prism.acp._server_delta_stream", side_effect=fake_stream), \
             patch("prism.acp._write", side_effect=captured.append):
            session_id = self._new_session()
            session = acp._sessions[session_id]
            session.model = "ollama:qwen2.5-coder:7b"
            session.client_capabilities = {"readTextFile": True, "writeTextFile": True}
            thread = acp.handle_session_prompt(
                {"sessionId": session_id, "prompt": [{"type": "text", "text": "hello"}]}, 1)
            thread.join(2.0)
            self.assertFalse(thread.is_alive())

        self.assertIsNotNone(passed_tools)
        tool_names = [t["function"]["name"] for t in passed_tools]
        self.assertIn("read_file", tool_names)
        self.assertIn("write_file", tool_names)

    def test_response_frame_without_result_or_error_raises_client_error(self):
        event = threading.Event()
        slot = [None, None]
        with acp._pending_lock:
            acp._pending[99] = (event, slot)
        try:
            acp._dispatch({"jsonrpc": "2.0", "id": 99})
            self.assertTrue(event.is_set())
            self.assertIsInstance(slot[1], acp.AcpClientError)
            self.assertIn("neither 'result' nor 'error'", str(slot[1]))
        finally:
            with acp._pending_lock:
                acp._pending.pop(99, None)

    def test_send_request_clears_pending_future_on_write_exception(self):
        with patch("prism.acp._write", side_effect=TypeError("cannot serialize")):
            with self.assertRaises(TypeError):
                acp._send_request("test", {})
        self.assertEqual(len(acp._pending), 0)

    def test_tool_loop_exhaustion_appends_assistant_message_to_history(self):
        def fake_stream(*_a, **_k):
            return iter([{"content": '<tool_call>{"name":"read_file","arguments":{"path":"/loop"}}</tool_call>'}])

        captured = []
        with patch("prism.acp._server_delta_stream", side_effect=fake_stream), \
             patch("prism.acp._write", side_effect=captured.append), \
             patch("prism.acp._send_request", return_value={"content": "data"}):
            session_id = self._new_session()
            session = acp._sessions[session_id]
            session.client_capabilities = {"readTextFile": True}
            thread = acp.handle_session_prompt(
                {"sessionId": session_id, "prompt": [{"type": "text", "text": "loop"}]}, 1)
            thread.join(2.0)
            self.assertFalse(thread.is_alive())

        self.assertTrue(session.messages)
        self.assertEqual(session.messages[-1]["role"], "assistant")
        self.assertIn("tool loop budget exhausted", session.messages[-1]["content"])

    def test_permission_json_rpc_error_reported_in_tool_content(self):
        turn = 0
        def fake_stream(*_a, **_k):
            nonlocal turn
            turn += 1
            if turn == 1:
                return iter([{"content": '<tool_call>{"name":"write_file","arguments":{"path":"/x","content":"y"}}</tool_call>'}])
            return iter([{"content": "understood"}])

        def fake_send_request(method, params, **kwargs):
            if method == "session/request_permission":
                raise acp.AcpClientError(-32000, "Permission daemon unavailable")
            return {}

        captured = []
        with patch("prism.acp._server_delta_stream", side_effect=fake_stream), \
             patch("prism.acp._write", side_effect=captured.append), \
             patch("prism.acp._send_request", side_effect=fake_send_request):
            session_id = self._new_session()
            session = acp._sessions[session_id]
            session.client_capabilities = {"writeTextFile": True}
            thread = acp.handle_session_prompt(
                {"sessionId": session_id, "prompt": [{"type": "text", "text": "write"}]}, 1)
            thread.join(2.0)
            self.assertFalse(thread.is_alive())

        tool_msgs = [m for m in session.messages if m.get("role") == "tool"]
        self.assertTrue(tool_msgs)
        self.assertIn("Permission daemon unavailable", tool_msgs[0]["content"])

    def test_write_file_validates_object_response_schema(self):
        turn = 0
        def fake_stream(*_a, **_k):
            nonlocal turn
            turn += 1
            if turn == 1:
                return iter([{"content": '<tool_call>{"name":"write_file","arguments":{"path":"/x","content":"y"}}</tool_call>'}])
            return iter([{"content": "understood"}])

        def fake_send_request(method, params, **kwargs):
            if method == "session/request_permission":
                return {"outcome": {"outcome": "selected", "optionId": "allow_once"}}
            return "not-a-dict"

        captured = []
        with patch("prism.acp._server_delta_stream", side_effect=fake_stream), \
             patch("prism.acp._write", side_effect=captured.append), \
             patch("prism.acp._send_request", side_effect=fake_send_request):
            session_id = self._new_session()
            session = acp._sessions[session_id]
            session.client_capabilities = {"writeTextFile": True}
            thread = acp.handle_session_prompt(
                {"sessionId": session_id, "prompt": [{"type": "text", "text": "write"}]}, 1)
            thread.join(2.0)
            self.assertFalse(thread.is_alive())

        tool_updates = [m for m in captured if m.get("method") == "session/update" and m.get("params", {}).get("update", {}).get("sessionUpdate") == "tool_call_update"]
        self.assertEqual(tool_updates[-1]["params"]["update"]["status"], "failed")
        tool_msgs = [m for m in session.messages if m.get("role") == "tool"]
        self.assertIn("Invalid response from fs/write_text_file", tool_msgs[0]["content"])

    def test_write_file_handles_null_content_as_empty_string(self):
        turn = 0
        def fake_stream(*_a, **_k):
            nonlocal turn
            turn += 1
            if turn == 1:
                return iter([{"content": '<tool_call>{"name":"write_file","arguments":{"path":"/empty","content":null}}</tool_call>'}])
            return iter([{"content": "understood"}])

        sent_params = []
        def fake_send_request(method, params, **kwargs):
            if method == "session/request_permission":
                return {"outcome": {"outcome": "selected", "optionId": "allow_once"}}
            sent_params.append(params)
            return {}

        captured = []
        with patch("prism.acp._server_delta_stream", side_effect=fake_stream), \
             patch("prism.acp._write", side_effect=captured.append), \
             patch("prism.acp._send_request", side_effect=fake_send_request):
            session_id = self._new_session()
            session = acp._sessions[session_id]
            session.client_capabilities = {"writeTextFile": True}
            thread = acp.handle_session_prompt(
                {"sessionId": session_id, "prompt": [{"type": "text", "text": "write empty"}]}, 1)
            thread.join(2.0)
            self.assertFalse(thread.is_alive())

        self.assertTrue(sent_params)
        self.assertEqual(sent_params[0]["content"], "")

    def test_initialize_handles_non_dict_client_capabilities(self):
        resp = acp.handle_initialize({"protocolVersion": 1, "clientCapabilities": "invalid_string"})
        self.assertEqual(resp["protocolVersion"], 1)


if __name__ == "__main__":
    unittest.main()
