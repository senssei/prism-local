import io
import shutil
import tempfile
import threading
import unittest
import urllib.error
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
