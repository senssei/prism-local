import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
from unittest.mock import patch

import prism.mcp as mcp
from prism.catalog import ModelCatalog
from prism.mcp import handle_list_tools, handle_tool_call
from prism.server import ActiveEngineManager
from tests.fakes import FakeEngine, make_model

class TestPrismMcp(unittest.TestCase):
    def test_handle_list_tools(self):
        tools = handle_list_tools()
        self.assertIsInstance(tools, list)
        self.assertEqual(len(tools), 5)
        names = [t["name"] for t in tools]
        self.assertIn("prism_ask_coder", names)
        self.assertIn("prism_code_review", names)
        self.assertIn("prism_list_models", names)
        self.assertIn("prism_get_status", names)
        self.assertIn("prism_benchmark", names)

        for t in tools:
            self.assertIn("description", t)
            self.assertIn("inputSchema", t)
            self.assertEqual(t["inputSchema"]["type"], "object")

    def test_handle_tool_call_list_models(self):
        output = handle_tool_call("prism_list_models", {})
        self.assertIsInstance(output, str)
        self.assertIn("Discovered", output)
        self.assertIn("NAME / ID", output)

    def test_handle_tool_call_get_status(self):
        output = handle_tool_call("prism_get_status", {})
        self.assertIsInstance(output, str)
        self.assertIn("PRISM SYSTEM & HARDWARE TELEMETRY", output)

    def test_handle_tool_call_insufficient_resources_hint(self):
        import io
        err_body = io.BytesIO(b'{"error":{"message":"RAM shortage","code":"insufficient_resources"}}')
        http_err = urllib.error.HTTPError("http://localhost:5272/v1/chat/completions", 503, "Server Error", {}, err_body)
        with patch("urllib.request.urlopen", side_effect=http_err):
            output = handle_tool_call("prism_ask_coder", {"task": "test", "model": "qwen"})
            self.assertIn("Hint:", output)
            self.assertIn("PRISM_RESOURCE_CHECK=off", output)

class TestServerlessFallback(unittest.TestCase):
    """With no `prism serve` reachable, tool calls run the model in this process; it must be loaded once, not once per call."""

    def setUp(self):
        FakeEngine.reset()
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        make_model(self.tmp, "qwen-coder-gpu", "qwen2")
        self.loads = []

        def factory(path):
            self.loads.append(path)
            return FakeEngine(path)

        catalog = ModelCatalog(search_paths=[self.tmp])
        manager = ActiveEngineManager(catalog=catalog, engine_factory=factory)
        for target, value in (("_catalog_instance", catalog), ("_manager", manager)):
            p = patch.object(mcp, target, value)
            p.start()
            self.addCleanup(p.stop)
        p = patch("prism.catalog.list_ollama_models", return_value=[])
        p.start()
        self.addCleanup(p.stop)
        p = patch("urllib.request.urlopen", side_effect=urllib.error.URLError("connection refused"))
        p.start()
        self.addCleanup(p.stop)
        self.addCleanup(self._cancel_timer)

    @staticmethod
    def _cancel_timer():
        if mcp._idle_timer is not None:
            mcp._idle_timer.cancel()
            mcp._idle_timer = None

    def test_two_calls_load_the_model_once(self):
        for _ in range(2):
            out = handle_tool_call("prism_ask_coder", {"task": "hi", "model": "qwen-coder-gpu"})
            self.assertIn("Hello world", out)
        self.assertEqual(len(self.loads), 1)

    def test_the_model_is_released_after_idling(self):
        handle_tool_call("prism_ask_coder", {"task": "hi", "model": "qwen-coder-gpu"})
        self.assertIsNotNone(mcp._idle_timer)  # armed: unloads the engine after IDLE_UNLOAD_SEC
        self.assertIsNotNone(mcp._manager.engine)
        mcp._manager.unload()
        self.assertIsNone(mcp._manager.engine)


class TestLazySingletonDeadlock(unittest.TestCase):
    """`_engine_manager()` calls `_catalog()` while already holding `_state_lock`; a plain
    (non-reentrant) `Lock` self-deadlocks on that nested acquire, the very first time the fallback
    path runs (found while reviewing prism/acp.py's identical pattern, which uses an RLock)."""

    def setUp(self):
        mcp._manager = None
        mcp._catalog_instance = None

    def tearDown(self):
        mcp._manager = None
        mcp._catalog_instance = None

    def test_engine_manager_does_not_deadlock_when_it_calls_catalog(self):
        result = {}

        def call():
            result["manager"] = mcp._engine_manager()

        t = threading.Thread(target=call, daemon=True)
        t.start()
        t.join(3.0)
        self.assertFalse(t.is_alive(), "mcp._engine_manager() appears to have deadlocked")
        self.assertIn("manager", result)


class TestMcpErrorHints(unittest.TestCase):
    def test_503_server_busy_does_not_suggest_resource_check(self):
        import io
        fp = io.BytesIO(b'{"error": {"message": "Server busy", "code": "server_busy"}}')
        err = urllib.error.HTTPError("http://localhost:5272/v1", 503, "Server busy", {}, fp)
        with patch("urllib.request.urlopen", side_effect=err):
            out = mcp.call_prism_server("phi-4-mini", [{"role": "user", "content": "hi"}], {})
            self.assertIn("Server busy", out)
            self.assertNotIn("PRISM_RESOURCE_CHECK", out)

    def test_503_insufficient_resources_suggests_resource_check(self):
        import io
        fp = io.BytesIO(b'{"error": {"message": "Insufficient RAM", "code": "insufficient_resources"}}')
        err = urllib.error.HTTPError("http://localhost:5272/v1", 503, "Insufficient resources", {}, fp)
        with patch("urllib.request.urlopen", side_effect=err):
            out = mcp.call_prism_server("phi-4-mini", [{"role": "user", "content": "hi"}], {})
            self.assertIn("Insufficient RAM", out)
            self.assertIn("PRISM_RESOURCE_CHECK=off", out)


class TestMcpAutoStop(unittest.TestCase):
    """Phase 6 / P10: $PRISM_MCP_AUTO_STOP_SEC makes the stdio MCP server exit after N seconds
    without a `tools/call`. Intended for test harnesses that start the MCP process but never drive it;
    without this knob the process keeps its Python/CUDA startup overhead for the rest of the session.
    """

    _PYTHON = sys.executable

    def _spawn_mcp(self, env_overrides, *, stdin_open=True):
        env = os.environ.copy()
        # Sandbox: don't leak the parent's ollama URLs, telemetry env, etc.
        for k in ("PRISM_BASE_URL", "PRISM_API_KEY", "PRISM_OLLAMA_URL", "PRISM_MAX_QUEUE",
                  "PRISM_QUEUE_TIMEOUT", "PRISM_LOAD_TIMEOUT", "PRISM_RESOURCE_CHECK"):
            env.pop(k, None)
        env.update(env_overrides)
        return subprocess.Popen(
            [self._PYTHON, "-m", "prism.cli", "mcp"],
            stdin=subprocess.PIPE if stdin_open else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )

    @staticmethod
    def _send_tools_list(proc, req_id=1):
        msg = json.dumps({"jsonrpc": "2.0", "id": req_id, "method": "tools/list"}) + "\n"
        proc.stdin.write(msg.encode("utf-8"))
        proc.stdin.flush()
        # Read the response line (one JSON object terminated by newline)
        proc.stdout.readline()

    def test_auto_stop_exits_process_after_idle(self):
        """No tool calls, `PRISM_MCP_AUTO_STOP_SEC=1` → process exits within ~3 s with status 0.
        Note: stdin must stay open — closing it would end the JSON-RPC `for line in sys.stdin`
        loop immediately and bypass the auto-stop path."""
        proc = self._spawn_mcp({"PRISM_MCP_AUTO_STOP_SEC": "1"})
        try:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                self.fail("MCP process did not exit after PRISM_MCP_AUTO_STOP_SEC=1 of idle")
            self.assertEqual(proc.returncode, 0,
                             f"MCP exited with {proc.returncode}")
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            stderr = proc.stderr.read().decode("utf-8", "replace")
            self.assertIn("PRISM_MCP_AUTO_STOP_SEC", stderr,
                          f"stderr should name the env var; got: {stderr!r}")

    def test_auto_stop_resets_on_tool_call(self):
        """`PRISM_MCP_AUTO_STOP_SEC=2`. After a `tools/list`, the timer resets: the process
        stays alive past the original 2 s deadline and only exits ~2 s after the last call."""
        proc = self._spawn_mcp({"PRISM_MCP_AUTO_STOP_SEC": "2"})
        try:
            self._send_tools_list(proc, req_id=1)  # at this point the timer is armed for ~2 s
            # Wait just before the original deadline and assert still alive.
            time.sleep(1.7)
            self.assertIsNone(proc.poll(),
                              "process exited before the auto-stop deadline could elapse")
            # Another tool call should reset the timer.
            self._send_tools_list(proc, req_id=2)
            t_after_second = time.monotonic()
            try:
                proc.wait(timeout=3.5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                self.fail("MCP did not exit ~2s after the second tool call")
            elapsed = time.monotonic() - t_after_second
            self.assertGreater(elapsed, 1.5,
                               f"exited too soon ({elapsed:.2f}s) after second call")
            self.assertLess(elapsed, 3.0,
                            f"exited too late ({elapsed:.2f}s) after second call")
            self.assertEqual(proc.returncode, 0)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    def test_default_zero_does_not_auto_stop(self):
        """Without the env var, MCP keeps running. After 1.5 s we must still be alive — regression."""
        proc = self._spawn_mcp({})
        try:
            time.sleep(1.5)
            self.assertIsNone(proc.poll(),
                              f"MCP process exited without the env var set (rc={proc.returncode})")
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


class TestLogging(unittest.TestCase):
    """Tool-call dispatch and `call_prism_server`'s fallback behavior can be traced through
    `logging.getLogger("prism.mcp")`, mirroring `prism/acp.py`'s equivalent tracing."""

    def test_tool_call_logs_the_tool_name(self):
        with self.assertLogs("prism.mcp", level="DEBUG") as cm:
            mcp.handle_tool_call("prism_list_models", {})
        joined = "\n".join(cm.output)
        self.assertIn("tool call", joined)
        self.assertIn("prism_list_models", joined)

    def test_dispatch_logs_the_method_and_id(self):
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            with self.assertLogs("prism.mcp", level="DEBUG") as cm:
                mcp.call_prism_server("hi", model="qwen")
        joined = "\n".join(cm.output)
        self.assertIn("call_prism_server", joined)
        self.assertIn("qwen", joined)

    def test_fallback_trigger_logs_a_warning(self):
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")), \
             patch("prism.catalog.list_ollama_models", return_value=[]):
            with self.assertLogs("prism.mcp", level="WARNING") as cm:
                mcp.call_prism_server("hi", model="nonexistent-model")
        joined = "\n".join(cm.output)
        self.assertIn("falling back to the direct engine", joined)

    def test_http_error_logs_a_warning(self):
        import io
        err_body = io.BytesIO(b'{"error":{"message":"boom"}}')
        http_err = urllib.error.HTTPError("http://localhost:5272/v1/chat/completions", 500, "Server Error", {}, err_body)
        with patch("urllib.request.urlopen", side_effect=http_err):
            with self.assertLogs("prism.mcp", level="WARNING") as cm:
                mcp.call_prism_server("hi", model="qwen")
        joined = "\n".join(cm.output)
        self.assertIn("call_prism_server failed", joined)


if __name__ == "__main__":
    unittest.main()
