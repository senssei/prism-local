import json
import shutil
import tempfile
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


if __name__ == "__main__":
    unittest.main()
