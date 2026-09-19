import json
import unittest
from prism.mcp import handle_list_tools, handle_tool_call

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

if __name__ == "__main__":
    unittest.main()
