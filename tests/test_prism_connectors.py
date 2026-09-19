import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from prism.connectors import test_mcp_protocol, connect_cursor, connect_cline, connect_mcp

class TestPrismConnectors(unittest.TestCase):
    def setUp(self):
        self.orig_cwd = os.getcwd()
        self.test_dir = tempfile.mkdtemp()
        os.chdir(self.test_dir)

    def tearDown(self):
        os.chdir(self.orig_cwd)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_live_mcp_protocol_handshake(self):
        res = test_mcp_protocol()
        self.assertTrue(res.get("success"), f"MCP handshake failed: {res.get('error')}")
        self.assertEqual(res.get("protocol_version"), "2024-11-05")
        self.assertEqual(res.get("server_name"), "prism-mcp")
        self.assertGreaterEqual(res.get("tools_count"), 5)

    def test_cursor_export_rules_and_mcp(self):
        connect_cursor(export_rules=True, export_mcp=True, test=False)
        self.assertTrue(Path(".cursorrules").exists())
        self.assertTrue((Path(".cursor") / "mcp.json").exists())

    def test_cline_export_mcp(self):
        connect_cline(export_mcp=True, test=False)
        self.assertTrue(Path("cline_mcp_settings.json").exists())

    def test_mcp_export_cursor_target(self):
        connect_mcp(target="cursor", write=True, test=False)
        self.assertTrue((Path(".cursor") / "mcp.json").exists())

    def test_cline_export_keeps_other_servers_and_backs_up(self):
        Path("cline_mcp_settings.json").write_text(json.dumps({"mcpServers": {"other": {"command": "x"}}}))
        connect_cline(export_mcp=True, test=False)
        data = json.loads(Path("cline_mcp_settings.json").read_text())
        self.assertEqual(set(data["mcpServers"]), {"other", "prism"})
        self.assertIn("autoApprove", data["mcpServers"]["prism"])
        backup = json.loads(Path("cline_mcp_settings.json.bak").read_text())
        self.assertEqual(list(backup["mcpServers"]), ["other"])

    def test_cline_mcp_target_merges_too(self):
        Path("cline_mcp_settings.json").write_text(json.dumps({"mcpServers": {"other": {"command": "x"}}}))
        connect_mcp(target="cline", write=True, test=False)
        data = json.loads(Path("cline_mcp_settings.json").read_text())
        self.assertEqual(set(data["mcpServers"]), {"other", "prism"})

    def test_invalid_existing_json_is_never_overwritten(self):
        Path("cline_mcp_settings.json").write_text("{ not json")
        connect_cline(export_mcp=True, test=False)
        self.assertEqual(Path("cline_mcp_settings.json").read_text(), "{ not json")
        self.assertFalse(Path("cline_mcp_settings.json.bak").exists())


if __name__ == "__main__":
    unittest.main()
