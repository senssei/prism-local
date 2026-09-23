import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from prism.connectors import test_mcp_protocol, test_acp_protocol, connect_cursor, connect_cline, connect_mcp, connect_acp

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


class TestAcpConnector(unittest.TestCase):
    def setUp(self):
        self.orig_cwd = os.getcwd()
        self.test_dir = tempfile.mkdtemp()
        os.chdir(self.test_dir)
        patcher = patch("pathlib.Path.home", return_value=Path(self.test_dir))
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        os.chdir(self.orig_cwd)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _cfg_path(self) -> Path:
        return Path(self.test_dir) / ".config" / "zed" / "settings.json"

    def test_write_creates_agent_servers_entry(self):
        connect_acp(write=True, test=False)
        data = json.loads(self._cfg_path().read_text())
        self.assertIn("Prism", data["agent_servers"])
        self.assertEqual(data["agent_servers"]["Prism"]["args"], ["acp"])

    def test_write_preserves_other_keys_and_backs_up(self):
        cfg = self._cfg_path()
        cfg.parent.mkdir(parents=True)
        cfg.write_text(json.dumps({"theme": "one-dark", "agent_servers": {"other": {"command": "y"}}}))
        connect_acp(write=True, test=False)
        data = json.loads(cfg.read_text())
        self.assertEqual(data["theme"], "one-dark")
        self.assertEqual(set(data["agent_servers"]), {"other", "Prism"})
        backup = json.loads(cfg.with_name("settings.json.bak").read_text())
        self.assertEqual(list(backup["agent_servers"]), ["other"])

    def test_without_write_only_prints_snippet(self):
        connect_acp(write=False, test=False)
        self.assertFalse(self._cfg_path().exists())

    def test_live_acp_protocol_handshake(self):
        res = test_acp_protocol()
        self.assertTrue(res.get("success"), f"ACP handshake failed: {res.get('error')}")
        self.assertEqual(res.get("protocol_version"), 1)
        self.assertIsNotNone(res.get("session_id"))


if __name__ == "__main__":
    unittest.main()
