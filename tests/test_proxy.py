import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from foundry_wsl.proxy import resolve_daemon_url

class TestProxy(unittest.TestCase):
    def test_resolve_daemon_url_not_found(self):
        with patch("pathlib.Path.home") as mock_home:
            mock_home.return_value = Path("/nonexistent/path/for/test")
            url = resolve_daemon_url()
            self.assertIsNone(url)

    def test_resolve_daemon_url_success(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_home = Path(tmp_dir)
            foundry_dir = tmp_home / ".foundry"
            foundry_dir.mkdir()
            daemon_json = foundry_dir / "daemon.json"
            daemon_json.write_text(json.dumps({"urls": ["http://127.0.0.1:45678/"], "pid": 1234}))

            with patch("pathlib.Path.home", return_value=tmp_home):
                url = resolve_daemon_url()
                self.assertEqual(url, "http://127.0.0.1:45678")

if __name__ == "__main__":
    unittest.main()
