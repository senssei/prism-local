import json
import threading
import time
import unittest
import urllib.request
from foundry_ng.server import start_server

class TestNgServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.port = 5289
        cls.server_thread = threading.Thread(
            target=start_server,
            kwargs={"port": cls.port, "host": "127.0.0.1"},
            daemon=True
        )
        cls.server_thread.start()
        time.sleep(1.0)

    def test_health_endpoint(self):
        url = f"http://127.0.0.1:{self.port}/health"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["status"], "ok")
            self.assertIn("hardware", data)

    def test_list_models_endpoint(self):
        url = f"http://127.0.0.1:{self.port}/v1/models"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["object"], "list")
            self.assertIsInstance(data["data"], list)
            self.assertGreaterEqual(len(data["data"]), 1)

if __name__ == "__main__":
    unittest.main()
