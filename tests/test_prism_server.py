import json
import threading
import unittest
import urllib.request

from prism.catalog import ModelCatalog
from prism.server import ActiveEngineManager, create_server, start_server
from tests.fakes import FakeEngine


class TestPrismServerSmoke(unittest.TestCase):
    """Boots the real server class on an ephemeral port (no fixed port, no sleeps)."""

    @classmethod
    def setUpClass(cls):
        manager = ActiveEngineManager(catalog=ModelCatalog(search_paths=[]), engine_factory=FakeEngine)
        cls.server = create_server(port=0, host="127.0.0.1", manager=manager)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def _get(self, path):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))

    def test_ephemeral_port_is_assigned(self):
        self.assertGreater(self.port, 0)

    def test_health_endpoint(self):
        status, data = self._get("/health")
        self.assertEqual(status, 200)
        self.assertEqual(data["status"], "ok")
        self.assertIn("hardware", data)

    def test_list_models_endpoint(self):
        status, data = self._get("/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual(data["object"], "list")
        self.assertIsInstance(data["data"], list)

    def test_start_server_is_exported_for_the_cli(self):
        self.assertTrue(callable(start_server))


if __name__ == "__main__":
    unittest.main()
