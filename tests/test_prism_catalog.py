import unittest
from prism.catalog import ModelCatalog

class TestPrismCatalog(unittest.TestCase):
    def setUp(self):
        self.catalog = ModelCatalog()

    def test_discover_onnx_models(self):
        models = self.catalog.discover_onnx_models()
        self.assertIsInstance(models, list)
        self.assertGreaterEqual(len(models), 1)
        # Verify Phi-4 GPU model is discovered
        names = [m["name"] for m in models]
        self.assertTrue(any("phi-4" in n.lower() for n in names))

    def test_list_all_models_includes_ollama(self):
        models = self.catalog.list_all_models(include_ollama=True)
        self.assertIsInstance(models, list)
        backends = [m.get("backend") for m in models]
        self.assertIn("onnx", backends)
        if any(b == "ollama" for b in backends):
            self.assertIn("ollama", backends)

    def test_resolve_model(self):
        resolved = self.catalog.resolve_model("Phi-4-mini-instruct-cuda-gpu")
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved["backend"], "onnx")

    def test_resolve_ollama_model(self):
        resolved = self.catalog.resolve_model("ollama:phi4-mini:latest")
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved["backend"], "ollama")

if __name__ == "__main__":
    unittest.main()
