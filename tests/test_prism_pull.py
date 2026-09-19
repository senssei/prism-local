import unittest
from unittest.mock import patch, MagicMock
from prism.catalog import ModelCatalog, KNOWN_HF_MODELS

class TestPrismPull(unittest.TestCase):
    def setUp(self):
        self.catalog = ModelCatalog()

    def test_known_hf_models_variants(self):
        # Verify curated models have cuda variants
        for key in ["phi-4-mini", "phi-4", "phi-3.5-mini"]:
            self.assertIn(key, KNOWN_HF_MODELS)
            info = KNOWN_HF_MODELS[key]
            self.assertIn("variants", info)
            self.assertIn("cuda", info["variants"])
            self.assertIn("cpu", info["variants"])
            self.assertIn("pattern", info["variants"]["cuda"])

    @patch("prism.catalog.pull_ollama_model")
    def test_pull_ollama_model_routing(self, mock_pull_ollama):
        mock_pull_ollama.return_value = True
        res = self.catalog.pull_model("ollama:qwen2.5-coder:7b")
        mock_pull_ollama.assert_called_once_with("ollama:qwen2.5-coder:7b")
        self.assertEqual(res, "ollama:qwen2.5-coder:7b")

    @patch("prism.catalog.pull_ollama_model")
    def test_pull_backend_flag_routing(self, mock_pull_ollama):
        mock_pull_ollama.return_value = True
        res = self.catalog.pull_model("qwen2.5-coder:7b", backend="ollama")
        mock_pull_ollama.assert_called_once_with("qwen2.5-coder:7b")
        self.assertEqual(res, "qwen2.5-coder:7b")

if __name__ == "__main__":
    unittest.main()
