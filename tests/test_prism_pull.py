import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from prism.catalog import ModelCatalog, KNOWN_HF_MODELS

GPU_SUB = "gpu/gpu-int4-rtn-block-32"
CPU_SUB = "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4"


def fake_download(files, subfolder=None, record=None):
    """Builds a snapshot_download stand-in that writes `files` (under `subfolder`) into local_dir."""
    def _download(**kwargs):
        if record is not None:
            record.update(kwargs)
        base = Path(kwargs["local_dir"])
        if subfolder:
            base = base / subfolder
        base.mkdir(parents=True, exist_ok=True)
        for name in files:
            (base / name).write_bytes(b"x" * 16)
    return _download


class TestPrismPull(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.catalog = ModelCatalog(search_paths=[self.tmp])

    def pull(self, name, download, **kwargs):
        hub = type("Hub", (), {"snapshot_download": staticmethod(download)})
        with patch.dict("sys.modules", {"huggingface_hub": hub}):
            return self.catalog.pull_model(name, output_dir=self.tmp, **kwargs)

    def test_known_hf_models_variants(self):
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

    @patch("prism.catalog.pull_ollama_model", return_value=False)
    def test_failed_ollama_pull_returns_none(self, _):
        self.assertIsNone(self.catalog.pull_model("ollama:nope"))

    def test_known_alias_downloads_flattens_subfolder_and_becomes_discoverable(self):
        record = {}
        files = ["genai_config.json", "model.onnx", "model.onnx.data", "tokenizer.json"]
        res = self.pull("phi-4-mini", fake_download(files, GPU_SUB, record), ep="cuda")
        dest = Path(self.tmp) / "Phi-4-mini-instruct-cuda-gpu"
        self.assertEqual(res, str(dest))
        self.assertEqual(record["repo_id"], "microsoft/Phi-4-mini-instruct-onnx")
        self.assertEqual(record["allow_patterns"], [f"{GPU_SUB}/*"])
        for f in files:
            self.assertTrue((dest / f).is_file(), f)
        self.assertFalse((dest / "gpu").exists(), "empty nested folders should be pruned")
        found = self.catalog.resolve_model("Phi-4-mini-instruct-cuda-gpu")  # cache was invalidated
        self.assertEqual(found["path"], str(dest))

    def test_cpu_variant_selected_explicitly(self):
        res = self.pull("phi-4-mini", fake_download(["genai_config.json", "model.onnx"], CPU_SUB), ep="cpu")
        self.assertTrue(res.endswith("Phi-4-mini-instruct-generic-cpu"))

    def test_ep_auto_detects_from_gpu(self):
        files = ["genai_config.json", "model.onnx"]
        with patch("prism.catalog.get_gpu_info", return_value={"available": False}):
            self.assertTrue(self.pull("phi-4-mini", fake_download(files, CPU_SUB)).endswith("generic-cpu"))
        with patch("prism.catalog.get_gpu_info", return_value={"available": True}):
            self.assertTrue(self.pull("phi-4-mini", fake_download(files, GPU_SUB)).endswith("cuda-gpu"))

    def test_raw_repo_id_uses_repo_name_and_no_pattern(self):
        record = {}
        res = self.pull("someorg/Some-Model-onnx", fake_download(["genai_config.json", "m.onnx"], None, record), ep="cpu")
        self.assertTrue(res.endswith("Some-Model-onnx"))
        self.assertNotIn("allow_patterns", record)

    def test_missing_genai_config_fails(self):
        self.assertIsNone(self.pull("phi-4-mini", fake_download(["model.onnx"], GPU_SUB), ep="cuda"))

    def test_missing_weights_fails(self):
        self.assertIsNone(self.pull("phi-4-mini", fake_download(["genai_config.json"], GPU_SUB), ep="cuda"))

    def test_download_error_returns_none(self):
        def boom(**kwargs):
            raise OSError("network down")
        self.assertIsNone(self.pull("phi-4-mini", boom, ep="cuda"))

    def test_missing_huggingface_hub_returns_none(self):
        with patch.dict("sys.modules", {"huggingface_hub": None}):
            self.assertIsNone(self.catalog.pull_model("phi-4-mini", output_dir=self.tmp, ep="cuda"))


if __name__ == "__main__":
    unittest.main()
