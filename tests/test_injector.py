import json
import tempfile
import unittest
from pathlib import Path
from foundry_wsl.injector import configure_cuda_genai_config, create_inference_model_json, mark_model_cached_in_modelinfo

class TestInjector(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_configure_cuda_genai_config(self):
        cfg_file = self.tmp_path / "genai_config.json"
        cfg_file.write_text(json.dumps({"model": {"type": "phi3"}}))
        
        ok = configure_cuda_genai_config(cfg_file, device_id=0)
        self.assertTrue(ok)
        
        data = json.loads(cfg_file.read_text())
        self.assertIn("session_options", data)
        self.assertEqual(data["session_options"]["provider_options"], [{"cuda": {"device_id": "0"}}])

    def test_create_inference_model_json(self):
        target = self.tmp_path / "inference_model.json"
        ok = create_inference_model_json(target, "test-model:1")
        self.assertTrue(ok)
        data = json.loads(target.read_text())
        self.assertEqual(data["Name"], "test-model:1")
        self.assertIn("PromptTemplate", data)

    def test_mark_model_cached(self):
        models_dir = self.tmp_path / "models"
        models_dir.mkdir(parents=True)
        info_file = models_dir / "foundry.modelinfo.json"
        info_file.write_text(json.dumps({"models": [{"id": "test-model:1", "cached": False}]}))
        
        ok = mark_model_cached_in_modelinfo(self.tmp_path, "test-model:1")
        self.assertTrue(ok)
        
        data = json.loads(info_file.read_text())
        self.assertTrue(data["models"][0]["cached"])

if __name__ == "__main__":
    unittest.main()
