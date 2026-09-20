import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from prism.catalog import AmbiguousModelError, ModelCatalog, planned_device
from tests.fakes import make_model


class TestPrismCatalog(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        make_model(self.tmp, "Phi-4-mini-instruct-cuda-gpu")
        make_model(self.tmp, "qwen2.5-coder-7b-onnx", "qwen2")
        make_model(self.tmp, "qwen2.5-coder-1.5b-onnx", "qwen2")
        patcher = patch("prism.catalog.list_ollama_models", return_value=[
            {"id": "ollama:phi4-mini:latest", "name": "phi4-mini:latest", "engine": "Ollama (llama.cpp)",
             "backend": "ollama"},
        ])
        patcher.start()
        self.addCleanup(patcher.stop)
        self.catalog = ModelCatalog(search_paths=[self.tmp])

    def test_discover_onnx_models(self):
        models = {m["name"]: m for m in self.catalog.discover_onnx_models()}
        self.assertEqual(set(models), {"Phi-4-mini-instruct-cuda-gpu", "qwen2.5-coder-7b-onnx",
                                       "qwen2.5-coder-1.5b-onnx"})
        phi = models["Phi-4-mini-instruct-cuda-gpu"]
        self.assertEqual(phi["backend"], "onnx")
        self.assertEqual(phi["device"], "CUDA (GPU)")
        self.assertEqual(phi["template"], "phi4")
        self.assertEqual(models["qwen2.5-coder-7b-onnx"]["template"], "chatml")

    def _planned(self, model, device=None, gpu=True):
        env = {"PRISM_DEVICE": device} if device else {}
        with patch.dict(os.environ, env, clear=False), patch("prism.catalog.get_gpu_info", return_value={"available": gpu}):
            if not device:
                os.environ.pop("PRISM_DEVICE", None)
            return planned_device(model)

    def test_planned_device_is_where_the_model_will_run_not_what_it_was_exported_for(self):
        cpu_variant = {"backend": "onnx", "device": "CPU", "name": "Phi-4-mini-instruct-generic-cpu"}
        cuda_variant = {"backend": "onnx", "device": "CUDA (GPU)", "name": "Phi-4-mini-instruct-cuda-gpu"}
        # auto: any ONNX model runs on CUDA when a GPU is present, including a `generic-cpu` variant
        self.assertEqual(self._planned(cpu_variant, gpu=True), "CUDA (GPU)")
        self.assertEqual(self._planned(cpu_variant, gpu=False), "CPU")
        self.assertEqual(self._planned(cuda_variant, gpu=False), "CPU")
        # an explicit device wins over both the files and the hardware
        self.assertEqual(self._planned(cuda_variant, device="cpu"), "CPU")
        self.assertEqual(self._planned(cpu_variant, device="cuda", gpu=False), "CUDA (GPU)")

    def test_planned_device_keeps_the_label_of_ollama_models_and_survives_a_bad_setting(self):
        ollama = {"backend": "ollama", "device": "GPU"}
        self.assertEqual(self._planned(ollama, device="cpu"), "GPU")
        self.assertEqual(self._planned({"backend": "onnx", "device": "CPU"}, device="tpu"), "CPU")

    def test_list_all_models_includes_ollama(self):
        backends = {m["backend"] for m in self.catalog.list_all_models(include_ollama=True)}
        self.assertEqual(backends, {"onnx", "ollama"})
        backends = {m["backend"] for m in self.catalog.list_all_models(include_ollama=False)}
        self.assertEqual(backends, {"onnx"})

    def test_resolve_exact_and_unique_substring(self):
        self.assertEqual(self.catalog.resolve_model("Phi-4-mini-instruct-cuda-gpu")["backend"], "onnx")
        self.assertEqual(self.catalog.resolve_model("phi-4-mini")["name"], "Phi-4-mini-instruct-cuda-gpu")
        self.assertEqual(self.catalog.resolve_model("1.5b")["name"], "qwen2.5-coder-1.5b-onnx")

    def _alias_catalog(self, gpu):
        if not os.path.isdir(os.path.join(self.tmp, "Phi-4-mini-instruct-generic-cpu")):
            make_model(self.tmp, "Phi-4-mini-instruct-generic-cpu")
            # Foundry-cache layout: <name>/v5, discovered as "<name>:v5"
            make_model(os.path.join(self.tmp, "Phi-4-mini-instruct-generic-cpu-5"), "v5")
        p = patch("prism.catalog.get_gpu_info", return_value={"available": gpu})
        p.start()
        self.addCleanup(p.stop)
        self.catalog.invalidate_cache()
        return self.catalog

    def test_curated_alias_prefers_the_variant_for_this_machine(self):
        self.assertEqual(self._alias_catalog(gpu=True).resolve_model("phi-4-mini")["name"],
                         "Phi-4-mini-instruct-cuda-gpu")
        self.assertEqual(self._alias_catalog(gpu=False).resolve_model("phi-4-mini")["name"],
                         "Phi-4-mini-instruct-generic-cpu")

    def test_curated_alias_falls_back_to_whichever_variant_is_installed(self):
        with patch("prism.catalog.get_gpu_info", return_value={"available": False}):
            self.assertEqual(self.catalog.resolve_model("phi-4-mini")["name"], "Phi-4-mini-instruct-cuda-gpu")

    def test_curated_alias_does_not_mask_real_ambiguity(self):
        self._alias_catalog(gpu=True)
        with self.assertRaises(AmbiguousModelError):
            self.catalog.resolve_model("phi-4-mini-instruct")  # not an alias: still a plain substring

    def test_resolve_ambiguous_raises(self):
        with self.assertRaises(AmbiguousModelError) as ctx:
            self.catalog.resolve_model("qwen2.5-coder")
        self.assertEqual(len(ctx.exception.candidates), 2)

    def test_resolve_unknown_and_empty(self):
        self.assertIsNone(self.catalog.resolve_model("nope"))
        self.assertIsNone(self.catalog.resolve_model(""))

    def test_resolve_ollama_model(self):
        self.assertEqual(self.catalog.resolve_model("ollama:anything:7b")["backend"], "ollama")
        resolved = self.catalog.resolve_model("phi4-mini:latest")  # unprefixed installed Ollama model
        self.assertEqual(resolved["id"], "ollama:phi4-mini:latest")

    def test_discovery_cache_and_invalidation(self):
        self.assertEqual(len(self.catalog.discover_onnx_models()), 3)
        make_model(self.tmp, "llama-3.2-3b-instruct-cuda-gpu", "llama")
        self.assertEqual(len(self.catalog.discover_onnx_models()), 3)  # served from cache
        self.catalog.invalidate_cache()
        models = {m["name"]: m for m in self.catalog.discover_onnx_models()}
        self.assertEqual(len(models), 4)
        self.assertEqual(models["llama-3.2-3b-instruct-cuda-gpu"]["template"], "llama3")

    def test_cached_list_is_not_mutated_by_callers(self):
        self.catalog.list_all_models(include_ollama=True)  # extends the returned list internally
        self.assertEqual(len(self.catalog.discover_onnx_models()), 3)


if __name__ == "__main__":
    unittest.main()
