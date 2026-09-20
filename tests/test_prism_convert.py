import io
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from prism.convert import build_command, default_name, find_model_dir, hf_cache_dir, missing_dependencies, convert_model


def fake_builder(nested=None, files=("genai_config.json", "model.onnx"), returncode=0, record=None):
    """A subprocess.run stand-in that writes a builder result (optionally under `nested`) into the -o folder."""
    def _run(command, **kwargs):
        if record is not None:
            record.append(command)
        if returncode == 0:
            out = Path(command[command.index("-o") + 1])
            target = out / nested if nested else out
            target.mkdir(parents=True, exist_ok=True)
            for name in files:
                (target / name).write_bytes(b"x" * 16)
        return SimpleNamespace(returncode=returncode)
    return _run


class TestPrismConvert(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        for target, value in (
            ("prism.convert.missing_dependencies", []),
            ("prism.convert.get_gpu_info", {"available": False}),
            ("prism.convert.has_hf_token", True),
        ):
            patcher = patch(target, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_convert(self, runner, *args, **kwargs):
        out = io.StringIO()
        with patch("prism.convert.subprocess.run", runner), redirect_stdout(out):
            result = convert_model(*args, output_dir=self.tmp, **kwargs)
        return result, out.getvalue()

    def test_default_name_contains_the_provider(self):
        self.assertEqual(default_name("Qwen/Qwen2.5-0.5B-Instruct", "cuda", "int4"), "Qwen2.5-0.5B-Instruct-cuda-int4")
        self.assertEqual(default_name("/models/my-llm/", "cpu", "int4"), "my-llm-cpu-int4")

    def test_build_command_for_a_hub_model(self):
        command = build_command("org/m", "/out", "cuda", "fp16")
        self.assertEqual(command[:3], [sys.executable, "-m", "onnxruntime_genai.models.builder"])
        cuda = command[3:]  # the builder's own arguments
        self.assertEqual(cuda[cuda.index("-m") + 1], "org/m")
        self.assertNotIn("-i", cuda)
        self.assertEqual(cuda[cuda.index("-o") + 1], "/out")
        self.assertEqual(cuda[cuda.index("-e") + 1], "cuda")
        self.assertEqual(cuda[cuda.index("-p") + 1], "fp16")
        self.assertEqual(cuda[cuda.index("-c") + 1], hf_cache_dir())
        self.assertNotIn("--extra_options", cuda)
        cpu = build_command("org/m", "/out", "cpu", "int4", trust_remote_code=True)[3:]
        self.assertEqual(cpu[cpu.index("-e") + 1], "cpu")
        self.assertEqual(cpu[cpu.index("--extra_options") + 1:], ["hf_remote=true"])

    def test_build_command_for_a_local_folder(self):
        command = build_command(self.tmp, "/out", "cpu", "int4")[3:]
        self.assertEqual(command[command.index("-i") + 1], self.tmp)
        self.assertNotIn("-m", command)

    def test_build_command_disables_the_token_when_none_is_configured(self):
        with patch("prism.convert.has_hf_token", return_value=False):
            command = build_command("org/m", "/out", "cpu", "int4", trust_remote_code=True)
        self.assertEqual(command[command.index("--extra_options") + 1:], ["hf_remote=true", "hf_token=false"])

    def test_hf_cache_dir_follows_the_hugging_face_variables(self):
        with patch.dict("os.environ", {"HF_HUB_CACHE": "/hub", "HF_HOME": "/home"}):
            self.assertEqual(hf_cache_dir(), "/hub")
        with patch.dict("os.environ", {"HF_HOME": "/home"}, clear=True):
            self.assertEqual(hf_cache_dir(), str(Path("/home") / "hub"))

    def test_missing_dependencies_names_the_pip_packages(self):
        installed = {"onnxruntime_genai", "torch"}
        with patch("prism.convert.importlib.util.find_spec", side_effect=lambda m: object() if m in installed else None):
            self.assertEqual(missing_dependencies(), ["transformers", "onnx-ir", "safetensors"])

    def test_installs_the_result_and_removes_staging(self):
        calls = []
        result, out = self.run_convert(fake_builder(record=calls), "org/model", ep="cpu")
        dest = Path(self.tmp) / "model-cpu-int4"
        self.assertEqual(result, str(dest))
        self.assertTrue((dest / "genai_config.json").exists())
        self.assertEqual([p.name for p in Path(self.tmp).iterdir()], ["model-cpu-int4"])
        self.assertIn("prism run model-cpu-int4", out)
        self.assertEqual(len(calls), 1)

    def test_nested_output_is_flattened(self):
        result, _ = self.run_convert(fake_builder(nested="model"), "org/model", ep="cpu", name="custom")
        self.assertEqual(result, str(Path(self.tmp) / "custom"))
        self.assertTrue((Path(self.tmp) / "custom" / "model.onnx").exists())
        self.assertFalse((Path(self.tmp) / "custom" / "model").exists())

    def test_ep_defaults_to_cuda_when_a_gpu_is_detected(self):
        calls = []
        with patch("prism.convert.get_gpu_info", return_value={"available": True}):
            self.run_convert(fake_builder(record=calls), "org/model")
        self.assertEqual(calls[0][calls[0].index("-e") + 1], "cuda")

    def test_cpu_fp16_is_rejected_without_running_the_builder(self):
        calls = []
        result, out = self.run_convert(fake_builder(record=calls), "org/model", ep="cpu", quant="fp16")
        self.assertIsNone(result)
        self.assertEqual(calls, [])
        self.assertIn("fp16 is not supported on CPU", out)

    def test_missing_dependencies_give_install_hint(self):
        calls = []
        with patch("prism.convert.missing_dependencies", return_value=["torch", "onnx-ir"]):
            result, out = self.run_convert(fake_builder(record=calls), "org/model", ep="cpu")
        self.assertIsNone(result)
        self.assertEqual(calls, [])
        self.assertIn("torch, onnx-ir", out)
        self.assertIn("prism-local[convert]", out)

    def test_builder_failure_cleans_up(self):
        result, out = self.run_convert(fake_builder(returncode=2), "org/model", ep="cpu")
        self.assertIsNone(result)
        self.assertIn("exit code 2", out)
        self.assertEqual(list(Path(self.tmp).iterdir()), [])

    def test_output_without_genai_config_is_rejected(self):
        result, out = self.run_convert(fake_builder(files=("model.onnx",)), "org/model", ep="cpu")
        self.assertIsNone(result)
        self.assertIn("no genai_config.json", out)
        self.assertEqual(list(Path(self.tmp).iterdir()), [])

    def test_output_without_weights_is_rejected(self):
        result, _ = self.run_convert(fake_builder(files=("genai_config.json",)), "org/model", ep="cpu")
        self.assertIsNone(result)
        self.assertEqual(list(Path(self.tmp).iterdir()), [])

    def test_existing_model_is_kept_unless_forced(self):
        existing = Path(self.tmp) / "model-cpu-int4"
        existing.mkdir()
        (existing / "marker").write_text("old")
        calls = []
        result, out = self.run_convert(fake_builder(record=calls), "org/model", ep="cpu")
        self.assertIsNone(result)
        self.assertEqual(calls, [])
        self.assertIn("--force", out)
        self.assertTrue((existing / "marker").exists())

        result, _ = self.run_convert(fake_builder(), "org/model", ep="cpu", force=True)
        self.assertEqual(result, str(existing))
        self.assertFalse((existing / "marker").exists())
        self.assertTrue((existing / "genai_config.json").exists())

    def test_failed_forced_run_keeps_the_existing_model(self):
        existing = Path(self.tmp) / "model-cpu-int4"
        existing.mkdir()
        (existing / "marker").write_text("old")
        result, _ = self.run_convert(fake_builder(returncode=1), "org/model", ep="cpu", force=True)
        self.assertIsNone(result)
        self.assertTrue((existing / "marker").exists())

    def test_keyboard_interrupt_cleans_up(self):
        def interrupted(command, **kwargs):
            Path(command[command.index("-o") + 1], "partial").write_text("x")
            raise KeyboardInterrupt
        result, out = self.run_convert(interrupted, "org/model", ep="cpu")
        self.assertIsNone(result)
        self.assertIn("interrupted", out)
        self.assertEqual(list(Path(self.tmp).iterdir()), [])

    def test_find_model_dir(self):
        root = Path(self.tmp)
        self.assertIsNone(find_model_dir(root))
        (root / "a" / "b").mkdir(parents=True)
        (root / "a" / "b" / "genai_config.json").write_text("{}")
        self.assertEqual(find_model_dir(root), root / "a" / "b")


if __name__ == "__main__":
    unittest.main()
