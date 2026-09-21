import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, mock_open

from prism.engine import ModelLoadError
from prism.resources import (
    InsufficientResourcesError,
    check_can_load,
    dir_size_mb,
    estimate_load_mb,
    ram_available_mb,
    vram_free_mb,
    DEFAULT_RAM_RESERVE_MB,
    DEFAULT_VRAM_RESERVE_MB,
)


class TestPrismResources(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.model_dir = Path(self.tmp.name)
        # Create a fake model file of 10 MB (10 * 1024 * 1024 bytes)
        with open(self.model_dir / "model.onnx", "wb") as f:
            f.write(b"\0" * (10 * 1024 * 1024))

    def tearDown(self):
        self.tmp.cleanup()

    def test_dir_size_mb(self):
        size = dir_size_mb(self.model_dir)
        self.assertAlmostEqual(size, 10.0, places=1)

    def test_ram_available_mb_from_meminfo(self):
        fake_meminfo = (
            "MemTotal:       32800000 kB\n"
            "MemFree:         8000000 kB\n"
            "MemAvailable:   16384000 kB\n"
            "Buffers:          500000 kB\n"
        )
        with patch("builtins.open", mock_open(read_data=fake_meminfo)):
            avail = ram_available_mb()
            # 16384000 kB / 1024 = 16000.0 MB
            self.assertEqual(avail, 16000.0)

    def test_vram_free_mb_when_gpu_available(self):
        fake_gpu = {
            "available": True,
            "devices": [{"index": 0, "name": "Fake GPU", "vram_free_mb": 8192.0}],
        }
        with patch("prism.resources.get_gpu_info", return_value=fake_gpu) as mock_gpu:
            free = vram_free_mb()
            mock_gpu.assert_called_once_with(max_age=0)
            self.assertEqual(free, 8192.0)

    def test_vram_free_mb_when_gpu_unavailable(self):
        fake_gpu = {"available": False, "devices": []}
        with patch("prism.resources.get_gpu_info", return_value=fake_gpu):
            self.assertIsNone(vram_free_mb())

    def test_estimate_load_mb(self):
        # 10 MB weights + 1024 * 1.4 MB prefill headroom = 10.0 + 1433.6 = 1443.6 MB
        est = estimate_load_mb(self.model_dir, prefill_chunk=1024)
        self.assertAlmostEqual(est, 1443.6, places=1)

    def test_check_can_load_passes_when_sufficient(self):
        with patch("prism.resources.ram_available_mb", return_value=32000.0), \
             patch("prism.resources.vram_free_mb", return_value=12000.0):
            # Should not raise
            check_can_load(self.model_dir, device="cuda")
            check_can_load(self.model_dir, device="cpu")

    def test_check_can_load_fails_on_insufficient_ram(self):
        # Needed: ~1443.6 MB + 2048 MB reserve = ~3491.6 MB
        with patch("prism.resources.ram_available_mb", return_value=2000.0), \
             patch("prism.resources.vram_free_mb", return_value=12000.0):
            with self.assertRaises(InsufficientResourcesError) as ctx:
                check_can_load(self.model_dir, device="cpu")
            self.assertIn("RAM", str(ctx.exception))
            self.assertTrue(issubclass(InsufficientResourcesError, ModelLoadError))

    def test_check_can_load_fails_on_insufficient_vram(self):
        # Needed: ~1443.6 MB + 1536 MB reserve = ~2979.6 MB
        with patch("prism.resources.ram_available_mb", return_value=32000.0), \
             patch("prism.resources.vram_free_mb", return_value=1000.0):
            with self.assertRaises(InsufficientResourcesError) as ctx:
                check_can_load(self.model_dir, device="cuda")
            self.assertIn("VRAM", str(ctx.exception))

    def test_check_can_load_honors_env_override(self):
        with patch.dict(os.environ, {"PRISM_RESOURCE_CHECK": "off"}), \
             patch("prism.resources.ram_available_mb", return_value=10.0), \
             patch("prism.resources.vram_free_mb", return_value=10.0):
            # Should not raise when disabled
            check_can_load(self.model_dir, device="cuda")

    def test_direct_import_has_no_circular_dependency(self):
        import subprocess, sys
        res = subprocess.run([sys.executable, "-c", "import prism.resources"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)

    def test_system_memory_info_structure_and_values(self):
        from prism.resources import system_memory_info
        fake_meminfo = (
            "MemTotal:       32800000 kB\n"
            "MemAvailable:   16400000 kB\n"
            "SwapTotal:       8000000 kB\n"
            "SwapFree:        4000000 kB\n"
        )
        with patch("builtins.open", mock_open(read_data=fake_meminfo)):
            info = system_memory_info()
            self.assertEqual(info["ram_total_mb"], 32031.2)
            self.assertEqual(info["ram_available_mb"], 16015.6)
            self.assertEqual(info["ram_used_mb"], 16015.6)
            self.assertEqual(info["swap_total_mb"], 7812.5)
            self.assertEqual(info["swap_free_mb"], 3906.2)
            self.assertEqual(info["swap_used_mb"], 3906.3)

    def test_check_can_load_default_device_auto(self):
        fake_gpu = {"available": True, "devices": [{"index": 0, "name": "Fake GPU", "vram_free_mb": 100.0}]}
        with patch("prism.resources.ram_available_mb", return_value=32000.0), \
             patch("prism.resources.get_gpu_info", return_value=fake_gpu):
            with self.assertRaises(InsufficientResourcesError) as ctx:
                check_can_load(self.model_dir)  # default device="auto"
            self.assertIn("VRAM", str(ctx.exception))

    def test_check_can_load_reports_active_holder_in_error_message(self):
        with patch("prism.resources.ram_available_mb", return_value=100.0), \
             patch("prism.resources.vram_free_mb", return_value=12000.0), \
             patch("prism.resources.active_holder_summary", return_value="PID 4242 (model: phi-4-mini)"):
            with self.assertRaises(InsufficientResourcesError) as ctx:
                check_can_load(self.model_dir, device="cpu")
            self.assertIn("PID 4242 (model: phi-4-mini)", str(ctx.exception))

    def test_inspect_wslconfig_ignores_settings_outside_wsl2_section(self):
        from prism.resources import inspect_wslconfig
        cfg_file = self.model_dir / ".wslconfig"
        cfg_file.write_text("[other]\nmemory=16GB\nautoMemoryReclaim=gradual\n[wsl2]\nvmIdleTimeout=-1\n")
        res = inspect_wslconfig(cfg_file)
        self.assertTrue(res["found"])
        self.assertFalse(res["has_memory"])
        self.assertFalse(res["has_reclaim"])


if __name__ == "__main__":
    unittest.main()
