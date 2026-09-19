import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import prism.telemetry as telemetry
from prism.telemetry import get_gpu_info, bootstrap_cuda_env


class TestPrismTelemetry(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _fake_site(self, *pkgs):
        for pkg in pkgs:
            os.makedirs(os.path.join(self.tmp, "nvidia", pkg, "lib"))
        return self.tmp

    def test_find_nvidia_lib_dirs_is_generic(self):
        site_dir = self._fake_site("cublas", "cudnn")
        with patch.object(telemetry, "_site_package_dirs", return_value=[site_dir]):
            found = telemetry.find_nvidia_lib_dirs()
        self.assertEqual(found, [os.path.join(site_dir, "nvidia", "cublas", "lib"),
                                 os.path.join(site_dir, "nvidia", "cudnn", "lib")])

    def test_find_nvidia_lib_dirs_empty(self):
        with patch.object(telemetry, "_site_package_dirs", return_value=[self.tmp]):
            self.assertEqual(telemetry.find_nvidia_lib_dirs(), [])

    def test_bootstrap_updates_ld_library_path_and_preloads(self):
        site_dir = self._fake_site("cublas")
        lib_dir = os.path.join(site_dir, "nvidia", "cublas", "lib")
        with patch.object(telemetry, "_BOOTSTRAPPED", False), \
                patch.object(telemetry, "_site_package_dirs", return_value=[site_dir]), \
                patch.object(telemetry, "_preload_cuda_libs") as preload, \
                patch.dict(os.environ, {"LD_LIBRARY_PATH": "/existing"}):
            bootstrap_cuda_env()
            entries = os.environ["LD_LIBRARY_PATH"].split(":")
            self.assertIn(lib_dir, entries)
            self.assertEqual(entries[-1], "/existing")  # pre-existing value is preserved, after ours
            self.assertIn(lib_dir, preload.call_args[0][0])

    def test_bootstrap_runs_once(self):
        with patch.object(telemetry, "_BOOTSTRAPPED", False), \
                patch.object(telemetry, "find_nvidia_lib_dirs", return_value=[]) as finder:
            bootstrap_cuda_env()
            bootstrap_cuda_env()
        self.assertEqual(finder.call_count, 1)

    def test_preload_skips_missing_and_unloadable_libs(self):
        lib_dir = os.path.join(self._fake_site("cublas"), "nvidia", "cublas", "lib")
        open(os.path.join(lib_dir, "libcublas.so.12"), "wb").write(b"not an elf")
        self.assertEqual(telemetry._preload_cuda_libs([lib_dir]), [])  # OSError swallowed

    def _fake_onnxruntime(self, with_lib=True):
        capi = os.path.join(self.tmp, "onnxruntime", "capi")
        os.makedirs(capi)
        if with_lib:
            open(os.path.join(capi, "libonnxruntime_providers_cuda.so"), "wb").write(b"not an elf")
        spec = type("Spec", (), {"submodule_search_locations": [os.path.join(self.tmp, "onnxruntime")]})()
        return patch("prism.telemetry.importlib.util.find_spec", return_value=spec)

    def test_probe_cuda_provider_reports_the_missing_library(self):
        with self._fake_onnxruntime(), patch.object(telemetry, "bootstrap_cuda_env"):
            res = telemetry.probe_cuda_provider()
        self.assertEqual((res["checked"], res["loadable"]), (True, False))
        self.assertIn("libonnxruntime_providers_cuda.so", res["error"])

    def test_probe_cuda_provider_when_not_applicable(self):
        with patch("prism.telemetry.importlib.util.find_spec", return_value=None):
            self.assertFalse(telemetry.probe_cuda_provider()["checked"])
        with self._fake_onnxruntime(with_lib=False):
            res = telemetry.probe_cuda_provider()
        self.assertFalse(res["checked"])
        self.assertIn("CPU-only", res["reason"])

    def test_get_gpu_info(self):
        info = get_gpu_info()
        self.assertIn("available", info)
        if info["available"]:
            self.assertGreaterEqual(info["device_count"], 1)
            dev = info["devices"][0]
            self.assertIn("name", dev)
            self.assertIn("compute_capability", dev)
            self.assertIn("vram_total_mb", dev)


if __name__ == "__main__":
    unittest.main()
