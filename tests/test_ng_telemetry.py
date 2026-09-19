import unittest
from foundry_ng.telemetry import get_gpu_info, bootstrap_cuda_env

class TestNgTelemetry(unittest.TestCase):
    def test_bootstrap_cuda_env(self):
        bootstrap_cuda_env()
        import os
        self.assertIn("LD_LIBRARY_PATH", os.environ)

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
