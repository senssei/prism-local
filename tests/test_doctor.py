import unittest
from foundry_wsl.doctor import check_nvml_detection, check_cuda_libraries, get_daemon_info, run_doctor_report

class TestDoctor(unittest.TestCase):
    def test_check_nvml_detection(self):
        res = check_nvml_detection()
        self.assertIn("status", res)
        if res["status"] == "ok":
            self.assertGreaterEqual(res["device_count"], 1)
            self.assertIn("devices", res)
            self.assertTrue(res["devices"][0]["qualifying"])

    def test_check_cuda_libraries(self):
        res = check_cuda_libraries()
        self.assertIn("status", res)
        # On this machine, all libraries should be resolved
        if res["status"] == "ok":
            self.assertEqual(len(res.get("missing_libraries", [])), 0)

    def test_run_doctor_report(self):
        report = run_doctor_report()
        self.assertIn("nvml", report)
        self.assertIn("cuda_libraries", report)
        self.assertIn("daemon", report)

if __name__ == "__main__":
    unittest.main()
