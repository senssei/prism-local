import contextlib
import io
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from prism import cli
from prism.catalog import AmbiguousModelError, ModelCatalog
from tests.fakes import make_model


def run_cli(*argv, env=None):
    """Runs prism.cli.main() and returns (exit_code, stdout)."""
    out = io.StringIO()
    code = 0
    env = {"PRISM_API_KEY": "", "PRISM_QUEUE_TIMEOUT": "", "PRISM_MAX_QUEUE": "", **(env or {})}
    with patch.object(sys, "argv", ["prism", *argv]), patch.dict(os.environ, env), \
            contextlib.redirect_stdout(out):
        try:
            cli.main()
        except SystemExit as ex:
            code = ex.code
    return code, out.getvalue()


class TestServeArgs(unittest.TestCase):
    def test_defaults_are_loopback_no_auth_no_cors(self):
        with patch.object(cli, "start_server") as start:
            run_cli("serve")
        start.assert_called_once_with(port=5272, host="127.0.0.1", api_key=None, cors_origins=[],
                                      queue_timeout=300.0, max_queue=8)

    def test_flags(self):
        with patch.object(cli, "start_server") as start:
            run_cli("serve", "--port", "6000", "--host", "0.0.0.0", "--api-key", "k",
                    "--cors-origin", "http://a", "--cors-origin", "http://b")
        start.assert_called_once_with(port=6000, host="0.0.0.0", api_key="k",
                                      cors_origins=["http://a", "http://b"], queue_timeout=300.0, max_queue=8)

    def test_queue_timeout_flag_and_environment(self):
        with patch.object(cli, "start_server") as start:
            run_cli("serve", "--queue-timeout", "5")
            self.assertEqual(start.call_args.kwargs["queue_timeout"], 5.0)
            run_cli("serve", env={"PRISM_QUEUE_TIMEOUT": "12"})
            self.assertEqual(start.call_args.kwargs["queue_timeout"], 12.0)
            run_cli("serve", env={"PRISM_QUEUE_TIMEOUT": "0"})  # 0 means wait forever
            self.assertIsNone(start.call_args.kwargs["queue_timeout"])
            run_cli("serve", "--queue-timeout", "7", env={"PRISM_QUEUE_TIMEOUT": "12"})  # the flag wins
            self.assertEqual(start.call_args.kwargs["queue_timeout"], 7.0)

    def test_max_queue_flag_and_environment(self):
        with patch.object(cli, "start_server") as start:
            run_cli("serve", "--max-queue", "4")
            self.assertEqual(start.call_args.kwargs["max_queue"], 4)
            run_cli("serve", env={"PRISM_MAX_QUEUE": "16"})
            self.assertEqual(start.call_args.kwargs["max_queue"], 16)
            run_cli("serve", env={"PRISM_MAX_QUEUE": "0"})  # 0 means unlimited
            self.assertIsNone(start.call_args.kwargs["max_queue"])
            run_cli("serve", "--max-queue", "2", env={"PRISM_MAX_QUEUE": "16"})  # the flag wins
            self.assertEqual(start.call_args.kwargs["max_queue"], 2)
            run_cli("serve", "--max-queue", "0")  # 0 flag means unlimited
            self.assertIsNone(start.call_args.kwargs["max_queue"])

    def test_api_key_from_environment(self):
        with patch.object(cli, "start_server") as start:
            run_cli("serve", env={"PRISM_API_KEY": "from-env"})
        self.assertEqual(start.call_args.kwargs["api_key"], "from-env")

    def test_cli_flag_wins_over_environment(self):
        with patch.object(cli, "start_server") as start:
            run_cli("serve", "--api-key", "flag", env={"PRISM_API_KEY": "from-env"})
        self.assertEqual(start.call_args.kwargs["api_key"], "flag")


class TestAcpCommand(unittest.TestCase):
    def test_acp_command_runs_the_acp_server(self):
        with patch("prism.acp.run_acp_server") as run:
            code, _ = run_cli("acp")
        self.assertEqual(code, 0)
        run.assert_called_once_with()


class TestDeviceFlag(unittest.TestCase):
    def test_flag_sets_environment_for_the_engine(self):
        seen = {}
        with patch.object(cli, "start_server", side_effect=lambda **kw: seen.update(device=os.environ.get("PRISM_DEVICE"))):
            run_cli("serve", "--device", "cpu")
            self.assertEqual(seen["device"], "cpu")
            run_cli("serve")
            self.assertNotEqual(seen["device"], "cpu")  # not sticky across invocations

    def test_invalid_device_is_rejected(self):
        code, _ = run_cli("serve", "--device", "tpu")
        self.assertEqual(code, 2)  # argparse error


class TestModelCommands(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        make_model(self.tmp, "alpha-phi-cuda-gpu")
        make_model(self.tmp, "beta-phi-cuda-gpu")
        for target, value in (("prism.cli.ModelCatalog", lambda: ModelCatalog(search_paths=[self.tmp])),
                              ("prism.catalog.list_ollama_models", lambda: [])):
            p = patch(target, value)
            p.start()
            self.addCleanup(p.stop)

    def test_list_shows_local_models(self):
        code, out = run_cli("list")
        self.assertEqual(code, 0)
        self.assertIn("alpha-phi-cuda-gpu", out)
        self.assertIn("beta-phi-cuda-gpu", out)

    def test_run_ambiguous_model_exits_1_with_candidates(self):
        code, out = run_cli("run", "phi", "hello")
        self.assertEqual(code, 1)
        self.assertIn("ambiguous", out)
        self.assertIn("alpha-phi-cuda-gpu", out)

    def test_run_model_load_failure_is_a_clean_error_not_a_traceback(self):
        from prism.engine import ModelLoadError
        with patch.object(cli, "OnnxGenAiEngine", side_effect=ModelLoadError("CUDA execution provider unavailable: x")):
            code, out = run_cli("run", "alpha-phi-cuda-gpu", "hello", "--device", "cuda")
        self.assertEqual(code, 1)
        self.assertIn("CUDA execution provider unavailable", out)

    def test_run_unknown_model_exits_1(self):
        code, out = run_cli("run", "nope", "hello")
        self.assertEqual(code, 1)
        self.assertIn("not found", out)

    def test_no_subcommand_prints_help_and_exits_1(self):
        code, out = run_cli()
        self.assertEqual(code, 1)
        self.assertIn("usage", out.lower())

    def test_pull_reports_success_and_failure(self):
        with patch.object(ModelCatalog, "pull_model", return_value="/some/dir") as pull:
            _, out = run_cli("pull", "phi-4-mini", "--ep", "cpu")
        self.assertIn("completed", out)
        self.assertEqual(pull.call_args.args, ("phi-4-mini",))
        self.assertEqual(pull.call_args.kwargs["ep"], "cpu")
        self.assertIsNone(pull.call_args.kwargs["output_dir"])  # falls back to the default model dir
        with patch.object(ModelCatalog, "pull_model", return_value=None):
            _, out = run_cli("pull", "phi-4-mini")
        self.assertIn("failed", out)

    def test_convert_passes_flags_and_reports_success(self):
        with patch.object(cli, "convert_model", return_value="/some/dir") as opt:
            code, out = run_cli("convert", "org/model", "--ep", "cpu", "--quant", "int4", "--name", "x",
                                "--force", "--trust-remote-code")
        self.assertEqual(code, 0)
        self.assertIn("completed", out)
        opt.assert_called_once_with("org/model", output_dir=None, ep="cpu", quant="int4", name="x",
                                    force=True, trust_remote_code=True)

    def test_convert_failure_exits_1(self):
        with patch.object(cli, "convert_model", return_value=None):
            code, out = run_cli("convert", "org/model")
        self.assertEqual(code, 1)
        self.assertIn("failed", out)


class TestStatusCommand(unittest.TestCase):
    def test_status_displays_ram_and_swap(self):
        fake_mem = {
            "ram_total_mb": 16384.0,
            "ram_available_mb": 12288.0,
            "ram_used_mb": 4096.0,
            "swap_total_mb": 8192.0,
            "swap_free_mb": 6144.0,
            "swap_used_mb": 2048.0,
        }
        with patch("prism.cli.system_memory_info", return_value=fake_mem):
            code, out = run_cli("status")
        self.assertEqual(code, 0)
        self.assertIn("System RAM:", out)
        self.assertIn("4096.0 MB used / 16384.0 MB total", out)
        self.assertIn("System Swap:", out)
        self.assertIn("2048.0 MB used / 8192.0 MB total", out)


class TestDoctorWslconfig(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_doctor_warns_when_wslconfig_missing_settings(self):
        cfg_path = os.path.join(self.tmp, ".wslconfig")
        with open(cfg_path, "w") as f:
            f.write("[wsl2]\nvmIdleTimeout=-1\n")
        code, out = run_cli("doctor", env={"PRISM_WSLCONFIG_PATH": cfg_path, "PRISM_FORCE_WSL": "1"})
        self.assertEqual(code, 0)
        self.assertIn("WSL configuration", out)
        self.assertIn("Missing 'memory='", out)
        self.assertIn("Missing 'autoMemoryReclaim=gradual'", out)

    def test_doctor_reports_ok_when_wslconfig_has_settings(self):
        cfg_path = os.path.join(self.tmp, ".wslconfig")
        with open(cfg_path, "w") as f:
            f.write("[wsl2]\nmemory=16GB\nautoMemoryReclaim=gradual\n")
        code, out = run_cli("doctor", env={"PRISM_WSLCONFIG_PATH": cfg_path, "PRISM_FORCE_WSL": "1"})
        self.assertEqual(code, 0)
        self.assertIn("WSL configuration", out)
        self.assertIn("memory limit and autoMemoryReclaim configured", out)

    def test_doctor_warns_when_wslconfig_missing_entirely_under_wsl(self):
        with patch("prism.cli.inspect_wslconfig", return_value={"is_wsl": True, "found": False, "path": None}):
            code, out = run_cli("doctor")
        self.assertEqual(code, 0)
        self.assertIn("WSL configuration", out)
        self.assertIn("no .wslconfig found", out)

    def test_doctor_skips_wslconfig_when_not_wsl(self):
        code, out = run_cli("doctor", env={"PRISM_FORCE_WSL": "0", "PRISM_WSLCONFIG_PATH": ""})
        self.assertEqual(code, 0)
        self.assertNotIn("WSL configuration", out)


if __name__ == "__main__":
    unittest.main()
