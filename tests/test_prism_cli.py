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
    env = {"PRISM_API_KEY": "", **(env or {})}
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
        start.assert_called_once_with(port=5272, host="127.0.0.1", api_key=None, cors_origins=[])

    def test_flags(self):
        with patch.object(cli, "start_server") as start:
            run_cli("serve", "--port", "6000", "--host", "0.0.0.0", "--api-key", "k",
                    "--cors-origin", "http://a", "--cors-origin", "http://b")
        start.assert_called_once_with(port=6000, host="0.0.0.0", api_key="k",
                                      cors_origins=["http://a", "http://b"])

    def test_api_key_from_environment(self):
        with patch.object(cli, "start_server") as start:
            run_cli("serve", env={"PRISM_API_KEY": "from-env"})
        self.assertEqual(start.call_args.kwargs["api_key"], "from-env")

    def test_cli_flag_wins_over_environment(self):
        with patch.object(cli, "start_server") as start:
            run_cli("serve", "--api-key", "flag", env={"PRISM_API_KEY": "from-env"})
        self.assertEqual(start.call_args.kwargs["api_key"], "flag")


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


if __name__ == "__main__":
    unittest.main()
