"""The SDLC gate helpers: `--red` (a new test must fail first)."""

import contextlib
import importlib.util
import io
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest import mock
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "sdlc_check.py"
spec = importlib.util.spec_from_file_location("sdlc_check", SCRIPT)
sdlc_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sdlc_check)

FIXTURE = textwrap.dedent(
    """
    import unittest

    class T(unittest.TestCase):
        def test_fails(self):
            self.assertEqual(1, 2)

        def test_errors(self):
            raise AttributeError("no attribute 'new_feature'")

        def test_passes(self):
            self.assertTrue(True)

        @unittest.skip("needs gpu")
        def test_skipped(self):
            pass

    NOT_A_TEST = 5

    class S(unittest.TestCase):
        def test_sub(self):
            for i in range(2):
                with self.subTest(i=i):
                    self.assertEqual(i, 99)

    class B(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            raise RuntimeError("no fixture")

        def test_a(self):
            pass

    class X(unittest.TestCase):
        @unittest.expectedFailure
        def test_ef(self):
            self.assertEqual(1, 2)

        def test_multi(self):
            self.assertEqual({"a": 1, "b": [1, 2]}, {"a": 1, "b": [1, 3]})

        def test_prints(self):
            print("HELLO STDOUT")
            self.assertEqual(1, 2)

        def test_hangs(self):
            import time
            time.sleep(5)
    """
)


class TestRed(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.name = f"red_fixture_{id(self)}"
        (Path(tmp.name) / f"{self.name}.py").write_text(FIXTURE)
        sys.path.insert(0, tmp.name)
        self.addCleanup(sys.path.remove, tmp.name)
        self.addCleanup(sys.modules.pop, self.name, None)

    def red(self, *tests, **kw):
        ids = [f"{self.name}.{t if '.' in t else 'T.' + t}" for t in tests]
        with contextlib.redirect_stdout(io.StringIO()):
            return sdlc_check.run_red(ids, **kw)

    def test_failing_and_erroring_tests_are_red(self):
        ok, lines = self.red("test_fails", "test_errors")
        self.assertTrue(ok)
        text = "\n".join(lines)
        self.assertIn("AssertionError", text)
        self.assertIn("AttributeError", text)

    def test_passing_test_is_not_red(self):
        ok, lines = self.red("test_fails", "test_passes")
        self.assertFalse(ok)
        self.assertRegex("\n".join(lines), r"test_passes.*passes")

    def test_skipped_test_is_not_red(self):
        ok, lines = self.red("test_skipped")
        self.assertFalse(ok)
        self.assertRegex("\n".join(lines), r"test_skipped.*skipped")

    def test_unknown_test_is_not_red(self):
        ok, lines = self.red("test_does_not_exist")
        self.assertFalse(ok)
        self.assertRegex("\n".join(lines), r"test_does_not_exist.*not found")

    def test_unknown_module_is_not_red(self):
        with contextlib.redirect_stdout(io.StringIO()):
            ok, lines = sdlc_check.run_red(["no_such_module_xyz.T.test_a"])
        self.assertFalse(ok)
        self.assertIn("not found", "\n".join(lines))

    def test_no_ids_is_not_red(self):
        ok, _ = sdlc_check.run_red([])
        self.assertFalse(ok)

    def test_failing_subtest_is_red(self):
        ok, lines = self.red("S.test_sub")
        self.assertTrue(ok, lines)
        self.assertIn("AssertionError", "\n".join(lines))

    def test_setupclass_failure_is_red_with_its_reason(self):
        ok, lines = self.red("B.test_a")
        self.assertTrue(ok, lines)
        self.assertIn("RuntimeError: no fixture", "\n".join(lines))

    def test_expected_failure_is_not_red(self):
        ok, lines = self.red("X.test_ef")
        self.assertFalse(ok)
        self.assertRegex("\n".join(lines), r"test_ef.*expectedFailure")

    def test_headline_is_the_exception_line_not_the_last_traceback_line(self):
        _, lines = self.red("X.test_multi")
        self.assertRegex(lines[0], r"test_multi: AssertionError: ")
        self.assertNotIn("^", lines[0])

    def test_test_output_does_not_leak_into_the_report(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            ok, lines = sdlc_check.run_red([f"{self.name}.X.test_prints"])
        self.assertTrue(ok)
        self.assertNotIn("HELLO STDOUT", buf.getvalue() + "\n".join(lines))

    def test_non_test_object_is_not_found_instead_of_crashing(self):
        with contextlib.redirect_stdout(io.StringIO()):
            ok, lines = sdlc_check.run_red([f"{self.name}.NOT_A_TEST"])
        self.assertFalse(ok)
        self.assertIn("not found", "\n".join(lines))

    def test_hanging_test_times_out(self):
        ok, lines = self.red("X.test_hangs", timeout=1)
        self.assertFalse(ok)
        self.assertIn("timed out", "\n".join(lines))

    def test_red_cannot_be_combined_with_only_or_base(self):
        for extra in (["--only", "tests"], ["--base", "origin/main"]):
            with self.subTest(extra=extra):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as cm:
                    sdlc_check.main(["--red", f"{self.name}.T.test_fails", *extra])
                self.assertEqual(cm.exception.code, 2)

    def test_cli_exit_codes(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(sdlc_check.main(["--red", f"{self.name}.T.test_fails"]), 0)
            self.assertEqual(sdlc_check.main(["--red", f"{self.name}.T.test_passes"]), 1)


@unittest.skipUnless(shutil.which("git"), "needs git")
class TestChangedFiles(unittest.TestCase):
    def test_paths_with_spaces_and_non_ascii_are_reported_verbatim(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
            subprocess.run([*git, "init", "-q"], cwd=repo, check=True)
            (repo / "seed.txt").write_text("x")
            subprocess.run([*git, "add", "."], cwd=repo, check=True)
            subprocess.run([*git, "commit", "-qm", "seed"], cwd=repo, check=True)
            (repo / "prism").mkdir()
            (repo / "prism" / "é.py").write_text("x")
            (repo / "a b.txt").write_text("x")
            (repo / "seed.txt").write_text("changed")
            files = sdlc_check.changed_files("HEAD", cwd=repo)
        self.assertEqual(files, sorted(["a b.txt", "prism/é.py", "seed.txt"]))


HOOK = SCRIPT.parent.parent / ".githooks" / "pre-commit"


@unittest.skipUnless(shutil.which("git") and shutil.which("sh"), "needs git and sh")
class TestHook(unittest.TestCase):
    """The opt-in pre-commit hook must run the gate from the repository root and pass its verdict through."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.repo = Path(tmp.name)
        subprocess.run(["git", "init", "-q"], cwd=self.repo, check=True)
        (self.repo / "scripts").mkdir()
        (self.repo / "sub").mkdir()

    def run_hook(self, gate_exit, cwd="."):
        (self.repo / "scripts" / "sdlc_check.py").write_text(
            "import os, sys\n"
            "open(os.path.join(os.path.dirname(__file__), 'args.txt'), 'w').write(' '.join(sys.argv[1:]) + '|' + os.getcwd())\n"
            f"sys.exit({gate_exit})\n"
        )
        env = dict(os.environ, PYTHON=sys.executable)
        return subprocess.run(["sh", str(HOOK)], cwd=self.repo / cwd, env=env, capture_output=True, text=True)

    def test_hook_is_executable(self):
        self.assertTrue(HOOK.is_file())
        self.assertTrue(os.access(HOOK, os.X_OK), "run: chmod +x .githooks/pre-commit")

    def test_blocks_commit_when_gate_fails_and_passes_when_it_passes(self):
        self.assertNotEqual(self.run_hook(1).returncode, 0)
        self.assertEqual(self.run_hook(0).returncode, 0)

    def test_runs_the_three_checks_from_the_repo_root(self):
        self.run_hook(0, cwd="sub")
        args, cwd = (self.repo / "scripts" / "args.txt").read_text().split("|")
        self.assertEqual(args, "--only compile --only tests --only changelog")
        self.assertEqual(Path(cwd).resolve(), self.repo.resolve())


if __name__ == "__main__":
    unittest.main()
