"""The SDLC gate helpers: `--red` (a new test must fail first)."""

import contextlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

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


FAKE_RUFF = textwrap.dedent(
    """
    import json, os, sys
    args = sys.argv[1:]
    with open(os.environ["FAKE_RUFF_LOG"], "a") as f:
        f.write(json.dumps(args) + "\\n")
    if os.environ.get("FAKE_RUFF_EXIT") == "2":
        sys.stderr.write("ruff: invalid config")
        sys.exit(2)
    rows = [int(r) for r in os.environ.get("FAKE_RUFF_ROWS", "").split(",") if r]
    out = [
        {"code": "F401", "message": "`os` imported but unused", "filename": os.path.abspath(f),
         "location": {"row": r, "column": 1}, "end_location": {"row": r, "column": 10}}
        for f in args if f.endswith(".py") for r in rows
    ]
    print(json.dumps(out))
    sys.exit(1 if out else 0)
    """
)
GIT = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]


@unittest.skipUnless(shutil.which("git"), "needs git")
class LintFixture(unittest.TestCase):
    """A throwaway git repo with a committed 10-line a.py and a fake ruff: one finding per FAKE_RUFF_ROWS row."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.repo = Path(tmp.name) / "repo"
        self.repo.mkdir()
        self.log = Path(tmp.name) / "ruff.log"
        fake = Path(tmp.name) / "fake_ruff.py"  # outside the repo, or it would be linted itself
        fake.write_text(FAKE_RUFF)
        self.git("init", "-q")
        (self.repo / "a.py").write_text("".join(f"x{i} = {i}\n" for i in range(1, 11)))
        (self.repo / "notes.md").write_text("notes\n")
        self.git("add", ".")
        self.git("commit", "-qm", "seed")
        patcher = mock.patch.object(sdlc_check, "_ruff_cmd", return_value=[sys.executable, str(fake)])
        patcher.start()
        self.addCleanup(patcher.stop)
        env = mock.patch.dict(os.environ, {"FAKE_RUFF_LOG": str(self.log), "FAKE_RUFF_ROWS": "", "FAKE_RUFF_EXIT": ""})
        env.start()
        self.addCleanup(env.stop)

    def git(self, *args):
        subprocess.run([*GIT, *args], cwd=self.repo, check=True)

    def edit_line(self, n, name="a.py"):
        lines = (self.repo / name).read_text().splitlines(keepends=True)
        lines[n - 1] = f"changed{n} = 0\n"
        (self.repo / name).write_text("".join(lines))

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []


class TestLint(LintFixture):
    def lint(self, rows="", **kw):
        os.environ["FAKE_RUFF_ROWS"] = rows
        return sdlc_check.check_lint("HEAD", cwd=self.repo, **kw)

    def test_lint_is_a_registered_check(self):
        self.assertIn("lint", [name for name, _ in sdlc_check.CHECKS])

    def test_skips_when_ruff_is_not_installed(self):
        self.edit_line(5)
        with mock.patch.object(sdlc_check, "_ruff_cmd", return_value=None):
            status, detail = sdlc_check.check_lint("HEAD", cwd=self.repo)
        self.assertEqual(status, "skip")
        self.assertIn("ruff", detail)

    def test_skips_when_git_cannot_diff_against_the_base(self):
        status, _ = sdlc_check.check_lint("no-such-ref", cwd=self.repo)
        self.assertEqual(status, "skip")

    def test_passes_without_calling_ruff_when_no_python_file_changed(self):
        (self.repo / "notes.md").write_text("more notes\n")
        self.assertEqual(self.lint("1")[0], "pass")
        self.assertEqual(self.calls(), [])

    def test_fails_on_a_violation_on_a_changed_line(self):
        self.edit_line(5)
        status, detail = self.lint("5")
        self.assertEqual(status, "fail")
        self.assertIn("a.py:5:1 F401", detail)

    def test_ignores_violations_on_untouched_lines_of_a_changed_file(self):
        self.edit_line(5)
        status, _ = self.lint("2,9")
        self.assertEqual(status, "pass")
        self.assertEqual(len(self.calls()), 1)  # ruff did run; the findings were filtered out

    def test_reports_only_the_changed_lines_among_several_findings(self):
        self.edit_line(5)
        status, detail = self.lint("2,5,9")
        self.assertEqual(status, "fail")
        self.assertIn("a.py:5:1", detail)
        self.assertNotIn("a.py:2:1", detail)
        self.assertNotIn("a.py:9:1", detail)

    def test_a_pure_deletion_changes_no_line(self):
        lines = (self.repo / "a.py").read_text().splitlines(keepends=True)
        del lines[4]
        (self.repo / "a.py").write_text("".join(lines))
        self.assertEqual(self.lint("4,5")[0], "pass")

    def test_untracked_python_file_is_checked_in_full(self):
        (self.repo / "b.py").write_text("y = 1\ny = 2\ny = 3\n")
        status, detail = self.lint("2")
        self.assertEqual(status, "fail")
        self.assertIn("b.py:2:1", detail)

    def test_deleted_python_file_is_not_passed_to_ruff(self):
        (self.repo / "a.py").unlink()
        self.assertEqual(self.lint("1")[0], "pass")
        self.assertEqual(self.calls(), [])

    def test_ruff_gets_only_the_changed_python_files_with_force_exclude_and_json(self):
        self.edit_line(5)
        (self.repo / "b.py").write_text("y = 1\n")
        (self.repo / "c.md").write_text("doc\n")
        self.lint("")
        (args,) = self.calls()
        files = [a for a in args if not a.startswith("-") and a not in ("check", "json")]
        self.assertEqual(sorted(files), ["a.py", "b.py"])
        self.assertIn("--force-exclude", args)
        self.assertEqual(args[args.index("--output-format") + 1], "json")
        self.assertNotIn("--fix", args)

    def test_only_restricts_the_files(self):
        self.edit_line(5)
        (self.repo / "b.py").write_text("y = 1\n")
        self.lint("", only=["b.py"])
        (args,) = self.calls()
        self.assertIn("b.py", args)
        self.assertNotIn("a.py", args)

    def test_ruff_error_fails_with_its_stderr(self):
        self.edit_line(5)
        os.environ["FAKE_RUFF_EXIT"] = "2"
        status, detail = self.lint("")
        self.assertEqual(status, "fail")
        self.assertIn("invalid config", detail)


HOOK_SCRIPT = SCRIPT.parent / "lint_hook.py"


class TestLintHook(LintFixture):
    def setUp(self):
        super().setUp()
        # A missing script must fail these tests (red), not skip them, so it is loaded here rather than at import time.
        spec = importlib.util.spec_from_file_location("lint_hook", HOOK_SCRIPT)
        self.lint_hook = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.lint_hook)
        # The hook imports its own copy of sdlc_check; point that copy at the same fake ruff.
        patcher = mock.patch.object(self.lint_hook.sdlc_check, "_ruff_cmd", return_value=sdlc_check._ruff_cmd())
        patcher.start()
        self.addCleanup(patcher.stop)

    def hook(self, path, rows="", tool="Edit"):
        os.environ["FAKE_RUFF_ROWS"] = rows
        payload = {"tool_name": tool, "tool_input": {"file_path": str(path)}}
        return self.lint_hook.run(payload, root=self.repo, base="HEAD")

    def test_a_finding_on_a_changed_line_exits_2_with_the_report(self):
        self.edit_line(5)
        code, report = self.hook(self.repo / "a.py", "5")
        self.assertEqual(code, 2)
        self.assertIn("a.py:5:1 F401", report)

    def test_a_finding_on_an_untouched_line_exits_0(self):
        self.edit_line(5)
        self.assertEqual(self.hook(self.repo / "a.py", "2")[0], 0)

    def test_a_clean_file_exits_0(self):
        self.edit_line(5)
        self.assertEqual(self.hook(self.repo / "a.py", "")[0], 0)

    def test_only_the_edited_file_is_linted(self):
        self.edit_line(5)
        (self.repo / "b.py").write_text("y = 1\n")
        self.hook(self.repo / "a.py", "")
        (args,) = self.calls()
        self.assertIn("a.py", args)
        self.assertNotIn("b.py", args)

    def test_relative_paths_are_resolved_against_the_repo_root(self):
        self.edit_line(5)
        self.assertEqual(self.hook("a.py", "5")[0], 2)

    def test_everything_that_is_not_a_finding_exits_0_without_calling_ruff(self):
        (self.repo / "notes.md").write_text("more\n")
        outside = self.repo.parent / "out.py"
        outside.write_text("x = 1\n")
        cases = {"markdown": self.repo / "notes.md", "outside the repo": outside, "missing": self.repo / "nope.py"}
        for name, path in cases.items():
            with self.subTest(name):
                self.assertEqual(self.hook(path, "1"), (0, ""))
        self.assertEqual(self.lint_hook.run({"tool_input": {}}, root=self.repo, base="HEAD"), (0, ""))
        self.assertEqual(self.lint_hook.run("not a dict", root=self.repo, base="HEAD"), (0, ""))
        self.assertEqual(self.calls(), [])

    def test_missing_ruff_exits_0(self):
        self.edit_line(5)
        with mock.patch.object(self.lint_hook.sdlc_check, "_ruff_cmd", return_value=None):
            code, _ = self.lint_hook.run({"tool_input": {"file_path": "a.py"}}, root=self.repo, base="HEAD")
        self.assertEqual(code, 0)

    def test_script_ignores_invalid_json_on_stdin(self):
        proc = subprocess.run([sys.executable, str(HOOK_SCRIPT)], input="not json", capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0)


if __name__ == "__main__":
    unittest.main()
