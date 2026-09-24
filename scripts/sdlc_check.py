#!/usr/bin/env python3
"""Deterministic SDLC gate shared by every harness (Claude Code, Codex, Cursor, CI).

Skills tell an agent *when* to run this; the checks themselves live here so the verdict does not depend on which agent
runs them. Standard library only.

    python3 scripts/sdlc_check.py                 # all checks against the diff from `main`
    python3 scripts/sdlc_check.py --only tests    # one check: compile, tests, changelog, docs, lint
    python3 scripts/sdlc_check.py --base origin/main
    python3 scripts/sdlc_check.py --red tests.test_x.TestY.test_z   # red-first: these tests must FAIL now

Exit status is 0 when every selected check passed or was skipped (with --red: when every named test failed), 1 otherwise.
"""

import argparse
import contextlib
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
# Result: (status, detail) with status one of "pass", "fail", "skip".
Result = Tuple[str, str]
RED_TIMEOUT_S = 60  # per test; a hanging test must not hang the agent that runs `--red`


def _run(cmd: List[str], env_extra: Optional[dict] = None, cwd: Path = ROOT) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.update(env_extra or {})
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)


def _tail(proc: subprocess.CompletedProcess, lines: int = 25) -> str:
    out = (proc.stdout + proc.stderr).strip().splitlines()
    return "\n".join(out[-lines:])


def changed_files(base: str, cwd: Path = ROOT) -> Optional[List[str]]:
    """Files changed relative to `base` (committed, staged, unstaged) plus untracked ones; None if git cannot tell.

    NUL-separated (-z) so paths with spaces or non-ASCII characters come back verbatim instead of quoted.
    """
    mb = _run(["git", "merge-base", base, "HEAD"], cwd=cwd)
    if mb.returncode != 0:
        return None
    diff = _run(["git", "diff", "--name-only", "-z", mb.stdout.strip()], cwd=cwd)
    untracked = _run(["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=cwd)
    if diff.returncode != 0 or untracked.returncode != 0:
        return None
    return sorted({*filter(None, diff.stdout.split("\0")), *filter(None, untracked.stdout.split("\0"))})


_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def changed_line_ranges(base: str, files: List[str], cwd: Path = ROOT) -> Optional[Dict[str, List[Tuple[int, int]]]]:
    """Inclusive (first, last) line ranges of `files` changed since the merge-base with `base`; None if git cannot tell.

    Every line of an untracked file counts as changed. A pure deletion changes no line of the new file. One
    `git diff -U0` per file, so a path never has to be parsed back out of the diff.
    """
    mb = _run(["git", "merge-base", base, "HEAD"], cwd=cwd)
    untracked = _run(["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=cwd)
    if mb.returncode != 0 or untracked.returncode != 0:
        return None
    new = set(filter(None, untracked.stdout.split("\0")))
    ranges: Dict[str, List[Tuple[int, int]]] = {}
    for f in files:
        if f in new:
            ranges[f] = [(1, sys.maxsize)]
            continue
        diff = _run(["git", "diff", "-U0", "--no-color", "--no-ext-diff", mb.stdout.strip(), "--", f], cwd=cwd)
        if diff.returncode != 0:
            return None
        ranges[f] = []
        for line in diff.stdout.splitlines():
            hunk = _HUNK.match(line)
            if hunk:
                start, count = int(hunk.group(1)), int(hunk.group(2) or 1)
                if count:
                    ranges[f].append((start, start + count - 1))
    return ranges


def check_compile(_base: str) -> Result:
    proc = _run([sys.executable, "-m", "compileall", "-q", "prism", "foundry_wsl", "tests", "scripts"])
    return ("pass", "") if proc.returncode == 0 else ("fail", _tail(proc))


def check_tests(_base: str) -> Result:
    proc = _run([sys.executable, "-m", "unittest", "discover", "-s", "tests"])
    return ("pass", _tail(proc, 3)) if proc.returncode == 0 else ("fail", _tail(proc))


def check_changelog(base: str) -> Result:
    """Runtime changes (prism/, foundry_wsl/) need a CHANGELOG.md entry, as CONTRIBUTING.md requires."""
    files = changed_files(base)
    if files is None:
        return "skip", f"cannot diff against {base!r} (no such ref or not a git checkout)"
    code = [f for f in files if f.startswith(("prism/", "foundry_wsl/"))]
    if not code:
        return "pass", "no runtime code changed"
    if "CHANGELOG.md" in files:
        return "pass", f"{len(code)} runtime file(s) changed, CHANGELOG.md updated"
    return "fail", "runtime code changed but CHANGELOG.md was not touched:\n  " + "\n  ".join(code)


def check_docs(_base: str) -> Result:
    if shutil.which("mkdocs") is None and _run([sys.executable, "-c", "import mkdocs"]).returncode != 0:
        return "skip", 'mkdocs not installed (pip install -e ".[docs]")'
    proc = _run([sys.executable, "-m", "mkdocs", "build", "--strict", "--site-dir", str(ROOT / "scratch" / "site-check")])
    return ("pass", "") if proc.returncode == 0 else ("fail", _tail(proc))


def _ruff_cmd() -> Optional[List[str]]:
    """The command that runs ruff: `ruff` on PATH, the checkout's `.venv`, or `python -m ruff`; None if missing."""
    found = shutil.which("ruff")
    if found:
        return [found]
    venv = ROOT / ".venv" / "bin" / "ruff"
    if venv.is_file():
        return [str(venv)]
    if _run([sys.executable, "-m", "ruff", "--version"]).returncode == 0:
        return [sys.executable, "-m", "ruff"]
    return None


def check_lint(base: str, cwd: Path = ROOT, only: Optional[List[str]] = None) -> Result:
    """ruff on the changed Python files, reporting only violations on changed lines (old code is not this diff's debt).

    `only` restricts the files (repo-relative paths); the edit hook passes the one file it just saw written.
    Never modifies a file.
    """
    ruff = _ruff_cmd()
    if ruff is None:
        return "skip", 'ruff not installed (pip install -e ".[dev]")'
    files = changed_files(base, cwd)
    if files is None:
        return "skip", f"cannot diff against {base!r} (no such ref or not a git checkout)"
    py = [f for f in files if f.endswith(".py") and (cwd / f).is_file() and (only is None or f in only)]
    if not py:
        return "pass", "no Python file changed"
    ranges = changed_line_ranges(base, py, cwd)
    if ranges is None:
        return "skip", f"cannot diff against {base!r} (no such ref or not a git checkout)"
    proc = _run([*ruff, "check", "--force-exclude", "--output-format", "json", *py], cwd=cwd)
    try:
        findings = json.loads(proc.stdout or "[]")
    except ValueError:
        findings = None
    if proc.returncode not in (0, 1) or not isinstance(findings, list):
        return "fail", _tail(proc) or f"ruff exited {proc.returncode} without a report"
    report = []
    for item in findings:
        try:
            name = Path(item["filename"]).resolve().relative_to(cwd.resolve()).as_posix()
        except ValueError:
            name = item["filename"]
        row, col = item["location"]["row"], item["location"]["column"]
        last = (item.get("end_location") or {}).get("row", row)
        if any(row <= b and a <= last for a, b in ranges.get(name, [])):
            report.append((name, row, col, f"{name}:{row}:{col} {item['code']} {item['message']}"))
    if report:
        return "fail", "\n".join(line for *_, line in sorted(report))
    return "pass", f"{len(py)} Python file(s), no violation on a changed line"


def _flatten(suite: unittest.TestSuite) -> List[unittest.TestCase]:
    tests: List[unittest.TestCase] = []
    for item in suite:
        tests.extend(_flatten(item) if isinstance(item, unittest.TestSuite) else [item])
    return tests


def _headline(traceback_text: str) -> str:
    """The exception line of a traceback (`AssertionError: 1 != 2`), not the last line, which can be a diff or a caret marker."""
    lines = traceback_text.strip().splitlines()
    if not lines:
        return "(no message)"
    frames = [i for i, line in enumerate(lines) if line.startswith('  File "')]
    for line in lines[(frames[-1] + 1 if frames else 0):]:
        if line and not line[0].isspace():
            return line[:200]
    return lines[-1][:200]


class _Timeout(BaseException):
    pass


def _run_one(test: unittest.TestCase, timeout: float) -> Tuple[unittest.TestResult, bool]:
    """Run one test with its output swallowed and a wall-clock limit; returns (result, timed_out)."""
    result = unittest.TestResult()
    state = {"timed_out": False}

    def on_alarm(_signum, _frame):
        state["timed_out"] = True
        raise _Timeout()

    old = signal.signal(signal.SIGALRM, on_alarm)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            unittest.TestSuite([test]).run(result)
    except _Timeout:
        pass
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
    return result, state["timed_out"]


def run_red(test_ids: List[str], timeout: float = RED_TIMEOUT_S) -> Tuple[bool, List[str]]:
    """Red-first check: True only if every named test exists and fails or errors right now.

    A test that passes proves nothing, a skipped or expectedFailure one does not count, and an id that does not resolve to a
    test is a typo, not a red test. The exception line of each failure is returned so the caller can judge whether it fails
    for the *expected* reason. Test output is swallowed; each test has `timeout` seconds.
    """
    if not test_ids:
        return False, ["no test ids given"]
    loader = unittest.TestLoader()
    lines: List[str] = []
    ok = True
    for tid in test_ids:
        try:
            tests = _flatten(loader.loadTestsFromName(tid))
        except (TypeError, ImportError, AttributeError):  # e.g. the id names a constant or a non-test callable
            tests = []
        # A missing module or attribute comes back as a synthetic `_FailedTest` whose run raises the import error.
        if not tests or any(t.id().startswith("unittest.loader._FailedTest") for t in tests):
            lines.append(f"NOT RED  {tid}: not found or not a test (check the id: module.Class.method)")
            ok = False
            continue
        for t in tests:
            result, timed_out = _run_one(t, timeout)
            if timed_out:
                lines.append(f"NOT RED  {t.id()}: timed out after {timeout:g}s")
            elif result.failures or result.errors:
                lines.append(f"RED      {t.id()}: {_headline((result.failures + result.errors)[0][1])}")
                continue
            elif result.expectedFailures:
                lines.append(f"NOT RED  {t.id()}: marked expectedFailure, so the suite counts it as passing")
            elif result.unexpectedSuccesses:
                lines.append(f"NOT RED  {t.id()}: unexpected success")
            elif result.skipped:
                lines.append(f"NOT RED  {t.id()}: skipped, so it was not run")
            else:
                lines.append(f"NOT RED  {t.id()}: passes already, so it proves nothing")
            ok = False
    return ok, lines


CHECKS: List[Tuple[str, Callable[[str], Result]]] = [
    ("compile", check_compile),
    ("tests", check_tests),
    ("changelog", check_changelog),
    ("docs", check_docs),
    ("lint", check_lint),
]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", help="ref the diff is taken from (default: main)")
    ap.add_argument("--only", action="append", choices=[n for n, _ in CHECKS], help="run only this check (repeatable)")
    ap.add_argument("--red", nargs="+", metavar="TEST_ID", help="expect these tests to fail now (red-first); runs no other check")
    args = ap.parse_args(argv)

    if args.red is not None:
        if args.only or args.base:
            ap.error("--red runs no other check; drop --only / --base")
        sys.path.insert(0, str(ROOT))
        ok, lines = run_red(args.red)
        print("\n".join(lines))
        return 0 if ok else 1

    failed = False
    for name, fn in CHECKS:
        if args.only and name not in args.only:
            continue
        status, detail = fn(args.base or "main")
        print(f"[{status.upper():4}] {name}" + (f": {detail.splitlines()[0]}" if detail and status != "fail" else ""))
        if status == "fail":
            failed = True
            if detail:
                print("\n".join("       " + line for line in detail.splitlines()))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
