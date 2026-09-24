#!/usr/bin/env python3
"""Claude Code PostToolUse hook: lint the Python file that was just edited or written.

Wired in `.claude/settings.json` for `Edit|Write|MultiEdit`. Reads the hook JSON from stdin and runs the `lint` check of
`sdlc_check.py` on that one file, so only violations on lines changed since `main` are reported. Exit 2 with the
report on stderr when there is one (Claude Code hands it to the model); exit 0 for everything else. It must never
block an edit for any other reason.
Standard library only.
"""

import json
import sys
from pathlib import Path
from typing import Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sdlc_check  # noqa: E402


def run(payload: object, root: Path = sdlc_check.ROOT, base: str = "main") -> Tuple[int, str]:
    """(exit code, report) for one PostToolUse payload."""
    tool_input = payload.get("tool_input") if isinstance(payload, dict) else None
    path = tool_input.get("file_path") if isinstance(tool_input, dict) else None
    if not isinstance(path, str) or not path.endswith(".py"):
        return 0, ""
    target = (root / path).resolve()  # an absolute `path` replaces `root`
    try:
        rel = target.relative_to(root.resolve()).as_posix()
    except ValueError:
        return 0, ""
    if not target.is_file():
        return 0, ""
    status, detail = sdlc_check.check_lint(base, cwd=root, only=[rel])
    if status != "fail":
        return 0, ""
    return 2, f"ruff: violations on the lines you changed in {rel} (fix them; other lines are not checked):\n{detail}"


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except ValueError:
        return 0
    code, report = run(payload)
    if report:
        print(report, file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
