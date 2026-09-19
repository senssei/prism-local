"""Guards against docs drifting from the code: every subcommand and route must be documented."""

import argparse
import re
import unittest
from pathlib import Path

from prism import cli

DOCS = Path(__file__).resolve().parent.parent / "docs"


def _subcommands():
    captured = {}
    orig = argparse.ArgumentParser.parse_args

    def spy(self, *a, **k):
        captured["parser"] = self
        raise SystemExit(0)

    argparse.ArgumentParser.parse_args = spy
    try:
        try:
            cli.main()
        except SystemExit:
            pass
    finally:
        argparse.ArgumentParser.parse_args = orig
    action = next(a for a in captured["parser"]._actions if isinstance(a, argparse._SubParsersAction))
    return sorted(action.choices)


class TestDocsCoverage(unittest.TestCase):
    def test_every_subcommand_is_documented(self):
        text = (DOCS / "cli.md").read_text()
        names = _subcommands()
        self.assertGreaterEqual(len(names), 10)
        for name in names:
            self.assertIn(f"prism {name}", text, f"`prism {name}` is missing from docs/cli.md")

    def test_every_server_route_is_documented(self):
        source = Path(cli.__file__).with_name("server.py").read_text()
        routes = set(re.findall(r'"(/(?:v1/)?[a-z/]+)":\s*self\._handle', source))
        self.assertTrue({"/v1/models", "/v1/chat/completions", "/health"} <= routes, routes)
        text = (DOCS / "api.md").read_text()
        for route in routes:
            self.assertIn(route, text, f"{route} is missing from docs/api.md")

    def test_every_env_var_is_documented(self):
        text = (DOCS / "getting-started.md").read_text()
        for var in ("PRISM_MODEL_DIRS", "PRISM_DEVICE", "PRISM_API_KEY", "PRISM_BASE_URL", "PRISM_PYTHON"):
            self.assertIn(var, text)


if __name__ == "__main__":
    unittest.main()
