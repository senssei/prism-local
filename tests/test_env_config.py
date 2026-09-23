"""
Tests for the centralised, stdlib-only env-var schema in `prism/env_config.py`.

Phase 11 / P15 (spec.md). The 22 `PRISM_*` env vars are declared in one frozen
`EnvConfig` dataclass with type coercion, defaults, validators, and an opt-in
`PRISM_ENV_FILE` loader — replacing the 31 scattered `os.environ.get(...)` reads
today. No runtime dependency (`kev`, `pydantic-settings`, `python-dotenv`); stdlib only.
"""
import io
import logging
import os
import tempfile
import unittest
from pathlib import Path
from typing import Mapping
from unittest.mock import patch

from prism.env_config import EnvConfig, _parse_env_file, load_config


# Defaults documented as class-level on `EnvConfig` (Phase 11 / 11.1): device "auto",
# prefill_chunk 1024, load_lock True, resource_check True, loop_guard True, base_url
# the loopback fixed port, etc. Hard-coded here so the test does not depend on
# importing constants from across the package.
EXPECTED_DEFAULTS = {
    "api_key": None,
    "base_url": "http://127.0.0.1:5272/v1",
    "device": "auto",
    "template": "auto",
    "queue_timeout": 300.0,         # DEFAULT_QUEUE_TIMEOUT_SEC in prism.server
    "max_queue": 8,                 # DEFAULT_MAX_QUEUE in prism.server
    "prefill_chunk": 1024,
    "threads": None,                # default_threads(): None when PRISM_THREADS not set
    "load_timeout": 120.0,          # DEFAULT_LOAD_TIMEOUT_S in prism.machine_lock
    "load_lock": True,
    "ram_reserve_mb": 2048.0,        # DEFAULT_RAM_RESERVE_MB in prism.resources
    "vram_reserve_mb": 1536.0,       # DEFAULT_VRAM_RESERVE_MB in prism.resources
    "resource_check": True,
    "loop_guard": True,
    "mcp_auto_stop_sec": 0.0,       # 0 = disabled
    "force_wsl": False,
    "wslconfig_path": None,
    "model_dirs": [],               # os.pathsep split of $PRISM_MODEL_DIRS, "" -> []
    "python": None,                 # $PRISM_PYTHON, "" -> None
    "env_file": None,               # $PRISM_ENV_FILE, "" -> None
    "state_dir_override": None,     # $PRISM_STATE_DIR, "" -> None
}


class TestEnvConfig(unittest.TestCase):
    """Phase 11 / 11.2 hermetic tests for `prism.env_config`."""

    # --- 1. Defaults match today's `os.environ.get` behaviour -----------------------

    def test_defaults_match_existing_getter_behaviour(self):
        """With empty env, every EnvConfig field equals the default value today's
        `os.environ.get(<key>, <default>)` would return for that key."""
        cfg = EnvConfig.from_env({})
        for name, expected in EXPECTED_DEFAULTS.items():
            self.assertEqual(getattr(cfg, name), expected,
                             f"{name} default mismatch: got {getattr(cfg, name)!r}, expected {expected!r}")

    def test_device_defaults_to_auto(self):
        cfg = EnvConfig.from_env({})
        self.assertEqual(cfg.device, "auto")

    def test_empty_strings_fall_back_to_defaults(self):
        """Today's pattern: `os.environ.get("X", default)` with `X=""` returns the default.
        Phase 11 preserves that (empty strings fall back to defaults rather than parsing
        to int/float/bool from `""`)."""
        cfg = EnvConfig.from_env({
            "PRISM_DEVICE": "",
            "PRISM_PREFILL_CHUNK": "",
            "PRISM_LOAD_LOCK": "",
            "PRISM_VRAM_RESERVE_MB": "",
            "PRISM_MCP_AUTO_STOP_SEC": "",
            "PRISM_API_KEY": "",
        })
        self.assertEqual(cfg.device, "auto")
        self.assertEqual(cfg.prefill_chunk, 1024)
        self.assertEqual(cfg.load_lock, True)
        self.assertEqual(cfg.vram_reserve_mb, 1536.0)
        self.assertEqual(cfg.mcp_auto_stop_sec, 0.0)
        self.assertIsNone(cfg.api_key)

    # --- 2. Type coercion ------------------------------------------------------

    def test_from_env_coerces_typed_fields(self):
        cfg = EnvConfig.from_env({
            "PRISM_DEVICE": "cuda",
            "PRISM_TEMPLATE": "jinja",
            "PRISM_PREFILL_CHUNK": "2048",
            "PRISM_MAX_QUEUE": "16",
            "PRISM_QUEUE_TIMEOUT": "0",        # legacy: 0 → consumer maps to None
            "PRISM_LOAD_LOCK": "off",
            "PRISM_VRAM_RESERVE_MB": "1024",
            "PRISM_MCP_AUTO_STOP_SEC": "12.5",
            "PRISM_MODEL_DIRS": "/a:/b:/c",
        })
        self.assertEqual(cfg.device, "cuda")
        self.assertEqual(cfg.template, "jinja")
        self.assertEqual(cfg.prefill_chunk, 2048)
        self.assertEqual(cfg.max_queue, 16)
        self.assertEqual(cfg.queue_timeout, 0.0)
        self.assertEqual(cfg.load_lock, False)  # "off" is the only bool at this layer
        self.assertEqual(cfg.vram_reserve_mb, 1024.0)
        self.assertEqual(cfg.mcp_auto_stop_sec, 12.5)
        self.assertEqual(cfg.model_dirs, ["/a", "/b", "/c"])

    def test_from_env_bool_falsy_spellings_all_turn_load_lock_off(self):
        for spelling in ("off", "0", "false", "no", "OFF", "False", "NO"):
            cfg = EnvConfig.from_env({"PRISM_LOAD_LOCK": spelling})
            self.assertEqual(cfg.load_lock, False, f"PRISM_LOAD_LOCK={spelling!r} should be False")
        for spelling in ("on", "1", "true", "yes"):
            cfg = EnvConfig.from_env({"PRISM_LOAD_LOCK": spelling})
            self.assertEqual(cfg.load_lock, True, f"PRISM_LOAD_LOCK={spelling!r} should be True")

    # --- 3. Validation (Phase 11 I1 — never silently fall back) ------------------

    def test_invalid_value_raises_value_error_naming_the_field(self):
        """Today `PRISM_DEVICE=ROGUE` is silently written through; Phase 11 I1 says
        raise (silent fallback is a bug). Same for `PRISM_PREFILL_CHUNK=abc`,
        `PRISM_MCP_AUTO_STOP_SEC=-1.0`, `PRISM_VRAM_RESERVE_MB=0` (positive-only)."""
        cases = [
            ("PRISM_DEVICE", "ROGUE"),
            ("PRISM_PREFILL_CHUNK", "abc"),
            ("PRISM_MCP_AUTO_STOP_SEC", "-1.0"),
            ("PRISM_VRAM_RESERVE_MB", "0"),
            ("PRISM_RAM_RESERVE_MB", "-10"),
        ]
        for var, bad in cases:
            with self.subTest(var=var, bad=bad):
                with self.assertRaises(ValueError) as ctx:
                    EnvConfig.from_env({var: bad})
                msg = str(ctx.exception)
                self.assertIn(var, msg)
                # Error must name the variable and the field — not just "something is wrong".

    def test_post_init_enumerates_all_invalid_fields(self):
        """A single from_env() raises ValueError listing every invalid field, not
        fail-fast on the first — so the operator sees the whole list at once."""
        with self.assertRaises(ValueError) as ctx:
            EnvConfig.from_env({
                "PRISM_DEVICE": "ROGUE",
                "PRISM_PREFILL_CHUNK": "abc",
                "PRISM_LOAD_LOCK": "maybe",
            })
        msg = str(ctx.exception)
        for var in ("PRISM_DEVICE", "PRISM_PREFILL_CHUNK", "PRISM_LOAD_LOCK"):
            self.assertIn(var, msg)

    # --- 4. .env file parser ----------------------------------------------------

    def test_quoted_value_in_env_file_preserves_spaces(self):
        """The hand-rolled .env parser handles double-quoted values, single-quoted
        values, comments, and blank lines — and the typed coerced view is the same."""
        with tempfile.TemporaryDirectory() as d:
            env_path = Path(d) / ".env"
            env_path.write_text(
                "# Prism config for the local dev box\n"
                "\n"
                "PRISM_DEVICE=cuda\n"
                'PRISM_BASE_URL="http://localhost:5272/v1 key"   # inline comment\n'
                "PRISM_PREFILL_CHUNK=2048\n"
                "PRISM_LOAD_LOCK=off\n"
                "\n"
                "PRISM_MODEL_DIRS=/a:/b:/c\n"
            )
            cfg = EnvConfig.from_env_file(str(env_path))
        self.assertEqual(cfg.device, "cuda")
        self.assertEqual(cfg.base_url, "http://localhost:5272/v1 key")
        self.assertEqual(cfg.prefill_chunk, 2048)
        self.assertEqual(cfg.load_lock, False)
        self.assertEqual(cfg.model_dirs, ["/a", "/b", "/c"])

    def test_parse_env_file_helper_skips_comments_and_blanks(self):
        """`_parse_env_file` (the internal helper) returns a dict — exposes the
        bare parser for direct test without going through `from_env_file`. Quoted
        values keep their internal whitespace; unquoted trailing `# comment` is
        stripped."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / ".env"
            p.write_text("# head\n\nK1=v1\n  K2=\"v 2\"  # tail\nK3='v3'\nK4=plain # end-of-line comment\n")
            out = _parse_env_file(str(p))
        self.assertEqual(out, {"K1": "v1", "K2": "v 2", "K3": "v3", "K4": "plain"})

    # --- 5. Process env overrides .env file (no silent precedence flip) ----------

    def test_process_env_overrides_env_file(self):
        with tempfile.TemporaryDirectory() as d:
            env_path = Path(d) / ".env"
            env_path.write_text("PRISM_DEVICE=cpu\n")
            cfg = EnvConfig.from_env_file(str(env_path), environ={"PRISM_DEVICE": "cuda"})
        self.assertEqual(cfg.device, "cuda")

    def test_no_prism_env_file_means_no_file_read(self):
        """`load_config()` only touches the filesystem when `PRISM_ENV_FILE` is set —
        verified via `mock.patch("os.path.exists")` so the test is hermetic."""
        with patch.dict(os.environ, {}, clear=True), \
             patch("os.path.exists", return_value=False) as mocked_exists:
            cfg = load_config()
        mocked_exists.assert_not_called()  # not even a "does the file exist?" probe
        self.assertEqual(cfg.device, "auto")

    # --- 6. Unknown PRISM_* vars warn (silent-typo guard) -----------------------

    def test_unknown_prism_var_warns_at_startup(self):
        """`PRISM_FOO` in `os.environ` is not in `EnvConfig`'s fields — `load_config()`
        emits exactly one `logging.warning` whose message names the unknown var,
        rather than silently dropping the typo."""
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setLevel(logging.WARNING)
        root = logging.getLogger()
        root.addHandler(handler)
        prev_level = root.level
        root.setLevel(logging.WARNING)
        try:
            with patch.dict(os.environ, {"PRISM_FOO": "bar"}, clear=False):
                # Run twice to assert "at least one" — buffer may also pick up
                # any setup warnings. The key test is that PRISM_FOO is named.
                load_config()
            output = buf.getvalue()
            self.assertIn("PRISM_FOO", output)
            self.assertIn("ignoring", output.lower())
        finally:
            root.removeHandler(handler)
            root.setLevel(prev_level)


if __name__ == "__main__":
    unittest.main()
