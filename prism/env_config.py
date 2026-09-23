"""
prism.env_config: A single, frozen, stdlib-only schema for every PRISM_* env var.

Phase 11 / P15 (spec.md §4). Replaces 31 scattered `os.environ.get(...)` calls across
five modules (`prism/cli.py`, `prism/resources.py`, `prism/templates.py`,
`prism/mcp.py`, `prism/server.py`) with one typed, validated dataclass plus an opt-in
`PRISM_ENV_FILE` loader. Invariant I4 ("No runtime dependencies") and `intent.md §3.2`
hold: stdlib only — no `kev`, no `pydantic-settings`, no `python-dotenv`.

Use `EnvConfig.from_env(environ)` to build from a dict (production callers pass
`os.environ` via `load_config()`), or `EnvConfig.from_env_file(path)` to layer a
hand-rolled `.env` file under process env. `__post_init__` validates every
field — bad values raise a single `ValueError` listing every error, not a
deferred `TypeError` at first use.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, fields
from typing import Dict, List, Mapping, Optional

logger = logging.getLogger("prism.env_config")

# Defaults pulled from the existing readers in prism.server / prism.resources /
# prism.machine_lock / prism.templates, so a clean env produces identical behaviour.
DEFAULT_QUEUE_TIMEOUT_SEC: float = 300.0       # prism.server.DEFAULT_QUEUE_TIMEOUT_SEC
DEFAULT_MAX_QUEUE: int = 8                    # prism.server.DEFAULT_MAX_QUEUE
DEFAULT_LOAD_TIMEOUT_S: float = 120.0         # prism.machine_lock.DEFAULT_LOAD_TIMEOUT_S
DEFAULT_RAM_RESERVE_MB: float = 2048.0        # prism.resources.DEFAULT_RAM_RESERVE_MB
DEFAULT_VRAM_RESERVE_MB: float = 1536.0       # prism.resources.DEFAULT_VRAM_RESERVE_MB
DEFAULT_PREFILL_CHUNK: int = 1024
DEFAULT_BASE_URL: str = "http://127.0.0.1:5272/v1"

ALLOWED_DEVICES: tuple = ("auto", "cpu", "cuda")
ALLOWED_TEMPLATES: tuple = ("auto", "jinja", "builtin")
FALSY_BOOL: tuple = ("off", "0", "false", "no", "")
TRUTHY_BOOL: tuple = ("on", "1", "true", "yes")


# The set of env var names this module understands. Anything starting with PRISM_
# that is NOT in this set is logged at WARN by `load_config()` — silent-typo guard.
DECLARED_PRISM_VARS: frozenset = frozenset({
    "PRISM_API_KEY",
    "PRISM_BASE_URL",
    "PRISM_DEVICE",
    "PRISM_TEMPLATE",
    "PRISM_QUEUE_TIMEOUT",
    "PRISM_MAX_QUEUE",
    "PRISM_PREFILL_CHUNK",
    "PRISM_THREADS",
    "PRISM_LOAD_TIMEOUT",
    "PRISM_LOAD_LOCK",
    "PRISM_RAM_RESERVE_MB",
    "PRISM_VRAM_RESERVE_MB",
    "PRISM_RESOURCE_CHECK",
    "PRISM_LOOP_GUARD",
    "PRISM_MCP_AUTO_STOP_SEC",
    "PRISM_FORCE_WSL",
    "PRISM_WSLCONFIG_PATH",
    "PRISM_MODEL_DIRS",
    "PRISM_PYTHON",
    "PRISM_ENV_FILE",
    "PRISM_STATE_DIR",
})


@dataclass(frozen=True)
class EnvConfig:
    """All `PRISM_*` env vars in one schema. Construct via `EnvConfig.from_env(environ)`;
    load once at startup via `load_config()`. Frozen so a config snapshot cannot be
    mutated after the CLI/server captures it."""

    api_key: Optional[str] = None
    base_url: str = DEFAULT_BASE_URL
    device: str = "auto"
    template: str = "auto"
    queue_timeout: float = DEFAULT_QUEUE_TIMEOUT_SEC
    max_queue: int = DEFAULT_MAX_QUEUE
    prefill_chunk: int = DEFAULT_PREFILL_CHUNK
    threads: Optional[int] = None
    load_timeout: float = DEFAULT_LOAD_TIMEOUT_S
    load_lock: bool = True
    ram_reserve_mb: float = DEFAULT_RAM_RESERVE_MB
    vram_reserve_mb: float = DEFAULT_VRAM_RESERVE_MB
    resource_check: bool = True
    loop_guard: bool = True
    mcp_auto_stop_sec: float = 0.0
    force_wsl: bool = False
    wslconfig_path: Optional[str] = None
    model_dirs: List[str] = field(default_factory=list)
    python: Optional[str] = None
    env_file: Optional[str] = None
    state_dir_override: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate ranges only (enums are checked in `from_env` so the operator sees
        every problem at once, not fail-fast on the first enum violation)."""
        errors: List[str] = []
        if self.prefill_chunk < 0:
            errors.append(f"PRISM_PREFILL_CHUNK={self.prefill_chunk} must be >= 0")
        if self.mcp_auto_stop_sec < 0:
            errors.append(f"PRISM_MCP_AUTO_STOP_SEC={self.mcp_auto_stop_sec} must be >= 0")
        if self.vram_reserve_mb <= 0:
            errors.append(f"PRISM_VRAM_RESERVE_MB={self.vram_reserve_mb} must be > 0")
        if self.ram_reserve_mb <= 0:
            errors.append(f"PRISM_RAM_RESERVE_MB={self.ram_reserve_mb} must be > 0")
        if self.threads is not None and self.threads <= 0:
            errors.append(f"PRISM_THREADS={self.threads} must be a positive integer or unset")
        if errors:
            raise ValueError("Invalid EnvConfig: " + "; ".join(errors))

    # --- Coercion helpers (classmethods keep the from_env() body readable) ----

    @staticmethod
    def _coerce_int(environ: Mapping[str, str], key: str, default: int, errors: List[str],
                    min_val: Optional[int] = None) -> int:
        raw = environ.get(key, "").strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            errors.append(f"{key}={raw!r} (expected int)")
            return default
        if min_val is not None and value < min_val:
            errors.append(f"{key}={value} must be >= {min_val} (got {raw!r})")
        return value

    @staticmethod
    def _coerce_float(environ: Mapping[str, str], key: str, default: float, errors: List[str],
                      min_val: Optional[float] = None) -> float:
        raw = environ.get(key, "").strip()
        if not raw:
            return default
        try:
            value = float(raw)
        except ValueError:
            errors.append(f"{key}={raw!r} (expected float)")
            return default
        if min_val is not None and value < min_val:
            errors.append(f"{key}={value} must be >= {min_val} (got {raw!r})")
        return value

    @staticmethod
    def _coerce_bool(environ: Mapping[str, str], key: str, default: bool, errors: List[str]) -> bool:
        raw = environ.get(key, "").strip().lower()
        if not raw:
            return default
        if raw in FALSY_BOOL:
            return False
        if raw in TRUTHY_BOOL:
            return True
        errors.append(f"{key}={raw!r} (expected one of {list(FALSY_BOOL)[:-1]}|{list(TRUTHY_BOOL)})")
        return default

    @staticmethod
    def _coerce_optional_str(environ: Mapping[str, str], key: str) -> Optional[str]:
        """Returns None when the env var is unset or empty (matches today's
        `os.environ.get(key) or None` idiom)."""
        raw = environ.get(key, "").strip()
        return raw or None

    @staticmethod
    def _coerce_list(environ: Mapping[str, str], key: str, sep: str = ":") -> List[str]:
        raw = environ.get(key, "").strip()
        if not raw:
            return []
        return [s.strip() for s in raw.split(sep) if s.strip()]

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> "EnvConfig":
        """Build an `EnvConfig` from `environ` (a Mapping[str, str]). Defaults to
        `os.environ` when `environ` is None. Bad values are accumulated across all
        fields and raised as a single `ValueError`."""
        if environ is None:
            environ = os.environ
        errors: List[str] = []

        device = environ.get("PRISM_DEVICE", "auto").strip().lower() or "auto"
        if device not in ALLOWED_DEVICES:
            errors.append(
                f"PRISM_DEVICE={device!r} must be one of {list(ALLOWED_DEVICES)}"
            )
        template = environ.get("PRISM_TEMPLATE", "auto").strip().lower() or "auto"
        if template not in ALLOWED_TEMPLATES:
            errors.append(
                f"PRISM_TEMPLATE={template!r} must be one of {list(ALLOWED_TEMPLATES)}"
            )

        # Build via cls() — __post_init__ catches range errors (prefill_chunk, etc.).
        # We catch anything that escapes __post_init__ and append its message so enum
        # + range + coercion errors surface together.
        try:
            instance = cls(
                api_key=cls._coerce_optional_str(environ, "PRISM_API_KEY"),
                base_url=environ.get("PRISM_BASE_URL", DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL,
                device=device,
                template=template,
                queue_timeout=cls._coerce_float(environ, "PRISM_QUEUE_TIMEOUT",
                                                DEFAULT_QUEUE_TIMEOUT_SEC, errors),
                max_queue=cls._coerce_int(environ, "PRISM_MAX_QUEUE", DEFAULT_MAX_QUEUE, errors,
                                         min_val=0),
                prefill_chunk=cls._coerce_int(environ, "PRISM_PREFILL_CHUNK",
                                              DEFAULT_PREFILL_CHUNK, errors, min_val=0),
                threads=cls._coerce_optional_threads(environ, errors),
                load_timeout=cls._coerce_float(environ, "PRISM_LOAD_TIMEOUT",
                                               DEFAULT_LOAD_TIMEOUT_S, errors),
                load_lock=cls._coerce_bool(environ, "PRISM_LOAD_LOCK", True, errors),
                ram_reserve_mb=cls._coerce_float(environ, "PRISM_RAM_RESERVE_MB",
                                                 DEFAULT_RAM_RESERVE_MB, errors, min_val=0.01),
                vram_reserve_mb=cls._coerce_float(environ, "PRISM_VRAM_RESERVE_MB",
                                                  DEFAULT_VRAM_RESERVE_MB, errors, min_val=0.01),
                resource_check=cls._coerce_bool(environ, "PRISM_RESOURCE_CHECK", True, errors),
                loop_guard=cls._coerce_bool(environ, "PRISM_LOOP_GUARD", True, errors),
                mcp_auto_stop_sec=cls._coerce_float(environ, "PRISM_MCP_AUTO_STOP_SEC",
                                                    0.0, errors, min_val=0.0),
                force_wsl=cls._coerce_bool(environ, "PRISM_FORCE_WSL", False, errors),
                wslconfig_path=cls._coerce_optional_str(environ, "PRISM_WSLCONFIG_PATH"),
                model_dirs=cls._coerce_list(environ, "PRISM_MODEL_DIRS", sep=":"),
                python=cls._coerce_optional_str(environ, "PRISM_PYTHON"),
                env_file=cls._coerce_optional_str(environ, "PRISM_ENV_FILE"),
                state_dir_override=cls._coerce_optional_str(environ, "PRISM_STATE_DIR"),
            )
        except ValueError as ex:
            # Range errors from __post_init__ survive as the single error reason; surface
            # them alongside the enum errors we already collected.
            raise ValueError(
                "Invalid env vars: " + "; ".join(errors + [str(ex).removeprefix("Invalid EnvConfig: ")])
            ) from ex
        if errors:
            raise ValueError("Invalid env vars: " + "; ".join(errors))
        return instance

    @classmethod
    def _coerce_optional_threads(cls, environ: Mapping[str, str], errors: List[str]) -> Optional[int]:
        raw = environ.get("PRISM_THREADS", "").strip()
        if not raw:
            return None
        try:
            value = int(raw)
        except ValueError:
            errors.append(f"PRISM_THREADS={raw!r} (expected a positive integer)")
            return None
        if value <= 0:
            errors.append(f"PRISM_THREADS={value} must be a positive integer > 0 (got {raw!r})")
            return None
        return value

    @classmethod
    def from_env_file(cls, path: str, environ: Optional[Mapping[str, str]] = None) -> "EnvConfig":
        """Read the .env file at `path`, then layer `environ` (default `os.environ`)
        on top so the process env wins. The env file is opt-in: `load_config()` only
        reads it when `PRISM_ENV_FILE` is explicitly set in the process env."""
        if environ is None:
            environ = {}
        try:
            file_values = _parse_env_file(path)
        except OSError as ex:
            raise ValueError(f"Could not read env file {path!r}: {ex}") from ex
        merged: Dict[str, str] = {**file_values, **environ}
        return cls.from_env(merged)


def _parse_env_file(path: str) -> Dict[str, str]:
    """Hand-rolled `KEY=VALUE` reader — handles whitespace, `# comment`, blank lines,
    and double- or single-quoted values. Stdlib only; no `python-dotenv`.

    Returns a dict; invalid lines (no `=`, blank, comment-only) are skipped. Stricter
    coercion errors are reported by `EnvConfig.from_env()`, not here.
    """
    import re
    _INLINE_COMMENT_RE = re.compile(r"\s#.*$")
    out: Dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, raw_value = line.partition("=")
            key = key.strip()
            if not key:
                continue
            value = raw_value.lstrip()
            # Quoted value: take content between matching quotes; comment chars inside
            # the quotes are literal.
            if value and value[0] in ('"', "'"):
                quote = value[0]
                end = value.find(quote, 1)
                if end >= 0:
                    value = value[1:end]
                else:
                    value = value[1:]  # unterminated quote — take what's there
            else:
                # Unquoted: strip trailing inline `# comment` (whitespace-before-`#`).
                value = _INLINE_COMMENT_RE.sub("", value).rstrip()
            out[key] = value
    return out


def load_config() -> EnvConfig:
    """Read `PRISM_ENV_FILE` if set (opt-in only — no auto-discovery of `.env` in
    CWD, security: never load a file the operator did not name), parse it, layer the
    file values under `os.environ` so process env wins, build an `EnvConfig`, and emit
    one `logging.warning` per `PRISM_*` key in the merged env that is not declared.

    Returns a fresh `EnvConfig`. Operators should call this once at process start
    and pass the result down so later changes to `os.environ` do not affect the
    captured config.
    """
    process_env = dict(os.environ)

    env_file = process_env.get("PRISM_ENV_FILE")
    if env_file:
        merged: Dict[str, str] = {**_parse_env_file(env_file), **process_env}
    else:
        merged = process_env

    # Silent-typo guard: any PRISM_* key in the merged env that is not in the
    # declared set is logged once at WARN. Process env continues to win on
    # conflicts (unknown keys are simply ignored).
    seen: set = set()
    for key in merged:
        if key.startswith("PRISM_") and key not in DECLARED_PRISM_VARS and key not in seen:
            logger.warning(
                "%s is set in the environment but is not a declared EnvConfig field; ignoring",
                key,
            )
            seen.add(key)

    return EnvConfig.from_env(merged)
