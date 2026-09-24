---
name: python-conventions
description: How to write and edit Python in prism-local (`prism/`, `foundry_wsl/`, `scripts/`, `tests/`). Use before writing or changing any Python file here: typing and logging style, error handling, locks, stdio-protocol safety, hermetic tests, and running ruff on your change.
---

# Python conventions for prism-local

The short rules are in `AGENTS.md` (`## Python conventions`). This skill has the reasoning and the patterns to copy. Match the
module you are editing over anything written here; the code base was never machine-formatted, so consistency with the
neighbouring code is the rule and a drive-by restyle is a defect (one logical change per commit).

## Before you finish

```bash
ruff check path/to/changed.py                 # or: python3 scripts/sdlc_check.py --only lint
```

The gate reports only violations **on lines you changed** (spec section 5), so a clean run means your diff is clean, not the whole
file. In Claude Code a `PostToolUse` hook (`scripts/lint_hook.py`) runs the same check after each edit and feeds findings back.
Do not run `ruff format` or `ruff check --fix` on a whole file: it rewrites lines you did not mean to touch. `ruff format --diff FILE`
is fine for looking.

## Compatibility

- Python **3.10** is the floor (CI runs 3.10, 3.11, 3.12). Do not use 3.11+ features: `tomllib`, `ExceptionGroup`/`except*`,
  `typing.Self`, `enum.StrEnum`, `typing.override`, `asyncio.TaskGroup`.
- The runtime is standard library only. Engine packages (`onnxruntime_genai`, `huggingface_hub`, `jinja2`, ...) are optional
  extras: import them lazily, inside the function that needs them, and raise an error that names the extra to install.

## Types and docstrings

- Annotate public functions. This code base spells types `Optional[X]`, `List[X]`, `Dict[K, V]` from `typing` (`ruff` is configured
  to allow it). Do not mix in `X | None` or `list[X]` in a module that does not already use them.
- A module starts with a docstring `prism.<name>: one line of purpose`. A function gets one sentence about behavior, and about the
  exception it raises when that is part of the contract (see `prism/resources.py`).
- Line length is 120. Use `pathlib.Path` in new code; leave existing `os.path` code alone.

## Errors and logging

- No bare `except:`. Catch the narrowest exception. `except Exception` belongs only at a boundary that must not die (a request
  handler, the ACP/MCP read loop) and it logs with `logger.exception(...)` or `logger.warning(...)`; it never ends in `pass`.
- A silent hardware fallback is a bug (`spec.md` invariant I1): report it to the user.
- Chain on re-raise: `raise NewError("...") from exc`.
- `logger = logging.getLogger("prism.<module>")` at module level (`prism.server`, `prism.acp`, `prism.cli`, ...). `print` is for
  output the user asked for from the CLI. Diagnostics go through the logger.
- **stdout is the protocol** in `prism acp` and `prism mcp`: only JSON-RPC frames, written through the module's locked writer
  (`sys.stdout.write` plus `flush` under `_stdout_lock` in `prism/acp.py`). A stray `print` corrupts the stream.
- Custom exceptions subclass the nearest existing one (`InsufficientResourcesError(ModelLoadError)`).

## Concurrency

Shared mutable state sits behind a `threading.Lock`. When a code path can re-enter the same lock (the lazy singletons
`_engine_manager()` calling `_catalog()` in `prism/acp.py` and `prism/mcp.py`), use `threading.RLock`; a plain `Lock` self-deadlocks
on the nested acquire (fixed in `238c3ff`). Build a lazy singleton under its lock, and say in a comment why a lock is an `RLock`.

## Security

No `shell=True`, `eval`, `exec` or `pickle` on data that did not come from us. Pass `subprocess` an argument list. Treat paths from a
client (ACP `fs/*`, HTTP routes) as untrusted and check them against the allowed root. Do not log secrets or prompt text at INFO.
`ruff` rule set `S` enforces the mechanical part; the noisy `S310`, `S603`, `S607`, `S104`, `S108` are switched off on purpose in
`pyproject.toml` because this is a local server that fetches URLs and runs `ollama` and `nvidia-smi`.

## Tests

- `unittest`, hermetic: no GPU, Ollama, network or model files. Use `tests/fakes.py` (`FakeEngine`, `FakeOg`), temp dirs,
  `create_server(port=0)`. `FakeEngine` keeps class-level state, so call `reset()` in `setUp`.
- Name the file `tests/test_prism_<module>.py`. One behavior per test method, named for the behavior.
- A new test must be seen red for the right reason before the code exists: `python3 scripts/sdlc_check.py --red <test id>`.
