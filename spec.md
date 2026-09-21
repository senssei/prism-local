# Specification: Prism (`prism-local`)

This file records what must stay true (invariants), what happens when things go wrong (failure modes), behavior that is planned
but not built, and the contract of the development tooling. It does not repeat the wire formats: `docs/api.md` (HTTP),
`docs/cli.md` (commands) and `docs/getting-started.md` (environment variables) are **normative** for those. Change them in the
same commit as the behavior.

## 1. Components

`docs/architecture.md` describes the modules and the request flow. In short: `cli.py` turns subcommands into calls; `catalog.py`
resolves model names; `engine.py` (`OnnxGenAiEngine`) loads and streams ONNX models; `ollama_bridge.py` talks to Ollama;
`server.py` is the OpenAI-compatible server and owns the engine lock (`ActiveEngineManager`); `telemetry.py` reads the GPU;
`mcp.py` and `connectors.py` serve and configure MCP and editor clients.

## 2. Invariants

Tests and reviews cite these by number. Changing one needs operator approval.

| # | Invariant |
|---|---|
| I1 | **Device honesty.** The execution provider actually used is reported (`prism run` stderr, `GET /health` `active_device`, `telemetry.device`, `prism benchmark`). A fallback is never silent: `--device auto` that falls back to CPU warns with the underlying error; `--device cuda` fails with an error naming the cause. |
| I2 | **One resident ONNX model, one lock.** Loading and generation are serialized in `ActiveEngineManager.use_engine`; asking for another ONNX model unloads the current one. Ollama requests are not serialized by Prism. |
| I3 | **Errors before headers.** Model loading and the first Ollama chunk are resolved before any response header is sent, so failures are JSON errors, not dropped connections. |
| I4 | **No runtime dependencies.** The package imports only the standard library at import time; optional engines are imported lazily and their absence is handled. |
| I5 | **Safe defaults.** Loopback bind, no CORS headers, non-loopback `Host` rejected on a loopback bind, 10 MB body cap, constant-time bearer comparison, and a warning for a non-loopback bind without an API key. |
| I6 | **MCP never touches user files.** The MCP tools generate, list, report status and benchmark; they do not read or write the user's files. |
| I7 | **Templates are sandboxed.** A model's `chat_template` is rendered only inside `ImmutableSandboxedEnvironment`; a template that fails or is blocked is logged and replaced by the built-in format. |
| I8 | **Tests are hermetic.** They use fakes (`FakeEngine`, `FakeOg`), temp directories and `create_server(port=0)`; only `tests/test_prism_gpu_integration.py` uses hardware, and it skips itself. |

## 3. Failure modes (current behavior)

| Situation | Behavior | Where specified |
|---|---|---|
| ONNX model cannot be loaded (e.g. `--device cuda` with a broken CUDA setup) | `500 model_load_failed` | `docs/api.md` Errors |
| Model stays busy longer than `--queue-timeout` (default 300 s) | `503 server_busy`, `Retry-After: 30` | `docs/api.md` Concurrency |
| Ollama unreachable or failing | `502 backend_unavailable` | `docs/api.md` Errors |
| Prompt fills the context window (ONNX, known `context_length`) | `400 context_length_exceeded` | `docs/api.md` Errors |
| Name matches several models | `400 ambiguous_model`, message lists them | `docs/api.md` Errors |
| Non-zero `frequency_penalty` / `presence_penalty` on an ONNX model | `400 unsupported_parameter` | `CHANGELOG.md` |
| An ONNX generation ends in a repeating token cycle | Stopped after a warning, reported as `finish_reason: length`; `PRISM_LOOP_GUARD=off` disables | `CHANGELOG.md` |
| Long prompt on ONNX | Prefill in chunks of `PRISM_PREFILL_CHUNK` (default 1024) to bound GPU memory | `docs/devices.md` |

## 4. Planned behavior (not implemented; `plan.md` Phase 1)

**Parallel use must not exhaust the machine.** Hypothesis, not yet measured: an overflowing VRAM on WDDM spills into host RAM,
and WSL `MemAvailable` cannot see that, so guarding VRAM matters most.

- **P1 Resource budget** (`prism/resources.py`): before an ONNX model is loaded, `check_can_load(model_path, device)` compares free
  RAM (always) and free VRAM (`cuda` only) with the estimated need (weights plus prefill headroom of `PRISM_PREFILL_CHUNK` x about
  1.4 MB per token) and a reserve. It raises `InsufficientResourcesError`, a `ModelLoadError`, whose message gives free versus
  needed memory, which Prism process holds the memory, and how to override. Environment: `PRISM_VRAM_RESERVE_MB` (1536),
  `PRISM_RAM_RESERVE_MB` (2048), `PRISM_RESOURCE_CHECK=off`.
- **P2 Cross-process load lock** (`prism/machine_lock.py`): `flock` on `~/.prism/load.lock`, held only while checking and loading
  (never during generation). A second process waits up to `PRISM_LOAD_TIMEOUT` (120 s) and then fails naming the holder's PID and
  model. The lock vanishes with its holder.
- **P3 Entry point**: `OnnxGenAiEngine._load_model` does `lock; check; load`, so `serve`, `mcp`, `run`, `chat` and `benchmark` are all
  covered. A refusal never calls the library's model loader.
- **P4 HTTP**: `InsufficientResourcesError` becomes `503 insufficient_resources` with `Retry-After`; more than `--max-queue` /
  `$PRISM_MAX_QUEUE` (default 8) waiters get `503 server_busy` immediately.
- **P5 Diagnostics**: `prism status` shows RAM and swap; a load logs the VRAM and RAM delta; `prism doctor` warns, read-only, when
  a Windows `.wslconfig` has no `memory=` or `autoMemoryReclaim=gradual`. Prism never writes `.wslconfig`.
- **Not covered**: Ollama (GGUF) shares the GPU and bypasses the guard; `vram_free_mb()` still sees its usage.

## 5. Development tooling contract

- **Gate** `python3 scripts/sdlc_check.py [--only compile|tests|changelog|docs ...] [--base REF]`: `compile` byte-compiles `prism`,
  `foundry_wsl`, `tests`, `scripts`; `tests` runs the unit suite; `changelog` requires `CHANGELOG.md` to change when runtime code
  (`prism/`, `foundry_wsl/`) changed against `--base` (default `main`); `docs` runs `mkdocs build --strict` and is skipped when
  MkDocs is not installed. Exit `0` unless a check failed. `changed_files` returns paths verbatim (NUL-separated).
- **Red-first** `python3 scripts/sdlc_check.py --red TEST_ID...`: exit `0` only if every named test fails or errors now. A passing,
  skipped, `expectedFailure`, timed-out (60 s) or non-existent id is not red. It prints the exception line of each failure, swallows
  test output, and cannot be combined with `--only` / `--base`.
- **Pre-commit hook** `.githooks/pre-commit` (opt-in: `git config core.hooksPath .githooks`) runs `compile`, `tests` and `changelog`
  from the repository root and blocks the commit when one fails.
- **CI** runs the changelog check on every pull request in addition to tests, packaging and docs.
