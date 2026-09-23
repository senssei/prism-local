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
| Client disconnects while waiting for the engine (queued behind another request) | No HTTP response is attempted (the peer is gone); the request is removed from the queue immediately and the engine lock is left for the next caller; logged at debug | `docs/api.md` Concurrency |
| `prism mcp` reaches `$PRISM_MCP_AUTO_STOP_SEC` seconds without a tool call | Process exits with status 0 and a one-line stderr notice; intended for test harnesses that start MCP and want it to disappear after the run | `docs/getting-started.md` Environment |

## 4. Planned behavior

### Phase 2: Reasoning content separation (`plan.md` Phase 2)

- **P7 Reasoning content separation**:
  Models emitting `<think>...</think>` blocks (such as Qwen 2.5/3 with thinking, DeepSeek-R1) have their thoughts separated from final answer text for `/v1/chat/completions`.
  - **Non-streaming**: `choices[0].message.reasoning_content` carries the thinking text (stripped of `<think>` and `</think>` tags). `choices[0].message.content` carries the remaining answer text. If `<think>` is unclosed at the end of generation (e.g. truncated by `max_tokens`), the thinking text goes to `reasoning_content` and `content` is `""`. If no `<think>` block is generated, `reasoning_content` is omitted.
  - **Streaming**: Tokens generated inside `<think>...</think>` are emitted as `delta: {"reasoning_content": piece}`. Neither opening nor closing tags are emitted in the stream. Subsequent tokens are emitted as `delta: {"content": piece}`. A lookahead buffer (up to 8 characters) cleanly handles tags split across chunk boundaries without leaking tag fragments.
  - **Ollama**: If Ollama emits `message.thinking` or `<think>` tags, it is handled identically, delivering `reasoning_content`.
  - **Legacy text completions (`/v1/completions`)**: Unaffected; returns raw text.
  - **Tool calling**: If a reasoning model emits `<think>...</think>` before a tool call, `reasoning_content` is extracted and provided alongside `tool_calls`.

### Phase 4: Operator-driven model unload (`plan.md` Phase 4)

- **P8 Operator unload** (`POST /v1/unload`): drops the ONNX model currently held by `ActiveEngineManager` and frees its VRAM/RAM, so an external tool (e.g. a benchmark script) can swap to a different large model without restarting `prism serve`.
  - **Auth**: requires the API key (default behavior of `_dispatch` for any path not listed as `public`); without `--api-key`, every caller is on loopback and the endpoint is open there.
  - **Request body**: optional. With `Content-Length: 0` (or absent) the body is treated as `{}`; any JSON object is accepted and ignored (reserved for future options such as a timeout). A non-empty body that is not a JSON object returns `400`.
  - **Response**: `200 {"unloaded": bool, "model": string|null}`. `unloaded` is `true` if a model was resident and is now released, `false` otherwise (idempotent: repeated calls succeed and report `false` once nothing is loaded). `model` is the id of the released model, or `null` when none was held.
  - **Concurrency**: `manager.unload()` acquires the engine lock, so if a generation is in flight the unload waits for it to finish. The HTTP call is therefore synchronous and may take seconds; a request that wants a fast yes/no should check `GET /health` (`active_model`) before calling.
  - **Scope**: only ONNX models go through `ActiveEngineManager`; Ollama models are not loaded into the server process and are unaffected. Loading another model after `unload()` re-runs the resource budget check (Phase 1, P1).
  - **Failure modes**: `400 invalid_request_error` for a non-empty, non-object body or for an invalid `Content-Length` header; `401 invalid_api_key` when the bearer token does not match `--api-key`; `404 not_found` for any other path. Non-`POST` methods inherit the stdlib behavior: `GET /v1/unload` returns `404` (the path is unknown to `do_GET`'s route map), other methods return `501 Unsupported` from `BaseHTTPRequestHandler`. No new state on the server, no new environment variable.

### Phase 5: Cancel queued request when client disconnects (`plan.md` Phase 5)

- **P9 Queue cancellation on client disconnect**: A request that is queued behind another in `ActiveEngineManager.use_engine` no longer blocks until `--queue-timeout` when its HTTP connection has already been closed by the peer.
  - **Trigger**: a chat/completions POST whose thread is parked in `ActiveEngineManager.use_engine` waiting for `self.lock`. Today, the wait is a single blocking `acquire(timeout=queue_timeout)`, so the queue slot (`waiting_count`) is held for up to `--queue-timeout` seconds (default 300) after disconnect; with `--max-queue 1` this starves the next caller and reserves a slot for a peer that will never read the response.
  - **Behavior**: `ActiveEngineManager.use_engine(resolved, is_alive=None)` gains an optional `is_alive: Callable[[], bool]`. When `None` the manager behaves exactly as today. When supplied, the manager poll-iterates with a 50 ms tick: each iteration checks `is_alive()` first (short-circuit: if the peer is gone, raise), then tries `self.lock.acquire(timeout=tick)`. If the lock is acquired but the peer has gone in the meantime (between the `is_alive` check and the acquire returning), the manager releases the lock and raises — the slot is freed before any model work. The exception is `prism.server.ClientDisconnectedError`; it is caught by `OpenAIApiHandler._dispatch` next to `BrokenPipeError` and `ConnectionResetError` and logged at debug; no HTTP response is written because the peer is gone.
  - **Detection (`is_connection_alive`)**: a module-level helper `is_connection_alive(conn) -> bool` that temporarily sets the socket non-blocking and calls `conn.recv(1, socket.MSG_PEEK)`. Empty bytes mean the peer performed an orderly shutdown (FIN observed). `BlockingIOError` means alive with no data pending. Any other `OSError` (`ConnectionResetError`, `BrokenPipeError`, etc.) means dead. The socket's previous blocking state is restored before returning. The handler passes a closure `lambda: is_connection_alive(self.connection)` into `use_engine`.
  - **Scope**: the `is_alive` closure is wired in `_generate`, which is the shared body of both `/v1/chat/completions` (`_handle_chat_completions`) and `/v1/completions` (`_handle_completions`). `/v1/embeddings` and `/v1/unload` do not call `use_engine`. Ollama backends do not enter `ActiveEngineManager` and are unaffected. Mid-generation disconnects remain covered by the SSE write catching `BrokenPipeError`/`ConnectionResetError` in `_generate` (see `TestClientDisconnect::test_generation_stops_when_client_goes_away`) — this change does not touch that path.
  - **`queue_timeout == 0`**: the poll loop normalises `self.queue_timeout <= 0` to `None` ("wait forever"), matching the env-var path of `default_queue_timeout()` (which maps `0` to `None`) and the pre-P9 behaviour of the `is_alive=None` branch (`acquire(timeout=-1)`). A CLI `--queue-timeout 0` therefore still means "wait forever"; only an `is_alive` peer that is gone short-circuits the wait.

### Phase 6: MCP auto-stop for test harnesses (`plan.md` Phase 6)

- **P10 MCP auto-stop** (`$PRISM_MCP_AUTO_STOP_SEC`, default `0` = disabled): the stdio MCP server exits with status 0 after the configured number of seconds with no `tools/call`. The timer is reset on every successful tool call (sliding window). Intended for test harnesses that start `prism mcp` to satisfy editor integrations but never drive it; without this knob the process lingers and holds its Python/CUDA startup overhead (~1.6 GB VRAM passive) for the rest of the session.
  - **Semantics**: a positive float sets the idle window. `0` or unset = no auto-stop (current behaviour). Negative = rejected at startup (`ValueError`). The exit goes through `os._exit(0)` so no further requests on stdin are processed; a one-line stderr notice names the timeout and the env var so the cause is obvious in a test log.
  - **Independence**: this is a separate timer from the existing `_schedule_idle_unload()` (which only fires after a model has been loaded into the MCP process — i.e. when MCP is serving tool calls itself, not going through `prism serve`). The auto-stop timer fires regardless of whether a model was ever loaded, and covers the case the existing timer misses (MCP idle since startup).
  - **Scope**: `prism.cli mcp` only; no effect on `prism serve`, `prism chat`, `prism benchmark` or any other subcommand. No HTTP endpoint. No CLI flag (env var only) so the subprocess invocation stays minimal in test harnesses.
  - **Failure modes**: there is no recoverable error path — when the timer fires, the process exits. A request already in flight when the timer fires still gets its response (the timer uses a daemon thread and does not interrupt `handle_tool_call`). A `tools/call` after the timer has fired is impossible because stdin is closed by `os._exit`.
  - **Configuration**: `$PRISM_MCP_AUTO_STOP_SEC` (positive float, seconds). No CLI flag, no entry in `docs/cli.md` for `mcp`; documented in `docs/getting-started.md` Environment.
  - **Failure modes**: no new HTTP status. The peer is gone, so there is no response to write. Disconnect during the *lock-acquire* poll is silent (debug log). Disconnect detected *after* acquire but before generation starts releases the lock and is silent. Disconnect detected *during* generation still surfaces as a `BrokenPipeError` on the next `wfile.write` and is caught as today. The existing `503 server_busy` row is unchanged: a peer that is still connected but waited too long still gets the same `503` with `Retry-After: 30`.
  - **Configuration**: no new flag, no new environment variable. `--queue-timeout` keeps its meaning for callers that are connected; `--max-queue` keeps its meaning for callers that are connected.

### Phase 7: No silent tool-drop on chat-template render failure (`plan.md` Phase 7)

- **P11 No silent tool-drop on render failure**: when a chat completion request carries `tools` and the model's Jinja
  chat template raises during `render_chat_template(text, messages, tools=tools)` (e.g. the template references a
  field the caller's tool definitions do not provide), Prism no longer falls back to the built-in `format_prompt`,
  which has no `tools` argument and would silently hand the model a prompt without the tool definitions. The failure
  now surfaces as `400 invalid_request_error` with `error.code = "template_render_failed"` and a `message` naming
  the model id and the underlying jinja/template exception.
  - **Why**: a silent fallback that drops a request's `tools` argument is a silent change to what the model sees
    (`intent.md` §3.3 calls a silent fallback a bug; the same principle applies here — the caller asked for tools
    and got none without warning).
  - **Scope**: ONNX backend only; Ollama handles tools itself and does not call `render_prompt` for tool placement.
  - **No-tools case**: unchanged. When `tools is None`, the existing "log and fall back to the built-in format"
    path stays (invariant I7 still applies — a template that fails on the messages alone is harmless, the built-in
    format renders the same conversation).
  - **Where the check fires**: `_dispatch` already turns `ApiError` into the JSON error body, so the natural place
    to convert the new exception is the single call to `render_prompt(resolved, messages, tools)` in `_generate`
    (`prism/server.py`). Other callers of `render_prompt` (`cli.py`, `chat.py`, `mcp.py`, `benchmark.py`) do not
    pass `tools`, so the no-tools path is unchanged for them.
  - **Failure modes**: `400 template_render_failed` with `error.message` naming the model id and the underlying
    jinja/template exception text. No `5xx`; the failure is a client-side issue (the template does not accept the
    provided tool definitions), not a server fault. The engine lock is released by `ActiveEngineManager.use_engine`
    on the way out, so no slot is held.
  - **Docs**: `docs/api.md` Errors table gets a row for `template_render_failed`. `CHANGELOG.md` `[Unreleased]`
    gets a `### Changed` entry.

### Phase 8: Drain endpoint and structured holder detail (`plan.md` Phase 8)

- **P12 Drain the server and structured holder info**:
  - **`POST /v1/drain`** asks the running `prism serve` to finish its current request (the engine lock waits for the
    in-flight generation), unload the ONNX model, and exit with status 0. The HTTP response is sent first; the
    process exits through `os._exit(0)` on a daemon thread ~250 ms later so the response can be flushed. The
    response body is `{"drained": bool, "model": str|null, "exit_in_ms": int}`. Idempotent (no model → `drained:
    false, model: null, exit_in_ms: 250`).
    - **Why**: a benchmarking or evaluation tool that needs a different model on the same VRAM cannot ask a
      running `prism serve` to step aside today. `POST /v1/unload` releases the model but keeps the server alive,
      so the bench would still hit the same process on the next request. `POST /v1/drain` lets the orchestrator
      replace the server (e.g. start its own `prism serve` with a large model, run the eval, restart the original
      if needed). Stdlib-only (`os._exit`, `threading.Timer`).
    - **Auth**: same as `/v1/unload` (default: loopback open; `--api-key` makes it `Bearer`-required).
    - **Request body**: optional. `Content-Length: 0` (or absent) → `{}`. Any JSON object is accepted and ignored
      today (reserved for a future `timeout_ms`). A non-empty body that is not a JSON object → `400`.
    - **Failure modes**: `400 invalid_request_error` for a non-empty non-object body or invalid
      `Content-Length`; `401 invalid_api_key` when the bearer does not match `--api-key`; `404 not_found` for any
      other path. Non-`POST` methods inherit the stdlib behaviour (`GET /v1/drain` → `404`, other → `501`). No
      new state on the server, no new environment variable, no new CLI flag.
    - **Interaction with the engine queue**: a drain waits for the current `use_engine` context to exit
      (`manager.unload()` already holds the engine lock; the next request that arrives in the 250 ms window
      before `os._exit(0)` enters `_generate`, sees `manager.engine is None`, and starts a fresh model load
      that is interrupted mid-flight when `os._exit(0)` fires. The dropped request sees a TCP reset or a partial
      response. The 250 ms window is sized for a TCP close handshake, not for new requests — clients must
      treat the drain reply as the last response they will see from this server).
    - **Docs**: `docs/api.md` adds the route to the routing table and the endpoint to the Errors / Endpoints
      table. `docs/devices.md` "Parallel use" gets a short paragraph on the orchestrator pattern. `CHANGELOG.md`
      `[Unreleased]` gets a `### Added` entry.
  - **`error.holder` on `503 insufficient_resources`**: when the model load lock is held by another process, the
    `503` JSON now carries a structured `error.holder = {"pid": int, "model": str}` in addition to the existing
    prose in `error.message` (which stays — logs and humans still parse it). The shape matches the on-disk lock
    metadata (`prism/machine_lock.py::_read_holder_info`). When no holder is recorded, `error.holder` is omitted
    (not `null`), so the orchestrator's `if "holder" in error:` is the right check. Same auth, same `Retry-After:
    30`, same `code: "insufficient_resources"`. No new files; the change is in `prism/server.py`
    (`ApiError.__init__` accepts `holder: Optional[Dict[str, Any]]`, `_send_api_error` includes it when set; the
    `InsufficientResourcesError` handler reads `_read_holder_info()` and passes the dict). `docs/api.md` Errors
    table row for `503 insufficient_resources` documents the new field. `CHANGELOG.md` `[Unreleased]` gets a
    `### Changed` entry.

### Phase 9: Graceful shutdown during in-flight SSE responses (`plan.md` Phase 9)

- **P13 Graceful shutdown while a streamed response is in flight**:
  - When `prism serve` is interrupted (Ctrl+C / SIGTERM with default handler; `POST /v1/drain` is its own path,
    covered by P12), the server prints `Shutting down server...`, waits for the in-flight generation under the
    engine lock (`manager.unload()`), and exits `0` once `with server:` closes the listening socket.
  - `OpenAIApiHandler._begin_sse`, `_sse`, `_sse_error`, the streaming loops in `_generate` / `_generate_ollama`,
    and the JSON helper `_send_json` (plus the CORS preflight `do_OPTIONS`) must not raise an unhandled exception
    when the HTTP connection is closed by shutdown underneath them:
    - `send_response`, `send_header`, `end_headers` and `wfile.write` calls sit inside
      `try/except (OSError, ValueError)` (the implementation is a single helper,
      `OpenAIApiHandler._safe_write`). Hits are logged at debug, the response is dropped, no synthetic `5xx` is
      written (headers may already be gone or the next write would fail anyway). `OSError` covers
      `BrokenPipeError`, `ConnectionResetError`, and `EBADF`; `ValueError` covers the `BufferedWriter` "I/O
      operation on closed file." case that is not an `OSError` subclass.
    - The streaming loops' existing `(BrokenPipeError, ConnectionResetError)` clauses widen to the same
      `(OSError, ValueError)` tuple. `_sse` re-raises `ConnectionResetError` when its write was swallowed so the
      streaming loops' existing catch aborts generation as before — the loop does not run the engine to completion
      with no reader.
    - `_dispatch`'s `(BrokenPipeError, ConnectionResetError, ClientDisconnectedError)` catch is **deliberately
      not** widened to four classes: a handler may legitimately raise `OSError` / `ValueError` from its own
      logic (file paths, int parsing, …) and those must reach the `except Exception` branch, not be silently
      treated as "client disconnected". I/O failures the response writers raise are pre-empted at the point of
      the failing write.
  - Failure modes:
    - Client disconnects between dispatch and `_begin_sse` → no response written, debug log only.
      (Invariant I3 still holds: model work and the first Ollama chunk finish before headers go out.)
    - Streaming response interrupted by shutdown → SSE write fails inside `_safe_write`, debug-logged; `_sse`
      re-raises `ConnectionResetError`; the streaming loop's outer `except (OSError, ValueError)` debug-logs,
      closes the engine generator in `finally`, and the handler thread exits cleanly. The process exits `0`
      once `manager.unload()` returns and the lock is released.
    - Non-streaming response raced by shutdown → same swallow at the `_safe_write` site, debug log only.
  - Why: today only `BrokenPipeError`/`ConnectionResetError` are caught inside the streaming loops; `_begin_sse`,
    `_send_json`, and `_sse_error` are unguarded. Writes that hit a closed-by-shutdown file raise
    `OSError([Errno 9] Bad file descriptor)` or `ValueError("I/O operation on closed file.")`, the traceback prints
    at the same time as `Shutting down server...`, and the operator reads it as a crash even though
    `daemon_threads = True` keeps the process alive.
  - Stdlib-only. No new env var, no new CLI flag, no new endpoint.
  - No invariant changes (I1–I8 still hold).
  - Docs: `docs/api.md` Concurrency paragraph gains one sentence naming the shutdown behaviour and the `0` exit
    code. `CHANGELOG.md` `[Unreleased]` gets a `### Fixed` entry.

### Phase 10: Honor `tool_choice` on the agent path (`plan.md` Phase 10)

- **P14 `tool_choice` is honored; `supports_tools` is a structural check**:
  - Today, `OpenAIApiHandler._tools_param` only special-cases `tool_choice == "none"`
    (`prism/server.py`); every other value, including `"required"`, the structured
    `{"type": "function", "function": {"name": ...}}`, and the OpenAI default
    `"auto"`, is silently treated as `"auto"` — the full tools list goes to the
    engine even when the caller explicitly named one tool. That makes
    `tool_choice: {"type": "function", "function": {"name": "get_weather"}}` a
    silent fallback on every model that would have picked a different tool, exactly
    the failure mode `intent.md §3.3` calls "a silent fallback is a bug" on the
    agent path.
  - Behaviors:
    - `tool_choice == "none"` → tools list returns `None` (unchanged).
    - `tool_choice == "required"` → tools list returned in full; the engine's own
      tool-aware template is the only way to "force" a call on the open-source
      models Prism targets, and that is what today's rendering already does. The
      docstring makes this contract explicit instead of hiding the auto-fallback
      behind a verbose internal name.
    - `tool_choice == "auto"` (or absent) → tools returned in full (unchanged).
    - `tool_choice == {"type": "function", "function": {"name": "X"}}` →
      `_tools_param` narrows the list to the tool whose name matches `X`. The
      model only sees that one tool definition, which is what OpenAI promises.
    - Any other shape → `400 invalid_request_error` with a message naming the
      expected shapes (`"none" | "auto" | "required" | {"type": "function", ...}`).
  - `supports_tools` is no longer a `re.search(r"\btools\b", text)` text scan
    (`prism/templates.py`). It is replaced with a structural check that only
    matches the `tools` symbol when it appears inside a Jinja expression or block:
    `\{\{[^}]*\btools\b[^}]*\}\}|\{\%[^%]*\btools\b[^%]*%\}`. The first branch is
    the `{{ … tools … }}` expression form; the second covers every Jinja
    block that uses `tools` (`{% if tools %}`, `{% for t in tools %}`,
    `{% for tools in x %}`, `{% set t = tools %}`, `{% if not tools %}` — the
    only requirement is that the block contains the symbol). A `\btools\b` word
    boundary excludes `tool_registry`, `tools_dict`, etc. Jinja comment blocks
    (`{# … #}`) are excluded because their delimiters are `{# … #}`, not
    `{% … %}`. Templates whose chat_template file is missing the keyword in any
    Jinja block return `False`, so the existing `400 tools_not_supported` path
    is reached on the same models as today — and True results now correspond to
    genuine template support, not prose.
  - No invariant changes (I1–I8 still hold; I1 strengthens: the new heuristic is
    a structural match, not a textual one, so a model whose template documents
    "tools" without rendering them does not falsely claim support).
  - Stdlib-only. No new env var, no new CLI flag, no new endpoint.
  - No engine changes; the contract is honoured in the HTTP layer.
  - Docs: `docs/api.md` "Tool calling" gains a paragraph listing exactly which
    `tool_choice` values Prism honours and which it 400s. `CHANGELOG.md`
    `[Unreleased]` gets a `### Changed` entry.

### Phase 11: Central env-var schema, stdlib-only (`plan.md` Phase 11)

- **P15 Centralised, validated, type-safe env config (no runtime deps)**:
  - Today, 22 `PRISM_*` env vars are read with `os.environ.get("X", default)`
    scattered across `prism/cli.py`, `prism/resources.py`, `prism/templates.py`,
    `prism/mcp.py`, and `prism/server.py` (31 callsites in total). Defaults are
    defined alongside the readers, type coercion is hand-rolled (`float(...)`,
    `raw.strip().lower() not in ("off","0","false","no")`, `or "auto"`), and a
    typo'd var (`PRISM_DEVIC=...`) is caught only at first use, silently.
  - Add `prism/env_config.py` — stdlib only (`dataclasses`, `os`, `pathlib`,
    `logging`, `typing`, `re`; `configparser` optional for the `.env` reader if a
    fragment form is later chosen). Single frozen dataclass `EnvConfig` declaring
    every `PRISM_*` field with the right type, default, and a `__post_init__`
    validator (e.g. `PRISM_PREFILL_CHUNK >= 0`, `PRISM_DEVICE in
    {"auto","cpu","cuda"}`, `PRISM_TEMPLATE` in `TEMPLATE_MODES`).
  - `EnvConfig.from_env(environ: Optional[Mapping[str, str]] = None) -> EnvConfig`
    builds an instance from `os.environ` (or the supplied mapping for tests).
    `EnvConfig.from_env_file(path)` parses a hand-rolled `KEY=VALUE` reader
    (whitespace, `# comment`, blank lines, optional double-quoted value) and
    merges it under `environ` so the process env still overrides on conflicts.
  - A module-level `load_config() -> EnvConfig` runs at CLI/server start. Fails
    fast: a malformed value (e.g. `PRISM_PREFILL_CHUNK=abc`) raises one
    `ValueError` listing every bad field, not a deferred `TypeError` at first use.
  - `.env` file loading is opt-in only. No auto-discovery in CWD; the only way
    to load one is `PRISM_ENV_FILE=/path/to/.env` in the process env (security:
    never load a file the operator did not name — prevents a third-party `cd`
    from silently swapping API keys).
  - Unknown vars in the process env that begin with `PRISM_` but are NOT in the
    dataclass are logged at WARN at startup (silent-typo guard, not an error;
    keeps backward compatibility with `PRISM_*` vars a shell export may have
    brought along).
  - This phase migrates `prism/cli.py` only (the largest user, 5 callsites).
    The other four modules keep their existing `os.environ.get` reads until a
    follow-up phase migrates them one at a time.
  - No invariant changes (I1–I8 still hold; I4 strengthens: the schema proves
    "no runtime dependencies" by listing exactly what is read and how it's
    coerced, with every reader in one file).
  - Stdlib-only. No `kev`, no `pydantic-settings`, no `python-dotenv`.
  - Docs: `docs/getting-started.md` Environment section gains a one-row-per-var
    table (var, type, default, valid range) and a short "Loading from a file"
    subsection that shows the `PRISM_ENV_FILE=/path/to/.env` opt-in flow with a
    5-line `.env` example. `CHANGELOG.md [Unreleased]` `### Added` (the loader)
    plus `### Changed` (the `prism/cli.py` migration).

### Phase 12: ACP agent, first milestone — handshake and a plain streamed prompt turn (`plan.md` Phase 12)

- **P16 `prism acp`**: a new stdio JSON-RPC 2.0 subcommand, `prism/acp.py`, structurally a sibling of `prism/mcp.py` (same
  `for line in sys.stdin` shape) but speaking the [Agent Client Protocol](https://agentclientprotocol.com) instead of MCP.
  This phase covers only a plain, tool-free chat session; file/terminal mediation is Phase 13/14 (not yet planned — see
  "Not yet planned" below and `intent.md` §4.5).
  - **Concurrency, the one real departure from `mcp.py`**: MCP's loop is fully synchronous — one request, one blocking
    handler, one response. ACP cannot be: a `session/prompt` may stream for tens of seconds, and `session/cancel` must be
    able to interrupt it while the main loop is still free to read other stdin lines. So `run_acp_server()`'s stdin loop
    dispatches `session/prompt` onto its own daemon thread (one per in-flight prompt) and keeps reading; all other methods
    (`initialize`, `session/new`, `session/cancel`) are handled inline on the main thread since they never block. A single
    `threading.Lock` (`_stdout_lock`) guards every `sys.stdout.write` so a streamed `session/update` notification from the
    worker thread can never interleave with another response mid-line.
  - **`initialize`** (client→agent, request): params `{"protocolVersion": int, "clientCapabilities": {...}}`. Response:
    `{"protocolVersion": <same or the highest Prism supports, whichever is lower>, "agentCapabilities": {"loadSession":
    false, "promptCapabilities": {"image": false, "audio": false, "embeddedContext": false}}, "authMethods": []}`.
    `loadSession: false` and no auth methods are permanent for this milestone (intent.md non-goal §4.5: no session
    persistence). No auth is required — same reasoning as the MCP server: loopback, single-user (I5). A `protocolVersion`
    that is not a positive integer (a bool — `True`/`False` are `int` subclasses in Python and must not silently round-trip
    as JSON `true`/`false` — a negative number, a float, or a string) is treated as absent: the response falls back to
    `ACP_PROTOCOL_VERSION` rather than propagating the malformed value.
  - **`session/new`** (client→agent, request): params `{"cwd": str, "mcpServers": [...]}`. `mcpServers` is accepted and
    ignored (Prism does not host or proxy MCP servers inside its ACP agent — non-goal §4.5). Response: `{"sessionId": str}`
    (a `uuid4` hex string). Session state (`_Session`: `sessionId`, `messages: List[dict]`, `model: str`,
    `cancel_event: threading.Event`, `busy: bool`) lives in an in-process dict; it does not survive process restart.
    `model` is resolved once, at `session/new` time, via a new shared helper `prism.catalog.pick_default_model(all_models)`
    (extracted from the picking logic duplicated today only in `mcp.py::call_prism_server`: prefer a CUDA ONNX model, else
    the first available model), overridable by `$PRISM_ACP_MODEL` (an unset or empty value keeps the auto-pick). If the
    catalog is completely empty (no ONNX or Ollama models installed at all) and `$PRISM_ACP_MODEL` is unset,
    `prism.catalog.DEFAULT_FALLBACK_MODEL_ID` (`"Phi-4-mini-instruct-cuda-gpu"`, the same literal `mcp.py::call_prism_server`
    has always fallen back to) is used; `session/new` still succeeds in this case, and the absence of any real model only
    surfaces on the first `session/prompt`, as a normal generation error (see below) — there is no separate "no models
    installed" failure mode at `session/new` time.
  - **`session/prompt`** (client→agent, request): params `{"sessionId": str, "prompt": [{"type": "text", "text": str}, ...]}`.
    Any content block whose `"type"` is not `"text"` (e.g. `"image"`, `"resource"`, `"resource_link"` — all legal in the
    protocol but not yet supported here) makes the whole request fail immediately with `-32602 invalid_params` naming the
    unsupported type; nothing is sent to the model. A second `session/prompt` for a session whose previous prompt has not
    yet resolved (`busy is True`) also fails immediately with `-32602 invalid_params` ("a prompt is already in progress for
    this session"); the first milestone is one in-flight prompt per session.
    - The text blocks are joined and appended to `messages` as a `{"role": "user", ...}` turn (same message-list shape as
      `/v1/chat/completions`).
    - Generation prefers `$PRISM_BASE_URL/chat/completions` with `"stream": true` (same default URL and `$PRISM_API_KEY`
      auth as `mcp.py`); it parses the SSE stream (`data: {...}` lines, terminated by `data: [DONE]`) and, for every chunk
      whose `choices[0].delta.content` is non-empty, sends a `session/update` **notification** (no `id`, so no response is
      expected): `{"sessionId": ..., "update": {"sessionUpdate": "agent_message_chunk", "content": {"type": "text",
      "text": <piece>}}}`. `reasoning_content` deltas (P7) are forwarded the same way as `"agent_thought_chunk"` updates,
      not `"agent_message_chunk"` — an ACP client that renders thoughts and answers differently gets that distinction for
      free from Prism's existing reasoning-content separation.
    - **Server unreachable** (`urllib.error.URLError`, matching `mcp.py`'s existing fallback condition): falls back to
      `prism.engine.OnnxGenAiEngine.stream_generate` on the in-process `ActiveEngineManager` (same fallback the MCP server
      already uses, same caveat that this bypasses the HTTP server's cross-request serialization — pre-existing behavior,
      not new to ACP).
    - **Cancellation**: `cancel_event` is checked between chunks in both the SSE-reading loop and the direct-engine
      fallback loop. On seeing it set, the loop stops reading (closing the HTTP connection or the generator), and the
      worker thread resolves the still-pending `session/prompt` request with `result: {"stopReason": "cancelled"}` instead
      of `"end_turn"`. If generation had already finished naturally in the race, the `"end_turn"` response — sent first —
      wins, and a `session/cancel` that arrives after is a silent no-op (idempotent; ACP does not require an
      acknowledgement for `session/cancel`, it is a notification). A cancel that arrives before any content has been
      generated still appends `{"role": "assistant", "content": ""}` to `session.messages` (rather than nothing at
      all) — an empty turn, not a missing one, so strict user/assistant alternation (I7) holds for the next prompt.
    - **Errors** (backend unavailable, resource limits, template render failure, context length exceeded — the same
      failure modes `docs/api.md` Errors table lists for `/v1/chat/completions`) resolve the `session/prompt` request with
      a JSON-RPC error object, `code: -32000`, `message` naming the underlying Prism error (HTTP status code and body, or
      the direct-engine exception text) — not a silently truncated answer. Matches the "no silent fallback" reasoning
      behind I1 and P11. On any such error, the user turn appended at the start of this prompt is popped back off
      `session.messages` before the error is sent, so the session's history has no unanswered, dangling user turn — the
      next `session/prompt` on the same session starts from the last *complete* exchange, not from two consecutive user
      turns (which would break strict alternating-role chat templates, I7).
  - **`session/cancel`** (client→agent, **notification**, no `id`, no response): params `{"sessionId": str}`. Sets
    `cancel_event` for that session if it exists and has a prompt in flight; unknown or idle session ids are ignored
    (idempotent, matches `/v1/unload`'s idempotence style).
    - **Known limitation — cancel is session-scoped, not turn-scoped.** ACP's `session/cancel` carries only a `sessionId`,
      no id correlating it to a specific `session/prompt` call. If a client sends `session/cancel` for a turn that has
      already resolved (the client raced its own next `session/prompt` ahead of receiving that resolution, or sent a
      cancel after already reading the response), and a new prompt on the same session is already in flight by the time
      the stale cancel is processed, the stale cancel takes effect against that *new* prompt instead of being a no-op.
      This is only reachable if the client itself sends a new `session/prompt` before it has received the previous one's
      `stopReason` — an ACP client that waits for each turn to resolve before starting the next (the expected pattern)
      never hits it. Fixing this fully would need a per-turn correlation id, which ACP's `session/cancel` does not carry;
      not addressed in this milestone.
  - **Unknown methods** (including `session/load`, any `fs/*` or `terminal/*` call arriving from a client that assumes
    Phase 13/14 capabilities Prism has not advertised) get MCP's existing pattern: `-32601 Method not found` when the
    request carries an `id`; a notification with no `id` and an unknown method is silently dropped.
  - **Malformed-but-JSON-valid input never crashes the process, and is classified by an explicit contract, not by
    exception type.** `AcpRequestError(code, message)` is raised only at points that have explicitly validated the
    incoming shape: a non-dict `params` for any of the four methods, a `sessionId` that is not a string
    (`_require_session_id`), or a `prompt` that is not a list of objects with a `"text"` type (`_extract_prompt_text`).
    `_dispatch` catches `AcpRequestError` first and sends its carried `code` (always `-32602`) verbatim. A *second,
    separate* `except Exception` catches everything else and always reports `-32603` — this is deliberately not the
    same code path: an earlier revision classified by exception *type* (`AttributeError`/`TypeError`/`IndexError`/
    `KeyError` → `-32602`, else `-32603`), which is unsound, because a genuine internal bug (e.g. a catalog entry whose
    `device` field is `None` instead of a string) can raise the exact same `AttributeError` a malformed request raises,
    and would have been mislabeled `-32602 Invalid params` for a perfectly well-formed request. A request that is not
    even a JSON object, or a notification with no `id`, is silently absorbed with no response. This applies to the
    synchronous handlers (`initialize`, `session/new`, `session/cancel`, and the prompt-extraction half of
    `session/prompt` that runs before its worker thread is spawned); the worker thread itself (`_run_prompt`) already
    has its own `try/except Exception` (see the Errors bullet above), so a fault there was already isolated to that one
    `session/prompt` request.
  - **A pathologically nested (but syntactically valid) JSON line must not crash the process either.** `json.loads` on
    a deeply nested structure (thousands of nested `[`) raises `RecursionError`, not `json.JSONDecodeError` — a third
    crash vector distinct from a malformed *shape* (caught by `_dispatch`, above) because it happens before `_dispatch`
    is ever reached. `run_acp_server`'s per-line body (`_handle_line`) therefore wraps both `json.loads` and the
    `_dispatch` call in one `try/except Exception`, logging and dropping the line instead of propagating.
  - **A broken stdout never crashes the process either.** `_write` (the single choke point every response and
    notification goes through) wraps its `sys.stdout.write`/`flush` in `try/except (BrokenPipeError, OSError,
    ValueError)`, logging to stderr and dropping the line instead of raising — mirrors `prism/server.py`'s `_safe_write`
    (spec P13). Without this, an editor that closed its end of the pipe (crashed, or tore the ACP session down) would
    take down `run_acp_server`'s stdin loop on the very next response — including the `-32602`/`-32603` error responses
    `_dispatch`'s own catch-all sends, which would otherwise raise from inside that catch-all and propagate out
    uncaught. The stderr log line itself is wrapped in its own `try/except Exception: pass` — if stderr is *also*
    broken (both streams torn down together), there is nothing left to report to, but the write still must not raise.
  - **The lazy `_catalog()`/`_engine_manager()` singletons are safe under concurrent daemon threads.** Both are guarded
    by one module-level `threading.RLock` (`_state_lock`), not a plain `Lock`: `_engine_manager()` calls `_catalog()`
    while already holding `_state_lock`, so a non-reentrant lock would self-deadlock the very first time it runs. Two
    sessions can independently hit the direct-engine fallback on their own daemon threads at the same time (e.g. both
    see `prism serve` unreachable); without the lock, both could pass the "is it `None`" check before either finished
    constructing, producing two `ActiveEngineManager`s — i.e. two engine locks, defeating I2's "one resident model, one
    lock" and reaching exactly the concurrent-model-load scenario `AGENTS.md` calls out as having hung the reference
    Windows host.
  - **A late failure while reporting a result must not discard a completed answer.** `_run_prompt` tracks whether the
    assistant turn was actually appended to `session.messages` (a local `answered` flag) before deciding whether to pop
    the dangling user turn on an exception. Only `answered is False` pops — if the failure happens *after* the answer
    was appended (e.g. `_send_result` itself raising), the completed answer stays in history and only the response to
    the client is lost, rather than silently discarding the model's actual answer and leaving the same one dangling
    unanswered user turn the pop was introduced to prevent, just via a different trigger.
  - **`prism connect acp`**: mirrors `prism connect mcp --target ...` — prints (and with `--write`, merges into) a Zed
    `settings.json` `agent_servers` entry pointing at `prism acp` (no network config needed; ACP is stdio-launched by the
    editor, like MCP). `docs/integrations.md` gets an "ACP (Zed)" section next to "MCP server" documenting the table of
    methods above and the connector command.
  - **Not yet planned** (tracked in `plan.md` Phase 3 backlog, gated by `intent.md` non-goal §4.5 until promoted): fs-mediated
    `session/prompt` tool calls (`fs/read_text_file`, `fs/write_text_file`, agent-initiated `session/update` of type
    `"tool_call"` / `"tool_call_update"`, and `session/request_permission` before a write) and `terminal/*` command
    execution. Constraint §3.7 ("no direct file or process access from the ACP agent") only becomes load-bearing once
    those phases are built; this milestone does no file or process I/O of any kind, so it is vacuously satisfied.
  - No invariant changes (I1–I8 still hold: I2's serialization is unaffected because ACP goes through the same
    `ActiveEngineManager`/HTTP path as every other client; I4 stays stdlib-only — no ACP SDK dependency, hand-rolled
    JSON-RPC like `mcp.py`). Stdlib-only.
  - Docs: `docs/integrations.md` new "ACP (Zed)" section; `docs/cli.md` new `prism acp` and `prism connect acp` entries
    (enforced by `tests/test_docs.py`); `docs/getting-started.md` Environment table gets a row for `$PRISM_ACP_MODEL`.
    `CHANGELOG.md [Unreleased]` gets a `### Added` entry.

### Implemented (Phase 1: Parallel use must not exhaust the machine)

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
- **P6 Thread control** (`PRISM_THREADS`): A positive integer setting intra-op thread count for ONNX models. When set,
  `_model_for` overlays `model.decoder.session_options.intra_op_num_threads`. Unset (default) leaves thread allocation to the engine.
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
