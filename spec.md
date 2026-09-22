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
