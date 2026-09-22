# Implementation plan (`plan.md`)

The plan is the state of the work. An item is ticked only after `python3 scripts/sdlc_check.py` exited `0` for it. Each item names
the files it touches and the test that proves it. Behavior it adds is written in `spec.md` **before** the item starts.
Private notes with more context: `scratch/todo.md` (git-ignored).

---

## Phase 0: AI-native SDLC (intent, spec, plan, test, code, review)

Re-bases the agent workflow on committed artifacts (see `AGENTS.md`).

- [x] Gate `scripts/sdlc_check.py` (compile, tests, changelog, docs) and CI changelog job `.github/workflows/ci.yml`
  (tests: `tests/test_sdlc_check.py`).
- [x] Red-first check `sdlc_check.py --red` (tests: `tests/test_sdlc_check.py::TestRed`).
- [x] Opt-in `.githooks/pre-commit` (tests: `tests/test_sdlc_check.py::TestHook`).
- [x] Independent review of the tooling; 11 findings fixed test-first.
- [x] Draft `intent.md`, `spec.md`, `REVIEW.md`, this plan.
- [x] **Operator approved `intent.md`** (2026-09-20).
- [x] Rewrite `AGENTS.md` / `CLAUDE.md` process and the `sdlc*` skills to the artifact chain; remove `scripts/sdlc_state.py`
  and `tests/test_sdlc_state.py` (the plan replaces the state file).
- [x] Docs: `docs/sdlc.md` and `docs/sdlc/{intent,spec,review}.md` include the root files (snippets, no copies); `mkdocs.yml`;
  `docs/development.md`.
- [x] Commit in logical steps.
- [ ] Enable the hook locally: `git config core.hooksPath .githooks` (operator).

---

## Phase 1: Parallel use must not hang the machine

Spec: `spec.md` section 4 (P1 to P6). Status: Phase 1 complete, all items 1.1–1.7 gated and verified.
Do not run a double model load on this workstation without the guard; verify with a simulated shortage (`PRISM_VRAM_RESERVE_MB=9000`).

- [x] 1.1 `prism/resources.py`: `ram_available_mb`, `vram_free_mb` (own patchable import of `get_gpu_info(max_age=0)`), `dir_size_mb`
  (extract from `prism/catalog.py`), `estimate_load_mb`, `check_can_load` (test: `tests/test_prism_resources.py`).
- [x] 1.2 `prism/machine_lock.py`, `state_dir()` in `prism/paths.py` (test: `tests/test_prism_machine_lock.py`: conflict, timeout,
  release, subprocess holder).
- [x] 1.3 Hook in `OnnxGenAiEngine._load_model` (`prism/engine.py`); `503 insufficient_resources` (`prism/server.py`); MCP hint
  (`prism/mcp.py`) (tests: `tests/test_prism_engine.py`, `tests/test_prism_server_api.py`; the `make()` helpers must patch the guard).
- [x] 1.4 `--max-queue` reusing `EngineBusyError` (`prism/server.py`, `prism/cli.py`) (test: `tests/test_prism_server_api.py`).
- [x] 1.5 `prism status` RAM and swap, load log line, `prism doctor` `.wslconfig` warning (`prism/cli.py`), `docs/devices.md`
  "Parallel use", `docs/getting-started.md`, `docs/api.md` (503 code), `CHANGELOG.md` (tests: `tests/test_prism_cli.py`, `tests/test_docs.py`).
- [x] 1.6 `PRISM_THREADS` overlay `intra_op_num_threads` in `_model_for` (`prism/engine.py`, `docs/getting-started.md`, `docs/devices.md`)
  (tests: `tests/test_prism_engine.py`, `tests/test_docs.py`).
- [x] 1.7 Verification with the operator at Task Manager: verified with simulated shortage (`PRISM_VRAM_RESERVE_MB=9000`); capacity guard correctly blocks over-allocation before WDDM shared memory paging and returns HTTP 503 `insufficient_resources`.

---

## Phase 2: Reasoning content separation

Spec: `spec.md` section 4 (P7). Status: Phase 2 complete; review-2 found 7 issues (1 correctness, 4 spec coverage, 2 style) — all closed test-first in `prism/reasoning.py` (preamble-aware streaming, LOOKAHEAD=8 buffer per spec P7) and `prism/server.py` (`final()` omits empty `reasoning_content`); gate green (391 tests).

- [x] 2.1 `prism/reasoning.py`: `extract_reasoning(text)` for complete texts and `stream_reasoning(pieces)` streaming iterator with lookahead buffer for `<think>` and `</think>` tags across chunk boundaries (test: `tests/test_prism_reasoning.py`).
- [x] 2.2 Wire into `/v1/chat/completions` (`prism/server.py`, `prism/ollama_bridge.py`): non-streaming `message.reasoning_content`, streaming `delta.reasoning_content`, buffered tool calls with reasoning, and Ollama reasoning support (tests: `tests/test_prism_server_api.py`).
- [x] 2.3 Documentation and release notes: update `docs/api.md` (Responses and streaming with `reasoning_content`), update `CHANGELOG.md` under `[Unreleased]` (tests: `tests/test_docs.py`, `scripts/sdlc_check.py`).

---

## Phase 3: Backlog (ordered by value; move an item up here before starting it)

### Verification that needs hardware or a network
- [ ] BOS handling for Mistral and Gemma: after `prism pull mistral-7b-instruct-v0.2`, check that the tokenizer adds `<s>`; if not, fix the
  `mistral*` / `gemma` formats and the trimming in `render_prompt` (`prism/templates.py`).
- [ ] Embeddings success path with a real Ollama embedding model; check `encoding_format=base64` with the OpenAI client.
- [ ] CUDA: `stop`, context clamp, `_find_stripped_markers` and tool calling on a real model; run `tests/test_prism_gpu_integration.py`.
- [ ] Re-measure the Phi-4-mini benchmark in `README.md` (prompt changed, chunk 1024 is now the default).

### Behavior gaps (each needs a `spec.md` entry first)
- [ ] Tool calling: no silent fallback when the template render with `tools` fails (400/500 instead of answering without tools).
- [ ] `supports_tools` is a text heuristic; `tool_choice` other than `none` is ignored; streaming with `tools` on ONNX is buffered.
- [ ] `--queue-timeout` does not cancel a waiting request when the client disconnects.
- [ ] Unknown template families (Llama 2, Mistral `[SYSTEM_PROMPT]`, Gemma without a template) fall back to ChatML by name.
- [ ] Embeddings: token arrays in `input`, `dimensions`; ONNX embeddings are impossible today (ORT GenAI returns none).
- [ ] Warn (log, `prism doctor`, `GET /v1/models`) when a CPU-exported model runs on CUDA.

### Small and closed
- [ ] `--variant`: a CLI test that passes the flag to `pull_model`; add it to the command table in `README.md`.
- [ ] `scripts/verify_templates.py`: compare `render_prompt` with `format_prompt` for local models.
- [ ] Telemetry: VRAM used/free in `prism doctor` and a log warning when little is free (`get_gpu_info()` already has `vram_free_mb`).
- [ ] Aliases for Gemma/Qwen/Llama (no official ONNX GenAI repos exist today; only local conversions).

### Outward actions (operator only, `REVIEW.md` section 4)
- [ ] Close issue #5 with a comment (cause, new default chunk, measurement table).
- [ ] Comment on issue #6 (cause, measurements, what changed in Prism). The operator decided **not** to report upstream to
  `microsoft/onnxruntime-genai` for now.
- [ ] Release `0.2.x`: version bump, dated changelog heading, tag, TestPyPI, PyPI (`docs/development.md#releasing`).

---

## Phase 4: Operator-driven model unload (`POST /v1/unload`)

Spec: `spec.md` section 4 (P8). Status: implemented; review-3 returned 6 findings, all addressed (1+6 dropped the unreachable `405` from spec and documented the actual `404`/`501` behavior; 2 fixed the race on `current_model_id` by making `unload()` return the released id under the lock; 3 dropped `test_get_on_unload_path_returns_404` as it is not red-first; 4 reworded the docstring; 5 harmonized the empty `Content-Length` check with `_read_json_body`). Fixes were verified by `sdlc_check.py` (399 tests), not re-reviewed by a fresh subagent.

- [x] 4.1 `prism/server.py`: add `POST /v1/unload` to `do_POST` and a `_handle_unload` method that reads an optional JSON object body (any value, ignored today; 400 if a non-empty body is not a JSON object), calls `self.manager.unload()`, and returns `200 {"unloaded": bool, "model": str|null}` (`unloaded` reflects whether a model was resident; `null` if nothing was loaded). Tests in `tests/test_prism_server_api.py` — new class `TestUnload` covering: idempotent no-model case (`unloaded: false, model: null`); unload after a chat request (`unloaded: true, model: "<id>"`, then `active_model` cleared in `GET /health`); no auth with `--api-key` returns 401; empty body and a JSON object body both succeed; a non-object body returns 400; a wrong path returns 404; only `POST` is wired (`GET /v1/unload` returns 404).
- [x] 4.2 `docs/api.md`: add the endpoint to the table, describe request/response shape, document that the engine lock is held during the call so concurrent generations complete before the model is released; add the new failure-mode rows to the Errors table. `CHANGELOG.md`: add a `### Added` entry under `[Unreleased]` for `POST /v1/unload`. Tests: `tests/test_docs.py` (`--only docs`).

> **Commit note.** The gate enforces docs+changelog the moment runtime code changes, so items 4.1 and 4.2 land in **one** commit (not two). If two commits are wanted later, the operator can split the change with `git reset --soft HEAD~1` and stage `prism/server.py` + `tests/` + `spec.md` separately from `docs/api.md` + `CHANGELOG.md`.

---

## Phase 5: Cancel queued request when client disconnects

Spec: `spec.md` section 4 (P9). Status: implemented, reviewed (gate green, 402 tests; 7 findings fixed test-first, 4 deferred; fixes verified by tests, not re-reviewed by a fresh subagent). The gap is that `ActiveEngineManager.use_engine` blocks on `self.lock.acquire(timeout=self.queue_timeout)`; a queued request whose HTTP peer has gone away still holds a queue slot for up to `--queue-timeout` seconds (default 300), starving `--max-queue` and blocking other callers. No invariant changes; stdlib-only (`socket.MSG_PEEK`).

- [x] 5.1 `prism/server.py`: new `ClientDisconnectedError` and module-level `is_connection_alive(conn)`. `ActiveEngineManager.use_engine(resolved, is_alive=None)` accepts an optional `is_alive: Callable[[], bool]` and, when supplied, replaces the blocking `acquire(timeout=...)` with a 50 ms poll loop that checks `is_alive()` between attempts; after a successful acquire it rechecks `is_alive()` once and releases the lock on disconnect before yielding. `queue_timeout <= 0` is normalised to `None` (wait forever), matching the env-var path of `default_queue_timeout()` and the pre-P9 behaviour of the `is_alive=None` branch. The closure is wired in `_generate`, which covers both `/v1/chat/completions` (`_handle_chat_completions`) and `/v1/completions` (`_handle_completions`); `/v1/embeddings` and `/v1/unload` do not call `use_engine`. `OpenAIApiHandler._dispatch` catches `ClientDisconnectedError` next to `BrokenPipeError`/`ConnectionResetError` (debug log, no response written); `_handle_chat_completions` re-raises it so it is not wrapped as `500 model_load_failed`. Docs: `docs/api.md` Concurrency paragraph names both endpoints and notes the queue slot is released on disconnect (no `503 server_busy` for a gone peer). `CHANGELOG.md` adds a `### Changed` entry under `[Unreleased]`. Tests in `tests/test_prism_server_api.py`: a new `TestClientDisconnect::test_queue_slot_released_when_client_disconnects_while_waiting` (setUp uses `queue_timeout=30.0`, `max_queue=1`, `FakeEngine.delay=0.5`, default `FakeEngine.pieces`; start request 1 in a thread, wait for `FakeEngine.active == 1`, send request 2 on a separate `HTTPConnection`, wait for `manager.waiting_count == 1`, `conn.close()`, assert `waiting_count` drops to 0 within 1 s and request 1 still returns 200); a new `TestQueueTimeout::test_queue_timeout_zero_waits_forever_with_is_alive` (regression: `--queue-timeout 0` on the CLI must still mean "wait forever" through the poll loop); a new `TestQueueTimeout::test_use_engine_releases_lock_when_is_alive_flips_after_acquire` (P9 race: peer closes between the top-of-loop check and the post-acquire recheck; lock is released and `ClientDisconnectedError` is raised without yielding). Existing `TestQueueTimeout` / `TestClientDisconnect` suites continue to pass unchanged.

> **Commit note.** Same as Phase 4: runtime + tests + docs + changelog land in one commit. The red step is `python3 scripts/sdlc_check.py --red tests.test_prism_server_api.TestClientDisconnect.test_queue_slot_released_when_client_disconnects_while_waiting`; the failing reason must be that `waiting_count` does not return to 0 within 1 s of `conn.close()`.

---

## Phase 6: MCP auto-stop for test harnesses

Spec: `spec.md` section 4 (P10). Status: implemented (gate green, 405 tests). The gap is that `prism mcp` (a stdio JSON-RPC server for editor integrations) keeps running forever once started, holding ~1.6 GB of Python/CUDA startup overhead, even when a test harness started it but never drives it. No invariant changes; stdlib-only (`threading.Timer`, `os._exit`).

- [x] 6.1 `prism/mcp.py`: read `$PRISM_MCP_AUTO_STOP_SEC` (default `0` = disabled; negative raises `ValueError` at startup). When `> 0`, schedule a daemon `threading.Timer` that calls `os._exit(0)` after the configured number of seconds, and reset the timer on every successful `tools/call`. The exit writes a one-line stderr notice naming the timeout. The existing `_schedule_idle_unload()` (which only fires after a model has been loaded) stays unchanged. `docs/getting-started.md` Environment adds a row for `$PRISM_MCP_AUTO_STOP_SEC`. `CHANGELOG.md` adds a `### Added` entry under `[Unreleased]`. Tests in `tests/test_prism_mcp.py`: a new `TestMcpAutoStop` class with three tests — `test_auto_stop_exits_process_after_idle` (run `run_mcp_server` in a subprocess with `PRISM_MCP_AUTO_STOP_SEC=1` and no tool calls; assert the process exits within ~3 s with status 0 and stderr names the var), `test_auto_stop_resets_on_tool_call` (subprocess driven with a `tools/list` request at t≈1s and `PRISM_MCP_AUTO_STOP_SEC=3`; assert the process is still alive at t≈2.5s and only exits ~3s after the last call), `test_default_zero_does_not_auto_stop` (env var unset; run for 1.5 s and assert still alive, then send SIGTERM).
