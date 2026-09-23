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

### Future ACP milestones (gated by `intent.md` non-goal §4.5; promote one at a time after Phase 12 ships)
- [ ] Phase 13: fs-mediated tool calls in `session/prompt` — agent-initiated `fs/read_text_file` / `fs/write_text_file`
  requests to the client, `session/update` of type `tool_call` / `tool_call_update`, and `session/request_permission`
  before a write. Needs its own `spec.md` entry before it starts.
- [ ] Phase 14: `terminal/*` command execution mediated by the ACP client, plus the permission-request flow for running a
  command. Needs its own `spec.md` entry before it starts.

### Behavior gaps (each needs a `spec.md` entry first)
- [ ] `supports_tools` heuristic → structural check + `tool_choice={"function": {"name": ...}}` narrows tools (→ Phase 10 / P14, in progress).
- [ ] `--queue-timeout` does not cancel a waiting request when the client disconnects. *(Done in Phase 5 / P9; remove on Phase 3 cleanup.)*
- [ ] Unknown template families (Llama 2, Mistral `[SYSTEM_PROMPT]`, Gemma without a template) fall back to ChatML by name.
- [ ] Embeddings: token arrays in `input`, `dimensions`; ONNX embeddings are impossible today (ORT GenAI returns none).
- [ ] Warn (log, `prism doctor`, `GET /v1/models`) when a CPU-exported model runs on CUDA.
- [ ] Tool calling: no silent fallback when the template render with `tools` fails (400/500 instead of answering without tools). *(Done in Phase 7 / P11; remove on Phase 3 cleanup.)*

### Small and closed
- [x] `prism/mcp.py` `_state_lock` self-deadlock: `_engine_manager()` calls `_catalog()` while already holding
  `_state_lock`; switched `threading.Lock()` to `threading.RLock()` so the nested acquire doesn't hang forever the
  first time `prism mcp`'s direct-engine fallback runs. Found and fixed separately from Phase 12 (which has the same
  pattern in `prism/acp.py`, already using `RLock`) per the operator's explicit request, not bundled into that phase's
  diff. Test: `tests/test_prism_mcp.py::TestLazySingletonDeadlock` (bounded `Thread.join` timeout, proven red before
  the fix — the un-fixed lock hangs the call). Gate green (487 tests).
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

---

## Phase 7: No silent tool-drop on chat-template render failure

Spec: `spec.md` section 4 (P11). Status: implemented, reviewed (gate green, 410 tests; 5 new tests; 3 review findings around the spec/impl/docs drift on the exception-text in the HTTP message — all fixed test-first: tightened `test_template_failure_with_tools_returns_400_template_render_failed` to assert the underlying exception text is in `error.message`, server now passes `str(ex)` into `ApiError.message`, `docs/api.md` row and `CHANGELOG.md` aligned; fixes verified by tests, not re-reviewed by a fresh subagent). Awaiting operator commit.

---

## Phase 8: Drain endpoint and structured holder detail

Spec: `spec.md` section 4 (P12). Status: implemented, reviewed (gate green, 422 tests; 11 new tests; 4 review findings — `error.holder` leaked `time` from lock file, spec described non-existent `400 model_not_loaded`, subprocess exit test was mocked instead of real, dead `with patch` block in 400 test — all fixed test-first: `prism/server.py` projects holder to the documented 2-key shape, `spec.md` "Interaction with the engine queue" now describes the actual behaviour, new `TestDrainProcessExit` boots a real `prism serve` and asserts exit 0, `test_drain_with_non_object_body_returns_400` now asserts `threading.Timer` was not scheduled; fixes verified by tests, not re-reviewed by a fresh subagent). Awaiting operator commit. The gap is that an orchestrator (e.g. `benchrig --runtime prism`) cannot replace a running `prism serve` with a different model on the same GPU: `/v1/unload` (P8) drops the resident ONNX model but the server stays alive and holds the loopback endpoint, so the orchestrator's next request is answered by the same process. A separate `POST /v1/drain` finishes the in-flight generation, releases the model, and exits with status 0 so the orchestrator can start its own `prism serve`. The 503 `insufficient_resources` response today embeds the holder only as prose (`"Held by PID 481017 (loading 'v2')"`); an orchestrator needs a structured field (`error.holder = {pid, model}`) to decide between wait / drain / smaller-model automatically. No invariant changes. Stdlib-only.

- [x] 8.1 `prism/server.py`: (a) extend `ApiError.__init__` with `holder: Optional[Dict[str, Any]] = None` and store it; `_send_api_error` includes `holder` in the JSON only when set. (b) Wire `POST /v1/drain` in `do_POST` → new `_handle_drain` method that reads an optional JSON object body (any value ignored; 400 on non-empty non-object), calls `self.manager.unload()` (atomic — same as `_handle_unload`), and schedules a daemon `threading.Timer(0.25, os._exit, args=[0])` to flush the response. The response body is `{"drained": bool, "model": str|null, "exit_in_ms": 250}`. (c) In `_generate`'s `except InsufficientResourcesError as ex:` branch (already raises `ApiError(503, ..., code="insufficient_resources", ...)`), call `_read_holder_info(state_dir() / "load.lock")` and pass `holder=info` when it carries a `pid`. Tests in `tests/test_prism_server_api.py` (new `TestDrain`, sibling to `TestUnload`): `test_drain_with_no_model_returns_200_and_exit_marker` (no prior request: `200`, `drained: false`, `model: null`, `exit_in_ms: 250`); `test_drain_after_a_request_unloads_and_reports_model` (issue a chat request, assert `current_model_id` set in `/health`, then drain, assert `drained: true`, `model: "<id>"`, `exit_in_ms: 250`, and `GET /health` would now show no active model — note the process exits, so the `GET /health` after `POST /v1/drain` must be sent in a fresh server, or skipped); `test_drain_with_api_key_requires_bearer` (new server with `api_key="s3cret"`, no `Authorization` → `401`, with `Authorization: Bearer s3cret` → `200`); `test_drain_with_empty_body_works` and `test_drain_with_json_object_body_works` (both `200`); `test_drain_with_non_object_body_returns_400` (a JSON array `[]` → `400 invalid_request_error`, message names the path); `test_drain_wrong_path_returns_404` (POST to `/v1/not_drain` → `404`); `test_drain_only_post_is_wired` (GET → `404`); `test_drain_process_actually_exits` (subprocess boots `prism serve --port 0` via a helper, POST `/v1/drain`, assert the subprocess exits within ~3 s with status 0). Tests in `tests/test_prism_server_api.py` (new `TestInsufficientResourcesHolder`, sibling to `TestResources` if it exists; otherwise a fresh class): `test_503_insufficient_resources_carries_error_holder_when_lock_held` — seed `state_dir()/load.lock` with a fake holder JSON `{"pid": 481017, "model": "v2"}` via `_read_holder_info` (or write the lock file directly and patch `_read_holder_info` to read it), then trigger a load that fails the resource check (`FakeEngine.fail_load = True` won't do — use a separate FakeEngine variant that raises `InsufficientResourcesError("` to mirror the production path) and assert `data["error"]["holder"] == {"pid": 481017, "model": "v2"}` and that `data["error"]["message"]` still contains `"Held by PID 481017 (loading 'v2')"`; `test_503_without_holder_omits_error_holder_field` (no lock file: assert `"holder" not in data["error"]`, not `data["error"]["holder"] is None`).
- [x] 8.2 `docs/api.md`: add `/v1/drain` to the routing table (next to `/v1/unload`) with method `POST`, request/response shape, error rows; add a one-line description for `503 insufficient_resources` mentioning `error.holder` when the lock is held; add `POST /v1/drain` to the Errors/Endpoints table. `docs/devices.md` "Parallel use": a short paragraph describing the orchestrator pattern (drain the active `prism serve`, start a fresh one, swap back when done) with a 3-line example. `CHANGELOG.md` `[Unreleased]`: add `### Added` for `POST /v1/drain` and `### Changed` for `error.holder` on `503 insufficient_resources`. Tests: `tests/test_docs.py` (`--only docs`).

> **Commit note.** Same as Phase 4–7: runtime + tests + docs + changelog land in one commit. The red step is `python3 scripts/sdlc_check.py --red tests.test_prism_server_api.TestDrain.test_drain_after_a_request_unloads_and_reports_model`; the failing reason must be that `_handle_drain` does not exist yet (405/404 from the routing table) and `ApiError(holder=...)` is not yet supported. The gap is that `render_prompt(resolved, messages, tools=tools)` (`prism/templates.py`) catches every exception from `render_chat_template` and returns `format_prompt(messages, resolved.get("template"))`, which has no `tools` argument and therefore drops the request's tool definitions without telling the caller. `intent.md` §3.3 calls a silent fallback a bug; the same principle applies here — the caller asked for tools and the model answered without them. No invariant changes; `tools=None` keeps the silent fallback per I7. Stdlib-only.

- [x] 7.1 `prism/templates.py`: new `TemplateToolRenderError(TemplateError)`. In `render_prompt`, when `tools is not None` and the inner `render_chat_template(text, ..., tools=tools)` raises (any exception), raise `TemplateToolRenderError` from the original, with the model id in the message and the original exception as `__cause__`. The no-tools path keeps the existing "log warning + `format_prompt`" fallback (matches I7 and the existing `tests/test_prism_templates.py` cases). `prism/server.py`: in `_generate`, wrap the single `render_prompt(resolved, messages, tools)` call in `try/except TemplateToolRenderError as ex: logger.warning("chat template of %s failed to render with tools: %s", model_id, ex); raise ApiError(400, f"Chat template of '{model_id}' failed to render with the provided tools; Prism does not silently drop tools from the prompt.", code="template_render_failed")`. The wrap is on the line just before `use_engine` is entered, so a failure returns 400 without ever acquiring the engine lock. Tests in `tests/test_prism_templates.py` (new `TestTemplateToolRenderError`): `test_render_prompt_without_tools_still_falls_back_on_template_error` (a template that raises unconditionally: with `tools=None` it still returns the built-in format — guards I7), `test_render_prompt_with_tools_raises_template_tool_render_error` (same template, now `tools=[WEATHER_TOOL]`; assert the new exception is raised and `__cause__` is the original jinja error), `test_render_prompt_with_tools_returns_prompt_when_template_accepts_tools` (the existing `TOOL_TEMPLATE` shape: with tools, returns the rendered string without raising — guards against over-catching). Tests in `tests/test_prism_server_api.py` (new `TestToolTemplateFailure`, sibling to `TestToolsRejected`): `test_template_failure_with_tools_returns_400_template_render_failed` (a model whose jinja template references `tools` so `supports_tools` returns True, but the template errors on a real tool definition; assert `400 template_render_failed`, message names the model id, `FakeEngine.prompts` is empty — the engine was never loaded because `render_prompt` runs before `use_engine` enters), `test_template_failure_without_tools_still_returns_200_via_builtin_fallback` (same model, no tools: assert `200` — the silent fallback still works when the caller did not ask for tools).
- [x] 7.2 `docs/api.md`: in the Errors table, add a row `400 template_render_failed` with a one-line description ("the model's chat template failed to render with the provided `tools`; Prism does not silently drop tools from the prompt") and link to `spec.md` P11. `CHANGELOG.md`: under `[Unreleased]`, add `### Changed` with a single bullet naming the new `400 template_render_failed` and the previous silent behaviour. Tests: `tests/test_docs.py` (`--only docs`).

> **Commit note.** Same as Phase 4–6: runtime + tests + docs + changelog land in one commit. The red step is `python3 scripts/sdlc_check.py --red tests.test_prism_server_api.TestToolTemplateFailure.test_template_failure_with_tools_returns_400_template_render_failed`; the failing reason must be that `render_prompt` returns the built-in format silently (assertEqual on `resp.status == 400` fails with the actual response being `200` and a tool-less prompt in `FakeEngine.prompts`).

---

## Phase 10: `tool_choice` is honored and `supports_tools` is a structural check

Spec: `spec.md` section 4 (P14). Status: implemented (gate green: 434 tests; both items ticked). Items 10.1 and 10.2 land as one commit, matching Phases 4–9. The gap is that `_tools_param` (`prism/server.py`) only special-cases `tool_choice == "none"` today; every other value (including the OpenAI-shaped `{"type":"function","function":{"name":"X"}}`) is silently treated as `"auto"` and the full tools list is sent to the engine even when the caller explicitly named one tool. And `supports_tools` is a `re.search(r"\btools\b", text)` text scan that matches prose as easily as Jinja usages, so a model whose template documents "tools" without rendering them is reported as supporting them. Both are silent-fallback paths on the agent path that intent.md §3.3 calls a bug. `POST /v1/unload` (P8), `POST /v1/drain` (P12), `POST /v1/chat/completions` `400 template_render_failed` (P11), and the `prism serve` Ctrl+C / SIGTERM race (P13) all stay unchanged. No invariant changes (I1–I8 still hold; I1 strengthens for `supports_tools`); stdlib-only.

- [x] 10.1 `prism/server.py` + `prism/templates.py`:
  - (a) `OpenAIApiHandler._tools_param(req)` (`prism/server.py` ~:335) accepts `tool_choice == {"type":"function","function":{"name": "<tool_name>"}}` and narrows the returned list to the single tool whose `name` matches `<tool_name>`. Any other shape (other than `"none"`, `"auto"`, `"required"`, the structured-function form, or absent) becomes a `400 invalid_request_error` whose message names the four accepted shapes. The docstring now describes the four exactly instead of hiding the auto-fallback behind an internal-name. (b) `prism.templates.supports_tools(resolved)` (`prism/templates.py` ~:244) replaces `re.search(r"\btools\b", text)` with a structural check: a regex matching `tools` inside `{{ ... }}` or `{% ... %}` Jinja blocks (`\{\{[^}]*\btools\b[^}]*\}\}|\{\%[^%]*\btools\b[^%]*%\}`), with word boundaries so `tool_registry`, `tools_dict`, etc. don't match. Templates whose `chat_template.jinja` does not use `tools` in those positions return `False`; templates that do return `True`; templates with no chat template still return `False` (unchanged from the builtin-template-mode branch). Stdlib-only.
  Tests in `tests/test_prism_server_api.py` (new `TestToolChoice`, sibling to `TestToolsRejected`):
  - `test_tool_choice_function_narrows_tools_to_named_one` — POST `/v1/chat/completions` with `tools=[get_weather, get_time, send_email]` and `tool_choice={"type":"function","function":{"name":"get_weather"}}`; assert the rendered prompt contains `get_weather;` and does NOT contain `get_time;` or `send_email;`.
  - `test_tool_choice_unknown_function_name_returns_400` — same setup but `tool_choice={"type":"function","function":{"name":"nonexistent"}}`; assert `400`, `data["error"]["type"] == "invalid_request_error"`, the message names the missing tool, and `FakeEngine.prompts` is empty (the engine was never touched — the lock was never acquired).
  - `test_tool_choice_invalid_shape_returns_400` — `tool_choice` of `42`, `"sometimes"`, and `["array"]` are all `400` with `data["error"]["type"] == "invalid_request_error"` and a message naming `"tool_choice"`. Engine untouched.
  - `test_tool_choice_none_still_drops_tools` — regression: `tool_choice="none"` plus non-empty `tools` continues to render without tools (no `TOOLS:` line in `FakeEngine.prompts[-1]`). Guards Phase 7's tests.
  Tests in `tests/test_prism_templates.py` (new `TestSupportsTools`, sibling to the existing template-render tests):
  - `test_true_when_template_uses_tools_in_jinja_expression` — `{{ tools }}`; `True`.
  - `test_true_when_template_uses_tools_in_jinja_if` — `{% if tools %}has{% else %}hasnt{% endif %}`; `True`.
  - `test_true_when_template_uses_tools_in_jinja_for` — `{% for t in tools %}{{ t.name }}{% endfor %}`; `True` (the case the original Phase-9 plan regex missed; the broader `{% ... tools ... %}` half of the new regex catches both `for t in tools` and `for tools in X`).
  - `test_false_when_template_documents_tools_in_prose_only` — `{# tools are optional; the caller may omit them #}` (Jinja comment, no Jinja `{%`/`%}` or `{{`/`}}` block around the symbol); `False` (regression: today's `re.search(r"\btools\b", text)` returns `True` here).
  - `test_false_when_template_uses_a_different_symbol` — `{{ tool_registry }}`; `False` (today returns `True`; new returns `False`).
  - `test_false_when_template_lacks_chat_template_file` — empty folder; `False` (regression: today's behaviour is preserved).

- [x] 10.2 `docs/api.md` + `CHANGELOG.md`:
  - `docs/api.md` "Tool calling" gains a paragraph listing the four accepted `tool_choice` shapes (`"none" | "auto" | "required" | {"type": "function", "function": {"name": ...}}`) and explaining the structural `supports_tools` check.
  - `CHANGELOG.md [Unreleased]` `### Changed` with two bullets: (a) the new `tool_choice={"function":{"name":...}}` narrowing; (b) the structural `supports_tools` check replacing the prose scan.
  - Tests: `python3 scripts/sdlc_check.py --only docs` (mkdocs strict + changelog gate).

> **Commit note.** Same as Phases 4–9: runtime + tests + docs + changelog land in **one** commit. The red step is `python3 scripts/sdlc_check.py --red tests.test_prism_server_api.TestToolChoice.test_tool_choice_function_narrows_tools_to_named_one`; failing reason must be that `_tools_param` returns the full tools list (assertEqual on a one-element list fails because the captured list has all three tools). Then `tests.test_prism_templates.TestSupportsTools.test_false_when_template_documents_tools_in_prose_only` must also fail (today's `supports_tools` returns `True` on `# tools are optional`, the new structural check returns `False`).

---

## Phase 11: Centralised env-var schema (stdlib-only, no runtime dep)

Spec: `spec.md` section 4 (P15). Status: implemented (gate green: 446 tests, all four items ticked). Items 11.1, 11.2, 11.3, 11.4 land in one commit, matching Phases 4–10. The gap was that 22 `PRISM_*` env vars were read with `os.environ.get(...)` in 31 callsites across 5 modules (`prism/cli.py`, `prism/resources.py`, `prism/templates.py`, `prism/mcp.py`, `prism/server.py`); defaults were duplicated next to each reader, type coercion hand-rolled, and a typo'd var caught silently at first use. The fix introduces `prism/env_config.py` — a single frozen `EnvConfig` dataclass that declares every var with its type, default, and validators — without violating invariant I4 ("No runtime dependencies"). The `.env` file flow is opt-in only via `PRISM_ENV_FILE` (no auto-discovery in CWD — security: never load a file the operator did not name). Phase 11 migrates `prism/cli.py` (1 callsite; the plan overestimated 5 — the line at `cli.py:358` is a write, not a read); the other four modules stay as-is until a follow-up phase migrates them one at a time.

- [x] 11.1 `prism/env_config.py` (new file):
  - `@dataclass(frozen=True) class EnvConfig` with one field per `PRISM_*` var, each with the right type, default, and a docstring naming the source field. Concrete fields: `api_key: Optional[str] = None`, `base_url: str = "http://127.0.0.1:5272/v1"`, `device: str = "auto"` (`auto|cpu|cuda`), `template: str = "auto"` (must be in `TEMPLATE_MODES`), `queue_timeout: Optional[float] = None`, `max_queue: Optional[int] = None`, `prefill_chunk: int = 1024` (today `0|off` = whole-prompt prefill), `threads: Optional[int] = None`, `load_timeout: float = DEFAULT_LOAD_TIMEOUT_S`, `load_lock: bool = True`, `ram_reserve_mb: float = DEFAULT_RAM_RESERVE_MB`, `vram_reserve_mb: float = DEFAULT_VRAM_RESERVE_MB`, `resource_check: bool = True`, `loop_guard: bool = True`, `mcp_auto_stop_sec: float = 0.0` (0 = disabled), `force_wsl: bool = False`, `wslconfig_path: Optional[str] = None`, `model_dirs: List[str] = field(default_factory=list)`, `python: Optional[str] = None`, `env_file: Optional[str] = None`, `state_dir_override: Optional[str] = None`.
  - `@classmethod from_env(cls, environ: Optional[Mapping[str, str]] = None) -> EnvConfig` reads each var, coerces (str → int / float / bool / `List[str]`), and constructs. Raises `ValueError` listing every bad field at once (collect errors, don't fail-fast on the first). Stricter semantics than today: `PRISM_DEVICE=ROGUE` raises; `PRISM_PREFILL_CHUNK=abc` raises; empty strings fall back to defaults.
  - `@classmethod from_env_file(cls, path: str, environ: Optional[Mapping[str, str]] = None) -> EnvConfig` parses a hand-rolled `KEY=VALUE` reader (whitespace, `# comment`, blank lines, optional double-quoted value, `\\`-escape inside quotes). Stdlib only — no `python-dotenv`.
  - `__post_init__` validates ranges and enums (e.g. `PRISM_PREFILL_CHUNK >= 0`, `PRISM_TEMPLATE in TEMPLATE_MODES`, `PRISM_MCP_AUTO_STOP_SEC >= 0`, `PRISM_VRAM_RESERVE_MB > 0`, `PRISM_RAM_RESERVE_MB > 0`).
  - Module-level `load_config() -> EnvConfig`: reads `PRISM_ENV_FILE` if set, parses it, layers the values under `os.environ`, then runs `from_env(...)`. Emits one `logging.warning("PRISM_FOO is set but is not a declared env var; ignoring")` per unknown `PRISM_*` key in the merged env.
  - Stdlib-only; no new imports beyond `dataclasses`, `os`, `pathlib`, `logging`, `typing`, `re`.

- [x] 11.2 `tests/test_env_config.py` (new file, hermetic):
  - `test_defaults_match_existing_getter_behaviour` — constructs `EnvConfig.from_env({})` and asserts every field equals the default value the current `os.environ.get` would return for that key.
  - `test_from_env_coerces_typed_fields` — `from_env({"PRISM_DEVICE": "cuda"})` → `EnvConfig(device="cuda")`; `from_env({"PRISM_PREFILL_CHUNK": "1024"})` → `prefill_chunk=1024`; `from_env({"PRISM_MAX_QUEUE": "16"})` → `max_queue=16`; `from_env({"PRISM_LOAD_LOCK": "off"})` → `load_lock=False`; same for `0`, `false`, `no` (matching today's strict set).
  - `test_invalid_value_raises_value_error_naming_the_field` — `from_env({"PRISM_DEVICE": "ROGUE"})` raises `ValueError` whose message names `PRISM_DEVICE` and lists the allowed enum; same for `PRISM_PREFILL_CHUNK=abc`, `PRISM_MCP_AUTO_STOP_SEC=-1.0`, `PRISM_VRAM_RESERVE_MB=0`.
  - `test_quoted_value_in_env_file_preserves_spaces` — write a temp `.env` with `PRISM_DEFAULT_URL="http://localhost:5272/v1 key"` (whitespace inside quotes) and `PRISM_DEVICE='cuda'` (single-quoted); both parse correctly; `# comment` and blank lines are skipped.
  - `test_unknown_prism_var_warns_at_startup` — `load_config()` with `os.environ` patched to include `PRISM_FOO=bar` emits exactly one `logging.warning` whose message contains `PRISM_FOO` and does NOT raise (silent-typo guard).
  - `test_process_env_overrides_env_file` — both `PRISM_ENV_FILE` and the process env set `PRISM_DEVICE`; the process env wins.
  - `test_no_prism_env_file_means_no_file_read` — `load_config()` with no `PRISM_ENV_FILE` does not touch the filesystem (verified via `mock.patch("os.path.exists")`).

- [x] 11.3 `prism/cli.py` migrate (1 callsite — `api_key` only; the plan overestimated 5; `os.environ["PRISM_DEVICE"] = ...` line was a write, not a read):
  - `prism.cli.get_config()` returns a fresh `EnvConfig` per call (no module-load cache — tests patch `os.environ` between `run_cli()` invocations and would otherwise see a stale snapshot).
  - `cmd_serve` reads `_config.api_key` instead of `os.environ.get("PRISM_API_KEY")`. CLI flag `--api-key` keeps precedence (`args.api_key or get_config().api_key`).
  - The remaining 4 modules (server, resources, templates, mcp) keep their existing `os.environ.get` reads; future Phase migrates them one at a time.
  - Stdlib-only; no behaviour change for users whose env is well-formed.

- [x] 11.4 `docs/getting-started.md` + `CHANGELOG.md`:
  - `docs/getting-started.md` Configuration section gains a "Loading settings from a file" subsection that documents `PRISM_ENV_FILE=/path/to/.env` opt-in, the `KEY=VALUE` format with `# comment` and quoted whitespace, and the precedence rules (process env > file; CLI flags > env; unknown `PRISM_*` warns).
  - `CHANGELOG.md [Unreleased]`: `### Added` — the `prism/env_config.py` module with `EnvConfig.from_env()` / `from_env_file()` and the `PRISM_ENV_FILE=/path/to/.env` opt-in (stdlib-only, no runtime dep). `### Changed` — `prism/cli.py` reads `PRISM_API_KEY` through `EnvConfig.from_env()` (defaults unchanged for users; typos are caught at startup instead of at first use).
  - Tests: `python3 scripts/sdlc_check.py --only docs`.

> **Commit note.** Same as Phases 4–10: runtime + tests + docs + changelog land in one commit. The red step is `python3 scripts/sdlc_check.py --red tests.test_env_config.EnvConfigTests.test_unknown_prism_var_warns_at_startup`; the failing reason must be that `load_config()` silently ignores `PRISM_FOO` in `os.environ` (the test sets the env, calls `load_config()`, and asserts exactly one `logging.warning` whose message contains `PRISM_FOO` — fails today because no warning fires).

---

## Phase 9: Graceful shutdown during in-flight SSE responses

Spec: `spec.md` section 4 (P13). Status: implemented, reviewed (gate green: 426 tests; review found 3 MINOR + 6 NIT findings, all fixed test-first: catch tuples simplified to `(OSError, ValueError)` and updated in `spec.md`; test 2 split into a thin integration test (catch-tuple widening) plus a unit test (`test_sse_raises_connection_reset_when_safe_write_returns_none`) that locks in the `_sse` re-raise contract without driving a real streaming request — the prior one-shot patch leaked `FakeEngine.produced` into `TestStopSequences`; `_sse_error` except narrowed; `_dispatch` asymmetry documented with a comment; `/health` follow-up added to test 3; stale docstring in `_sse_error` updated to the new tuple). Verifying the fixes with the tests above; not re-running an independent reviewer in this session. Items 9.1 and 9.2 land as one commit, matching Phases 4–8. The gap is that Ctrl+C or a default-handler SIGTERM during an in-flight streamed response produces an unhandled I/O exception inside the handler thread (`OpenAIApiHandler._begin_sse`, `_sse_error`, or the streaming loops after `with server:` has closed the connection underneath them). The daemon thread does not kill the process, but the traceback surfaces to the operator's terminal at the same time as `Shutting down server...`, making a clean shutdown look like a crash. Today only `(BrokenPipeError, ConnectionResetError)` is caught; `OSError` ("Bad file descriptor") or `ValueError` ("I/O operation on closed file.") from a write after the connection file is closed propagates. `POST /v1/drain` (P12) covers its own path and is unaffected. No invariant changes (I1–I8 still hold); stdlib-only.

- [x] 9.1 `prism/server.py`:
  - (a) Module-level helper `_safe_write(handler, payload: bytes)` that calls `handler.wfile.write(payload); handler.wfile.flush()` inside `try/except (BrokenPipeError, ConnectionResetError, OSError, ValueError)`, logs at debug, returns `None`.
  - (b) `_begin_sse` wraps `send_response(200)`, the CORS header loop, the three explicit `send_header` calls, and `end_headers` in the same `try/except (BrokenPipeError, ConnectionResetError, OSError, ValueError)`. `_sse` and `_sse_error` route through `_safe_write` instead of writing `wfile` directly. The `(BrokenPipeError, ConnectionResetError)` clauses in the streaming loops in `_generate` (~server.py:919) and `_generate_ollama` (~server.py:977) widen to the same four-class tuple with the same debug log.
  - (c) `_send_json` and the CORS-preflight `do_OPTIONS` get the same wrapping for `send_response`/`send_header`/`end_headers`/`wfile.write`, so a non-streaming response raced by shutdown exits cleanly too.
  Stdlib-only; no new imports.
  Tests in `tests/test_prism_server_api.py` (new `TestShutdownRace`, sibling to `TestClientDisconnect`):
  - `test_begin_sse_after_client_close_does_not_raise` — open `/v1/chat/completions` with `stream=True`, close the client socket before the server reaches `_begin_sse`, assert the server logs at debug and `server.shutdown()` still returns normally.
  - `test_sse_write_after_connection_close_is_swallowed` — issue a streaming request, read the first chunk, then call `_safe_write` directly with crafted bytes after closing the underlying `connection`, assert no exception and a debug log line.
  - `test_ctrl_c_during_streaming_request_exits_cleanly` — subprocess boots `prism serve --port 0` via the existing `TestDrainProcessExit._spawn_serve` helper, opens a streaming connection in the parent, reads one chunk, sends `SIGINT` to the child; assert the child exits within ~30 s with status 0 and that stderr contains no `Traceback (most recent call last)`.

- [x] 9.2 `docs/api.md`: in the Concurrency paragraph (next to the existing text on queue cancellation), add one sentence: "Ctrl+C / SIGTERM during an in-flight streamed response finishes the current generation under the engine lock and exits `0`; no client-side connection error surfaces as an unhandled traceback." `CHANGELOG.md` `[Unreleased]` gets a `### Fixed` entry with one bullet naming the shutdown race and that `_begin_sse`, `_sse_error`, the streaming loops, and `_send_json` swallow shutdown-time I/O errors at debug. Tests: `tests/test_docs.py --only docs`.

> **Commit note.** Same as Phases 4–8: runtime + tests + docs + changelog land in one commit. The red step is `python3 scripts/sdlc_check.py --red tests.test_prism_server_api.TestShutdownRace.test_begin_sse_after_client_close_does_not_raise`; the failing reason must be that `_begin_sse` raises `ConnectionResetError`/`OSError` on the closed socket and the test sees an unhandled exception.

---

## Phase 12: ACP agent, first milestone (handshake + a plain streamed prompt turn)

Spec: `spec.md` section 4 (P16). Status: Phase 12 complete and reviewed. Independent review (fresh subagent, scoped to this phase's diff) found 5 findings:
(1) critical — malformed-but-JSON-valid input (wrong-typed `params`/`prompt`) crashed the whole `prism acp` process via an
unhandled exception in the main stdin loop, killing every open session; (2) high — `session/cancel` is session-scoped
with no per-turn correlation id, so a stale cancel can in a narrow race affect the wrong (later) prompt on the same
session; (3) high — a failed generation left a dangling, unanswered user turn in `session.messages`, corrupting the next
prompt's role alternation; (4) medium — the hardcoded model-id fallback for an empty catalog was undocumented and
duplicated as a literal in both `mcp.py` and `acp.py`; (5) low — `initialize`'s `protocolVersion` accepted booleans
(round-tripping as JSON `true`/`false`) and negative numbers unvalidated. (1), (3), (5) fixed test-first (9 new tests in
`tests/test_prism_acp.py`, all proven red before the fix); (4) fixed by adding `prism.catalog.DEFAULT_FALLBACK_MODEL_ID`
as the single source of truth and documenting it in `docs/cli.md` / `docs/integrations.md`; (2) accepted as a documented
limitation per operator decision (spec.md P16 "Known limitation" bullet under `session/cancel`) rather than a code fix,
since ACP's `session/cancel` carries no turn-correlation id and a real fix would need a protocol extension Prism does not
control — only reachable if the client itself sends a new `session/prompt` before receiving the previous one's
`stopReason`. Gate green after fixes (474 tests); fixes verified by tests, not re-reviewed by a second fresh subagent.
Phase 13 (fs-mediated tool calls) and Phase 14 (terminal execution) remain in the Phase 3 backlog, gated by `intent.md`
non-goal §4.5, to be promoted one at a time.

**Review round 3** (fresh subagent, verifying round 2's fixes and doing a new full pass): confirmed round-2 fixes (1),
(4), (5) fully correct and (2) still a sound accepted trade-off, but found round-2 fix (3) (error-code classification)
only partially fixed, plus 2 new issues introduced by round 2's own fixes: (new-1, high) `run_acp_server`'s stdin loop
only caught `json.JSONDecodeError` around `json.loads`, so a syntactically-valid but pathologically deep JSON line
(thousands of nested `[`) raises `RecursionError` instead and crashes the whole process — a third crash vector distinct
from malformed shape (round 1) or a broken pipe (round 2), reachable because it happens before `_dispatch` is ever
called; (new-2, medium) `_write`'s own `except`-block `sys.stderr.write` was unprotected, so a stdout-and-stderr-both-
broken scenario would still crash; (partial-3, medium) round 2's `(AttributeError, TypeError, IndexError, KeyError)` →
`-32602` heuristic in `_dispatch` is an unsound proxy — a genuine internal bug (e.g. a catalog entry with a non-string
`device` field, reproduced via `pick_default_model`) raises the same `AttributeError` a malformed request raises and
was still mislabeled `-32602` for a well-formed request. The reviewer also independently found (new-4, high, not
previously reported in any round) that `_catalog()`/`_engine_manager()` are unguarded lazy singletons with no lock at
all, while `acp.py` is the one file in the codebase that actually calls them from concurrent daemon threads (one per
`session/prompt`) — two sessions hitting the direct-engine fallback at once could each construct their own
`ActiveEngineManager`, i.e. two engine locks, risking the concurrent-model-load scenario AGENTS.md and invariant I2
forbid. All four fixed test-first (9 new/strengthened tests, proven red against the round-2-committed baseline via
`git stash` + `git checkout stash@{0} -- <file>` to isolate old-implementation/new-tests): `_handle_line` wraps both
`json.loads` and `_dispatch` in one `try/except Exception`; `_write`'s stderr fallback is itself wrapped in
`try/except: pass`; `_catalog()`/`_engine_manager()` are now guarded by a shared `threading.RLock` (`_state_lock`) —
`RLock`, not `Lock`, because `_engine_manager()` calls `_catalog()` while already holding the lock, so a plain `Lock`
would self-deadlock (a mistake the new `test_engine_manager_does_not_deadlock_when_it_calls_catalog` test, with a
bounded `Thread.join` timeout, is specifically designed to catch); the error-code split was replaced with an explicit
`AcpRequestError(code, message)` contract — raised only where a shape is actually validated
(`_require_session_id`, `_extract_prompt_text`, the `params`-is-a-dict check in `_dispatch`) — with every other
exception defaulting to `-32603`, instead of guessing from exception type. Gate green after round-3 fixes (486 tests).
Fixes verified by tests only, not re-reviewed by a fourth independent pass.

**Separate, out-of-scope finding reported to the operator (not fixed here):** round 3's reviewer's phrasing about
`acp.py` lacking a lock led to discovering that `prism/mcp.py`'s *existing*, already-committed
`_catalog()`/`_engine_manager()` (same lazy-singleton pattern, predates Phase 12) uses a plain `threading.Lock()` for
`_state_lock`, and `_engine_manager()` calls `_catalog()` while holding it — the exact self-deadlock shape fixed in
`acp.py` above, just with a `Lock` instead of an `RLock`. This means `prism mcp`'s direct-engine fallback path
(`call_prism_server`'s `except urllib.error.URLError` branch) would hang forever the first time it runs, in real usage,
not just under concurrency — `tests/test_prism_mcp.py::TestServerlessFallback` never hits it because it patches
`mcp._manager` directly, bypassing `_engine_manager()`'s body entirely. This is a pre-existing bug from an earlier,
already-reviewed phase, out of scope for the Phase 12 diff (no drive-by fixes per `REVIEW.md` checklist item C) — flagged
for the operator to decide whether to fix as its own small item. **Fixed separately per operator request**: same
`threading.RLock` fix applied to `prism/mcp.py`'s `_state_lock`, with its own regression test
(`tests/test_prism_mcp.py::TestLazySingletonDeadlock`, proven red before the fix) and `CHANGELOG.md` entry, landed as
its own commit outside the Phase 12 diff (`plan.md` "Small and closed" backlog).

**Review round 4** (fresh subagent, full from-scratch pass over the now-fully-committed state — no uncommitted diff was
left to review incrementally): confirmed all of rounds 1-3's fixes are intact, and found no lock-ordering issue between
`_sessions_lock` and `_state_lock`. Found 2 new substantive issues plus 3 low-severity doc/hygiene ones: (1,
medium-high) `_direct_engine_delta_stream` had no backend guard — an Ollama model or an unresolvable model id fell
through to a raw `KeyError`/`AttributeError` deep in `render_prompt`/`use_engine`, reported to the client as an
unhelpful `-32000: "'path'"` instead of the clear "start `prism serve`" message `mcp.py`'s equivalent fallback already
gives for the identical situation — contradicting the docs'/CHANGELOG's own "same as `prism mcp`" parity claim; (2,
medium) the `except urllib.error.HTTPError` handler's `ex.read()` call was itself unguarded — if reading the error
body fails (e.g. the connection drops mid-read), the new exception is not caught by the sibling `except Exception` of
the same try/except (exceptions inside one except clause are never caught by another clause of the same statement),
so it escapes `_run_prompt` entirely on the worker thread and **no response of any kind is ever sent** for that
`session/prompt` — the client hangs forever on that one call. Both fixed test-first (3 new tests, proven red). Also
fixed: (3, low) `docs/architecture.md`'s module table never mentioned `prism/acp.py`; (4, low) `spec.md` cited
invariant I7 ("templates are sandboxed") for message-role alternation, which I7 does not cover — removed the
incorrect citation; (5, low, pre-existing, not caused by Phase 12) `tests/test_prism_mcp.py` had a misplaced
`if __name__ == "__main__":` block partway through the file (before the last test class), so running that one file
directly (`python3 tests/test_prism_mcp.py`) silently skipped `TestMcpAutoStop` — moved to the true end of the file;
(6b, low) `tests/test_prism_catalog.py::TestPickDefaultModel` fabricated a `"device"` key on a synthetic Ollama model
dict that real `list_ollama_models()` never sets — corrected to match production shape. A batch of "`thread.join()`
with no `assertFalse(is_alive())` follow-up" test-style notes across the suite was raised but dropped as a style nit,
per the operator's/reviewer's own read that it's cosmetic, not a defect. Gate green after round-4 fixes (490 tests).
Fixes verified by tests only, not re-reviewed by a fifth independent pass.

**Observability, per operator request ("add logs and traces")**: added `logging.getLogger("prism.acp")` tracing through
the session lifecycle — `session/new` at `INFO` (session id, model, cwd); `session/prompt` accept/reject/start/finish
at `DEBUG`; the direct-engine fallback trigger and any failed `session/prompt` at `WARNING`; `session/cancel` and
`_dispatch`'s per-message routing at `DEBUG` — mirroring `prism/server.py`'s existing `logging` convention. The
operator-facing stderr banner/notices already in the module were left untouched (lower-risk than rewiring already-
reviewed, already-tested error-recovery paths through the logger). 7 new tests using `assertLogs`, proven red by
stashing `prism/acp.py` back to its pre-logging state and confirming all 7 failed with "no logs ... triggered" before
restoring the change. Gate green (497 tests).

**Completed the logging pass, per operator follow-up request ("add rest of logs and traces")**: extended tracing to the
two `prism/acp.py` spots initially left as plain `sys.stderr` writes (`_write`'s dropped-response notice,
`_handle_line`'s dropped-malformed-request notice — both additive, `logging.warning` alongside the existing stderr
write, not a replacement) and to `prism/mcp.py`, which had zero `logging` calls before this: `logging.getLogger
("prism.mcp")` now traces `tools/call` dispatch and `call_prism_server`'s model resolution / direct-engine fallback
trigger / HTTP and internal error paths at `DEBUG`/`WARNING`, mirroring `prism/acp.py`'s convention. 6 new tests (2 for
the acp.py spots, 4 for mcp.py), all proven red via `git stash` before their fixes. Pure observability additions, no
behavior change. Gate green (503 tests).

Awaiting operator decision to commit.

**Review round 2** (fresh subagent, verifying round 1's fixes and doing a new full pass): confirmed (1), (4), (5) fixed
and (2) correctly documented as an accepted trade-off, but found (3) only partially fixed plus 2 new issues introduced by
round 1's own fixes: (new-1, high) `_write` had no protection against a broken stdout pipe, so an I/O failure while
sending *any* response — including the `-32602`/`-32603` error responses `_dispatch`'s own catch-all sends — would
propagate out of `_dispatch` uncaught and kill the whole stdio loop, reintroducing the exact crash class finding (1)
fixed, just via pipe failure instead of malformed JSON; (new-2, medium-high) the round-1 "pop the dangling user turn on
any exception" fix could pop the *assistant* turn instead if the failure happened after a successful generation (e.g.
`_send_result` itself raising), silently discarding a completed answer; (new-3, medium) `_dispatch`'s blanket
`except Exception` mislabeled genuine internal faults (e.g. a catalog I/O error in `_default_model()`) as
`-32602 Invalid params`. All three fixed test-first (5 new tests, all proven red against the round-1 code via
`git stash` before the round-2 fix): `_write` now swallows `(BrokenPipeError, OSError, ValueError)` and logs to stderr
(mirrors `prism/server.py`'s `_safe_write`, spec P13); `_run_prompt` tracks an `answered` flag and only pops when the
assistant turn was never appended; `_dispatch` now distinguishes `(AttributeError, TypeError, IndexError, KeyError)` →
`-32602` from any other exception → `-32603`. Gate green after round-2 fixes (479 tests). Fixes verified by tests only —
not re-reviewed by a third independent pass; the operator can request one if more assurance is wanted before shipping. `intent.md` §2
(outcome), §3.7 (constraint) and §4.5 (non-goal) approved 2026-09-23. This phase is deliberately narrow: a working `prism acp` an ACP-capable editor (Zed) can hold a plain-text
streamed conversation with, and cancel — no file or command access yet (that is Phase 13/14, listed under Phase 3
backlog, gated by non-goal §4.5). No invariant changes; stdlib-only (hand-rolled JSON-RPC over stdio, one
`threading.Lock` for stdout, one daemon thread per in-flight `session/prompt`, matching `prism/mcp.py`'s existing pattern
of a synchronous stdio loop plus daemon timers).

- [x] 12.1 `prism/catalog.py`: extract `pick_default_model(all_models) -> Optional[str]` from the model-picking logic
  duplicated today only inside `mcp.py::call_prism_server` (prefer a CUDA ONNX model, else the first available model).
  `mcp.py` calls the extracted helper instead of its inline logic (no behavior change). Test: `tests/test_prism_catalog.py`
  (new `TestPickDefaultModel`: empty list → `None`; a CUDA and a non-CUDA model present → the CUDA one; no CUDA model →
  the first one) plus the existing `tests/test_prism_mcp.py` fallback tests continuing to pass unchanged (regression
  guard that the extraction did not change `call_prism_server`'s behavior).
- [x] 12.2 `prism/acp.py` (new file): `run_acp_server()` stdio loop; `_Session` dataclass; `handle_initialize`,
  `handle_session_new`, `handle_session_prompt` (spawns the worker thread), `handle_session_cancel`; `_stdout_lock`;
  the SSE-parsing helper for the `$PRISM_BASE_URL` path and the `stream_generate` fallback path, both checking
  `cancel_event` between chunks. Tests in `tests/test_prism_acp.py` (new, hermetic, mirrors `tests/test_prism_mcp.py`'s
  structure): `test_initialize_returns_capabilities`; `test_session_new_returns_session_id_and_resolves_model`;
  `test_session_prompt_streams_agent_message_chunks_then_end_turn` (patch the SSE source with a fake iterable of chunks,
  assert the emitted `session/update` notifications and the final `stopReason: "end_turn"` response, using a
  `list`-backed fake stdout writer instead of real stdio); `test_session_prompt_rejects_non_text_content_block`;
  `test_session_prompt_rejects_concurrent_prompt_on_same_session`; `test_session_cancel_stops_in_flight_prompt_and_reports_cancelled`
  (drives `handle_session_prompt` on a background thread against a fake slow chunk source, calls `handle_session_cancel`
  mid-stream, asserts the response is `stopReason: "cancelled"` and no chunks are emitted after the cancel);
  `test_session_cancel_after_natural_completion_is_a_noop` (race guard: cancel arrives after `"end_turn"` already sent);
  `test_unknown_method_with_id_returns_method_not_found`; `test_unknown_method_without_id_is_dropped`;
  `test_generation_error_resolves_prompt_with_json_rpc_error` (patch the SSE path to raise / return a 503, assert
  `error.code == -32000` and the message names the underlying Prism error).
- [x] 12.3 `prism/cli.py`: `cmd_acp(args)` calling `prism.acp.run_acp_server()`, wired as `subparsers.add_parser("acp", ...)`
  next to `p_mcp`. Test: `tests/test_prism_cli.py::TestAcpCommand` (there was no pre-existing `prism mcp --help` test to
  mirror, as the original wording assumed; instead patches `prism.acp.run_acp_server` and asserts `cmd_acp` calls it and
  exits 0 — the same technique `TestServeArgs` uses for `start_server`).
- [x] 12.4 `prism/connectors.py` + `prism/cli.py`: `prism connect acp [--write] [--test]` prints (and optionally merges
  into) `~/.config/zed/settings.json`'s `agent_servers.Prism` entry for `prism acp`, via a new `_merge_zed_agent_entry`
  helper mirroring `merge_prism_mcp_entry` (validate-then-backup order, safer than the inline antigravity/cursor/claude
  branches in `connect_mcp`). `test_acp_protocol()` runs a live `initialize` + `session/new` handshake against a real
  `prism acp` subprocess, mirroring `test_mcp_protocol()`. Test: `tests/test_prism_connectors.py::TestAcpConnector`
  (write creates the entry; write preserves other top-level keys and other `agent_servers` and backs up first;
  printed-only leaves no file; live handshake test).
- [x] 12.5 Docs and changelog: `docs/integrations.md` "ACP (Zed)" section; `docs/cli.md` entries for `prism acp` and
  `prism connect acp`; `docs/getting-started.md` Environment row for `$PRISM_ACP_MODEL`; `CHANGELOG.md [Unreleased]`
  `### Added`. Tests: `tests/test_docs.py` (`--only docs`).

> **Commit note.** Same shape as Phases 4–11: runtime + tests + docs + changelog land in one commit (12.1 may land
> separately first since it is a pure, behavior-preserving extraction with its own regression test — the operator can
> decide at commit time). The red step is
> `python3 scripts/sdlc_check.py --red tests.test_prism_acp.TestAcp.test_session_prompt_streams_agent_message_chunks_then_end_turn`;
> the failing reason must be `ModuleNotFoundError: No module named 'prism.acp'`.

---

## Phase 13: ACP fs-mediated tool calls — `read_file`, `write_file`

> Status: Phase 13 complete and reviewed. Round 1 independent review found 16 findings (14 fixed test-first with 11 new tests, 1 not a defect, 1 nit dropped). Round 2 independent review found 9 findings (2 CRITICAL, 2 HIGH, 2 MEDIUM, 2 LOW, 1 NIT); all actionable findings fixed test-first (6 new tests in TestPhase13Round2ReviewFindings, proven red before fixing). Gate green: 539 tests passing, CHANGELOG updated, sdlc_check.py exits 0.

> **Goal.** Make `prism acp` advertise two model tools — `read_file(path)` and `write_file(path, content)` — and execute
> them by mediating the ACP client's `fs/read_text_file` and `fs/write_text_file` JSON-RPC methods. The ACP agent
> itself never opens a file or writes to disk; every byte goes through the client editor (intent.md §3.7, new spec
> invariant I9). The model never knows the difference from a normal `role:"tool"` turn.
>
> **Spec entry.** `spec.md` "Phase 13: ACP fs-mediated tool calls — `read_file`, `write_file`" (under "Planned
> behavior"). New invariant I9 "ACP agent never touches user files or runs commands" added to the invariants table.
> Phase 12 entry's "Not yet planned" bullet tightened to drop the Phase 13 parts and reference Phase 13 instead.
>
> **Backlog items closed by this phase.** Phase 3 backlog §3.7 ("no direct file or process access from the ACP
> agent") becomes load-bearing; the Phase 3 backlog entry "fs-mediated `session/prompt` tool calls" moves from
> "not planned" to "Phase 13".

- [x] 13.1 `spec.md`: write the Phase 13 entry under "Planned behavior" (done in this plan item; landing in the
  same commit as 13.2 so the spec describes shipped behavior, not a wish list). Add new invariant I9 to the
  invariants table: "ACP agent never touches user files or runs commands". Tighten Phase 12's "Not yet planned"
  bullet to keep only `terminal/*` (Phase 14) and reference Phase 13 instead. Test: `python3 scripts/sdlc_check.py
  --only compile` and a quick `tests/test_docs.py --only docs` run to confirm the spec still parses; no new test
  because the spec is prose.

- [x] 13.2 `prism/acp.py`: bidirectional JSON-RPC over stdio + `_send_request` helper. Replace the read-only main
  loop with a small frame dispatcher that handles three message kinds: (a) **request** (top-level `id` and
  `method`) → handled by the existing `handle_*` methods, response written to stdout by the new `_send_response(id,
  result_or_error)`; (b) **response** (top-level `id`, no `method`) → look up pending future in `_pending` and
  complete it with the result, or raise `AcpClientError(code, message, data)` on a JSON-RPC `error`; (c)
  **notification** (no `id`, has `method`) → handled by the existing notification paths. New module-level:
  `class AcpClientError(Exception)` carrying `code` and `message`; `_pending: dict[int, tuple[Event, list]]`
  (`list` so callers can stash an `[result, exc]` slot without a closure); `_next_request_id()` returning a
  monotonically increasing int via `itertools.count`; `_send_request(method, params, *, timeout=None) -> dict |
  raises AcpClientError` that parks a future, writes the request frame, and waits with a per-call timeout (None =
  no timeout, controlled by the caller — the tool loop's `cancel_event` is the actual escape hatch). Connection
  drop handler iterates `_pending`, sets `exc=ConnectionError("ACP connection closed")` on each, and removes the
  entry. Tests in `tests/test_prism_acp.py`:
  - `test_send_request_round_trip_writes_request_and_resolves_on_response` — fake stdout + fake stdin reader; send
    `{"method":"_test_echo","params":{"x":1}}`; assert stdout frame is `{"jsonrpc":"2.0","id":1,"method":...,
    "params":...}`; feed the matching response frame; assert `_send_request` returns the result.
  - `test_send_request_raises_AcpClientError_on_jsonrpc_error_response` — feed a response with `error: {code:
    -32600, message: "bad"}`; assert `AcpClientError(code=-32600, message="bad")` is raised.
  - `test_pending_futures_are_cancelled_when_connection_drops` — park two requests, simulate a connection drop,
    assert both raise `ConnectionError` (or `AcpClientError` — pick one and document), and `_pending` is empty.
  - `test_send_request_id_is_monotonic_across_calls` — five calls, assert ids are `1..5`.

- [x] 13.3 `prism/acp.py`: capability negotiation + tool definitions + tool-aware render + tool-call loop +
  `session/request_permission` + capability-mismatch fallback + cancellation guard + max-iterations guard. All in
  one commit because the loop is a single indivisible change. New module-level:
  - `_FS_TOOLS: list[dict]` = the two tool definitions (OpenAI function-calling JSON-schema shape:
    `{"type":"function","function":{"name":"read_file","description":..., "parameters":{...}}}` for both).
  - `MAX_ACP_TOOL_ROUNDS = 8` (module-level constant; no env var in v1).
  - `_class AcpClientError(Exception)` (added in 13.2; reused here).
  `handle_initialize(params)` now also reads `params.get("clientCapabilities", {}).get("fs", {})` and stores
  `{"readTextFile": bool(...), "writeTextFile": bool(...)}` in a connection-scoped dict keyed by session id
  (initialized to defaults at `session/new` if not already present; explicit `initialize` after `session/new` is
  a protocol error and falls through the existing "Method not found"-style reject). `_resolve_tools(session)` returns
  the filtered list (drop `read_file` if `readTextFile` is False, drop `write_file` if `writeTextFile` is False;
  empty list if neither is advertised). New helpers `_request_fs_read(session, path, line=None, limit=None)` and
  `_request_fs_write(session, path, content)` wrap `_send_request` with `method="fs/read_text_file"` /
  `method="fs/write_text_file"` and the right params. New helper `_request_fs_permission(session, call_id, path)`
  sends `session/request_permission` with `toolCall: {toolCallId, title: "Write {path}", kind: "edit"}` and
  `options: [{optionId: "allow_once", name: "Allow once", kind: "allow_once"}, {optionId: "reject_once", name:
  "Reject", kind: "reject_once"}]`; returns one of `"allow"`, `"reject"`, `"cancelled"`. Refactor `_run_prompt`
  into a bounded loop (max `MAX_ACP_TOOL_ROUNDS` iterations): each iteration resolves the session's model via
  `_catalog().resolve_model(session.model)`, checks `prism.templates.supports_tools(resolved)` (skip tools
  otherwise), renders with `_resolve_tools(session)`, runs the existing delta stream, `parse_tool_calls` for
  text + calls. If calls: for each call, generate `toolCallId = f"call_{session.session_id}_{n:04d}"` (per-session
  monotonic counter), emit `session/update` `tool_call` with `kind: "read"` (read_file) or `kind: "edit"`
  (write_file) and `status: "in_progress"`, dispatch:
  - `read_file`: `_request_fs_read(...)` → fold result into a `role:"tool"` message with `tool_call_id` and
    `content=<file text>`. Emit `tool_call_update` with `status:"completed"` and the content block, OR `failed`
    with the error text on `AcpClientError` or non-schema result.
  - `write_file`: `_request_fs_permission(...)` → on `"allow"` run `_request_fs_write(...)` and emit
    `tool_call_update` with `status:"completed"` and content `"wrote {N} bytes to {path}"`; on `"reject"` emit
    `tool_call_update` with `status:"failed"` and content `"permission denied"` (do NOT call
    `fs/write_text_file`); on `"cancelled"` resolve `session/prompt` with `stopReason:"cancelled"` and stop the
    loop (the prompt turn has been cancelled by the client).
  After dispatching all calls, append the resulting `role:"tool"` messages to the session's `messages` and iterate.
  Between iterations and immediately before each `_request_fs_*` / `_request_fs_permission` call, check
  `session.cancel_event.is_set()`; if set, resolve with `stopReason:"cancelled"`. On exceeding
  `MAX_ACP_TOOL_ROUNDS`, emit a final `agent_message_chunk` with `"tool loop budget exhausted ({N} rounds)"` and
  resolve with `stopReason:"end_turn"`. Capability-mismatch fallback: if a tool call arrives for a capability the
  client did not advertise (defense-in-depth — the `_resolve_tools` filter is the primary defense), return a
  `role:"tool"` message of `{"error": "fs.{readTextFile|writeTextFile} is not advertised by the client"}` and
  emit `tool_call_update` with `status:"failed"`. The defense-in-depth path is reachable only if a model emits a
  call despite the tool not being advertised (e.g., model remembers it from a prior turn where the client
  advertised it) — the `tests/test_prism_acp.py::TestAcpToolLoop::test_tool_call_for_non_advertised_capability_returns_error_content`
  test covers it. Tests in `tests/test_prism_acp.py`:
  - `test_initialize_captures_fs_capabilities_into_session` — `initialize` with
    `clientCapabilities.fs = {"readTextFile": True, "writeTextFile": False}`, then `session/new`; assert
    `_resolve_tools(session)` returns the `read_file`-only list.
  - `test_initialize_without_fs_capabilities_filters_tools` — `clientCapabilities = {}` → `_resolve_tools(session)
    == []`.
  - `test_session_prompt_with_read_file_tool_call_executes_via_fs_read_text_file` — model emits
    `<tool_call>{"name":"read_file","arguments":{"path":"/tmp/x"}}</tool_call>`, fake client replies to
    `fs/read_text_file` with `{"content":"hello"}`, model emits a final answer; assert: one `tool_call`
    notification with `kind:"read"` and `status:"in_progress"`, one `tool_call_update` with `status:"completed"`
    and content `"hello"`, the session's `messages` ends with a `role:"tool"` turn with content `"hello"`,
    `session/prompt` resolves with `stopReason:"end_turn"`, and no `_request_fs_write` was issued.
  - `test_session_prompt_with_write_file_requests_permission_then_writes` — model emits `write_file`, fake
    client replies to `session/request_permission` with `allow_once`, then to `fs/write_text_file` with `{}`;
    assert both calls in order, `tool_call_update` `status:"completed"`, and the tool result message has
    content `"wrote N bytes to ..."`.
  - `test_session_prompt_with_write_file_rejected_skips_fs_write` — same setup, client returns `reject_once`;
    assert NO `fs/write_text_file` request, `tool_call_update` `status:"failed"`, content `"permission denied"`,
    and the loop continues (the model can react).
  - `test_session_prompt_with_write_file_cancelled_resolves_session_prompt_with_cancelled` — same setup, client
    returns `cancelled`; assert `session/prompt` resolves with `stopReason:"cancelled"` and the loop stops.
  - `test_session_prompt_fs_read_text_file_error_returns_error_as_tool_content` — fake client returns
    `error: {code: -32600, message: "File not found"}`; assert `tool_call_update` `status:"failed"`, the tool
    result content carries the error text, and the loop continues.
  - `test_session_prompt_tool_loop_respects_max_iterations` — a model that loops `read_file` forever stops
    after `MAX_ACP_TOOL_ROUNDS` iterations and resolves with `end_turn` plus a final `agent_message_chunk`
    carrying `"tool loop budget exhausted"`.
  - `test_session_prompt_cancel_during_tool_loop_stops_cleanly` — start a prompt with a tool call, set
    `cancel_event` mid-execution, assert the prompt resolves with `stopReason:"cancelled"` and no further
    `_send_request` calls.
  - `test_tool_call_id_is_unique_within_session` — a model emits two `read_file` calls in one turn; assert
    two distinct `toolCallId`s.
  - `test_tool_call_notifications_carry_correct_kind` — `read_file` → `kind:"read"`, `write_file` →
    `kind:"edit"`.
  - `TestAcpAgentNeverTouchesFiles::test_agent_never_opens_file_or_runs_subprocess` — patch
    `builtins.open` and `subprocess.*` to record every call from inside the tool-call loop; run a turn that
    produces `read_file` and `write_file` calls; assert neither was called (this is the load-bearing test for
    invariant I9).

- [x] 13.4 `docs/integrations.md` "ACP (Zed)" section gains a "fs-mediated tool calls" subsection listing
  `read_file` and `write_file`, the `session/update` `tool_call` / `tool_call_update` shapes, and the
  `session/request_permission` UX. `CHANGELOG.md [Unreleased]` gets a `### Added` entry:
  `ACP agent now mediates model tool calls through the editor's fs/* capabilities (read_file, write_file)`;
  plus a `### Changed` entry noting the new spec invariant I9. Tests: `tests/test_docs.py --only docs`. No new
  env var row in `docs/getting-started.md` (Phase 13 deliberately does not add `$PRISM_ACP_MAX_TOOL_ROUNDS`; the
  constant is module-level).

> **Commit note.** Phase 13 lands in four commits in this order: 13.1 (spec only, may merge with 13.2 at
> commit time), 13.2 (bidirectional RPC + tests), 13.3 (tool loop + tests — the biggest commit), 13.4 (docs
> + changelog). The red step for 13.2 is
> `python3 scripts/sdlc_check.py --red tests.test_prism_acp.TestAcpRpc.test_send_request_round_trip_writes_request_and_resolves_on_response`
> — failing reason must be `AttributeError: module 'prism.acp' has no attribute '_send_request'`. The red step
> for 13.3 is the `TestAcpToolLoop::test_session_prompt_with_read_file_tool_call_executes_via_fs_read_text_file`
> test. All Phase 13 commits must keep the gate green
> (`python3 scripts/sdlc_check.py` exits 0) before the next one starts; in particular 13.2 must compile and
> pass tests before 13.3 begins.

> **Risk register.** (1) **Bidirectional stdio RPC** (13.2) is the trickiest piece — a wrong dispatch between
> responses and notifications can deadlock the loop. Mitigation: write `_send_request` with a `threading.Event`
> per id and a strict "id present → future; no id → notification; id + no method → response" dispatch in the
> main read loop. Hermetic tests cover the round-trip, error, and connection-drop paths. (2) **Permission flow
> cancellation** — if `session/request_permission` is in flight when `session/cancel` arrives, the cancel must
> be visible to the `_request_fs_permission` waiter. Mitigation: `_request_fs_permission` checks
> `cancel_event.is_set()` before the call AND is wrapped so a `cancelled` outcome short-circuits the turn.
> (3) **Tool-call loop budget** — `MAX_ACP_TOOL_ROUNDS=8` is a soft guard against misbehaving models, not a
> security control. The editor's permission model is the actual gate (invariant I9 + intent.md §3.7). The
> constant is module-level in v1; if the operator wants to tune it, that is a follow-up.

