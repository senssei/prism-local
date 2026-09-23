# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [SemVer](https://semver.org/) (pre-1.0: minor
versions may include breaking changes).

## [Unreleased]

### Fixed
- `prism acp`'s direct-engine fallback (used when `prism serve` is unreachable) gave an unhelpful raw
  `KeyError`/`AttributeError` message when the session's model was an Ollama model or unresolvable, instead of the
  clear "start the server" message `prism mcp`'s equivalent fallback already gives; it now checks the resolved
  model's backend first. A failure while reading an `HTTPError`'s body (e.g. the connection drops mid-read) is now
  itself guarded — previously it could escape `_run_prompt`'s error handler entirely and silently drop the response
  for that `session/prompt`, hanging the client with no reply at all.
- `prism.mcp`'s lazy `_catalog()`/`_engine_manager()` singletons used a plain `threading.Lock` for
  `_state_lock`, but `_engine_manager()` calls `_catalog()` while already holding it — a
  non-reentrant lock self-deadlocks on that nested acquire, so `prism mcp`'s direct-engine fallback
  (used when `prism serve` is unreachable) would hang forever the first time it ran. Switched to
  `threading.RLock`. Found while reviewing the equivalent, correctly-`RLock`'d pattern added for
  `prism acp` (Phase 12); no test previously exercised the real `_engine_manager()` body, since
  `TestServerlessFallback` patches `mcp._manager` directly.

### Changed
- Spec invariant I9 added: "ACP agent never touches user files or runs commands." Every read or write requested by the model is mediated through the client editor's `fs/*` capabilities.
- `prism.cli` now reads `PRISM_API_KEY` (and, by next phase, the rest of the `PRISM_*` vars) through a typed `EnvConfig` dataclass in `prism/env_config.py`. CLI flags keep precedence over the env. Defaults are unchanged for users; a typo in a var is caught once at startup instead of at first use.
- `tool_choice` is no longer silently rewritten to `"auto"` for every non-`"none"` value. The four accepted shapes (`"none" | "auto" | "required" | {"type": "function", "function": {"name": ...}}`) are honored as OpenAI specifies: the structured form narrows `tools` to the single named tool (a name not in the list is a hard `400 invalid_request_error` before the engine is touched; the unknown shape is also a `400`). The previous behaviour silently passed the full `tools` list to the model even when the caller explicitly named one tool.
- `prism.templates.supports_tools` no longer uses `re.search(r"\btools\b", text)`. The check is now structural: the `tools` symbol must appear inside a Jinja expression (`{{ ... tools ... }}`) or a Jinja block (`{% ... tools ... %}`, covering `{% if tools %}`, `{% for t in tools %}`, `{% set t = tools %}`, etc.). Templates that only mention `tools` in a `{# comment #}` or in prose no longer falsely report support (no False-Positives on neighbouring text); the same models as before reach the `400 tools_not_supported` path because real Jinja usage still matches.
- `PRISM_PREFILL_CHUNK` now defaults to `1024` tokens (`0` or `off` restores whole-prompt prefill). Without it, one long prompt left most of the GPU memory
  held after the model was unloaded, and the next model loaded then ran about 25 times slower (qwen2.5-coder-7b after a 4200-token Phi-4-mini prompt: TTFT
  83 s and 1.2 tok/s, against 0.34 s and 30 tok/s with the default).
- README: a "Prism" badge was added next to the existing version/CI/license badges, linking to the project repository.
- ONNX sampling: `temperature` and `top_p` now take effect. Most `genai_config.json` files say `top_k: 1`, and with that ONNX Runtime GenAI ignores both, so every
  request was greedy (Phi-4-mini at temperature 1.5: 1 distinct output in 4 runs before, 4 in 4 now). When sampling, `top_k` is the model's own value if above 1, else 40.
- Queue slot is released when the HTTP client disconnects while waiting for the engine lock: `ActiveEngineManager.use_engine` now accepts an `is_alive` callback and poll-iterates with a 50 ms tick, so a gone peer no longer holds a slot for the full `--queue-timeout` (default 300 s). The check uses `recv(1, MSG_PEEK)` with the socket temporarily set non-blocking; no new flag, no new environment variable. Disconnects mid-generation are still handled as before.
- Chat-template render failure with tools no longer falls back silently to the built-in format (which would drop the caller's tool definitions). `/v1/chat/completions` now returns `400 template_render_failed` instead, naming the model id and the underlying jinja/template exception text. The no-tools path keeps the silent fallback (I7).
- `503 insufficient_resources` now carries a structured `error.holder = {"pid", "model"}` when the model-load lock is held by another process, alongside the existing prose in `error.message`. Orchestrators (e.g. `benchrig --runtime prism`) can read the field to decide between wait / drain / smaller-model.

### Added
- ACP agent now mediates model tool calls through the editor's `fs/*` capabilities (`read_file`, `write_file`).
- `prism acp`: a stdio server speaking the Agent Client Protocol (ACP), so ACP-capable editors (Zed and others) can run a
  streamed, cancellable chat session against a local, zero-cost Prism model. First milestone: `initialize`, `session/new`,
  `session/prompt` (plain-text content blocks only) and `session/cancel` — no file or terminal access yet. Falls back to an
  in-process ONNX engine when `prism serve` is unreachable, same as `prism mcp`. New `$PRISM_ACP_MODEL` env var and
  `prism connect acp [--write] [--test]` to configure Zed's `agent_servers` settings. A malformed-but-JSON-valid message
  (wrong-typed `params`/`prompt`) returns a JSON-RPC error instead of crashing the stdio loop; a failed generation no
  longer leaves a dangling, unanswered user turn in the session history; `initialize`'s `protocolVersion` rejects
  booleans and negative numbers instead of echoing them back. `prism.catalog.DEFAULT_FALLBACK_MODEL_ID` centralizes the
  hardcoded model id `prism mcp` and `prism acp` fall back to when no local models are installed at all. A broken stdout
  pipe (the editor closed the connection) is logged and dropped instead of crashing the process; a late failure while
  reporting a completed answer no longer discards that answer; an internal fault (e.g. a catalog I/O error) is reported
  as `-32603 Internal error` instead of being mislabeled `-32602 Invalid params` — now via an explicit `AcpRequestError`
  contract rather than guessing from exception type, since a genuine internal bug can raise the same `AttributeError`/
  `TypeError` a malformed request raises. A pathologically deep (but syntactically valid) JSON line no longer crashes
  the process with an uncaught `RecursionError`. The lazy `_catalog()`/`_engine_manager()` singletons are now guarded by
  a shared `threading.RLock`, closing a race where two sessions hitting the direct-engine fallback at once could each
  construct their own engine manager (two engine locks instead of one, risking a concurrent model load). A session's
  lifecycle can now be traced through `logging.getLogger("prism.acp")` (session creation, prompt accept/start/finish,
  the direct-engine fallback trigger, failures, and cancellation), mirroring `prism/server.py`'s existing `logging`
  convention; standard `logging` configuration applies, no new env var or flag. The module's remaining
  operator-facing stderr notices (a dropped response when stdout is gone, a dropped malformed stdin line) now also
  emit the equivalent `logging.warning` alongside the existing `sys.stderr` write.
- `prism.mcp` gets the equivalent `logging.getLogger("prism.mcp")` tracing: `tools/call` dispatch and
  `call_prism_server`'s model resolution, the direct-engine fallback trigger, and its HTTP/internal error paths, at
  `DEBUG`/`WARNING`. Same convention as `prism.acp`; no behavior change.
- Centralised env-var schema (`prism/env_config.py`, stdlib only): one frozen `EnvConfig` dataclass declaring every `PRISM_*` variable with its type, default, and `__post_init__` validator. `EnvConfig.from_env(...)` builds an instance from a Mapping; `EnvConfig.from_env_file(path)` parses a hand-rolled `KEY=VALUE` file (whitespace, `# comment`, double/single-quoted values); `load_config()` runs at CLI / server startup, fails fast on bad values (one `ValueError` listing every problem), and emits one `logging.warning` for any unknown `PRISM_*` key. No runtime dependency — no `kev`, no `pydantic-settings`, no `python-dotenv`. `.env` file loading is opt-in only: `PRISM_ENV_FILE=/path/to/.env` (security: never auto-discover a file the operator did not name).
- Reasoning separation: `prism.reasoning` module extracting `<think>...</think>` blocks into `message.reasoning_content` (and streaming `delta.reasoning_content`) for OpenAI-compatible `/v1/chat/completions` across ONNX and Ollama backends, while keeping raw output intact for legacy `/v1/completions`.
- Thread control: `PRISM_THREADS` environment variable to set intra-op thread count via session_options overlay on ONNX models.
- Diagnostics and telemetry: `prism status` displays system RAM and swap usage; model loads log memory deltas ("VRAM +N MB, RAM +N MB"); `prism doctor` inspects Windows WSL2 `.wslconfig` and warns if `memory=` or `autoMemoryReclaim=gradual` are missing.
- Request queue limit: `--max-queue` CLI flag and `$PRISM_MAX_QUEUE` (default 8) to bound waiting requests for active ONNX model, immediately returning 503 `server_busy` when full.
- Guarded ONNX model loading: `OnnxGenAiEngine._load_model` acquires the machine lock and checks memory headroom before allocating library resources. Returns HTTP 503 `insufficient_resources` (`Retry-After: 30`) on `/v1/chat/completions` and provides actionable hints in MCP tool completions.
- Cross-process model load lock: `prism.machine_lock` module with `flock` on `~/.prism/load.lock` (via `state_dir()` in `prism.paths`) to serialize model loading across processes and prevent concurrent VRAM spikes (timeout configurable via `PRISM_LOAD_TIMEOUT`, `PRISM_LOAD_LOCK=off`).
- Resource budget and capacity guard: `prism.resources` module with `ram_available_mb()`, `vram_free_mb()`, `estimate_load_mb()`, and `check_can_load()` to guard against VRAM and RAM exhaustion before loading models (configurable via `PRISM_RAM_RESERVE_MB`, `PRISM_VRAM_RESERVE_MB`, `PRISM_RESOURCE_CHECK=off`).
- AI-native SDLC for coding agents (Claude Code, Codex, Cursor, ...): `intent.md`, `spec.md`, `plan.md` and `REVIEW.md` as committed artifacts, `AGENTS.md`, `sdlc*` skills in `.agents/skills/`, `scripts/sdlc_check.py` (gate, and `--red` to prove a new test fails first), an opt-in `.githooks/pre-commit`, and a CI changelog check on pull requests. See `docs/sdlc.md`.
- `top_k` and `repetition_penalty` request fields (ONNX and Ollama); `frequency_penalty` / `presence_penalty` go to Ollama, and on ONNX a non-zero value is a `400`
  `unsupported_parameter` instead of being silently ignored.
- Loop guard: an ONNX generation that ends in a repeating token cycle (period up to 64 tokens, over at least 200 tokens and 12 repetitions) is stopped after a warning
  in the log and reported as `finish_reason: length`. `PRISM_LOOP_GUARD=off` disables it.
- `POST /v1/drain`: an orchestrator (e.g. a benchmark that needs a different model on the same GPU) can ask the running `prism serve` to finish its current request, unload the model, and exit with status 0. Reply is `{"drained": bool, "model": str|null, "exit_in_ms": 250}`. Idempotent; same auth as `/v1/unload`. Stdlib-only (`os._exit`, `threading.Timer`).
- `POST /v1/unload`: drops the ONNX model currently held by `prism serve` and frees its VRAM/RAM. The body is optional (any JSON object is accepted and ignored),
  the call is idempotent (`{"unloaded": bool, "model": string|null}`), and it acquires the engine lock so an in-flight generation finishes before the model is released.
  Useful between benchmark runs of different large models, so each one starts from cold VRAM. Requires the API key when `--api-key` is set; without it, the endpoint is open on
  loopback like every other route.
- `PRISM_MCP_AUTO_STOP_SEC` (a positive float, seconds): `prism mcp` exits with status 0 after that many seconds without a `tools/call`; the timer resets on every call. Intended
  for test harnesses that start the MCP process to satisfy editor integrations but never drive it, so the process stops holding its Python/CUDA startup overhead (~1.6 GB VRAM
  passive) after the run. `0` or unset = no auto-stop (current behaviour). One-line stderr message on exit names the env var so the cause is obvious in a test log.

### Fixed
- Tests: WSL config doctor tests in `tests/test_prism_cli.py` now run hermetically on non-WSL Linux environments (`is_wsl_system` honors `PRISM_WSLCONFIG_PATH` and `PRISM_FORCE_WSL=0`).
- ONNX engine dropped every token with id 0 (`!` in Phi-4: "Wow! Great!" came out as "Wow Great") and did not count it, so a reply cut off at `max_tokens` could be
  reported as `finish_reason: stop`.
- Ctrl+C / SIGTERM during an in-flight streamed response used to print a `Traceback` alongside the "Shutting down server…" message
  (`OSError("Bad file descriptor")` or `ValueError("I/O operation on closed file.")` from a `wfile.write` after `with server:` closed
  the connection underneath the daemon handler thread). The SSE / JSON writers — `_begin_sse`, `_sse`, `_sse_error`, `_send_json`,
  `_send_cors_headers`, `do_OPTIONS`, and the streaming loops in `_generate` / `_generate_ollama` — now swallow that family of
  I/O errors at debug; `_sse` re-raises `ConnectionResetError` when its write failed so the streaming loops' existing
  client-disconnect catch aborts generation as before. `POST /v1/drain` (P12) is its own exit path and is unchanged.

## [0.2.0] - 2026-09-20

### Added
- `prism convert MODEL`: converts and quantizes a Hugging Face model (or a local folder) to an ONNX Runtime GenAI folder with onnxruntime-genai's own model builder
  (CUDA or CPU, `int4` or `fp16`) and installs it next to pulled models. Optional `convert` extra; it runs as a subprocess, is verified like a pull, is built in a
  staging folder, and only replaces an existing model with `--force`. `prism doctor` reports whether its requirements are installed.
- `stream_options.include_usage` on streaming requests: a last chunk with empty `choices`, `usage` and the `telemetry` block (device, timings). Streaming
  responses carried neither before, so clients had to estimate token counts.
- `PRISM_PREFILL_CHUNK` (a positive integer of tokens): process the prompt in chunks. ONNX Runtime GenAI's GPU memory otherwise grows by
  about 1.4 MB per prompt token and is not released; on Phi-4-mini with 4500 prompt tokens the peak fell from 11.7 GB to 6.6 GB with a
  chunk of 256; on two other models the peak fell by 54% and 5% while time to first token rose by 110% and 33%, so it stays off by default. See [Devices & CUDA](https://senssei.github.io/prism-local/devices/#gpu-memory-and-long-prompts).

- `stop` on chat and legacy completions (a string or up to 4 strings): text is cut at the first match, held-back partial matches never leak, and
  generation aborts. Previously it was ignored.
- ONNX `max_tokens` is capped to the room left in the model's `context_length` (from `genai_config.json`); a prompt that fills the window is a
  `400 context_length_exceeded` instead of a failure inside the engine.
- Ollama responses carry `usage` and the real `finish_reason` (`length` when it hit `max_tokens`), and honour `stream_options.include_usage`.
- `$OLLAMA_HOST` selects the Ollama daemon.

- `prism pull owner/repo` downloads one model folder from a repo that keeps several side by side (Microsoft's `cuda/…`, `cpu_and_mobile/…`, `directml/…`), chosen by
  `--ep` and `--quant`, or by the new `--variant TEXT`. Before, such a repo was downloaded whole (every variant) and then rejected for lacking a root `genai_config.json`.
  Leftover `.azDownload*` files are skipped.
- Aliases `phi-4-mini-reasoning`, `phi-4-reasoning`, `phi-3-mini-4k` and `mistral-7b-instruct-v0.2`; `scripts/verify_aliases.py` now also requires an `.onnx` file and a tokenizer.
- Chat templates for Gemma and Mistral (v0.1/v0.2 and v0.3+), checked against the real Jinja templates of Gemma 2 and 3 and of Mistral v0.2 (Microsoft's ONNX export) and v0.3.

- `prism serve --queue-timeout SEC` / `$PRISM_QUEUE_TIMEOUT` (default 300, `0` = no limit): a request that cannot get the model in time is answered `503 server_busy` with
  `Retry-After` instead of hanging behind a slow generation.

- Optional `jinja` extra (`pip install "prism-local[jinja]"`): Prism renders the model's own chat template in Jinja's immutable sandbox instead of only recognising its family,
  which reproduces details like Qwen2.5's default system prompt. A leading BOS is dropped when the tokenizer adds it. A template that fails falls back to the built-in
  format with a warning. `$PRISM_TEMPLATE=auto|jinja|builtin`. Without jinja2 nothing changes.

- Tool calling on `/v1/chat/completions`: `tools`, `tool_choice: "none"`, `role: "tool"` messages and assistant `tool_calls`; replies carry OpenAI-shaped `tool_calls` and
  `finish_reason: "tool_calls"`, streaming or not. ONNX models need the `jinja` extra and a chat template that takes `tools` (otherwise `400 tools_not_supported`); their
  calls are recognised in the output in the Qwen/Hermes, Phi-4-mini, Mistral and Llama 3.1 conventions, and streaming with `tools` is buffered. Ollama models pass `tools` to the daemon.
  Checked end to end on Qwen3-0.6B (ONNX, CPU) and Llama 3.1 8B (Ollama).

- `POST /v1/embeddings` (float and base64), served by Ollama; ONNX models answer `400 embeddings_not_supported`.

### Fixed
- ONNX Runtime GenAI decodes special tokens to empty text, which removed the `<tool_call>`, `<|tool_call|>`, `[TOOL_CALLS]` and `<|python_tag|>` markers from the output. The engine now
  restores those (and only those) for models whose tokenizer drops them.
- `finish_reason` is `stop`, not `length`, when the model ends with an end-of-sequence token exactly at `max_tokens` (the EOS ids come from `genai_config.json`).
- The model scan no longer walks into the subfolders of a model folder.
- `prism mcp`, with no server running, loads the model once and keeps it for later tool calls (released after 2 idle minutes) instead of reloading it on every call.

### Changed
- The chat template is taken from the model's own `chat_template` (`chat_template.jinja`, `chat_template.json` or `tokenizer_config.json`) when it has one Prism
  knows, and only otherwise guessed from the name. Checked against the templates of Phi-3.5, Phi-4, Phi-4-mini, Qwen2.5-Coder and Qwen3: Prism's prompt is
  now identical to the rendered Jinja for all five.
- Phi-4 and Phi-4-mini get their own formats (`<|im_start|>…<|im_sep|>`, and `<|role|>…<|end|>` without newlines). Both were sent the Phi-3 format, and Phi-4 (14B)
  is not ChatML-compatible, so its prompts were malformed.
- Message `content` given as a list of parts (`[{"type": "text", ...}]`) is flattened to its text; it used to appear in the prompt as a Python repr, and Ollama
  received the raw list.
- GPU telemetry (NVML) is reused for 2 seconds, so `/health` and `/v1/models` no longer initialise NVML on every call; `prism benchmark` still reads it fresh.
- `GET /v1/models` and `prism list` show the device a model will run on (`CPU` or `CUDA (GPU)`, following `--device` and the hardware) instead of what its
  files were exported for; the latter is `exported_for` in the API. Under `--device auto` a `generic-cpu` model runs on CUDA when a GPU is present.

## [0.1.0] - 2026-09-19

First public version.

### Added
- `prism` CLI: `status`, `doctor`, `list`, `pull`, `run`, `chat`, `serve`, `benchmark`, `mcp`, `connect`.
- Engines: ONNX Runtime GenAI (CUDA or CPU) and Ollama (GGUF), behind one OpenAI-compatible server.
- `--device auto|cuda|cpu` / `$PRISM_DEVICE`; the device actually used is reported by `benchmark`, `/health` and responses.
- `prism doctor` checks (with `ldd`, never `dlopen`, which can crash) that ONNX Runtime's CUDA provider can find all its
  libraries and names any that are missing.
- Per-model chat templates (Phi, ChatML/Qwen, Llama 3, DeepSeek).
- Curated aliases (`phi-4-mini`, ...) resolve to the installed variant that suits the machine (CUDA if a GPU is present),
  so they are not ambiguous when several variants exist.
- Server: API key (`--api-key`), CORS allowlist (`--cors-origin`), Host-header check, 10 MB body cap, OpenAI-shaped JSON
  errors, streaming with role delta and `finish_reason`, tokenizer-based `usage`, client-disconnect cancellation.
- Cursor, Cline and MCP connectors.
- `pyproject.toml` with a `prism` console script and `cuda`, `pull`, `dev`, `docs` extras. The `cuda` extra installs a matched
  stack (`onnxruntime-genai-cuda` and `onnxruntime-gpu[cuda,cudnn]`, CUDA 13; Python 3.11+).
- Model search paths via `$PRISM_MODEL_DIRS` (default `~/.prism/models`).
- Documentation site (MkDocs Material) and CI (Python 3.10-3.12, wheel smoke test).

### Changed
- The server binds to `127.0.0.1` by default (was `0.0.0.0`) and sends no CORS headers by default.
- Inference is serialized behind a lock; concurrent requests queue instead of racing the engine.
- `prism pull` fails when the download lacks `genai_config.json` or weights, and prunes empty nested folders.
- Models are no longer discovered in `./models` or a sibling checkout; `bin/prism` no longer looks for a sibling virtualenv.
  Use `PRISM_MODEL_DIRS` and `PRISM_PYTHON`.
- `fng` and `foundry-ng` launchers are deprecated aliases of `prism`.

### Fixed
- The ONNX engine never selected an execution provider, so models whose `genai_config.json` has an empty provider list ran
  on the CPU while being labelled GPU. Provider selection is now explicit, with a warned fallback in `auto` mode.
- Wrong chat template for every non-Phi model.
- Unsynchronised engine access and unhandled errors that dropped connections.
- `usage.prompt_tokens` was a whitespace word count.
- CUDA library discovery relied on hard-coded paths and on `LD_LIBRARY_PATH` edits that cannot affect the running process.

[Unreleased]: https://github.com/senssei/prism-local/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/senssei/prism-local/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/senssei/prism-local/releases/tag/v0.1.0
