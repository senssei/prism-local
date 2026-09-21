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

## Phase 2: Backlog (ordered by value; move an item up here before starting it)

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
- [ ] `<think>` blocks of Qwen3 land in `content`; consider `reasoning_content`.

### Small and closed
- [ ] `--variant`: a CLI test that passes the flag to `pull_model`; add it to the command table in `README.md`.
- [ ] `scripts/verify_templates.py`: compare `render_prompt` with `format_prompt` for local models.
- [ ] Telemetry: VRAM used/free in `prism doctor` and a log warning when little is free (`get_gpu_info()` already has `vram_free_mb`).

### Outward actions (operator only, `REVIEW.md` section 4)
- [ ] Close issue #5 with a comment (cause, new default chunk, measurement table).
- [ ] Comment on issue #6 (cause, measurements, what changed in Prism). The operator decided **not** to report upstream to
  `microsoft/onnxruntime-genai` for now.
- [ ] Release `0.2.x`: version bump, dated changelog heading, tag, TestPyPI, PyPI (`docs/development.md#releasing`).
