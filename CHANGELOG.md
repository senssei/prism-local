# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [SemVer](https://semver.org/) (pre-1.0: minor
versions may include breaking changes).

## [Unreleased]

### Changed
- `PRISM_PREFILL_CHUNK` now defaults to `1024` tokens (`0` or `off` restores whole-prompt prefill). Without it, one long prompt left most of the GPU memory
  held after the model was unloaded, and the next model loaded then ran about 25 times slower (qwen2.5-coder-7b after a 4200-token Phi-4-mini prompt: TTFT
  83 s and 1.2 tok/s, against 0.34 s and 30 tok/s with the default).

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
