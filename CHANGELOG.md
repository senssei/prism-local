# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [SemVer](https://semver.org/) (pre-1.0: minor
versions may include breaking changes).

## [0.1.0] - Unreleased

First public version.

### Added
- `prism` CLI: `status`, `doctor`, `list`, `pull`, `run`, `chat`, `serve`, `benchmark`, `mcp`, `connect`.
- Engines: ONNX Runtime GenAI (CUDA or CPU) and Ollama (GGUF), behind one OpenAI-compatible server.
- `--device auto|cuda|cpu` / `$PRISM_DEVICE`; the device actually used is reported by `benchmark`, `/health` and responses.
- `prism doctor` checks whether the ONNX Runtime CUDA provider can actually load and names any missing library.
- Per-model chat templates (Phi, ChatML/Qwen, Llama 3, DeepSeek).
- Server: API key (`--api-key`), CORS allowlist (`--cors-origin`), Host-header check, 10 MB body cap, OpenAI-shaped JSON
  errors, streaming with role delta and `finish_reason`, tokenizer-based `usage`, client-disconnect cancellation.
- Cursor, Cline and MCP connectors.
- `pyproject.toml` with a `prism` console script and `cuda`, `pull`, `dev`, `docs` extras.
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
