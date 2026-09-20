<div align="center">
  <img src="docs/assets/logo.svg" width="110" height="110" alt="Prism Logo" />
  <h1>Prism</h1>
  <p><b>A multi-engine local AI CLI and OpenAI-compatible server for Linux and WSL2.</b></p>
  <p>
    <a href="https://pypi.org/project/prism-local/"><img src="https://img.shields.io/pypi/v/prism-local.svg" alt="PyPI" /></a>
    <a href="https://github.com/senssei/prism-local/actions/workflows/ci.yml"><img src="https://github.com/senssei/prism-local/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License: Apache-2.0" /></a>
    <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+" />
    <img src="https://img.shields.io/badge/status-alpha-orange.svg" alt="Status: alpha" />
  </p>
</div>

Prism puts **ONNX Runtime GenAI** (CUDA or CPU) and **Ollama / llama.cpp** (GGUF) behind one command line, one
`/v1/chat/completions` endpoint on a fixed port, and ready-made connectors for Cursor, Cline and MCP.

📖 **Documentation:** <https://senssei.github.io/prism-local/>

> **Status: alpha (v0.1.0).** It works, it is tested (~115 tests, no GPU needed), and its defaults are safe
> (loopback-only, no CORS). APIs and flags may still change. See [Known limitations](#known-limitations).

## Why

Prism began as an evaluation of [Microsoft Foundry Local](https://github.com/microsoft/foundry-local) on WSL2
(see the [research notes](docs/research/evaluation-report.md)). In that evaluation the official CLI (`0.10.3`) detected no GPU
under WSL2, ran on the CPU, and served on a random port. Prism keeps the useful part, running ONNX GenAI models locally, and adds:

| | Foundry CLI 0.10.3 (as evaluated) | Prism |
|---|---|---|
| GPU detection on WSL2 | Not detected (WMI-based) | Direct NVML (`libnvidia-ml.so.1`) |
| Engines | One | ONNX Runtime GenAI **and** Ollama (GGUF) |
| Server port | Ephemeral | Fixed, default `127.0.0.1:5272` |
| Execution provider | Not selectable | `--device auto\|cuda\|cpu`, with the device actually used reported |
| Model source | Microsoft catalog | Hugging Face ONNX repos, local folders, Ollama registry |
| IDE / agent integration | None | Cursor, Cline, MCP server |

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install "prism-local[cuda,pull]"   # GPU stack (needs Python 3.11+) + huggingface_hub; both optional
prism doctor                       # checks NVML, ONNX Runtime GenAI, the CUDA provider, Ollama
prism pull phi-4-mini              # downloads to ~/.prism/models
prism run phi-4-mini "Write a Fibonacci function in Python."
prism serve                        # http://127.0.0.1:5272/v1
```

Without a GPU, `pip install prism-local` is enough for CPU and Ollama use. You can also run from a checkout without
installing: `./bin/prism …`. The Python it uses comes from `$PRISM_PYTHON`, then
`./.venv`, then the active virtualenv, then `python3`.

```bash
curl -s http://127.0.0.1:5272/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "phi-4-mini", "stream": true,
  "messages": [{"role": "user", "content": "Count from 1 to 5."}]
}'
```

## Commands

| Command | Purpose |
|---|---|
| `prism status` / `prism doctor` | GPU and environment diagnostics; `doctor` also tests whether the CUDA provider can load |
| `prism list` | Local ONNX models plus installed Ollama models |
| `prism pull <model>` | Hugging Face ONNX (`phi-4-mini`, `owner/repo`) or Ollama (`ollama:qwen2.5-coder:7b`) |
| `prism run <model> [prompt]` | One-shot completion (reads stdin; no prompt starts chat) |
| `prism chat <model>` | Interactive streaming chat |
| `prism serve` | OpenAI-compatible REST server |
| `prism benchmark <model>` | Load time, TTFT, tokens/s, VRAM delta, and the execution provider used |
| `prism mcp` / `prism connect …` | MCP server and Cursor/Cline/MCP client setup |

`run`, `chat`, `serve` and `benchmark` accept `--device auto|cuda|cpu` (or `$PRISM_DEVICE`).
`auto` tries CUDA and falls back to CPU **with a warning that says why**; `cuda` fails instead of falling back.

## Configuration

| Variable | Purpose | Default |
|---|---|---|
| `PRISM_MODEL_DIRS` | `:`-separated model directories; the first is where `pull` writes | `~/.prism/models` |
| `PRISM_DEVICE` | `auto`, `cuda` or `cpu` | `auto` |
| `PRISM_API_KEY` | Bearer token for `serve`; also used by the MCP client | unset (no auth) |
| `PRISM_BASE_URL` | Server URL used by `prism mcp` | `http://localhost:5272/v1` |
| `PRISM_PYTHON` | Interpreter used by `bin/prism` | see above |

Models in the Foundry Local cache (`~/.foundry/cache/models`) are also discovered.

## Security defaults

`prism serve` binds to **127.0.0.1**, sends **no CORS headers**, rejects non-loopback `Host` headers (DNS-rebinding
defence) and caps request bodies at 10 MB. To expose it on a network, set a key:
`prism serve --host 0.0.0.0 --api-key "$(openssl rand -hex 16)"`. Details: [Security](docs/security.md).

## IDE and agent integration

```bash
prism connect cursor --test --export-rules --export-mcp   # Cursor: provider settings, .cursorrules, MCP
prism connect cline --test --export-mcp                   # Cline
prism connect mcp --target claude --write                 # Claude Desktop, Cursor, Antigravity
```

The MCP server exposes `prism_ask_coder`, `prism_code_review`, `prism_list_models`, `prism_get_status` and `prism_benchmark`.
See [Integrations](docs/integrations.md).

## Performance

Speed depends almost entirely on which execution provider runs. Measured with `prism benchmark` on 2026-09-19 (RTX 5070
12 GB, WSL2, driver 615.71, Phi-4-mini INT4, `onnxruntime-genai-cuda 0.16.0`, `onnxruntime-gpu 1.30.0`, CUDA 13 libraries,
two runs each):

| | Decode | Time to first token | VRAM added by the model |
|---|---|---|---|
| **CUDA** | 79–98 tok/s | 0.45–0.50 s | ~4.5 GB (released on unload) |
| CPU | 7–9 tok/s | ~0.6 s | none |

Earlier evaluation figures of 118–130 tok/s and a ~56 ms TTFT (see the
[reproducibility note](docs/research/evaluation-report.md)) were **not reproduced**; the GPU here was shared with about 6.7 GB
of other applications and the prompt differs. Measure your own machine with `prism benchmark <model>`; it prints the provider
it really used, and `prism doctor` shows a CUDA library mismatch.

## Known limitations

- **CUDA needs matching libraries and Python 3.11+.** `pip install "prism-local[cuda]"` installs a matched stack (ONNX Runtime GenAI, ONNX
  Runtime GPU and its CUDA 13 / cuDNN libraries, about 2.5 GB). If you bring your own environment, `prism doctor` names any
  missing library.
- One ONNX model is resident at a time and requests are serialized (a lock), so this is a single-user local server, not a
  high-concurrency one.
- No stop sequences, embeddings, or tool calling on `/v1/chat/completions` yet.
- Chat templates are detected from the model name and `genai_config.json` (Phi, Qwen/ChatML, Llama 3, DeepSeek); unknown
  families fall back to ChatML and may need a template added.
- Linux and WSL2 only.

## Development

```bash
pip install -e ".[dev]"
PYTHONPATH=. python3 -m unittest discover -s tests -v    # ~115 tests, ~6 s, no GPU/network/models needed
pip install -e ".[docs]" && mkdocs serve                  # docs site at http://127.0.0.1:8000
```

See [CONTRIBUTING.md](CONTRIBUTING.md). Repository layout: `prism/` (the product), `tests/`, `docs/`, and
[`foundry_wsl/`](docs/legacy-foundry-wsl.md), the earlier WSL2 bridge toolkit kept for reference.

## License

[Apache-2.0](LICENSE)
