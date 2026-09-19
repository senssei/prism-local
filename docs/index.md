# Prism

**A multi-engine local AI CLI and OpenAI-compatible server for Linux and WSL2.**

Prism puts **ONNX Runtime GenAI** (CUDA or CPU) and **Ollama / llama.cpp** (GGUF) behind one command line and one
`/v1/chat/completions` endpoint on a fixed port, with connectors for Cursor, Cline and the Model Context Protocol.

!!! warning "Alpha software (v0.1.0)"
    It works and is tested, and the defaults are safe (loopback-only, no CORS), but flags and APIs may still change.
    See [known limitations](#known-limitations).

## Try it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install "prism-local[cuda,pull]"
prism doctor                 # is the GPU, ONNX Runtime GenAI and the CUDA provider usable?
prism pull phi-4-mini
prism run phi-4-mini "Write a Fibonacci function in Python."
prism serve                  # OpenAI-compatible API on http://127.0.0.1:5272/v1
```

## What you get

<div class="grid cards" markdown>

- **One endpoint, two engines.** ONNX models and installed Ollama models are served from the same OpenAI-style API. See the [REST API](api.md).
- **You always know where it ran.** `--device auto|cuda|cpu`; the device actually used is printed by `run`, `benchmark` and reported by `/health`. See [Devices & CUDA](devices.md).
- **Safe by default.** Loopback only, no CORS, Host-header check, body cap, optional API key. See [Security](security.md).
- **IDE ready.** `prism connect cursor|cline|mcp` and a built-in MCP server. See [Integrations](integrations.md).

</div>

## Why it exists

Prism started as an evaluation of Microsoft Foundry Local on WSL2. The official CLI (`0.10.3`) did not detect the GPU under
WSL2, ran on the CPU and used random ports. The [research notes](research/index.md) document what was found and how the
findings shaped Prism.

## Known limitations

- **CUDA needs matching libraries and Python 3.11+.** The `cuda` extra installs a matched stack; if you bring your own,
  [`prism doctor`](devices.md#diagnosing-cuda-problems) tells you what is missing.
- One ONNX model is resident at a time and requests are serialized, so this is a single-user local server.
- No stop sequences, embeddings or tool calling on the chat endpoint yet.
- Only three model aliases are curated (`phi-4-mini`, `phi-4`, `phi-3.5-mini`); any other ONNX GenAI repo can be pulled as
  `owner/repo`. See [Models](models.md).
- Linux and WSL2 only.
