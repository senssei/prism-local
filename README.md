# 💎 Prism: Multi-Engine Local AI CLI & Inference Server

A high-performance, developer-first local AI CLI and OpenAI-compatible inference server for **Linux & WSL2**.

**Prism** refracts disparate local AI runtimes—**ONNX Runtime GenAI on NVIDIA CUDA** and **Ollama / llama.cpp (GGUF)**—into a single, unified developer experience with sub-millisecond dispatch, interactive streaming chat, automated benchmarking, zero-dependency REST serving, and IDE connectors for **Cursor, Cline, and Model Context Protocol (MCP)**.

---

## 🌟 Why Prism?

Prism was conceived following an in-depth architectural evaluation of **Microsoft Foundry Local** (`microsoft/foundry-local` v0.10.3 and the v2.0 C++ rewrite). Official tools suffer from WSL2 hardware blindness, CPU fallback locks, catalog gating, and ephemeral ports.

Prism replaces fragile single-engine tools with an open, multi-engine local runtime:

| Capability | Official `foundry` CLI (`0.10.3`) | 💎 **Prism** (`prism`) |
|---|---|---|
| **Engine Architecture** | Single-engine locked | 🔀 **Multi-Engine (ONNX Runtime GenAI + Ollama/GGUF)** |
| **GPU Detection on WSL2** | ❌ Fails (`GPU: Not detected` via WMI) | ⚡ **Direct NVML hardware detection (`libnvidia-ml.so.1`)** |
| **Hardware Execution** | ❌ Locks Linux to CPU (`CPUExecutionProvider`) | ⚡ **100% Native CUDA on RTX GPUs (`onnxruntime-genai-cuda`)** |
| **Decode Speed (Phi-4-mini)**| ❌ 9.1 – 15.3 tok/s (CPU) | 🚀 **118.6 – 130.2 tok/s (CUDA)** |
| **Time to First Token (TTFT)**| ❌ 5.0 – 7.7 seconds | 🚀 **50 – 60 milliseconds** |
| **REST Server Port** | ❌ Ephemeral random port (e.g. `:37863`) | 🟢 **Stable, fixed port (default: `:5272`)** |
| **API Standardization** | ⚠️ OpenAI `/v1` on random port | 🟢 Full OpenAI `/v1/chat/completions` + SSE streaming |
| **Model Ecosystem** | ❌ Microsoft catalog with 0 CUDA variants | 🟢 Pulls official ONNX weights from HF & GGUF from Ollama |
| **IDE & Agent Connectors**| ❌ None | 🟢 **Native Cursor, Cline, and MCP connectors** |
| **VRAM Lifecycle** | ❌ Leaks ~99% VRAM on unload (Issue #1079) | 🟢 Complete GPU memory disposal on unload |
| **Compatibility Aliases** | N/A | 🟡 `fng` / `foundry-ng` kept as deprecated aliases of `prism` |

---

## 📦 Installation

Prism has no runtime dependencies; the engines are optional extras.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[cuda,pull]"   # ONNX Runtime GenAI (CUDA) + Hugging Face downloads
prism doctor
```

Or run straight from a checkout with [`./bin/prism`](bin/prism) (no install needed). It picks its Python from `$PRISM_PYTHON`, then `./.venv`, then the active virtualenv, then `python3`.

### Configuration

| Variable | Purpose | Default |
|---|---|---|
| `PRISM_MODEL_DIRS` | `:`-separated directories to scan for ONNX models. The first one is where `prism pull` writes. | `~/.prism/models` |
| `PRISM_PYTHON` | Interpreter used by `bin/prism` | see above |
| `PRISM_API_KEY` | Bearer token for `prism serve`; also sent by the MCP client and connector probe | unset (no auth) |
| `PRISM_BASE_URL` | Server URL used by `prism mcp` | `http://localhost:5272/v1` |

Models are also discovered in the Foundry Local cache (`~/.foundry/cache/models`). CUDA and cuDNN libraries installed as pip `nvidia-*` wheels are found automatically in the active environment's site-packages and preloaded before ONNX Runtime starts.

> **Migrating:** earlier versions scanned `./models` and a sibling `../02-ollama-loadtest` checkout, and `bin/prism` used that checkout's virtualenv. Set `PRISM_MODEL_DIRS` and `PRISM_PYTHON` to keep using them.

## 🚀 Quickstart

The launcher script is [`./bin/prism`](bin/prism). The old names [`./bin/fng`](bin/fng) and [`./bin/foundry-ng`](bin/foundry-ng) still work but are deprecated and print a notice.

```bash
# Add bin to PATH (optional):
export PATH="$(pwd)/bin:$PATH"

# 1. System & GPU Telemetry
prism status

# 2. Environment Doctor Check
prism doctor

# 3. List All Discovered Models (ONNX + Ollama)
prism list

# 4. Run a Prompt Completion (ONNX CUDA GPU)
prism run Phi-4-mini-instruct-cuda-gpu "Write a Python function to compute Fibonacci numbers."

# 5. Run a Prompt Completion (Ollama GGUF)
prism run ollama:qwen2.5-coder:7b "Explain quicksort in two sentences."

# 6. Interactive Streaming Terminal Chat
prism chat Phi-4-mini-instruct-cuda-gpu

# 7. Start the OpenAI-Compatible REST Server (binds to 127.0.0.1 by default)
prism serve --port 5272
```

---

## 🔌 IDE & Agent Connectors (Cursor, Cline, MCP)

Prism provides dedicated connectors for zero-friction integration with your favorite AI coding tools:

### 1. Cursor IDE (`prism connect cursor`)
Generate custom OpenAI provider settings, export workspace rules, and configure Cursor MCP:
```bash
# Display setup instructions and test connectivity
prism connect cursor --test

# Export .cursorrules into the current workspace
prism connect cursor --export-rules

# Export Cursor MCP configuration (.cursor/mcp.json)
prism connect cursor --export-mcp
```

### 2. Cline Extension (`prism connect cline`)
Configure Cline (VS Code / Cursor extension) to use Prism as an OpenAI-compatible provider and register MCP tools:
```bash
# Display Cline UI settings and test server reachability
prism connect cline --test

# Export cline_mcp_settings.json
prism connect cline --export-mcp
```

### 3. Native Model Context Protocol (MCP) Server & Connector
Prism contains a built-in stdio JSON-RPC 2.0 MCP server exposing 5 tools:
- `prism_ask_coder`: Fast local code generation, bug fixing, test authoring with zero token cost.
- `prism_code_review`: Security, concurrency, and performance code review.
- `prism_list_models`: Discovered local models across both engines.
- `prism_get_status`: Real-time NVML GPU telemetry and server status.
- `prism_benchmark`: Automated latency and throughput micro-benchmark.

```bash
# Test MCP JSON-RPC protocol handshake
prism connect mcp --test

# Auto-wire Prism into Antigravity (~/.gemini/config/mcp_config.json)
prism connect mcp --target antigravity --write

# Auto-wire Cursor (.cursor/mcp.json)
prism connect mcp --target cursor --write

# Auto-wire Claude Desktop (~/.config/Claude/claude_desktop_config.json)
prism connect mcp --target claude --write
```

---

## 📥 Multi-Engine Model Pulling (`prism pull`)

Download genuine ONNX models from Hugging Face or GGUF models via Ollama with live streaming progress:

```bash
# Pull official Microsoft Phi-4 Mini (CUDA INT4 GPU by default):
prism pull phi-4-mini

# Pull CPU variant explicitly:
prism pull phi-4-mini --ep cpu

# Pull custom Hugging Face ONNX repo:
prism pull microsoft/Phi-4-mini-instruct-onnx

# Pull Ollama GGUF model directly via Ollama registry:
prism pull ollama:qwen2.5-coder:7b
prism pull deepseek-r1:14b --backend ollama
```

---

## 📡 OpenAI-Compatible REST API

`prism serve` provides a multi-threaded REST server on a predictable static port (default `127.0.0.1:5272`). Inference is serialized behind a lock, so concurrent requests queue rather than crash the engine.

**Security defaults:** the server binds to loopback only, rejects non-loopback `Host` headers (DNS-rebinding defence), sends no CORS headers, and caps request bodies at 10 MB. To expose it or call it from a browser:

```bash
# Listen on all interfaces, require a bearer token (also read from $PRISM_API_KEY)
prism serve --host 0.0.0.0 --api-key "$(openssl rand -hex 16)"

# Allow a specific browser origin (repeatable; '*' allows any)
prism serve --cors-origin http://localhost:3000
```
`/health` stays unauthenticated for liveness probes. Chat templates (Phi, ChatML/Qwen, Llama 3, DeepSeek) are chosen per model from its name and `genai_config.json`; unknown families fall back to ChatML.

### 1. List Available Models across Both Engines
```bash
curl -s http://127.0.0.1:5272/v1/models | jq .
```

### 2. Standard Chat Completion
```bash
curl -s http://127.0.0.1:5272/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Phi-4-mini-instruct-cuda-gpu",
    "messages": [{"role": "user", "content": "Write a hello world program in Rust."}],
    "max_tokens": 100,
    "stream": false
  }' | jq .
```

### 3. Server-Sent Events (SSE) Streaming Completion
```bash
curl -s -N http://127.0.0.1:5272/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Phi-4-mini-instruct-cuda-gpu",
    "messages": [{"role": "user", "content": "Count from 1 to 5."}],
    "stream": true
  }'
```

### 4. Health & Hardware Telemetry Endpoint
```bash
curl -s http://127.0.0.1:5272/health | jq .
```

---

## 🏎️ Built-in Micro-Benchmarking

Measure generation throughput, prefill latency, and VRAM memory footprint on your machine:

```bash
prism benchmark Phi-4-mini-instruct-cuda-gpu
```

**Verified RTX 5070 Telemetry:**
- **Time to First Token (TTFT)**: **56.2 ms (0.056s)**
- **Decode Speed**: **118.6 tok/s**
- **Token Count**: 256 tokens in 2.16s
- **Peak VRAM**: 100% within dedicated GPU memory

---

## 📑 Evaluation Whitepapers & Research

- 📘 [**`EVALUATION_REPORT.md`**](EVALUATION_REPORT.md): Complete evaluation whitepaper comparing Microsoft Foundry against Ollama, vLLM, and llama.cpp across 8 metrics.
- 🔬 [**`docs/UPSTREAM_CODE_ANALYSIS.md`**](docs/UPSTREAM_CODE_ANALYSIS.md): Source-level autopsy of `microsoft/foundry-local` examining the C++ `sdk_v2` rewrite vs legacy .NET CLI.
- 📋 [**`docs/FOUNDRY_WSL2_FEASIBILITY.md`**](docs/FOUNDRY_WSL2_FEASIBILITY.md): Feasibility analysis of WSL2 vs Windows Host Bridge architecture.

---

## 🗄️ Legacy: `foundry_wsl`

[`foundry_wsl/`](foundry_wsl/) is the earlier WSL2 bridge toolkit (doctor, cache injector, Windows-host proxy) from the original Foundry Local evaluation. It is kept as-is, is not part of the installable `prism-local` package, and is run from a checkout (`python -m foundry_wsl.cli`). New work goes into `prism/`.

---

## 🧪 Running the Test Suite

```bash
PYTHONPATH=. python3 -m unittest discover -s tests -v
```

The suite (about 100 tests, ~6 s) needs no GPU, Ollama, network, or models: the server is exercised through a fake engine on an ephemeral port, `OnnxGenAiEngine` against a fake `onnxruntime_genai`, and `pull` against a fake `huggingface_hub`. It covers chat templates, model resolution and caching, OpenAI response and SSE framing, error JSON, auth/CORS/Host checks, engine serialization, client-disconnect cancellation, Ollama routing, CLI argument handling, connectors, and the MCP handshake. GitHub Actions runs it on Python 3.10–3.12 and smoke-tests the built wheel (`.github/workflows/ci.yml`).

Two real-hardware smoke tests (`tests/test_prism_gpu_integration.py`) skip automatically unless `onnxruntime_genai`, an NVIDIA GPU, and a CUDA model are all present:

```bash
export PRISM_PYTHON=/path/to/venv/bin/python3 PRISM_MODEL_DIRS=/path/to/models
PYTHONPATH=. "$PRISM_PYTHON" -m unittest tests.test_prism_gpu_integration -v
```
