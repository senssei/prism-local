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
| **Compatibility Aliases** | N/A | 🟢 Fully accessible via `prism`, `fng`, or `foundry-ng` |

---

## 🚀 Quickstart

The launcher script is located at [`./bin/prism`](bin/prism) (with backward-compatible symlinks [`./bin/fng`](bin/fng) and [`./bin/foundry-ng`](bin/foundry-ng)).

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

# 7. Start the OpenAI-Compatible REST Server
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

`prism serve` provides a high-throughput, multi-threaded REST server running on a predictable static port (default `5272` on `0.0.0.0` for WSL2/Windows host reachability).

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

## 🧪 Running the Test Suite

```bash
PYTHONPATH=. python3 -m unittest discover -s tests -v
```
All **28 automated unit tests** verify NVML hardware detection, multi-engine catalog resolution, model pulling, prompt formatting, REST server endpoints, SSE streaming, MCP JSON-RPC protocol handshake, and Cursor/Cline connectors.
