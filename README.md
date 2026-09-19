# 💎 Prism: Multi-Engine Local AI CLI & Inference Server

A high-performance, developer-first local AI CLI and OpenAI-compatible inference server for **Linux & WSL2**.

**Prism** refracts disparate local AI runtimes—**ONNX Runtime GenAI on NVIDIA CUDA** and **Ollama / llama.cpp (GGUF)**—into a single, unified developer experience with sub-millisecond dispatch, interactive streaming chat, automated benchmarking, and zero-dependency REST serving.

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
| **Model Ecosystem** | ❌ Microsoft catalog with 0 CUDA variants | 🟢 Pulls official ONNX weights directly from Hugging Face |
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

## 📡 OpenAI-Compatible REST API

`prism serve` provides a high-throughput, multi-threaded REST server running on a predictable static port (default `5272`).

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

## 📥 Model Management & Pulling

Pull genuine GPU ONNX models directly from Hugging Face without Microsoft catalog gating:

```bash
# Pull official Microsoft Phi-4 Mini (INT4 AWQ GPU):
prism pull phi-4-mini

# Pull custom Hugging Face ONNX repo:
prism pull microsoft/Phi-4-mini-instruct-onnx
```

Models are stored in `./models/` and dynamically indexed alongside `~/.foundry/cache/models/` and `../02-ollama-loadtest/models/`.

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
All 18 automated unit tests verify NVML hardware detection, catalog resolution across both engines, prompt formatting, REST server endpoints, and SSE streaming.
