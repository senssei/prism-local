# 🏛️ Microsoft Foundry Local: Architectural Evaluation & `foundry-ng` Alternative CLI

A complete ecosystem evaluation and next-generation local AI CLI replacement for **Microsoft Foundry Local** (`microsoft/foundry-local`), optimized specifically for **Linux and WSL2** with native **NVIDIA GeForce RTX (CUDA)** acceleration and unified **Ollama (GGUF)** orchestration.

---

## 🌟 What is `foundry-ng` (`fng`)?

**`foundry-ng`** is an open-source, high-performance local AI CLI and OpenAI-compatible inference server. It was created to overcome the critical architectural shortcomings of the official Microsoft Foundry CLI (`0.10.3`) on Linux/WSL2:

| Capability | Official `foundry` CLI (`0.10.3`) | `foundry-ng` (`fng`) |
|---|---|---|
| **GPU Detection on WSL2** | ❌ Fails (`GPU: Not detected` via WMI) | ✅ Direct NVML hardware detection (`libnvidia-ml.so.1`) |
| **Hardware Target** | ❌ Locks Linux to CPU (`CPUExecutionProvider`) | ✅ 100% Native CUDA on RTX GPUs (`onnxruntime-genai-cuda`) |
| **Decode Speed (Phi-4-mini)**| ❌ 9.1 – 15.3 tokens/sec (CPU) | ⚡ **118.6 – 130.2 tokens/sec (CUDA)** |
| **Time to First Token (TTFT)**| ❌ 5.0 – 7.7 seconds | ⚡ **50 – 60 milliseconds** |
| **REST Server Port** | ❌ Ephemeral random port (e.g. `:37863`) | 🟢 **Stable, fixed port (default: `:5272`)** |
| **API Standardization** | ⚠️ OpenAI `/v1` on random port | 🟢 Full OpenAI `/v1/chat/completions` + SSE streaming |
| **Model Ecosystem** | ❌ Microsoft catalog with 0 CUDA variants | 🟢 Pulls official ONNX weights directly from Hugging Face |
| **Multi-Engine Routing** | ❌ Closed to ONNX runtime | 🟢 **Unified ONNX + Ollama (GGUF) orchestration** |
| **VRAM Lifecycle** | ❌ Leaks ~99% VRAM on unload (Issue #1079) | 🟢 Complete memory disposal on model exit |

---

## 🚀 Quickstart

The launcher script is located at [`./bin/foundry-ng`](bin/foundry-ng) (alias [`./bin/fng`](bin/fng)).

```bash
# Add bin to PATH (optional):
export PATH="$(pwd)/bin:$PATH"

# 1. System & GPU Telemetry
fng status

# 2. Environment Doctor Check
fng doctor

# 3. List All Local Models (ONNX + Ollama)
fng list

# 4. Run a Prompt Completion (ONNX CUDA GPU)
fng run Phi-4-mini-instruct-cuda-gpu "Write a Python function to compute Fibonacci numbers."

# 5. Run a Prompt Completion (Ollama GGUF)
fng run ollama:qwen2.5-coder:7b "Explain quicksort in two sentences."

# 6. Interactive Streaming Terminal Chat
fng chat Phi-4-mini-instruct-cuda-gpu

# 7. Start the OpenAI-Compatible REST Server
fng serve --port 5272
```

---

## 📡 OpenAI-Compatible REST API

`foundry-ng serve` provides a high-throughput, multi-threaded REST server running on a predictable static port (default `5272`).

### 1. List Available Models
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
fng benchmark Phi-4-mini-instruct-cuda-gpu
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
fng pull phi-4-mini

# Pull custom Hugging Face ONNX repo:
fng pull microsoft/Phi-4-mini-instruct-onnx
```

Models are stored in `./models/` and dynamically indexed alongside `~/.foundry/cache/models/` and `../02-ollama-loadtest/models/`.

---

## 📑 In-Depth Evaluation Whitepapers

- 📘 [**`EVALUATION_REPORT.md`**](EVALUATION_REPORT.md): Complete evaluation whitepaper comparing Microsoft Foundry against Ollama, vLLM, and llama.cpp across 8 metrics.
- 🔬 [**`docs/UPSTREAM_CODE_ANALYSIS.md`**](docs/UPSTREAM_CODE_ANALYSIS.md): Source-level autopsy of `microsoft/foundry-local` examining the C++ `sdk_v2` rewrite vs legacy .NET CLI.
- 📋 [**`docs/FOUNDRY_WSL2_FEASIBILITY.md`**](docs/FOUNDRY_WSL2_FEASIBILITY.md): Feasibility analysis of WSL2 vs Windows Host Bridge architecture.

---

## 🧪 Running the Test Suite

```bash
PYTHONPATH=. python3 -m unittest discover -s tests -v
```
All 18 automated unit tests verify NVML hardware detection, catalog resolution, prompt formatting, REST server endpoints, and SSE streaming.
