# 📋 Feasibility Analysis: Microsoft Foundry Local on WSL2

An operational and architectural feasibility study examining whether Microsoft Foundry Local is currently usable on WSL2, whether it will be natively supported in the future, and how alternative deployment topologies compare.

---

## 1. Executive Feasibility Matrix

| Evaluation Dimension | Out-of-the-Box CLI (`0.10.3`) | Hack / Cache Injection | Python SDK (`v2.0.1`) | Windows Host Gateway |
|:---|:---:|:---:|:---:|:---:|
| **WSL2 Usability Today** | ⚠️ **Severe (CPU-only)** | ✅ **Functional (Fast)** | ⚠️ **High Complexity** | ✅ **Optimal (Native WinML)** |
| **Decode Speed (7B/3.8B)** | 9.1 – 15.3 tok/s | **130.2 tok/s** | ~120 – 130 tok/s | ~130 – 160 tok/s |
| **GPU VRAM Acceleration** | ❌ No (Host RAM only) | ✅ Yes (RTX 5070 CUDA) | ✅ Yes (via PyPI ORT) | ✅ Yes (WinML / TRT-RTX) |
| **Setup Friction** | 🟢 Low (single binary) | 🔴 High (manual weights/json) | 🟡 Medium (venv + C ABI) | 🟡 Medium (port proxying) |
| **Maintenance Burden** | 🟢 Low | 🔴 Fragile (breaks on wipe) | 🟡 Medium (API changes) | 🟢 Stable |
| **Tool Calling / OpenAI API** | ✅ Reachable (HTTP) | ✅ Reachable (HTTP) | In-process or HTTP | ✅ Reachable (HTTP) |

---

## 2. Is Foundry CLI Currently Usable on WSL2?

### The "Out of the Box" Experience (Verdict: ❌ Unusable for Daily Coding)
When a developer runs `tar xzf foundry-0.10.3-linux-x64.tar.gz` and executes `foundry model run qwen2.5-coder-7b` inside WSL2:
1. The CLI queries WMI for GPU hardware, fails, and marks GPU as `Not detected`.
2. The catalog downloads `qwen2.5-coder-7b-instruct-generic-cpu`.
3. Inference runs on CPU vector instructions (`CPUExecutionProvider`).
4. **Performance**: Time-To-First-Token is **7.74 seconds**; generation speed is **9.1 tokens/second**.
5. **Conclusion**: At 9 tokens/second, it is too sluggish for responsive pair programming or multi-turn agent execution compared to Ollama (116–174 tok/s).

### The "Cache Injection" Experience (Verdict: ✅ Fast, but Fragile)
As demonstrated in `02-ollama-loadtest`:
1. Downloading true GPU ONNX weights from Hugging Face (`microsoft/Phi-4-mini-instruct-onnx`).
2. Modifying `genai_config.json` with `"provider_options": [{"cuda": {}}]`.
3. Patching `foundrylocald` with user-space CUDA 12 / cuDNN 9 paths.
4. **Performance**: Decode speed reaches **130.2 tokens/second** and TTFT drops to **90 ms** with 100% coding unit test pass rates.
5. **Conclusion**: Proves that the underlying ONNX GenAI engine is capable of stellar performance, but the manual maintenance overhead makes it impractical for regular developers without automated tooling.

---

## 3. Will Foundry CLI Be Usable on WSL2 in the Future?

### Roadmap Analysis
Microsoft's development trajectory reveals three critical patterns:

1. **Prioritization of Windows Copilot+ Architecture**:
   Microsoft Foundry Local is primarily designed as the runtime layer for Windows on-device AI. Features like NPU acceleration (Qualcomm Hexagon, Intel NPU, AMD Ryzen AI) and GeForce RTX optimization (`Microsoft.WinML.NVIDIA.TRT-RTX.EP`) are deeply coupled with Windows 11 (24H2+) system APIs.

2. **The SDK-First Shift**:
   The active C++ rewrite (`sdk_v2`) consolidates functionality into `libfoundry_local.so` for embedding directly inside desktop and server apps. The standalone CLI is treated as a secondary developer utility, not the primary product.

3. **WSL2 Virtualization Boundaries**:
   Because WinML and DirectML hardware acceleration run on the Windows host graphics driver, passing high-performance low-latency DirectML sessions across the WSL2 virtualization boundary is non-trivial. Microsoft explicitly recommends running Foundry Local directly on the Windows host.

**Forecast**: While Microsoft will eventually update the Linux CLI to use the `sdk_v2` C++ runtime (fixing NVML detection), Linux packaging and catalog availability will likely remain secondary to Windows.

---

## 4. The Recommended Alternative Topology: Windows Host Gateway

For developers working inside WSL2 who want genuine Microsoft Foundry Local hardware acceleration without cache hacking:

```
┌────────────────────────────────────────────────────────┐
│                   Windows 11 Host                      │
│                                                        │
│  foundry server start (DirectML / WinML TensorRT-RTX)  │
│  Listening on: http://0.0.0.0:5272                     │
│  Hardware: NVIDIA GeForce RTX 5070 (Native Windows)    │
└───────────────────────────▲────────────────────────────┘
                            │ Virtual Ethernet Bridge
┌───────────────────────────▼────────────────────────────┐
│                  WSL2 (Ubuntu 24.04)                   │
│                                                        │
│  export FOUNDRY_BASE_URL="http://$(hostname).local:5272/v1"
│  Antigravity / Coding Agents / MCP Tools               │
└────────────────────────────────────────────────────────┘
```

### Advantages of the Host Gateway:
- **100% Native Hardware Acceleration**: Uses official Microsoft WinML and TensorRT-RTX packages from the Windows Store without missing dynamic libraries.
- **Zero Cache Hacking**: Official GPU models (`-cuda-gpu`) download cleanly through Windows PowerShell.
- **Clean WSL2 Isolation**: Keeps large CUDA SDK wheels and virtual environment bloat out of the WSL2 Linux distribution.
