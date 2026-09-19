# 🏛️ Microsoft Foundry Local: Architectural Evaluation & WSL2 Bridge

This repository contains an in-depth technical evaluation of **Microsoft Foundry Local** (`microsoft/foundry-local`), analyzing its codebase, Linux and WSL2 hardware execution viability, upstream roadmap, and alternative local LLM inference engines (such as **Ollama**, **vLLM**, and **`llama.cpp`**).

It also includes the **`foundry-wsl`** toolkit: a set of diagnostic probes, an automated reverse proxy for ephemeral ports, and a CUDA cache injection utility.

---

## 📑 Core Documentation & Analysis

| Document | Description |
|---|---|
| 📘 [**`EVALUATION_REPORT.md`**](EVALUATION_REPORT.md) | **Master Evaluation Whitepaper**: Exhaustive architectural breakdown, 8-metric benchmark scorecard, analysis of upstream issues (#1109, #1079, PR #1110), and final recommendations. |
| 🔬 [**`docs/UPSTREAM_CODE_ANALYSIS.md`**](docs/UPSTREAM_CODE_ANALYSIS.md) | **Code-Level Autopsy**: Source walkthrough of `microsoft/foundry-local` comparing the new C++ `sdk_v2` against the legacy .NET CLI `0.10.3`, including NVML detection and manifest gaps. |
| 📋 [**`docs/FOUNDRY_WSL2_FEASIBILITY.md`**](docs/FOUNDRY_WSL2_FEASIBILITY.md) | **Feasibility Assessment**: In-depth review of out-of-the-box CLI usability, cache injection workarounds, and the recommended Windows Host Gateway pattern. |

---

## 🏆 Key Findings at a Glance

1. **The Architectural Disconnect ("The Two Foundries")**:
   - **The SDK (`sdk_v2`, v2.0.1)**: Open-source (MIT), rewritten in native C++20 (`libfoundry_local.so`), with modern `NvmlGpuDetector` that dynamically resolves `/usr/lib/wsl/lib/libnvidia-ml.so.1` on Linux.
   - **The CLI (`cli-preview-0.10.3`)**: Proprietary closed binary built on the legacy v1.0.0 .NET Core architecture. It relies on Windows WMI (`Win32_VideoController`), fails to detect GPUs in WSL2 (`GPU: Not detected`), and permanently locks downloads to `-generic-cpu` variants (running at 9 tok/s vs 130–174 tok/s on CUDA).

2. **The Linux CUDA Packaging Gap**:
   - In `cuda_ep_manifest.cc`, Microsoft bundles full CUDA 12 and cuDNN 9 binaries for Windows, but the Linux manifest **only bundles `libonnxruntime_providers_cuda.so`**.
   - Clean Ubuntu/WSL2 systems lack `libcublas.so.12` and `libcudnn.so.9` in default paths, requiring manual user-space configuration via PyPI wheels.

3. **Active Upstream Breakages (September 2026)**:
   - **Issue #1109**: Azure catalog currently returns **zero CUDA model variants** even when CUDA EP is registered.
   - **Issue #1079 / PR #1110**: Unloading a model in Foundry fails to reclaim ~99% of GPU VRAM due to retained GenAI memory pools (observed as ~5.5–6.0 GB VRAM retention on RTX 5070).

4. **Strategic Recommendation**:
   - For daily coding and agentic workflows: **Use Ollama (`ollama`) on WSL2** (turnkey CUDA, GGUF catalog, 174 tok/s decode, automatic VRAM eviction).
   - If evaluating Microsoft models: **Use the Windows Host Gateway** or **Direct ONNX Runtime GenAI**.

---

## 🛠️ The `foundry-wsl` Toolkit

### 1. Run the Hardware & Ecosystem Diagnostic Probe
Simulates `sdk_v2`'s `NvmlGpuDetector`, inspects dynamic library resolution, and checks active GPU memory:

```bash
python3 scripts/probe_foundry_ecosystem.py
```

### 2. Run the Health Doctor
Inspects NVML drivers, CUDA shared libraries, and daemon configuration:

```bash
PYTHONPATH=. python3 -m foundry_wsl.cli doctor
```

### 3. Stable Reverse Proxy for Ephemeral Ports
Microsoft's daemon assigns random ephemeral ports (e.g. `127.0.0.1:37863`). The proxy reads `~/.foundry/daemon.json` dynamically and exposes a stable static endpoint for agents and tools:

```bash
PYTHONPATH=. python3 -m foundry_wsl.cli proxy --port 5272
```

Connect your agent or client to: `http://127.0.0.1:5272/v1`

### 4. Cache Injector for Genuine CUDA Weights
Injects `"provider_options": [{"cuda": {}}]` into a model's `genai_config.json`:

```bash
PYTHONPATH=. python3 -m foundry_wsl.cli inject ~/.foundry/cache/models/Microsoft/<model_folder>/<ver>/genai_config.json
```

---

## 🌉 Windows Host Bridge Pattern

To run Foundry Local natively on Windows 11 (with full DirectML / WinML TensorRT-RTX acceleration) and access it seamlessly from WSL2:

1. **On Windows Host (PowerShell as Administrator)**:
   ```powershell
   .\windows_bridge\setup_windows_gateway.ps1
   ```
2. **Inside WSL2**:
   ```bash
   ./windows_bridge/test_connection.sh
   export FOUNDRY_BASE_URL="http://$(ip route show default | awk '{print $3}'):5272/v1"
   ```

---

## 🧪 Running Unit Tests

```bash
PYTHONPATH=. python3 -m unittest discover -s tests -v
```
