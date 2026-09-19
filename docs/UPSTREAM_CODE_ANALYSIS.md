# 🔬 Upstream Code Analysis: `microsoft/foundry-local`

A technical autopsy of Microsoft's `microsoft/foundry-local` repository, examining the source architecture of the **v2.0 C++ rewrite (`sdk_v2/`)** against the legacy **.NET CLI preview (`cli-preview-0.10.3`)**, with a specific focus on Linux and WSL2 hardware execution.

---

## 1. Repository Architectural Anatomy

The `microsoft/foundry-local` repository is currently bifurcated into two distinct generations:

```
microsoft/foundry-local/
├── sdk/                     # Legacy v1.x (.NET Core, C# Interop, Python ctypes shim)
│   ├── python/              # foundry-local-sdk <= 1.2.4
│   └── js/                  # foundry-local-sdk <= 1.2.4
├── sdk_v2/                  # New v2.x (Native C++20 Core Rewrite, v2.0.1)
│   ├── cpp/                 # Native runtime (libfoundry_local.so / C ABI)
│   │   └── src/
│   │       ├── ep_detection/ # Hardware detection & EP bootstrapping
│   │       ├── catalog/      # Azure & local catalog client
│   │       └── inferencing/  # ONNX Runtime GenAI session abstractions
│   ├── cs/                  # Modern .NET bindings over C ABI
│   ├── python/              # Modern Python bindings (foundry-local-sdk >= 2.0.0)
│   └── js/                  # Modern Node.js / TypeScript bindings
```

### Licensing Dichotomy
- **The SDKs (`sdk/` and `sdk_v2/`)**: Published under the open-source **MIT License**.
- **The Standalone CLI (`foundry` binary)**: Distributed exclusively via GitHub Releases under proprietary **Microsoft Software License Terms**.

---

## 2. Source Walkthrough: Hardware Detection on Linux

### A. The Legacy CLI Detection Failure (`cli-preview-0.10.3`)
In `cli-preview-0.10.3` (running `Foundry Local Core 1.0.0`), hardware detection relies on Windows Management Instrumentation (WMI) and DirectX/DirectML interfaces:

```text
WMI Query: Win32_VideoController.AdapterRAM
DirectX: DXCore / D3D12 Enumeration
```

**Why it fails on WSL2:**
WSL2 provides a virtualized Linux kernel. While NVIDIA exposes compute libraries into `/usr/lib/wsl/lib/` (`libcuda.so.1`, `libnvidia-ml.so.1`, `libdxcore.so`), WMI does not exist in Linux user-space. When the .NET Core CLI executes `foundry status`, the detector fails silently, reporting:
```text
System | GPU | Not detected
```
Consequently, the model catalog filter (`azure_catalog_client`) assumes a CPU-only environment and suppresses all GPU-accelerated variants.

---

### B. The v2.0 C++ NVML Detector (`sdk_v2/cpp/src/ep_detection/nvml_gpu_detector.cc`)
In the new C++ rewrite, Microsoft introduced a dedicated user-space detector querying the NVIDIA Management Library (`NVML`):

```cpp
// sdk_v2/cpp/src/ep_detection/nvml_gpu_detector.cc
#elif defined(__linux__)
LibraryHandle LoadNvmlLibrary() { 
  return dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL); 
}

void* GetSymbol(LibraryHandle lib, const char* name) { 
  return dlsym(lib, name); 
}
#endif
```

The detector dynamically resolves `nvmlInit_v2`, `nvmlDeviceGetCount_v2`, `nvmlDeviceGetHandleByIndex_v2`, and queries compute capability:

```cpp
bool HasQualifyingComputeCapability(const std::vector<std::pair<int, int>>& capabilities, 
                                    int min_major = 5, 
                                    int min_minor = 0) {
  for (const auto& [major, minor] : capabilities) {
    if (major > min_major || (major == min_major && minor >= min_minor)) {
      return true;
    }
  }
  return false;
}
```

**WSL2 Compatibility Status:**
Our diagnostic probe ([`scripts/probe_foundry_ecosystem.py`](../scripts/probe_foundry_ecosystem.py)) executed this exact sequence in WSL2:
- `dlopen("/usr/lib/wsl/lib/libnvidia-ml.so.1")` **succeeded**.
- Correctly identified: **`NVIDIA GeForce RTX 5070`**.
- Compute Capability: **`12.0`** (Qualifying: `True`).

**The Problem:**
Microsoft has **not yet released a CLI build based on `sdk_v2`**. The CLI remains frozen at `cli-preview-0.10.3`, meaning CLI users cannot benefit from this new C++ detector until Microsoft ships a v2.0 CLI binary.

---

## 3. The Linux CUDA Packaging Gap

In [`sdk_v2/cpp/src/ep_detection/cuda_ep_manifest.cc`](https://github.com/microsoft/foundry-local/blob/main/sdk_v2/cpp/src/ep_detection/cuda_ep_manifest.cc), the dynamic EP bootstrapper defines how execution provider assets are downloaded.

### Windows Manifest vs. Linux Manifest

| Component | Windows (`WindowsX64Manifest`) | Linux (`LinuxX64Manifest`) |
|---|---|---|
| **CUDA Toolkit Libraries** | `cublas64_12.dll`, `cublasLt64_12.dll`, `cudart64_12.dll` (Bundled, ~640 MB) | ❌ **NOT BUNDLED** |
| **cuDNN Libraries** | `cudnn64_9.dll`, `cudnn_ops64_9.dll`, `cudnn_adv64_9.dll`, etc. (Bundled, ~704 MB) | ❌ **NOT BUNDLED** |
| **ORT CUDA Provider** | `onnxruntime_providers_cuda.dll`, `onnxruntime-genai-cuda.dll` (Bundled, ~256 MB) | `libonnxruntime_providers_cuda.so`, `libonnxruntime-genai-cuda.so` (Bundled, ~448 MB) |

### Impact on Linux/WSL2
On Windows, Microsoft bundles the entire NVIDIA CUDA 12 and cuDNN 9 runtime inside the download. On Linux, Microsoft assumes the user already has system-wide CUDA and cuDNN libraries configured in `LD_LIBRARY_PATH`.

If a user does not manually install `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` (via pip or system packages) and export `LD_LIBRARY_PATH`, loading `libonnxruntime_providers_cuda.so` will crash with:
```text
The specified EP cuda is not registered
```

---

## 4. Upstream Issue Tracker Audit (September 2026)

### Issue #1109: "Foundry does not expose any CUDA-enabled model variants"
- **Opened**: September 15, 2026.
- **Affected Platforms**: Windows 11 & Linux.
- **Symptom**: Even when `CUDAExecutionProvider` is downloaded and confirmed registered in logs, `foundry model list --variants` and `foundry.modelinfo.json` contain **zero CUDA models**.
- **Root Cause**: The Azure AI Foundry catalog indexing service is failing to return CUDA variants for client queries, defaulting all returned models to `Device: CPU`.

### Issue #1079 & PR #1110: CUDA VRAM Retention Bug
- **Opened**: September 15, 2026 | **PR Submitted**: September 15, 2026 by Microsoft engineer `aciddelgado`.
- **Symptom**: When a model is unloaded (`foundry model unload`), the daemon retains ~99% of GPU VRAM.
- **Root Cause**: ONNX Runtime GenAI's internal memory allocator keeps the GPU device pool open in a long-lived dummy session.
- **Fix in PR #1110**: Adds explicit invocation of `OgaReleaseDeviceResources("CUDA")` after the last CUDA model unloads.

### PR #1114: Catalog v2 Migration
- **Submitted**: September 17, 2026 by Microsoft engineer `prathikr`.
- **Changes**: Migrates catalog fetching from `crossRegion/` endpoints to the new `asset-gallery/` backend, adding `minFLVersion` validation.

---

## 5. Architectural Conclusion

The `microsoft/foundry-local` repository is in the middle of a massive architectural rewrite. While the new C++ `sdk_v2` represents a massive leap forward in cross-platform cleanliness, the user-facing CLI on Linux/WSL2 remains caught in a transition gap:

1. The CLI is locked to legacy .NET v1.0.0 with non-functional Linux hardware detection.
2. The Linux CUDA bundle lacks essential runtime libraries.
3. The Azure catalog backend currently fails to serve CUDA variants.

For developers seeking turnkey local inference on WSL2 today, alternative engines like **Ollama** provide far superior out-of-the-box reliability.
