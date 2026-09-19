# Devices & CUDA

Where a model runs is decided by ONNX Runtime's **execution provider**. Prism makes that choice explicit and reports it,
because a silent CPU fallback looks exactly like a slow GPU.

## Selecting a device

```bash
prism run phi-4-mini "hi" --device auto   # default
prism run phi-4-mini "hi" --device cuda   # fail rather than fall back
prism run phi-4-mini "hi" --device cpu
PRISM_DEVICE=cuda prism serve
```

At load time Prism overrides the provider list in the model's `genai_config.json` (the `model.decoder.session_options`
section). This matters: many "GPU" model folders ship an **empty** `provider_options` list, which ONNX Runtime GenAI treats as
**CPU**. Prism does not trust folder names such as `…-cuda-gpu`.

| Mode | Outcome |
|---|---|
| `auto` and an NVIDIA GPU is detected | Try CUDA. If it fails to load, warn (with the underlying error) and run on CPU. |
| `auto` and no GPU is detected | CPU |
| `cuda` | CUDA, or an error naming the cause. |
| `cpu` | CPU |

The device actually in use is shown by:

- `prism run` (`[device: cuda]` on stderr) and `prism chat`
- `prism benchmark` (Execution provider, plus a warning if it is not the GPU)
- `GET /health` → `active_device`, and `telemetry.device` in non-streaming responses

With an older `onnxruntime-genai` that has no `Config` API, Prism cannot choose a provider; the device is then reported as
`default` and the model's own configuration decides.

## Diagnosing CUDA problems

```bash
prism doctor
```

`doctor` tries to load ONNX Runtime's CUDA provider library, so it catches version mismatches that only show up at model-load
time:

```text
✅ ONNX Runtime GenAI: Installed.
❌ CUDA execution provider: cannot load: libcublasLt.so.13: cannot open shared object file: No such file or directory
   Models will fall back to CPU. Install CUDA libraries matching your onnxruntime-genai build.
```

The typical cause is a **build/library mismatch**: an `onnxruntime-genai-cuda` wheel built for one major CUDA version while
only another version's runtime libraries (for example the `nvidia-*-cu12` wheels) are installed. Make them agree: either
install the runtime libraries for the CUDA version your wheel targets, or install a wheel built for the CUDA version you have.
Check the `onnxruntime-genai` release notes for which CUDA version each release targets.

### How Prism finds CUDA libraries

`nvidia-*` pip wheels put their libraries under `site-packages/nvidia/<pkg>/lib`. Prism discovers those directories from the
running interpreter and **preloads** the libraries into the process before ONNX Runtime starts, because setting
`LD_LIBRARY_PATH` after startup does not affect the current process. `/usr/lib/wsl/lib` (the WSL2 driver shim, with `libcuda`
and `libnvidia-ml`) is added to `LD_LIBRARY_PATH` for child processes such as the MCP server.

## WSL2 notes

- GPU access needs a recent NVIDIA Windows driver with WSL2 support; `nvidia-smi` should work inside WSL2.
- Prism reads GPU info directly from `libnvidia-ml.so.1` (NVML), so it works where WMI-based detection does not.
- The GPU is shared with Windows applications, so absolute VRAM numbers include them. `prism benchmark` therefore reports
  the *change* in VRAM caused by the model.

## Verifying a GPU run

```bash
prism benchmark phi-4-mini
```

Look for `Execution provider: CUDA` and a VRAM increase on the order of the model size. A model that "ran on GPU" but added
almost no VRAM and decoded at single-digit tokens per second was running on the CPU.
