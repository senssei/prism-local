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
❌ CUDA execution provider: cannot load: missing shared libraries: libcublas.so.13, libcublasLt.so.13, libcudart.so.13
   Models will fall back to CPU. Install CUDA libraries matching your onnxruntime-genai build.
```

The check runs `ldd` on ONNX Runtime's CUDA provider library, so it reports unresolved dependencies without running any of
the library's code. (Loading that library outside ONNX Runtime can crash the process, so Prism deliberately does not try.)

The typical cause is a **build/library mismatch**: a wheel built for one major CUDA version while only another version's
runtime libraries are installed. Make them agree by installing the stack below.

## A known-good CUDA setup

```bash
python3 -m venv .venv && source .venv/bin/activate     # Python 3.11+
pip install "prism-local[cuda,pull]"                    # or, without Prism's extra:
# pip install onnxruntime-genai-cuda "onnxruntime-gpu[cuda,cudnn]>=1.30"
prism doctor                                            # expect: CUDA execution provider ... resolve
prism benchmark phi-4-mini --device cuda
```

Verified 2026-09-19 with `onnxruntime-genai-cuda 0.16.0`, `onnxruntime-gpu 1.30.0` (a **CUDA 13** build), `nvidia-cublas 13.8`,
`nvidia-cudnn-cu13 9.26` and an NVIDIA driver reporting CUDA 13.4. A different `onnxruntime-genai` release may target a
different CUDA major version; ONNX Runtime GPU's `cuda` and `cudnn` extras pull the libraries that match it.

`bin/prism` uses `./.venv` automatically when it exists.

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

## GPU memory and long prompts

With ONNX Runtime GenAI the GPU memory a request needs grows with the **prompt length**, and it is not given back when the request
ends. Measured on Phi-4-mini (int4, CUDA) with an RTX 5070 under WSL2, one process per setting, greedy decoding, GPU memory in use
(whole GPU, about 1.1 GB of it before the model was loaded; 5.5 GB right after loading):

| Prompt tokens | Default | `PRISM_PREFILL_CHUNK=256` | `PRISM_PREFILL_CHUNK=1024` |
|---:|---:|---:|---:|
| 300 | 5.9 GB | 5.8 GB | 5.9 GB |
| 1000 | 7.4 GB | 6.5 GB | 7.4 GB |
| 2200 | 9.5 GB | 6.5 GB | 7.4 GB |
| 4500 | 11.7 GB | 6.6 GB | 7.4 GB |

That is about 1.4 MB per prompt token by default, roughly ten times the KV cache, so the growth is not the KV cache. It did not change
with the CUDA memory allocator settings that were tried (`arena_extend_strategy`, `gpu_mem_limit`), so those do not help.

**`PRISM_PREFILL_CHUNK`** (default `1024`; `0` or `off` disables it) makes ONNX Runtime GenAI process the prompt in chunks of that size, which
bounded the growth in these measurements: at 4500 tokens, 6.6 GB with 256 (44% less than the default), 7.4 GB with 1024 and 7.7 to 8.7 GB
with 512 (two runs). Decode speed (73 to 88 tok/s in every setting) and time to first token (about 1.0 s at 4500 tokens) stayed within a few
percent. For the longest prompt the first 48 greedy tokens differed from the unchunked run (shorter prompts were identical), which is expected
from different numerics but means outputs are not guaranteed bit-identical.

It is on by default because the memory is also **not returned when the model is unloaded**. With no chunking, a 4200-token prompt on Phi-4-mini
left 7.3 GB of the GPU held after `unload()` (1.3 GB with chunk 512, 2.3 GB with 1024); the next model, qwen2.5-coder-7b, then found the card
nearly full and needed 83 s to the first token at 1.2 tok/s, against 0.34 s and 30 tok/s with the default of 1024. Chunk 512 held less memory but
raised time to first token on qwen3-0.6b (17.5 s against 11.0 s), and 1024 did not on any of the three models measured (Phi-4-mini, qwen3-0.6b,
qwen2.5-coder-7b), which is why it is the default. If a model is slower with it, **check yours** and set `PRISM_PREFILL_CHUNK` accordingly.

Other models measured the same way (same GPU, 4500-token prompt, chunk of 256; the two `generic-cpu` variants come from the Foundry Local cache
and run on the CUDA provider under `--device auto`, slowly):

| Model | Vocabulary | Peak memory, default | With chunk 256 | Time to first token, default | With chunk 256 |
|---|---:|---:|---:|---:|---:|
| Phi-4-mini (`cuda-gpu`, int4) | 200k | 11.7 GB | 6.6 GB (-44%) | 1.0 s | 1.1 s (+4%) |
| qwen3-0.6b (`generic-cpu`) | 152k | 11.6 GB | 5.4 GB (-54%) | 7.7 s | 16.3 s (+110%) |
| qwen2.5-coder-7b (`generic-cpu`) | 152k | 11.6 GB | 11.0 GB (-5%) | 15.3 s | 20.4 s (+33%) |

Chunking lowered the peak for every model, but by very different amounts (the 7B model already filled the GPU after loading, 11.3 GB, so there
was little room to save) and it cost anywhere from 4% to 110% more time to first token. The growth per prompt token by default was about 1.4 MB
for Phi-4-mini and 1.5 MB for qwen3-0.6b, which fits a buffer that scales with vocabulary size times prompt length (both vocabularies are 150k to 200k)
rather than with the KV cache; that is an inference from the pattern, not something verified in the library. Phi-3.5-mini did not finish the
long prompt within five minutes.

## Verifying a GPU run

```bash
prism benchmark phi-4-mini
```

Look for `Execution provider: CUDA` and a VRAM increase on the order of the model size. On the reference machine a CUDA run added about 4.5 GB
of VRAM and decoded at 79–98 tok/s. A model that "ran on GPU" but added almost no VRAM and decoded at single-digit tokens per
second was running on the CPU.

## Parallel use

Running multiple inference processes or concurrent model loads on the same machine can exhaust host RAM or GPU VRAM,
potentially locking the operating system or crashing background tasks. Prism guards against parallel thrashing at several levels:

1. **Cross-process model load lock**: `prism.machine_lock` uses an OS-level file lock (`~/.prism/load.lock`) to serialize model loading
   across all local Prism commands (`run`, `chat`, `serve`, `benchmark`, `mcp`). If another process is currently loading a model, a second process
   waits up to `PRISM_LOAD_TIMEOUT` (default 120 s) instead of loading concurrently.
2. **Resource capacity guard**: Before loading an ONNX model, Prism checks live available RAM (`/proc/meminfo`) and free VRAM
   (NVML) against the model weight size and prefill headroom (`PRISM_PREFILL_CHUNK` × 1.4 MB/token), keeping safety reserves
   (`PRISM_RAM_RESERVE_MB`, default 2048 MB; `PRISM_VRAM_RESERVE_MB`, default 1536 MB). If headroom is insufficient, the load is refused
   immediately with `InsufficientResourcesError` (HTTP `503 insufficient_resources`). Disable with `PRISM_RESOURCE_CHECK=off`.
3. **Queue limits**: `prism serve` limits waiting requests for the active model with `--max-queue` / `$PRISM_MAX_QUEUE` (default 8). When
   full, additional requests fail immediately with `503 server_busy` rather than piling up indefinitely.
4. **WSL2 host memory bounding**: Under WSL2, Linux memory allocations may consume Windows host RAM without bounds unless configured.
   Create or edit `%USERPROFILE%\.wslconfig` (typically `C:\Users\<Username>\.wslconfig`) with:

```ini
[wsl2]
memory=16GB
autoMemoryReclaim=gradual
```

`prism doctor` inspects `.wslconfig` and reports whether these safety limits are active. Prism never modifies `.wslconfig`.
5. **Thread control (`PRISM_THREADS`)**: By default, ONNX Runtime allocates internal thread pools based on CPU core count. When running alongside heavy host workloads or in shared environments, set `PRISM_THREADS` (a positive integer, e.g. `PRISM_THREADS=4`) to restrict intra-op parallelism and avoid CPU saturation during token generation.
6. **Orchestrator drain (`POST /v1/drain`)**: A benchmark or evaluation tool that needs a different model on the same GPU can ask the running `prism serve` to step aside. The flow:

```bash
# 1. Drain the active server (finishes the in-flight request, unloads the model, exits with status 0).
curl -s -X POST http://127.0.0.1:5272/v1/drain

# 2. Start your own prism serve that owns the GPU until you are done.
prism serve --device cuda --model <the-model-you-need>

# 3. (Optional) restart the previous server when you are done — it was already shut down by the drain.
```

When the orchestrator's request is blocked by a holder, the `503 insufficient_resources` response carries `error.holder = {"pid", "model"}` so the orchestrator can target the right `prism serve` instance. See [Drain in the API reference](api.md#drain).
