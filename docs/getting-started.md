# Installation & configuration

## Requirements

- Linux or WSL2, Python 3.10+.
- Optional engines: `onnxruntime-genai` (GPU build for CUDA) and/or a running [Ollama](https://ollama.com) daemon.
- Prism itself has **no runtime dependencies**; engines are installed as extras.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install "prism-local[cuda,pull]"
```

The package is on [PyPI](https://pypi.org/project/prism-local/). Plain `pip install prism-local` (no extras) is enough for
CPU inference or Ollama-only use, and installs nothing else.

| Extra | Installs | Needed for |
|---|---|---|
| `cuda` | `onnxruntime-genai-cuda`, `onnxruntime-gpu[cuda,cudnn]` (with the CUDA 13 and cuDNN libraries) | ONNX inference on an NVIDIA GPU. Python 3.11+, about 2.5 GB. |
| `pull` | `huggingface_hub` | `prism pull` from Hugging Face |
| `convert` | `torch`, `transformers`, `onnx-ir`, `safetensors` (torch alone is GBs) | `prism convert`. Also needs `onnxruntime-genai`: combine with `cuda`, or install it for CPU. |
| `dev` | `pytest` | development |
| `docs` | `mkdocs-material` | building this site |

Without the `cuda` extra you can still use the CPU build of `onnxruntime-genai`, or only Ollama models.

### From source

```bash
git clone https://github.com/senssei/prism-local && cd prism-local
pip install -e ".[cuda,pull]"
```

### Running from a checkout without installing

`./bin/prism …` works with no install. It picks its Python from `$PRISM_PYTHON`, then `./.venv`, then the active virtualenv,
then `python3` on `PATH`. (`bin/fng` and `bin/foundry-ng` are deprecated aliases.)

## Verify

```bash
prism doctor
```

`doctor` reports NVML (the GPU driver), whether `onnxruntime-genai` is importable, **whether ONNX Runtime's CUDA provider
actually loads**, and whether Ollama is reachable. Fix anything marked ❌ before expecting GPU speed; see
[Devices & CUDA](devices.md).

## Configuration

| Variable | Purpose | Default |
|---|---|---|
| `PRISM_MODEL_DIRS` | `:`-separated model directories. The first one is where `prism pull` writes. | `~/.prism/models` |
| `PRISM_DEVICE` | `auto`, `cuda` or `cpu` | `auto` |
| `PRISM_PREFILL_CHUNK` | Prompt tokens processed per step (a positive integer; `0` or `off` processes the whole prompt at once); bounds GPU memory on long prompts, see [Devices & CUDA](devices.md#gpu-memory-and-long-prompts) | `1024` |
| `PRISM_LOOP_GUARD` | `off` or `0` lets an ONNX model that is stuck repeating a short token cycle (at most 64 tokens long, over at least 200 tokens and 12 repeats) run on to `max_tokens`; by default it is stopped, logged, and reported as `finish_reason: length` | on |
| `PRISM_TEMPLATE` | `auto`, `jinja` or `builtin`: whether to render a model's own Jinja chat template (needs the `jinja` extra); see [Models](models.md#rendering-the-template-itself-optional) | `auto` |
| `PRISM_API_KEY` | Bearer token for `prism serve`; also sent by the MCP client and connector probe | unset (no auth) |
| `OLLAMA_HOST` | Ollama daemon address (`host`, `host:port` or a URL), as in Ollama itself | `http://localhost:11434` |
| `PRISM_BASE_URL` | Server URL used by `prism mcp` | `http://localhost:5272/v1` |
| `PRISM_QUEUE_TIMEOUT` | Seconds a request may wait for the model before `503` (`0` = forever); same as `prism serve --queue-timeout` | `300` |
| `PRISM_MAX_QUEUE` | Maximum requests allowed to wait for the engine before immediate `503 server_busy` (`0` = unlimited); same as `prism serve --max-queue` | `8` |
| `PRISM_RAM_RESERVE_MB` | Host RAM in MB kept free when checking if a model can safely load | `2048` |
| `PRISM_VRAM_RESERVE_MB` | GPU VRAM in MB kept free when checking if a CUDA model can safely load | `1536` |
| `PRISM_RESOURCE_CHECK` | `off` or `0` disables the pre-load RAM and VRAM capacity guard | on |
| `PRISM_LOAD_LOCK` | `off` or `0` disables cross-process model load serialization (`load.lock`) | on |
| `PRISM_LOAD_TIMEOUT` | Seconds a process waits for the cross-process model load lock before failing | `120` |
| `PRISM_STATE_DIR` | Directory for state and runtime locks (`load.lock`) | `~/.prism` |
| `PRISM_PYTHON` | Interpreter used by `bin/prism` | see above |

Models are searched in `$PRISM_MODEL_DIRS`, then `~/.prism/models`, then the Foundry Local cache
(`~/.foundry/cache/models`).

!!! note "Upgrading from a pre-release checkout"
    Earlier snapshots also scanned `./models` and a sibling `../02-ollama-loadtest` checkout, and `bin/prism` used that
    checkout's virtualenv. Set `PRISM_MODEL_DIRS` and `PRISM_PYTHON` to keep using them.

## First run

```bash
prism pull phi-4-mini                      # a few GB; picks the CUDA variant if an NVIDIA GPU is detected
prism list
prism run phi-4-mini "Explain quicksort in two sentences."
prism chat phi-4-mini
prism serve
```
