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
| `PRISM_PREFILL_CHUNK` | Prompt tokens processed per step (a positive integer, e.g. `256`); bounds GPU memory on long prompts, see [Devices & CUDA](devices.md#gpu-memory-and-long-prompts) | unset (whole prompt at once) |
| `PRISM_API_KEY` | Bearer token for `prism serve`; also sent by the MCP client and connector probe | unset (no auth) |
| `PRISM_BASE_URL` | Server URL used by `prism mcp` | `http://localhost:5272/v1` |
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
