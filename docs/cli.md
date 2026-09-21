# CLI reference

```text
prism [-h] {status,doctor,list,pull,run,chat,serve,benchmark,mcp,connect} ...
```

Model arguments accept an id, a name, a path to a model folder, or a **unique** substring (`prism run phi-4-mini …`).
An ambiguous substring is an error that lists the candidates. Ollama models use the `ollama:` prefix
(`ollama:qwen2.5-coder:7b`), and installed Ollama models also work without it.

## Device selection

`run`, `chat`, `serve` and `benchmark` accept **`--device {auto,cuda,cpu}`** (default `$PRISM_DEVICE`, else `auto`):

| Value | Behavior |
|---|---|
| `auto` | Use CUDA if an NVIDIA GPU is detected **and** the CUDA provider loads; otherwise run on CPU and print a warning with the reason. |
| `cuda` | Use CUDA or fail with an actionable error. Never falls back. |
| `cpu` | Force the CPU provider. |

See [Devices & CUDA](devices.md).

## `prism status`

GPU name, compute capability and VRAM (via NVML), system RAM and swap usage (via `/proc/meminfo`), whether `onnxruntime-genai` and Ollama are available, and the number of
discovered ONNX models.

## `prism doctor`

Checks the NVML driver, `onnxruntime-genai`, whether ONNX Runtime's **CUDA provider can be loaded** (naming the missing
library if not), the Ollama daemon, model conversion dependencies, and WSL2 host `.wslconfig` memory limits.

## `prism list`

Lists ONNX models (from the [search paths](getting-started.md#configuration)) and installed Ollama models with engine, size and
a device hint. The hint comes from the model's name and `genai_config.json`; the device that actually runs is chosen at load
time and reported by `run`, `benchmark` and `/health`.

## `prism pull MODEL`

```bash
prism pull phi-4-mini                       # curated alias; CUDA variant if a GPU is detected
prism pull phi-4-mini --ep cpu              # force the CPU variant
prism pull microsoft/Phi-4-mini-instruct-onnx   # any Hugging Face repo; one folder is chosen from a multi-variant repo
prism pull microsoft/mistral-7b-instruct-v0.2-ONNX --variant cuda-fp16   # pick the folder yourself
prism pull ollama:qwen2.5-coder:7b          # via the Ollama daemon
prism pull deepseek-r1:14b --backend ollama
```

| Option | Meaning |
|---|---|
| `--output-dir DIR` | Destination (default: first `$PRISM_MODEL_DIRS` entry, else `~/.prism/models`) |
| `--ep {cuda,cpu}` | Which variant to download, for an alias or a multi-variant repo (default: detected from the GPU) |
| `--variant TEXT` | For a repo with several model folders: the one whose path contains TEXT. By default the folder is chosen from `--ep` and `--quant`; if that is ambiguous nothing is downloaded and the candidates are listed |
| `--backend {auto,onnx,ollama}` | Force the engine |
| `--quant {int4,fp16}` | Which quantization to prefer in a multi-variant repo (default `int4`); an alias's folder is fixed |

After a Hugging Face download, Prism checks for `genai_config.json` and `*.onnx` weights and **fails** if either is missing,
because such a folder cannot be loaded. See [Models](models.md).

## `prism convert MODEL`

Converts and quantizes a Hugging Face model (or a local model folder) to an ONNX Runtime GenAI folder with onnxruntime-genai's model builder and installs it
next to pulled models. Needs the optional `convert` extra (heavy: it imports torch and transformers) and `onnxruntime-genai`.

```bash
pip install "prism-local[cuda,convert]"
prism convert Qwen/Qwen2.5-0.5B-Instruct                 # int4, for CUDA if a GPU is detected, else CPU
prism convert Qwen/Qwen2.5-0.5B-Instruct --ep cpu
prism convert ./my-hf-model --ep cuda --quant fp16 --name my-model-fp16   # a local Hugging Face folder
```

| Option | Meaning |
|---|---|
| `--ep {cuda,cpu}` | Target execution provider (default: detected from the GPU) |
| `--quant {int4,fp16}` | Precision (default `int4`); `fp16` needs `--ep cuda` |
| `--output-dir DIR` | Destination (default: first `$PRISM_MODEL_DIRS` entry, else `~/.prism/models`) |
| `--name NAME` | Installed folder name (default `<model>-<ep>-<quant>`) |
| `--force` | Replace the folder if it exists (an existing model is kept if the run fails) |
| `--trust-remote-code` | Let Hugging Face run the model's own code while loading it |

The result is checked like a pulled model (`genai_config.json` and `*.onnx`), then `prism run <name>` works. See [Models](models.md#converting-your-own-models).

## `prism run MODEL [PROMPT]`

One-shot generation, streamed to stdout. The device in use is printed to stderr (`[device: cpu]`). With no prompt, it reads
stdin if piped; on a terminal it starts `chat`.

| Option | Meaning |
|---|---|
| `--max-tokens N` | Generation limit (default 512) |
| `--device …` | See above |

## `prism chat MODEL`

Interactive streaming chat with multi-turn history. `/clear` resets the context; `/exit`, `exit`, `quit` or Ctrl+C leaves.
Shows tokens, tok/s and elapsed time per reply, and the device the model loaded on.

## `prism serve`

Starts the [OpenAI-compatible server](api.md).

| Option | Default | Meaning |
|---|---|---|
| `--port` | `5272` | Port |
| `--host` | `127.0.0.1` | Interface. Anything non-loopback exposes the server; combine with `--api-key`. |
| `--api-key KEY` | `$PRISM_API_KEY` | Require `Authorization: Bearer KEY` |
| `--cors-origin ORIGIN` | none | Allow a browser origin (repeatable, or `*`); CORS is off by default |
| `--queue-timeout SEC` | `$PRISM_QUEUE_TIMEOUT`, else `300` | How long a request may wait for the model before it gets `503 server_busy`; `0` waits forever |
| `--max-queue N` | `$PRISM_MAX_QUEUE`, else `8` | Maximum waiting requests before immediate `503 server_busy`; `0` is unlimited |
| `--device …` | `auto` | See above |

## `prism benchmark MODEL`

Loads the model and reports the **execution provider used**, load time, TTFT, decode tokens/s over 256 tokens, and VRAM:
the amount the model added, and what remained after unload (both relative to the VRAM in use before loading, because other
applications share the GPU). If the model did not run on the GPU, it says so and why.

## `prism mcp`

Runs Prism as a stdio [Model Context Protocol](integrations.md#mcp-server) server. Normally launched by an MCP client.

## `prism connect {cursor,cline,mcp}`

Prints setup instructions and writes client configuration. See [Integrations](integrations.md).
