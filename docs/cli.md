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

GPU name, compute capability and VRAM (via NVML), whether `onnxruntime-genai` and Ollama are available, and the number of
discovered ONNX models.

## `prism doctor`

Checks the NVML driver, `onnxruntime-genai`, whether ONNX Runtime's **CUDA provider can be loaded** (naming the missing
library if not), and the Ollama daemon.

## `prism list`

Lists ONNX models (from the [search paths](getting-started.md#configuration)) and installed Ollama models with engine, size and
a device hint. The hint comes from the model's name and `genai_config.json`; the device that actually runs is chosen at load
time and reported by `run`, `benchmark` and `/health`.

## `prism pull MODEL`

```bash
prism pull phi-4-mini                       # curated alias; CUDA variant if a GPU is detected
prism pull phi-4-mini --ep cpu              # force the CPU variant
prism pull microsoft/Phi-4-mini-instruct-onnx   # any Hugging Face repo
prism pull ollama:qwen2.5-coder:7b          # via the Ollama daemon
prism pull deepseek-r1:14b --backend ollama
```

| Option | Meaning |
|---|---|
| `--output-dir DIR` | Destination (default: first `$PRISM_MODEL_DIRS` entry, else `~/.prism/models`) |
| `--ep {cuda,cpu}` | Which variant of a curated alias to download (default: detected from the GPU) |
| `--backend {auto,onnx,ollama}` | Force the engine |
| `--quant {int4,fp16}` | Informational for now; the variant is fixed by the alias's folder |

After a Hugging Face download, Prism checks for `genai_config.json` and `*.onnx` weights and **fails** if either is missing,
because such a folder cannot be loaded. See [Models](models.md).

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
| `--device …` | `auto` | See above |

## `prism benchmark MODEL`

Loads the model and reports the **execution provider used**, load time, TTFT, decode tokens/s over 256 tokens, and VRAM:
the amount the model added, and what remained after unload (both relative to the VRAM in use before loading, because other
applications share the GPU). If the model did not run on the GPU, it says so and why.

## `prism mcp`

Runs Prism as a stdio [Model Context Protocol](integrations.md#mcp-server) server. Normally launched by an MCP client.

## `prism connect {cursor,cline,mcp}`

Prints setup instructions and writes client configuration. See [Integrations](integrations.md).
