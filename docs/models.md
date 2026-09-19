# Models

## Where models come from

Prism discovers ONNX Runtime GenAI model folders (containing `genai_config.json`) under, in order:

1. each directory in `$PRISM_MODEL_DIRS`
2. `~/.prism/models`
3. the Foundry Local cache (`~/.foundry/cache/models`)

Installed **Ollama** models (GGUF) are listed too when the Ollama daemon is reachable at `localhost:11434`.
Discovery results are cached for 5 seconds.

## Name resolution

`prism run`, `serve` requests, and friends resolve a model name like this:

1. `ollama:<name>` goes straight to Ollama.
2. An exact match on a model id, name or folder path.
3. An exact match on an installed Ollama model name.
4. A **curated alias** (`phi-4-mini`, `phi-4`, `phi-3.5-mini`) resolves to the installed variant that suits the machine: the
   CUDA variant if an NVIDIA GPU is detected, otherwise the CPU variant, falling back to whichever is installed. This keeps
   `prism run phi-4-mini` working when both variants (or a Foundry-cache copy) exist.
5. A **unique** case-insensitive substring of an ONNX model name.

More than one substring match is an error that lists the candidates (HTTP 400 `ambiguous_model` from the server); no match
is "not found".

## Pulling models

```bash
prism pull phi-4-mini
prism pull owner/some-onnx-genai-repo
prism pull ollama:qwen2.5-coder:7b
```

### Curated aliases

Aliases are shortcuts to Hugging Face repos that were checked to contain a loadable `genai_config.json` for both variants:

| Alias | Repository | Installed as |
|---|---|---|
| `phi-4-mini` | `microsoft/Phi-4-mini-instruct-onnx` | `Phi-4-mini-instruct-cuda-gpu` / `-generic-cpu` |
| `phi-4` | `microsoft/phi-4-onnx` | `Phi-4-instruct-cuda-gpu` / `-generic-cpu` |
| `phi-3.5-mini` | `microsoft/Phi-3.5-mini-instruct-onnx` | `Phi-3.5-mini-instruct-cuda-gpu` / `-generic-cpu` |

The variant is chosen by `--ep`, or by whether an NVIDIA GPU is detected. Any other ONNX Runtime GenAI repository works
as `prism pull owner/repo` (for example a Qwen or Llama GenAI export); it is downloaded in full.

!!! note "Re-verifying aliases"
    `PYTHONPATH=. python3 scripts/verify_aliases.py` checks each alias against Hugging Face (needs network).

### Verification

After downloading, Prism requires `genai_config.json` and `*.onnx` weights. If either is missing (for example a
transformers.js-style ONNX export, which ONNX Runtime GenAI cannot load), `pull` reports failure instead of success.
The variant subfolder (for example `gpu/gpu-int4-rtn-block-32`) is flattened into the model folder and the empty nested folders are removed.

## Chat templates

ONNX GenAI models need the chat format applied by the caller. Prism picks a template per model from its name and the
`model.type` in `genai_config.json`:

| Family (match) | Template |
|---|---|
| `phi` | Phi (`<|system|>` … `<|end|>`) |
| `llama` | Llama 3 (`<|start_header_id|>` …) |
| `deepseek` | DeepSeek (`<｜User｜>` / `<｜Assistant｜>`) |
| anything else, including Qwen | ChatML (`<|im_start|>` …) |

The detected template is stored on each model (`prism/templates.py`). If a model family needs a different format, add a branch to
`format_prompt` and `detect_template` there; tests for both live in `tests/test_prism_templates.py`. Ollama models apply their own
templates server-side.
