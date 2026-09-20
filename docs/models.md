# Models

## Where models come from

Prism discovers ONNX Runtime GenAI model folders (containing `genai_config.json`) under, in order:

1. each directory in `$PRISM_MODEL_DIRS`
2. `~/.prism/models`
3. the Foundry Local cache (`~/.foundry/cache/models`)

Installed **Ollama** models (GGUF) are listed too when the Ollama daemon is reachable at `localhost:11434` (or `$OLLAMA_HOST`).
Discovery results are cached for 5 seconds.

## Name resolution

`prism run`, `serve` requests, and friends resolve a model name like this:

1. `ollama:<name>` goes straight to Ollama.
2. An exact match on a model id, name or folder path.
3. An exact match on an installed Ollama model name.
4. A **curated alias** (`phi-4-mini`, `phi-4`, `mistral-7b-instruct-v0.2`, …) resolves to the installed variant that suits the machine: the
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
| `phi-4-mini-reasoning` | `microsoft/Phi-4-mini-reasoning-onnx` | `Phi-4-mini-reasoning-cuda-gpu` / `-generic-cpu` |
| `phi-4-reasoning` | `microsoft/Phi-4-reasoning-onnx` | `Phi-4-reasoning-cuda-gpu` / `-generic-cpu` |
| `phi-3-mini-4k` | `microsoft/Phi-3-mini-4k-instruct-onnx` | `Phi-3-mini-4k-instruct-cuda-gpu` / `-generic-cpu` |
| `phi-3.5-mini` | `microsoft/Phi-3.5-mini-instruct-onnx` | `Phi-3.5-mini-instruct-cuda-gpu` / `-generic-cpu` |
| `mistral-7b-instruct-v0.2` | `microsoft/mistral-7b-instruct-v0.2-ONNX` | `mistral-7b-instruct-v0.2-cuda-int4-rtn-block-32` / `…-cpu-int4-rtn-block-32-acc-level-4` |

The variant is chosen by `--ep`, or by whether an NVIDIA GPU is detected.

### Any other repository

`prism pull owner/repo` works for any ONNX Runtime GenAI repository. Microsoft's repos keep several complete models side by side in
subfolders (`cuda/…`, `cpu_and_mobile/…`, `directml/…`, several quantizations), so Prism lists the repo first and downloads **one** folder:
the one for `--ep` (CUDA if a GPU is detected), then `--quant`, preferring Microsoft's `acc-level-4` CPU builds, and never a DirectML, NPU or web build
while anything else fits. If that still leaves several, nothing is downloaded and the candidates are listed; pick one with `--variant TEXT`
(a substring of the folder path). The model is installed under the folder's own name, for example
`mistral-7b-instruct-v0.2-cuda-int4-rtn-block-32`. A repo with a single model at its root is downloaded whole, and so is any repo whose file list
cannot be fetched.

```bash
prism pull microsoft/mistral-7b-instruct-v0.2-ONNX                      # the int4 build for this machine
prism pull microsoft/mistral-7b-instruct-v0.2-ONNX --variant cuda-fp16  # a specific folder
```

!!! note "Re-verifying aliases"
    `PYTHONPATH=. python3 scripts/verify_aliases.py` checks each alias against Hugging Face (needs network): every variant must contain
    `genai_config.json`, an `.onnx` file and a tokenizer.

### Verification

After downloading, Prism requires `genai_config.json` and `*.onnx` weights. If either is missing (for example a
transformers.js-style ONNX export, which ONNX Runtime GenAI cannot load), `pull` reports failure instead of success.
The variant subfolder (for example `gpu/gpu-int4-rtn-block-32`) is flattened into the model folder and the empty nested folders are removed.

## Converting your own models

When no ready-made ONNX build exists, `prism convert` builds one from the original Hugging Face weights with ONNX Runtime GenAI's model builder
(`python -m onnxruntime_genai.models.builder`, the same builder Microsoft Olive runs for text models):

```bash
pip install "prism-local[cuda,convert]"
prism convert Qwen/Qwen2.5-0.5B-Instruct --ep cpu
prism run Qwen2.5-0.5B-Instruct-cpu-int4 "Hello"
```

- The builder runs as a subprocess of Prism's Python and its output is shown as it runs. It downloads the weights (into the Hugging Face cache, so a second run reuses them),
  exports and quantizes them; expect minutes and several GB of RAM. Building a CUDA model needs no GPU.
- The build goes to a staging folder first. Only a folder with `genai_config.json` and `*.onnx` weights is installed, under `<model>-<ep>-<quant>` (the name carries the
  provider, so `prism list` shows the right device); a failed or interrupted run leaves nothing behind, and never touches an existing model.
- Without a Hugging Face token the builder is told not to look for one (`hf_token=false`); a gated model needs `hf auth login` or `$HF_TOKEN`.
- Only architectures the builder supports work; others fail with its own error.
- Prism does not use Microsoft Olive: for CUDA and CPU text models it would only wrap this same builder. Olive matters for other providers (OpenVINO, QNN, …), which Prism cannot run,
  or for quantization algorithms the builder lacks; models it produces load in Prism like any other ONNX GenAI folder.

## Chat templates

ONNX GenAI models need the chat format applied by the caller. Prism picks one of its templates per model, in this order:

1. **The model's own chat template**: `chat_template.jinja`, `chat_template.json`, or the `chat_template` key of `tokenizer_config.json`.
   It is what the model was trained with, so it outranks the name. Prism does not run the Jinja; it recognises the family from the
   special tokens the template writes.
2. **The name** and the `model.type` in `genai_config.json`, when there is no template file or it is in a format Prism has no
   template for (Llama 2, for example).

| Family | Recognised by | Template |
|---|---|---|
| `phi4` | `<|end|>`; name `phi` | Phi-3 / 3.5 (`<|system|>\n` … `<|end|>\n`) |
| `phi4_mini` | `<|end|>` with role-built tags; name `phi-4-mini` | Phi-4-mini (`<|user|>` … `<|end|>`, no newlines) |
| `phi4_im` | `<|im_sep|>`; name `phi-4` | Phi-4 (`<|im_start|>user<|im_sep|>` … `<|im_end|>`) |
| `llama3` | `<|start_header_id|>`; name `llama` | Llama 3 |
| `deepseek` | `<｜User｜>` / `<｜Assistant｜>`; name `deepseek` | DeepSeek |
| `gemma` | `<start_of_turn>`; name `gemma` | Gemma (`<start_of_turn>user` … `<end_of_turn>`, role `model`); the system prompt leads the first user turn |
| `mistral_v02` | `[INST]` with ` [/INST]`; name `mistral`/`mixtral` with `v0.1`/`v0.2` | Mistral v0.1/v0.2 (`[INST] … [/INST]`) |
| `mistral` | `[INST]`; name `mistral`/`mixtral` | Mistral v0.3 and later (`[INST] …[/INST]`); the system prompt leads the last user message |
| `chatml` | `<|im_start|>`; anything else, including Qwen | ChatML |

### Rendering the template itself (optional)

With the optional `jinja` extra (`pip install "prism-local[jinja]"`), Prism does not stop at recognising the family: it **renders the model's own
Jinja chat template**, exactly as the Hugging Face tokenizer would. That reproduces details a fixed format cannot, for example the default system
prompt Qwen2.5 adds when you send none. A leading BOS token is dropped when the model's `tokenizer_config.json` says the tokenizer adds it itself.
If a template refuses the conversation (Gemma 2 and Mistral v0.2 reject a system role), Prism logs a warning and uses its built-in format, which
folds the system prompt into a user turn.

`$PRISM_TEMPLATE` selects the behaviour: `auto` (default: render when jinja2 is installed and the model has a template), `jinja` (the same, and
warns when jinja2 is missing) or `builtin` (never render; use the formats above). Without jinja2 everything works as described above.

Chat templates are code that comes from downloaded files, so they run in Jinja's immutable **sandbox**; see [Security](security.md).

The prompts for Gemma and Mistral leave out BOS (`<bos>`, `<s>`), which the model's tokenizer adds itself, as for Llama 3.

`GET /v1/models` does not show it, but the catalog keeps the choice as `template` and where it came from as `template_source`
(`chat_template` or `name`). A model whose template Prism does not know falls back to the name, then to ChatML, which can be
wrong for it. To support a new format, add a branch to `format_prompt` and a marker to `classify_chat_template` in
`prism/templates.py`; tests for both live in `tests/test_prism_templates.py`. Ollama models apply their own templates server-side.
