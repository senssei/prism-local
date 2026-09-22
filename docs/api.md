# REST API

`prism serve` exposes an OpenAI-compatible API at `http://127.0.0.1:5272/v1` (fixed port, configurable with `--port`).

```bash
curl -s http://127.0.0.1:5272/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "phi-4-mini",
  "messages": [{"role": "user", "content": "Write hello world in Rust."}],
  "max_tokens": 100
}'
```

=== "OpenAI Python client"

    ```python
    from openai import OpenAI

    client = OpenAI(base_url="http://127.0.0.1:5272/v1", api_key="anything")  # the real key if --api-key is set
    resp = client.chat.completions.create(
        model="phi-4-mini",
        messages=[{"role": "user", "content": "Count from 1 to 5."}],
        stream=True,
    )
    for chunk in resp:
        print(chunk.choices[0].delta.content or "", end="")
    ```

=== "curl (streaming)"

    ```bash
    curl -sN http://127.0.0.1:5272/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"model": "phi-4-mini", "stream": true, "messages": [{"role": "user", "content": "Hi"}]}'
    ```

## Endpoints

| Method and path | Purpose |
|---|---|
| `GET /v1/models` | ONNX and Ollama models. Extra fields: `size_mb`, `owned_by` (engine), `device` (where the model will run: `CPU` or `CUDA (GPU)`, following `--device` and the hardware), and for ONNX models `exported_for` (what the model files were built for; a `generic-cpu` variant still runs on CUDA under `--device auto`). |
| `POST /v1/chat/completions` | Chat completion, streaming or not |
| `POST /v1/embeddings` | Embeddings, **Ollama models only** (see [Embeddings](#embeddings)) |
| `POST /v1/completions` | Legacy text completion. ONNX models get the raw prompt (no chat template). Ollama models get it as one user message. |
| `POST /v1/unload` | Drop the ONNX model held by the server, freeing its VRAM/RAM. Idempotent; see [Unload](#unload). |
| `POST /v1/drain` | Finish the in-flight request, unload the model, and exit the server with status 0. Lets an orchestrator replace the running `prism serve` with a different model on the same GPU; see [Drain](#drain). |
| `GET /health` (also `/v1/health`, `/v1/status`) | Status, `active_model`, `active_device` (`cuda`/`cpu`/`null` when nothing is loaded) and GPU telemetry. **No auth required.** |

Trailing slashes are accepted.

## Request fields

| Field | Notes |
|---|---|
| `model` | Required. Same resolution rules as the [CLI](cli.md). `ollama:` names, and installed Ollama names, route to Ollama. |
| `messages` | Required, non-empty list. The chat template is chosen per model ([Models](models.md#chat-templates)). |
| `stream` | `true` for server-sent events |
| `stream_options.include_usage` | With `stream`, add a last chunk that carries `usage` (and, for ONNX models, `telemetry`); see Responses. Ollama models only when the daemon reports counts |
| `stop` | A string or a list of up to 4 strings. Generation ends at the first match, which is not included in the text; `finish_reason` is `stop` |
| `max_tokens` / `max_completion_tokens` | Default 512, must be ≥ 1. For ONNX models whose `genai_config.json` states a `context_length`, it is capped to the room left after the prompt (`finish_reason` is then `length`); a prompt that fills the window is a `400` `context_length_exceeded` |
| `tools` | OpenAI function tools (`[{"type": "function", "function": {"name", "description", "parameters"}}]`); see [Tool calling](#tool-calling) |
| `tool_choice` | `"none"` hides the tools for this request. Any other value is treated as `"auto"`: Prism cannot force a call |
| `temperature` | Default 0.1; `0` means greedy decoding |
| `top_p` | Default 0.9 |
| `top_k` | Integer ≥ 1 (an extension to the OpenAI API). ONNX: applies when sampling, default the model's `search.top_k` from `genai_config.json` when that is above 1, else 40. Ollama: passed on |
| `repetition_penalty` | Number > 0, 1.0 = off (an extension). ONNX: a multiplier on the logits of every token already in the context, prompt included, so values much above 1.05 can wreck the output; it does not stop a model that loops. Ollama: sent as `repeat_penalty` |
| `frequency_penalty`, `presence_penalty` | −2 to 2. Sent to Ollama. ONNX Runtime GenAI has neither, so a non-zero value is a `400` `unsupported_parameter` on ONNX models (`0` is accepted, clients send it by default) |

Message `content` may be a string or a list of parts; text parts are used and others (images, …) are dropped. Other OpenAI fields (`n`, `response_format`, …) are **accepted and ignored**.

## Responses

Non-streaming responses follow the OpenAI shape. `finish_reason` is `stop` (end of sequence) or `length` (hit `max_tokens`, or, on ONNX models, the model was stopped because it kept repeating the same token cycle; see [`PRISM_LOOP_GUARD`](getting-started.md#configuration)),
and `usage` uses the model tokenizer. When reasoning models (e.g. Qwen 2.5/3, DeepSeek-R1) generate `<think>...</think>` blocks, the thinking is separated into `message.reasoning_content` and stripped from `message.content`. Ollama responses carry the `finish_reason` and `usage` the daemon reports (none if it reports none). ONNX responses add a Prism extension:

```json
"telemetry": { "ttft_sec": 0.45, "decode_tok_per_sec": 85.2, "device": "cuda" }
```

Streaming sends `chat.completion.chunk` events: a first chunk with `delta: {"role": "assistant"}`, deltas (during thinking, `delta: {"reasoning_content": "..."}`; afterwards, `delta: {"content": "..."}`), a final chunk
with the `finish_reason`, then `data: [DONE]`. With `stream_options: {"include_usage": true}` one more chunk with an empty `choices` list
comes after the `finish_reason` chunk and before `[DONE]`, carrying `usage` (`prompt_tokens`, `completion_tokens`, `total_tokens`, from the model
tokenizer) and the same `telemetry` block as non-streaming responses; without the option the stream is unchanged. If generation fails after streaming has begun, an event
`data: {"error": {...}}` is sent before `[DONE]`. If the client disconnects, generation stops.

## Tool calling

Send `tools` as you would to OpenAI. When the model calls one, the reply carries `message.tool_calls` (`id`, `type: "function"`, `function.name`, and
`function.arguments` as a JSON **string**), `content` holds any text that came before the call (or is `null`), and `finish_reason` is `tool_calls`. For reasoning models, any `<think>...</think>` block preceding the tool call is extracted into `message.reasoning_content` (or streamed in a `delta: {"reasoning_content": ...}` chunk) rather than leaked into `content`.
Send the result back as a `{"role": "tool", "tool_call_id": ..., "content": ...}` message after the assistant message that made the call.

- **ONNX models** need the model's own chat template to take tools, and Prism renders that template only with the optional
  `jinja` extra (`pip install "prism-local[jinja]"`, see [Models](models.md#rendering-the-template-itself-optional)). Without both, a request with `tools` gets
  `400 tools_not_supported` instead of a silent non-answer. The model writes its calls as text in its own convention (`<tool_call>…`, `<|tool_call|>…`,
  `[TOOL_CALLS]…`, `<|python_tag|>…`, or a bare JSON object); Prism recognises them in the output. Those markers are special tokens that ONNX Runtime GenAI decodes
  to nothing, so the engine puts them back.
- **Streaming** with `tools` on an ONNX model is **buffered**: the calls can only be recognised in the finished text, so the reply arrives at the end, as a reasoning
  chunk (if present), a content chunk and/or one `tool_calls` delta, then the `finish_reason` chunk. Without `tools` streaming is token by token as before.
- **Ollama models** get `tools` passed to the daemon, which parses the calls itself (use a model that supports tools, such as `llama3.1` or `qwen2.5`).
- Whether a call is *right* depends on the model; small models are unreliable. Prism guarantees the shape of the reply, not the choice of tool.

## Embeddings

`POST /v1/embeddings` follows OpenAI's shape and is served by Ollama, because ONNX Runtime GenAI does not produce embeddings. Use an Ollama embedding model:

```bash
prism pull ollama:nomic-embed-text
curl -s http://127.0.0.1:5272/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"model": "ollama:nomic-embed-text", "input": ["first text", "second text"]}'
```

| Field | Notes |
|---|---|
| `model` | An Ollama model (`ollama:` prefix or an installed name). An ONNX model gets `400 embeddings_not_supported` |
| `input` | A string, or a list of up to 256 non-empty strings. Token arrays are not supported |
| `encoding_format` | `float` (default) or `base64` (little-endian float32; the official OpenAI Python client asks for it) |
| `dimensions` | Not supported (`400 dimensions_not_supported`) |

The reply is `{"object": "list", "data": [{"object": "embedding", "index", "embedding"}], "model", "usage": {"prompt_tokens", "total_tokens"}}`. A missing or
unreachable daemon is `502 backend_unavailable`; a model Ollama does not have is `404 model_not_found`.

## Unload

`POST /v1/unload` releases the ONNX model currently held by the server. Useful between benchmark runs of different large
models (so each one starts from cold VRAM) and when switching from a big ONNX model to a small one. The endpoint is
idempotent: a second call when nothing is loaded still returns `200`.

```bash
curl -s -X POST http://127.0.0.1:5272/v1/unload \
  -H 'Authorization: Bearer YOUR_KEY'      # only when --api-key is set
```

The body is optional (`{}` if absent) and any JSON object is accepted and ignored. The reply is:

```json
{"unloaded": true, "model": "phi-4-mini"}
```

`unloaded` is `true` when a model was resident and is now released, `false` otherwise; `model` is the id of the released model
(`null` when nothing was loaded). `manager.unload()` acquires the engine lock, so if a generation is in flight the unload
waits for it to finish — the HTTP call is synchronous and may take seconds. Ollama models are not loaded into the server
process and are unaffected. Loading a different model after `unload()` re-runs the resource-budget check ([Concurrency](#concurrency)).

## Drain

`POST /v1/drain` tells the running `prism serve` to finish its current request, unload the model, and exit with status
0. Use it from a benchmark or evaluation orchestrator that needs a different model on the same GPU: drain the active
server, start a fresh one, swap back when done. The engine lock waits for the in-flight generation to complete; the
HTTP response is sent first; the process exits through `os._exit(0)` ~250 ms later so the response can flush.

```bash
curl -s -X POST http://127.0.0.1:5272/v1/drain \
  -H 'Authorization: Bearer YOUR_KEY'      # only when --api-key is set
```

The body is optional (`{}` if absent) and any JSON object is accepted and ignored (reserved for a future
`timeout_ms`). A non-empty body that is not a JSON object returns `400`. The reply is:

```json
{"drained": true, "model": "phi-4-mini", "exit_in_ms": 250}
```

`drained` is `true` when a model was resident and is now released, `false` otherwise; `model` is the id of the released
model (`null` when nothing was loaded); `exit_in_ms` is the delay before `os._exit(0)`. After `drained: true` the
process is on its way out — do not issue further requests to it.

`POST /v1/drain` is the orchestrator's partner to the `error.holder` field on `503 insufficient_resources`. When the
orchestrator sees that field, it knows which running `prism serve` is holding VRAM; `POST /v1/drain` to that
process frees the GPU so the orchestrator can start its own server with the larger model.

## Errors

All errors are JSON: `{"error": {"message", "type", "param", "code"}}`.

| Status | `code` | Meaning |
|---|---|---|
| 400 | *(none)* | Invalid JSON, missing/invalid `messages`, `model`, or a numeric field; `/v1/unload` body that is not a JSON object |
| 400 | `embeddings_not_supported` / `dimensions_not_supported` | `/v1/embeddings` with an ONNX model / with `dimensions` |
| 400 | `tools_not_supported` | `tools` sent to an ONNX model whose chat template cannot take them, or without jinja2 installed |
| 400 | `template_render_failed` | The model's chat template raised while rendering with the provided `tools`. The message names the model id and the underlying jinja/template exception text so the caller can identify the field that tripped the template; Prism does not silently drop tools from the prompt ([spec.md P11](sdlc/spec.md)) |
| 400 | `ambiguous_model` | The name matches several models; the message lists them |
| 400 | `context_length_exceeded` | The prompt fills the model's context window (ONNX models with a known `context_length`) |
| 401 | `invalid_api_key` | Missing or wrong bearer token |
| 403 | `host_not_allowed` | Non-loopback `Host` header on a loopback bind |
| 404 | `model_not_found` / `not_found` | Unknown model / unknown route |
| 413 | `body_too_large` | Body over 10 MB |
| 503 | `insufficient_resources` | Host RAM or GPU VRAM is insufficient to load the model safely; carries `Retry-After: 30`. When the model-load lock is held by another process, `error.holder` (`{"pid", "model"}`) names it; see [Drain](#drain) for the orchestrator pattern |
| 503 | `server_busy` | The model stayed busy with other requests longer than `--queue-timeout`, or waiting queue exceeded `--max-queue` (default 8); carries `Retry-After: 30` |
| 500 | `model_load_failed` | The ONNX model could not be loaded, for example `--device cuda` with a broken CUDA setup |
| 502 | `backend_unavailable` | Ollama is unreachable or failed |

## Concurrency

One ONNX model is resident at a time, and generation is **serialized behind a lock**: concurrent requests queue, for at most `--queue-timeout` seconds (default 300), after which they get `503 server_busy`. If the number of waiting requests exceeds `--max-queue` (default 8), new requests get `503 server_busy` immediately. If available memory is below safety thresholds, loading is refused with `503 insufficient_resources`. Requesting a
different ONNX model unloads the current one and loads the new one, which takes seconds. Ollama requests are not serialized by
Prism.

A request that closes its HTTP connection while it is queued behind another caller (waiting for the engine lock) is removed from the queue immediately and never produces a response; the slot it held is freed for the next caller, and the holder is unaffected. The same handling applies to both `/v1/chat/completions` and `/v1/completions` (they share the generation path). Disconnects mid-generation are still detected and stop the generation, as before.

## Authentication and CORS

See [Security](security.md). With `--api-key`, every endpoint except `/health` requires `Authorization: Bearer <key>`.
