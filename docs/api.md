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
| `POST /v1/completions` | Legacy text completion. ONNX models get the raw prompt (no chat template). Ollama models get it as one user message. |
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
| `temperature` | Default 0.1; `0` means greedy decoding |
| `top_p` | Default 0.9 |

Message `content` may be a string or a list of parts; text parts are used and others (images, …) are dropped. Other OpenAI fields (`n`, `tools`, `response_format`, …) are **accepted and ignored**.

## Responses

Non-streaming responses follow the OpenAI shape. `finish_reason` is `stop` (end of sequence) or `length` (hit `max_tokens`),
and `usage` uses the model tokenizer. Ollama responses carry the `finish_reason` and `usage` the daemon reports (none if it reports none). ONNX responses add a Prism extension:

```json
"telemetry": { "ttft_sec": 0.45, "decode_tok_per_sec": 85.2, "device": "cuda" }
```

Streaming sends `chat.completion.chunk` events: a first chunk with `delta: {"role": "assistant"}`, content deltas, a final chunk
with the `finish_reason`, then `data: [DONE]`. With `stream_options: {"include_usage": true}` one more chunk with an empty `choices` list
comes after the `finish_reason` chunk and before `[DONE]`, carrying `usage` (`prompt_tokens`, `completion_tokens`, `total_tokens`, from the model
tokenizer) and the same `telemetry` block as non-streaming responses; without the option the stream is unchanged. If generation fails after streaming has begun, an event
`data: {"error": {...}}` is sent before `[DONE]`. If the client disconnects, generation stops.

## Errors

All errors are JSON: `{"error": {"message", "type", "param", "code"}}`.

| Status | `code` | Meaning |
|---|---|---|
| 400 | *(none)* | Invalid JSON, missing/invalid `messages`, `model`, or a numeric field |
| 400 | `ambiguous_model` | The name matches several models; the message lists them |
| 400 | `context_length_exceeded` | The prompt fills the model's context window (ONNX models with a known `context_length`) |
| 401 | `invalid_api_key` | Missing or wrong bearer token |
| 403 | `host_not_allowed` | Non-loopback `Host` header on a loopback bind |
| 404 | `model_not_found` / `not_found` | Unknown model / unknown route |
| 413 | `body_too_large` | Body over 10 MB |
| 500 | `model_load_failed` | The ONNX model could not be loaded, for example `--device cuda` with a broken CUDA setup |
| 502 | `backend_unavailable` | Ollama is unreachable or failed |

## Concurrency

One ONNX model is resident at a time, and generation is **serialized behind a lock**: concurrent requests queue. Requesting a
different ONNX model unloads the current one and loads the new one, which takes seconds. Ollama requests are not serialized by
Prism.

## Authentication and CORS

See [Security](security.md). With `--api-key`, every endpoint except `/health` requires `Authorization: Bearer <key>`.
