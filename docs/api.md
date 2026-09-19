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
| `GET /v1/models` | ONNX and Ollama models. Extra fields: `size_mb`, `device` (a hint), `owned_by` (engine). |
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
| `max_tokens` / `max_completion_tokens` | Default 512, must be ≥ 1 |
| `temperature` | Default 0.1; `0` means greedy decoding |
| `top_p` | Default 0.9 |

Other OpenAI fields (`stop`, `n`, `tools`, `response_format`, …) are **accepted and ignored**.

## Responses

Non-streaming responses follow the OpenAI shape. `finish_reason` is `stop` (end of sequence) or `length` (hit `max_tokens`),
and `usage` uses the model tokenizer. Ollama responses have no `usage`. ONNX responses add a Prism extension:

```json
"telemetry": { "ttft_sec": 0.06, "decode_tok_per_sec": 118.6, "device": "cuda" }
```

Streaming sends `chat.completion.chunk` events: a first chunk with `delta: {"role": "assistant"}`, content deltas, a final chunk
with the `finish_reason`, then `data: [DONE]`. If generation fails after streaming has begun, an event
`data: {"error": {...}}` is sent before `[DONE]`. If the client disconnects, generation stops.

## Errors

All errors are JSON: `{"error": {"message", "type", "param", "code"}}`.

| Status | `code` | Meaning |
|---|---|---|
| 400 | *(none)* | Invalid JSON, missing/invalid `messages`, `model`, or a numeric field |
| 400 | `ambiguous_model` | The name matches several models; the message lists them |
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
