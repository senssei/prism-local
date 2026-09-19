# Integrations

Start the server first (`prism serve`), or let the MCP server load ONNX models directly when the server is offline.

## Cursor

```bash
prism connect cursor --test --export-rules --export-mcp
```

- Prints the settings for Cursor's **OpenAI-compatible provider**: base URL `http://localhost:5272/v1`, any API key
  (the real one if you use `--api-key`), and the model id.
- `--test` checks that the server is reachable.
- `--export-rules` writes `.cursorrules` in the current directory; `--export-mcp` writes `.cursor/mcp.json`.
- `--model ID` chooses the model shown in the instructions.

## Cline

```bash
prism connect cline --test --export-mcp
```

Prints the Cline provider settings (API Provider: *OpenAI Compatible*) and, with `--export-mcp`, writes
(merging into) `cline_mcp_settings.json` in the current directory. The generated MCP entry **auto-approves all five Prism tools**; remove
entries from `autoApprove` if you want to confirm each call.

## MCP server

`prism mcp` speaks JSON-RPC 2.0 over stdio (protocol `2024-11-05`) and exposes:

| Tool | Arguments | Purpose |
|---|---|---|
| `prism_ask_coder` | `task` (required), `context_code`, `model` | Code generation, debugging and tests with a local model |
| `prism_code_review` | `code` (required), `focus`, `model` | Review for security, concurrency and edge cases |
| `prism_list_models` | none | Local ONNX and Ollama models |
| `prism_get_status` | none | GPU telemetry and server status |
| `prism_benchmark` | `model` (required) | Latency and throughput micro-benchmark |

The generation tools call the Prism server at `$PRISM_BASE_URL` (default `http://localhost:5272/v1`, with
`$PRISM_API_KEY` if set). If the server is offline and the model is an ONNX model, they load it in-process instead.
None of the tools reads or writes your files.

```bash
prism connect mcp --test                        # run a protocol handshake against `prism mcp`
prism connect mcp --target claude --write       # Claude Desktop
prism connect mcp --target cursor --write       # .cursor/mcp.json
```

| `--target` | Config file written |
|---|---|
| `claude` | `~/.config/Claude/claude_desktop_config.json` |
| `cursor` | `.cursor/mcp.json` (current directory) |
| `cline` | `cline_mcp_settings.json` (current directory) |
| `antigravity` | `~/.gemini/config/mcp_config.json` |
| `all` (default) | all of the above |

Without `--write`, the snippet is only printed. With `--write`, an existing config is merged (your other servers are kept) and
a `.bak` copy is saved first; a file that is not valid JSON is left untouched. The generated entries assume the server is on
the default port `5272`.
