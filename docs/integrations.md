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

## ACP (Zed)

`prism acp` speaks the [Agent Client Protocol](https://agentclientprotocol.com) over stdio, so an ACP-capable editor (Zed and
others) can run a coding session against a local, zero-cost model. First milestone (spec.md P16): plain-text prompts and
streamed responses only — no file or terminal access yet (see `intent.md` non-goal §4.5 and `plan.md` Phase 3 backlog for the
follow-up phases).

| Method | Direction | Purpose |
|---|---|---|
| `initialize` | client → agent | Protocol version and capability negotiation. No auth required (loopback, single-user). |
| `session/new` | client → agent | Starts a session; resolves the model once (`$PRISM_ACP_MODEL`, else a CUDA ONNX model if one is installed, else the first available model, else the hardcoded `Phi-4-mini-instruct-cuda-gpu` if no local models are installed) and returns a `sessionId`. |
| `session/prompt` | client → agent | A turn of one or more `{"type": "text", "text": ...}` content blocks. Streams `session/update` notifications (`agent_message_chunk`, `agent_thought_chunk` for `<think>` content) and resolves with `stopReason: "end_turn"` or `"cancelled"`. Only one prompt may be in flight per session. |
| `session/cancel` | client → agent (notification) | Stops the in-flight prompt for a session; the pending `session/prompt` resolves with `stopReason: "cancelled"`. |

Generation talks to `$PRISM_BASE_URL` (default `http://localhost:5272/v1`, with `$PRISM_API_KEY` if set) the same way
`prism mcp` does; if the server is unreachable, it falls back to loading the ONNX model directly in-process.

`session/cancel` is scoped to the session, not to a specific turn — a well-behaved client that waits for each prompt's
result before sending the next is unaffected, but a cancel sent for a turn that has already resolved can, in a race, stop
the *next* turn instead (spec.md P16 documents this as a known limitation).

### fs-mediated tool calls

When the client editor advertises filesystem capabilities during `initialize` (`clientCapabilities.fs.readTextFile`, `clientCapabilities.fs.writeTextFile`), `prism acp` advertises two tools to the model:

- `read_file(path: str, line: Optional[int], limit: Optional[int])`: mediated by the editor's `fs/read_text_file` JSON-RPC method.
- `write_file(path: str, content: str)`: mediated by the editor's `fs/write_text_file` JSON-RPC method, always gated by a `session/request_permission` user prompt.

The agent never opens a file, never writes to disk, and never spawns a subprocess (invariant I9); every byte passes through the client editor over stdio.

#### Notification and update shapes

Tool executions stream `session/update` notifications to the client:

- `tool_call`: `{sessionUpdate: "tool_call", toolCallId: string, title: string, kind: "read" | "edit", status: "in_progress"}`
- `tool_call_update` (success): `{sessionUpdate: "tool_call_update", toolCallId: string, status: "completed", content: [{"type": "content", "content": {"type": "text", "text": string}}]}`
- `tool_call_update` (failure): `{sessionUpdate: "tool_call_update", toolCallId: string, status: "failed", content: [{"type": "content", "content": {"type": "text", "text": string}}]}`

#### Permission flow

Before executing `write_file`, `prism acp` sends `session/request_permission`:

```json
{
  "sessionId": "...",
  "toolCall": {"toolCallId": "call_..._0001", "title": "Write /path/to/file", "kind": "edit"},
  "options": [
    {"optionId": "allow_once", "name": "Allow once", "kind": "allow_once"},
    {"optionId": "reject_once", "name": "Reject", "kind": "reject_once"}
  ]
}
```

The user's response determines execution:
- `allow_once`: executes `fs/write_text_file` and reports success.
- `reject_once`: skips writing and reports `"permission denied"` back to the model as a failed tool call so the model can react.
- `cancelled`: cancels the current prompt turn with `stopReason: "cancelled"`.

```bash
prism connect acp --test              # run a protocol handshake against `prism acp`
prism connect acp --write             # merge an agent_servers entry into ~/.config/zed/settings.json
```

Without `--write`, the snippet is only printed. With `--write`, an existing `settings.json` is merged (your other settings
and agent servers are kept) and a `.bak` copy is saved first; a file that is not valid JSON is left untouched.
