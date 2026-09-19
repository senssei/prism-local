# Security

`prism serve` is designed as a **local, single-user** server. The defaults assume that; changing them is your decision.

## Defaults

| Protection | Default | How to change |
|---|---|---|
| Listen address | `127.0.0.1` only | `--host` |
| Authentication | none (loopback only) | `--api-key KEY` or `$PRISM_API_KEY` |
| CORS | **off**: no CORS headers are sent | `--cors-origin ORIGIN` (repeatable; `*` allows any) |
| `Host` header | Non-loopback hosts are rejected (403) when bound to loopback, as a DNS-rebinding defence | n/a |
| Request size | 10 MB cap (413) | n/a |
| Bearer comparison | constant-time | n/a |

## Exposing it on a network

```bash
prism serve --host 0.0.0.0 --api-key "$(openssl rand -hex 16)"
```

Prism prints a warning if you bind a non-loopback address without a key. It does not provide TLS: put a reverse proxy in
front of it if the traffic leaves a trusted network.

!!! warning "About `/health`"
    `/health` is unauthenticated so it can be used for liveness checks. It returns the GPU model, memory usage and the active
    model name.

!!! warning "About `--cors-origin '*'`"
    That lets any web page open in your browser call the server. Use it only together with `--api-key`, or list specific
    origins instead.

## MCP

The MCP server offers generation, listing, status and benchmark tools only. It runs locally over stdio and never touches
your files. `prism connect cline` marks all five tools as auto-approved; edit `autoApprove` if you prefer prompts.

## Reporting vulnerabilities

Use GitHub's [private vulnerability reporting](https://github.com/senssei/prism-local/security/advisories/new). See
[SECURITY.md](https://github.com/senssei/prism-local/blob/main/SECURITY.md).
