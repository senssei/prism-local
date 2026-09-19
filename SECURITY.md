# Security Policy

## Reporting a vulnerability

Please report security issues privately through GitHub's
[private vulnerability reporting](https://github.com/senssei/prism-local/security/advisories/new) rather than a public
issue. Include the version, how you run `prism serve` (host, flags), and steps to reproduce. This is a small alpha project, so
expect a best-effort response, but reports are taken seriously.

## Threat model

`prism serve` is meant to be a **local, single-user** inference server.

- It binds to `127.0.0.1` by default, sends no CORS headers, rejects non-loopback `Host` headers, and limits request bodies
  to 10 MB.
- Setting `--host` to a non-loopback address exposes the model, and your GPU time, to that network. Use `--api-key` and put
  a TLS-terminating reverse proxy in front of it; Prism does not speak TLS itself.
- `/health` is unauthenticated by design and returns GPU model, memory and the active model name.
- `--cors-origin '*'` lets any web page call the server from a browser; only use it together with `--api-key`.
- The MCP server exposes read-only tools (generation, listing, status, benchmark). It does not read or write files.

Supported versions: the latest release only.
