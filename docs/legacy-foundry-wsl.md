# Legacy: `foundry_wsl`

`foundry_wsl/` is the first tool in this repository: a toolkit for making **Microsoft Foundry Local** usable from WSL2, built
during the evaluation described in the [research notes](research/index.md). It is kept for reference, is **not** part of the
installable `prism-local` package, and is run from a checkout. New work goes into Prism.

```bash
python -m foundry_wsl.cli --help          # doctor, proxy, status, inject
```

| Command | What it does |
|---|---|
| `doctor` | Reports NVML GPU detection, whether CUDA libraries resolve, and the Foundry daemon state |
| `status` | Shows Foundry Local status information |
| `proxy [--host H] [--port P]` | A stable reverse proxy (default `127.0.0.1:5272`) in front of Foundry's ephemeral daemon port, resolved from `~/.foundry/daemon.json` |
| `inject CONFIG [--device-id N]` | Writes a CUDA `provider_options` entry into a model's `genai_config.json` |

!!! warning "Known issue with `inject`"
    `inject` writes the provider options under a **top-level** `session_options` key. In the current model configs
    inspected while writing these docs, the decoder's provider list lives under `model.decoder.session_options`, and it is
    not established that ONNX Runtime GenAI honors the top-level key. Prefer Prism's `--device cuda`, which selects the provider
    through the runtime API instead of by editing files.

## Windows host bridge

`windows_bridge/` holds two helpers for the alternative architecture of running Foundry Local on the Windows host and
reaching it from WSL2:

- `setup_windows_gateway.ps1`: run in an elevated PowerShell on Windows; configures the firewall and starts Foundry Local
  listening on all interfaces.
- `test_connection.sh`: run in WSL2; probes `http://<windows-host>:${FOUNDRY_PORT:-5272}/v1/models`.

The trade-offs are discussed in the [feasibility analysis](research/foundry-wsl2-feasibility.md).
