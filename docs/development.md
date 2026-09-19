# Development

See also [CONTRIBUTING.md](https://github.com/senssei/prism-local/blob/main/CONTRIBUTING.md).

## Setup and tests

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,docs]"
PYTHONPATH=. python3 -m unittest discover -s tests -v
```

The suite (about 120 tests, ~6 s) needs no GPU, network, Ollama or model files.

| Area | How it is tested without hardware |
|---|---|
| HTTP server | `create_server(port=0)` with a `FakeEngine` (`tests/fakes.py`): shapes, SSE framing, errors, auth/CORS/Host, concurrency, client disconnect |
| `OnnxGenAiEngine` | `FakeOg`, a stand-in `onnxruntime_genai` module: streaming, finish reasons, provider selection and fallback |
| Catalog / paths | Temp directories with stub `genai_config.json` |
| `pull` | A fake `huggingface_hub.snapshot_download` writing files, including the subfolder layout |
| CLI | `prism.cli.main()` with patched `argv`, capturing stdout |
| Connectors / MCP | Temp working directory and a live handshake against `bin/prism mcp` |

`tests/test_prism_gpu_integration.py` runs the real engine and server, and **skips itself** unless `onnxruntime_genai`, an
NVIDIA GPU and a model are present *and the CUDA provider actually loads*. To run it:

```bash
export PRISM_PYTHON=/path/to/venv/bin/python3 PRISM_MODEL_DIRS=/path/to/models
PYTHONPATH=. "$PRISM_PYTHON" -m unittest tests.test_prism_gpu_integration -v
```

`tests/test_docs.py` checks that every CLI subcommand and server route is mentioned in these docs.

## Documentation site

```bash
mkdocs serve             # http://127.0.0.1:8000, live reload
mkdocs build --strict    # CI runs this; broken links and missing pages fail the build
```

Pages are published to GitHub Pages by `.github/workflows/docs.yml` on every push to `main`. One-time repository setting:
**Settings → Pages → Build and deployment → Source: GitHub Actions**.

## Continuous integration

`.github/workflows/ci.yml` runs the unit tests on Python 3.10–3.12, byte-compiles the package, and builds a wheel that is
installed in a fresh virtualenv and smoke-tested with `prism --help` and `prism doctor`.

## Keeping the model aliases honest

```bash
PYTHONPATH=. python3 scripts/verify_aliases.py    # needs network; exits 1 if any alias is broken
```

## Releasing

1. Update `CHANGELOG.md` and `prism/__init__.py` (`__version__`).
2. `pip wheel . --no-deps -w dist/` and smoke-test the wheel in a clean virtualenv.
3. Tag `vX.Y.Z` and create a GitHub release.
