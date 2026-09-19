# Contributing

Thanks for helping out. Prism is a small alpha project; issues and pull requests are welcome.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,docs]"      # add ,cuda,pull if you want real inference / downloads
```

## Tests

```bash
PYTHONPATH=. python3 -m unittest discover -s tests -v
```

The suite needs no GPU, Ollama, network or model files. Server and engine logic run against fakes in
[`tests/fakes.py`](tests/fakes.py) (`FakeEngine`, and `FakeOg`, a stand-in for `onnxruntime_genai`). Please add tests with your
change, keeping them hermetic: use temp dirs, `create_server(port=0)`, and patch `prism.catalog.list_ollama_models`.
`tests/test_prism_gpu_integration.py` runs on real hardware and skips itself when CUDA is unusable.

## Docs

The site is built with MkDocs Material from `docs/`:

```bash
mkdocs serve             # live preview
mkdocs build --strict    # what CI runs; broken links fail the build
```

If you add a CLI command or server route, document it in `docs/cli.md` or `docs/api.md`; `tests/test_docs.py` checks that
every subcommand and route is mentioned.

## Guidelines

- Keep the runtime dependency-free (standard library only); engines are optional extras.
- Anything that could change what hardware runs the model must be reported to the user (see `--device` and `prism doctor`).
  Silent fallbacks are bugs.
- Update `CHANGELOG.md` under the unreleased version.

## Pull requests

Describe what changed and why, and note how you tested it. CI must pass (Python 3.10-3.12, wheel smoke test, docs build).
