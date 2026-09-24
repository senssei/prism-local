# Development

See also [CONTRIBUTING.md](https://github.com/senssei/prism-local/blob/main/CONTRIBUTING.md).

## Setup and tests

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,docs]"
PYTHONPATH=. python3 -m unittest discover -s tests -v
```

The suite (about 300 tests, ~20 s) needs no GPU, network, Ollama or model files.

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
installed in a fresh virtualenv and smoke-tested with `prism --help` and `prism doctor`. On pull requests it also runs the
changelog check and `ruff` on the lines the pull request changed (see [Lint](#lint)).

## Lint

`ruff` is configured in `pyproject.toml` (`pip install -e ".[dev]"`). Nothing is auto-formatted and old code is not a debt: the
gate reports only violations **on lines you changed**.

```bash
ruff check path/to/changed.py                      # everything in the file, for your information
python3 scripts/sdlc_check.py --only lint          # what the gate and CI enforce: changed lines only
```

The check skips itself when `ruff` is not installed. It is not part of the opt-in pre-commit hook.

## Working with AI coding agents

The repository carries one agent workflow that works across harnesses (Claude Code, Codex, Cursor, Copilot, Gemini CLI). Every
non-trivial change goes through intent, spec, plan, test, code and review, and each stage leaves a committed file:

| File | Read by | Purpose |
|---|---|---|
| `AGENTS.md` | Codex, Cursor, Copilot, Gemini CLI, ... | Commands, the process and the project rules; the single source of truth |
| `CLAUDE.md` | Claude Code | Imports `AGENTS.md`, adds Claude-only notes |
| `intent.md`, `spec.md`, `plan.md`, `REVIEW.md` | Agents and humans | Intent, specification, plan (the state of the work) and review policy |
| `.agents/skills/sdlc*` | Skill-aware agents; Claude Code through the `.claude/skills` symlink | One skill per stage group |
| `.agents/skills/python-conventions` | Same | How to write Python in this repository |
| `scripts/sdlc_check.py` | Agents, humans, CI | Deterministic gate (compile, tests, changelog, docs, lint) and `--red` |
| `.claude/settings.json`, `scripts/lint_hook.py` | Claude Code | Shared permissions and a hook that runs `ruff` on each edited Python file |
| `.editorconfig` | Editors | Encoding, line endings, indentation |
| `.githooks/pre-commit` | git (opt-in) | Runs the gate before each commit: `git config core.hooksPath .githooks` |

Details: [Agent SDLC workflow](sdlc.md).

## Keeping the model aliases honest

```bash
PYTHONPATH=. python3 scripts/verify_aliases.py    # needs network; exits 1 if any alias is broken
```

## Releasing

Releases are published from GitHub Actions with PyPI **trusted publishing** (OIDC), so no API tokens are stored anywhere.
The workflow is `.github/workflows/publish.yml` (manual: *Actions → Publish → Run workflow*).

**One-time setup**

1. On [test.pypi.org](https://test.pypi.org/manage/account/publishing/) (and later [pypi.org](https://pypi.org/manage/account/publishing/)),
   add a *pending publisher*: project `prism-local`, owner `senssei`, repository `prism-local`, workflow `publish.yml`,
   environment `testpypi` (respectively `pypi`).
2. In the GitHub repository, create the environments `testpypi` and `pypi` (*Settings → Environments*). Add yourself as a
   required reviewer on `pypi` so a release needs an explicit approval.

**Each release**

1. Update `CHANGELOG.md` and `prism/__init__.py` (`__version__`), and merge to `main` with CI green.
2. Run **Publish → target `testpypi`**. It builds the sdist and wheel, runs `twine check --strict`, uploads to TestPyPI, then
   installs the uploaded version into a clean virtualenv and smoke-tests `prism --help` and `prism doctor`.
3. Tag the release (`git tag -s vX.Y.Z && git push origin vX.Y.Z`) and run **Publish** on that tag with target `pypi`. The
   workflow refuses to publish to PyPI unless it runs from the tag `v<__version__>`.
4. Create a GitHub release for the tag.

!!! note "Versions are permanent"
    Neither index lets you re-upload a version, and PyPI never lets you reuse one. Use a pre-release version such as `0.1.0rc1`
    while rehearsing on TestPyPI if you expect to iterate.
