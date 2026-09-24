# AGENTS.md

Instructions for AI coding agents (Claude Code, Codex, Cursor, Copilot, Gemini CLI, ...). `CLAUDE.md` imports this file, so
there is one source of truth. Humans: see [CONTRIBUTING.md](CONTRIBUTING.md).

## Project

`prism-local`: a multi-engine local AI CLI and OpenAI-compatible server (ONNX Runtime GenAI on CUDA/CPU, plus Ollama) for Linux
and WSL2. Python >= 3.10, package in `prism/`, tests in `tests/`, docs in `docs/` (MkDocs Material).

## Commands

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev,docs]"   # setup
PYTHONPATH=. python3 -m unittest discover -s tests                                    # tests (~300, ~20 s, no GPU/network)
PYTHONPATH=. python3 -m unittest tests.test_prism_cli.TestX.test_y                    # one test
python3 scripts/sdlc_check.py                                                         # the gate: compile, tests, changelog, docs, lint
python3 scripts/sdlc_check.py --red tests.test_x.TestY.test_z                         # red-first: these new tests must FAIL now
mkdocs build --strict                                                                 # docs, as CI runs it
ruff check path/to/changed.py                                                         # lint; the gate flags only lines you changed
```

## Python conventions

Match the module you are editing; the code was never machine-formatted, so do not reformat files or restyle code you are not
changing. Patterns and reasoning: skill `python-conventions`.

- **Python 3.10 is the floor** (CI runs 3.10 to 3.12): no `tomllib`, `ExceptionGroup`/`except*`, `typing.Self`, `enum.StrEnum`,
  `typing.override`. Import optional engines lazily and name the extra in the error.
- **Types**: annotate public functions in the module's own spelling (`Optional[X]`, `List[X]`, `Dict[K, V]` from `typing`; no
  `X | None` in a module that has none). Module docstring `prism.<name>: purpose`. Lines up to 120. `pathlib` in new code.
- **Errors**: no bare `except:`; `except Exception` only at a boundary that must not die, and it logs; never `pass`. Chain with
  `raise ... from exc`.
- **Logging**: `logger = logging.getLogger("prism.<module>")`. `print` only for CLI output the user asked for. In `prism acp` and
  `prism mcp` stdout is the JSON-RPC stream: write only frames, through the module's locked writer.
- **Locks**: a lock that a code path can re-enter is an `RLock` (`prism/acp.py`, `prism/mcp.py`, commit `238c3ff`).
- **Security**: no `shell=True`, `eval`, `exec`, `pickle`; `subprocess` takes an argument list; check client-supplied paths against
  the allowed root.
- **Lint**: run `ruff check` on the files you changed before you say you are done. `scripts/sdlc_check.py --only lint` reports only
  violations on changed lines; never bulk-apply `ruff format` or `--fix`. The Claude Code edit hook runs the same check.

## Development process (AI-native SDLC)

Every non-trivial change follows **intent -> spec -> plan -> test -> code -> review**, in that order. A stage is finished only when
its artifact exists on disk, so the work survives `/clear`, context compaction and a switch of agent.

| # | Stage | Artifact | Finished when |
|---|---|---|---|
| 1 | Intent | `intent.md` | Problem, outcome, constraints, non-goals and success criteria still hold for the change. If the change contradicts them, update intent first and get operator approval. |
| 2 | Spec | `spec.md` | The new or changed behavior is written: invariants, failure modes, and the normative doc (`docs/api.md`, `docs/cli.md`) it changes. No code against undefined behavior. |
| 3 | Plan | `plan.md` | Work is unchecked `- [ ]` items under a phase, each naming the files it touches and the test that proves it. The operator approved the plan. |
| 4 | Test | `tests/` | A hermetic test exists and was **seen failing for the right reason**: `sdlc_check.py --red` exits 0 and its printed reason is the missing behavior. |
| 5 | Code | `prism/` | The smallest change that turns the tests green. No unrelated refactors. |
| 6 | Review | `REVIEW.md` | Independent review has no open finding, the gate exits 0, plan boxes are ticked, `CHANGELOG.md` is updated, and the operator decides. |

Each stage has a skill in `.agents/skills/` (`.claude/skills` is a symlink to it): `sdlc` (find the stage), `sdlc-plan` (1 to 3),
`sdlc-implement` (4 and 5), `sdlc-review` (6), `sdlc-release` (ship). Every harness that reads skills gets the same workflow.
`python-conventions` is the reference for writing Python here.

### Process rules

- **Gates decide, not opinion.** Tick a plan box only after `python3 scripts/sdlc_check.py` exited 0 in this session.
- **Bug fixes start at stage 4**: reproduce with a failing test, update `spec.md` first if the intended behavior was undefined.
- **Trivial changes** (typo, comment, docs wording) may skip stages 1 to 4; say so in the commit message.
- **Read before editing.** Read the code that owns the behavior and its tests before proposing a change.
- **Independent review.** The reviewer is a fresh subagent or session, never the one that wrote the change.
- **One logical change per commit**, message `type: description` as in `git log` (`feat:`, `bugfix: #N`, `docs:`, `chore:`).
  Commit and push only when the operator asks.
- **Docs are included, not copied**: `docs/sdlc/{intent,spec,review}.md` include the root files, so change the root file only.
- **Operator gates**: intent changes, invariant changes, releases (version, tag, PyPI), pushes, and anything on GitHub (issues,
  PRs, comments) need the operator's explicit ask **in that turn**. Stop and ask instead of assuming.

## Rules that are not obvious from the code

- The runtime is **standard library only**. Engines (`onnxruntime-genai`, `huggingface_hub`, `jinja2`, ...) are optional extras.
- Anything that can change which hardware runs a model must be reported to the user (`--device`, `prism doctor`). A silent
  fallback is a bug (invariant I1 in `spec.md`).
- Tests are hermetic: no GPU, Ollama, network or model files. Use `tests/fakes.py` (`FakeEngine`, `FakeOg`), temp dirs and
  `create_server(port=0)`. `tests/test_prism_gpu_integration.py` is the only hardware test and skips itself.
- A new CLI subcommand goes in `docs/cli.md`, a new server route in `docs/api.md` (`tests/test_docs.py` enforces both).
- User-visible changes get a `CHANGELOG.md` entry under `[Unreleased]`.
- Never run two model loads at once on the reference workstation without the resource guard; that has hung Windows (`plan.md` Phase 1).
- If `.githooks/pre-commit` is enabled (`git config core.hooksPath .githooks`), it gates every commit. Never bypass it with
  `--no-verify`; fix the failure. Commits may be GPG-signed and wait for a passphrase: do not disable signing, ask the operator
  to unlock the agent.
