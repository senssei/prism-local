# Intent: Prism (`prism-local`)

> **Status: approved by the operator, 2026-09-20.** Written from `README.md`, `docs/` and the operator's private notes. The operator
> approves any change to this file (see `AGENTS.md`, operator gates).

## 1. Problem

Running local coding models on a Linux or WSL2 workstation with an NVIDIA GPU is harder than it should be.

1. **The official tooling did not work here.** In the evaluation that started this project, Microsoft Foundry Local (CLI `0.10.3`) detected no GPU
   under WSL2, ran on the CPU, and served on a random port. A CPU run looks exactly like a slow GPU run, so the user could not tell.
2. **One engine per tool.** ONNX Runtime GenAI models and Ollama (GGUF) models need different commands and different endpoints, and editors and
   agents (Cursor, Cline, MCP clients) each want their own configuration.
3. **Nothing protects the machine.** Several processes can each load their own model copy. On a 12 GB GPU shared with the Windows desktop, two
   models overflow VRAM and the run slows about 25 times (issue #5); parallel use has hung the whole Windows host.

## 2. Outcome

One command line and one OpenAI-compatible endpoint on a fixed port (`127.0.0.1:5272`) in front of ONNX Runtime GenAI (CUDA or CPU) and Ollama, with
ready-made connectors for Cursor, Cline and MCP. Prism tells the truth about where a model runs, and its defaults are safe for a single-user
machine. Coding agents (including the AI development workflow in this repository) can rely on it as a local, zero-cost model backend.

## 3. Constraints

1. **Platform**: Linux and WSL2 only. Python 3.10+.
2. **Runtime dependencies**: none. The standard library only; engines (`onnxruntime-genai`, `huggingface_hub`, `jinja2`, ...) are optional extras.
3. **Honest hardware**: anything that can change which device runs a model is reported. A silent fallback is a bug.
4. **Tests need no hardware**: no GPU, Ollama, network or model files, so they run in CI.
5. **Safe by default**: loopback only, no CORS, no API key needed on loopback, and a warning when that is changed unsafely.
6. **Reference machine**: WSL2 with 32 GB RAM and an RTX 5070 (12 GB) that is shared with the Windows desktop and other GPU applications.

## 4. Non-goals

1. A high-concurrency or multi-user server. One ONNX model is resident at a time and requests are serialized.
2. Windows or macOS native support, TLS termination, or hosting Prism as a service for other people.
3. Training, fine-tuning, or model hosting. Prism runs models that exist; `prism convert` only wraps ONNX Runtime GenAI's builder.
4. Replacing Ollama or Foundry Local. Prism sits in front of engines; it does not compete with them.

## 5. Success criteria

| Criterion | Evidence |
|---|---|
| The device a request ran on is always visible, and no fallback is silent | `spec.md` invariant I1; tests in `tests/test_prism_engine.py` |
| One endpoint serves both ONNX and Ollama models | `docs/api.md`; server tests with `FakeEngine` |
| The test suite runs with no GPU, network or models | CI (`.github/workflows/ci.yml`) |
| `prism doctor` names the cause when CUDA cannot be used | `docs/devices.md`; `tests/test_doctor.py` |
| Parallel use cannot exhaust VRAM or RAM without a clear refusal | `plan.md` Phase 1 (not yet done) |
| A change is only "done" when the deterministic gate passes | `scripts/sdlc_check.py`, `.githooks/pre-commit` |
