# Research

Prism grew out of an evaluation of [Microsoft Foundry Local](https://github.com/microsoft/foundry-local) on Linux and WSL2
(September 2026). These documents are the record of that work. They are point-in-time snapshots and are kept as written,
apart from the notes marked below.

| Document | What it covers |
|---|---|
| [Foundry Local evaluation](evaluation-report.md) | Hardware/runtime measurements and a comparison with Ollama, vLLM and llama.cpp |
| [Upstream code analysis](upstream-code-analysis.md) | Source-level reading of the Foundry Local CLI and the C++ `sdk_v2` |
| [WSL2 feasibility](foundry-wsl2-feasibility.md) | Running on WSL2 versus bridging to the Windows host |
| [Legacy `foundry_wsl` toolkit](../legacy-foundry-wsl.md) | The first tooling built from these findings |

!!! warning "Reproducibility"
    The CUDA throughput figures for Prism in these documents (for example 118–130 tok/s) were captured with the CUDA execution
    provider active. They have **not been reproduced** since the machine's ONNX Runtime GenAI build and CUDA libraries stopped
    matching; CPU inference measured about 8 tok/s. See the note at the top of the [evaluation](evaluation-report.md). Measure your
    own hardware with `prism benchmark`.
