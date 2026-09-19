#!/usr/bin/env python3
"""
foundry_ng.cli: Main Command-Line Interface for foundry-ng (fng).
"""

import argparse
import json
import sys
from foundry_ng.catalog import ModelCatalog
from foundry_ng.chat import run_interactive_chat
from foundry_ng.benchmark import run_benchmark
from foundry_ng.server import start_server
from foundry_ng.telemetry import get_gpu_info, bootstrap_cuda_env
from foundry_ng.engine import OnnxGenAiEngine, format_prompt, OG_AVAILABLE
from foundry_ng.ollama_bridge import is_ollama_running, stream_ollama_chat


def cmd_status(args):
    print("\n🖥️  FOUNDRY-NG SYSTEM & HARDWARE STATUS")
    print("=" * 65)
    gpu = get_gpu_info()
    if gpu.get("available"):
        for dev in gpu.get("devices", []):
            print(f"✅ GPU #{dev['index']}: {dev['name']}")
            print(f"   • Compute Capability: {dev['compute_capability']} (Qualifying: {dev['qualifying_cuda']})")
            print(f"   • VRAM: {dev['vram_used_mb']:.1f} MB used / {dev['vram_total_mb']:.1f} MB total ({dev['vram_free_mb']:.1f} MB free)")
    else:
        print(f"❌ GPU: {gpu.get('error', 'Not detected via NVML')}")

    print(f"• ONNX Runtime GenAI Available: {OG_AVAILABLE}")
    print(f"• Ollama Daemon Available:      {is_ollama_running()}")

    catalog = ModelCatalog()
    onnx_models = catalog.discover_onnx_models()
    print(f"• Discovered ONNX Models:       {len(onnx_models)}")
    print("=" * 65)


def cmd_doctor(args):
    bootstrap_cuda_env()
    print("\n🩺 FOUNDRY-NG ENVIRONMENT DOCTOR")
    print("=" * 65)
    gpu = get_gpu_info()
    if gpu.get("available"):
        print(f"✅ NVML Driver: Connected ({gpu['driver_path']})")
    else:
        print(f"❌ NVML Driver: Missing or inaccessible ({gpu.get('error')})")

    if OG_AVAILABLE:
        print("✅ ONNX Runtime GenAI: Installed and ready for CUDA inference.")
    else:
        print("❌ ONNX Runtime GenAI: Not found in current python environment.")

    if is_ollama_running():
        print("✅ Ollama Service: Reachable at http://localhost:11434")
    else:
        print("ℹ️ Ollama Service: Offline (optional for GGUF models)")
    print("=" * 65)


def cmd_list(args):
    catalog = ModelCatalog()
    models = catalog.list_all_models(include_ollama=args.all)
    print(f"\n📦 AVAILABLE LOCAL MODELS ({len(models)}):")
    print("=" * 75)
    print(f"{'NAME / ID':<40} {'ENGINE':<22} {'SIZE':<10} {'DEVICE'}")
    print("-" * 75)
    for m in models:
        dev = m.get("device", "GPU" if "cuda" in m["name"].lower() else "CPU/GPU")
        size_str = f"{m.get('size_mb', 0)} MB"
        print(f"{m['id']:<40} {m.get('engine', 'Unknown'):<22} {size_str:<10} {dev}")
    print("=" * 75)


def cmd_pull(args):
    catalog = ModelCatalog()
    catalog.pull_model(args.model, output_dir=args.output_dir)
    print("✅ Model download complete.")


def cmd_run(args):
    prompt = args.prompt
    if not prompt:
        if not sys.stdin.isatty():
            prompt = sys.stdin.read().strip()
        else:
            # Fall back to interactive chat if no prompt provided
            run_interactive_chat(args.model)
            return

    catalog = ModelCatalog()
    resolved = catalog.resolve_model(args.model)
    if not resolved:
        print(f"❌ Model '{args.model}' not found. Run 'foundry-ng list' to inspect available models.")
        sys.exit(1)

    if resolved.get("backend") == "ollama":
        history = [{"role": "user", "content": prompt}]
        for chunk in stream_ollama_chat(resolved["id"], history):
            print(chunk, end="", flush=True)
        print()
    else:
        engine = OnnxGenAiEngine(resolved["path"])
        formatted = format_prompt([{"role": "user", "content": prompt}])
        try:
            for token, _, _ in engine.stream_generate(formatted, max_tokens=args.max_tokens):
                print(token, end="", flush=True)
            print()
        finally:
            engine.unload()


def cmd_chat(args):
    run_interactive_chat(args.model)


def cmd_serve(args):
    start_server(port=args.port, host=args.host)


def cmd_benchmark(args):
    run_benchmark(args.model)


def main():
    parser = argparse.ArgumentParser(
        prog="foundry-ng",
        description="foundry-ng (fng): Next-Generation Local LLM CLI & Inference Server for WSL2",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # status
    p_status = subparsers.add_parser("status", help="Show GPU telemetry, VRAM, and model status")
    p_status.set_defaults(func=cmd_status)

    # doctor
    p_doc = subparsers.add_parser("doctor", help="Inspect NVML and CUDA environment readiness")
    p_doc.set_defaults(func=cmd_doctor)

    # list
    p_list = subparsers.add_parser("list", help="List installed ONNX and local models")
    p_list.add_argument("--all", "-a", action="store_true", default=True, help="Include Ollama models")
    p_list.set_defaults(func=cmd_list)

    # pull
    p_pull = subparsers.add_parser("pull", help="Download ONNX models directly from Hugging Face")
    p_pull.add_argument("model", help="Hugging Face repo ID or alias (e.g. phi-4-mini)")
    p_pull.add_argument("--output-dir", default="models", help="Destination folder (default: models)")
    p_pull.set_defaults(func=cmd_pull)

    # run
    p_run = subparsers.add_parser("run", help="Run model completion or streaming generation")
    p_run.add_argument("model", help="Model name or alias")
    p_run.add_argument("prompt", nargs="?", default=None, help="Prompt text (optional, defaults to chat)")
    p_run.add_argument("--max-tokens", type=int, default=512, help="Max generation tokens")
    p_run.set_defaults(func=cmd_run)

    # chat
    p_chat = subparsers.add_parser("chat", help="Start an interactive streaming terminal chat session")
    p_chat.add_argument("model", help="Model name or alias")
    p_chat.set_defaults(func=cmd_chat)

    # serve
    p_serve = subparsers.add_parser("serve", help="Launch OpenAI-compatible REST server")
    p_serve.add_argument("--port", type=int, default=5272, help="Port to listen on (default: 5272)")
    p_serve.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    p_serve.set_defaults(func=cmd_serve)

    # benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run automated micro-benchmark (TTFT, tok/s, VRAM)")
    p_bench.add_argument("model", help="Model to benchmark (e.g. Phi-4-mini-instruct-cuda-gpu)")
    p_bench.set_defaults(func=cmd_benchmark)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
