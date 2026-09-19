#!/usr/bin/env python3
"""
prism.cli: Main Command-Line Interface for prism.
"""

import argparse
import json
import os
import sys

from prism.catalog import AmbiguousModelError, ModelCatalog
from prism.chat import run_interactive_chat
from prism.benchmark import run_benchmark
from prism.server import start_server
from prism.telemetry import get_gpu_info, bootstrap_cuda_env, probe_cuda_provider
from prism.engine import ModelLoadError, OnnxGenAiEngine, format_prompt, OG_AVAILABLE
from prism.ollama_bridge import is_ollama_running, stream_ollama_chat
from prism.connectors import connect_cursor, connect_cline, connect_mcp


def cmd_status(args):
    print("\n🖥️  PRISM SYSTEM & HARDWARE STATUS")
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
    print("\n🩺 PRISM ENVIRONMENT DOCTOR")
    print("=" * 65)
    gpu = get_gpu_info()
    if gpu.get("available"):
        print(f"✅ NVML Driver: Connected ({gpu['driver_path']})")
    else:
        print(f"❌ NVML Driver: Missing or inaccessible ({gpu.get('error')})")

    if OG_AVAILABLE:
        print("✅ ONNX Runtime GenAI: Installed.")
        cuda = probe_cuda_provider()
        if not cuda["checked"]:
            print(f"ℹ️ CUDA execution provider: not checked ({cuda['reason']}); models will run on CPU.")
        elif cuda["loadable"]:
            print("✅ CUDA execution provider: loads; GPU inference is available (--device cuda).")
        else:
            print(f"❌ CUDA execution provider: cannot load: {cuda['error']}")
            print("   Models will fall back to CPU. Install CUDA libraries matching your onnxruntime-genai build.")
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
    res = catalog.pull_model(
        args.model,
        output_dir=args.output_dir,
        ep=args.ep,
        quant=args.quant,
        backend=args.backend,
    )
    if res:
        print("✅ Model pull completed.")
    else:
        print("❌ Model pull failed or aborted.")


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
        print(f"❌ Model '{args.model}' not found. Run 'prism list' to inspect available models.")
        sys.exit(1)

    if resolved.get("backend") == "ollama":
        history = [{"role": "user", "content": prompt}]
        for chunk in stream_ollama_chat(resolved["id"], history):
            print(chunk, end="", flush=True)
        print()
    else:
        engine = OnnxGenAiEngine(resolved["path"])
        print(f"[device: {engine.device}]", file=sys.stderr)
        formatted = format_prompt([{"role": "user", "content": prompt}], resolved.get("template"))
        try:
            for token, _, _ in engine.stream_generate(formatted, max_tokens=args.max_tokens):
                print(token, end="", flush=True)
            print()
        finally:
            engine.unload()


def cmd_chat(args):
    run_interactive_chat(args.model)


def cmd_serve(args):
    start_server(
        port=args.port,
        host=args.host,
        api_key=args.api_key or os.environ.get("PRISM_API_KEY") or None,
        cors_origins=args.cors_origin,
    )


def cmd_benchmark(args):
    run_benchmark(args.model)


def cmd_mcp(args):
    from prism.mcp import run_mcp_server
    run_mcp_server()


def cmd_connect(args):
    target = getattr(args, "connect_target", None)
    if target == "cursor":
        connect_cursor(
            model=args.model,
            export_rules=args.export_rules,
            export_mcp=args.export_mcp,
            test=args.test,
        )
    elif target == "cline":
        connect_cline(
            model=args.model,
            export_mcp=args.export_mcp,
            test=args.test,
        )
    elif target == "mcp":
        connect_mcp(
            target=args.target,
            write=args.write,
            test=args.test,
        )
    else:
        print("Specify a connector target: 'prism connect cursor', 'prism connect cline', or 'prism connect mcp'.")


def _device_parent() -> argparse.ArgumentParser:
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "--device", choices=["auto", "cuda", "cpu"], default=None,
        help="ONNX execution provider: auto (CUDA if it loads, else CPU), cuda (fail if unavailable), cpu "
             "(default: $PRISM_DEVICE or auto)",
    )
    return parent


def main():
    device_parent = _device_parent()
    parser = argparse.ArgumentParser(
        prog="prism",
        description="prism: Next-Generation Multi-Engine Local AI CLI & Inference Server for WSL2/Linux",
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
    p_pull = subparsers.add_parser("pull", help="Download ONNX models (Hugging Face) or GGUF models (Ollama)")
    p_pull.add_argument("model", help="Model name, alias, or Hugging Face repo ID (e.g. phi-4-mini, ollama:qwen2.5-coder:7b)")
    p_pull.add_argument("--output-dir", default=None, help="Destination folder (default: first $PRISM_MODEL_DIRS entry, else ~/.prism/models)")
    p_pull.add_argument("--ep", choices=["cuda", "cpu"], default=None, help="Target execution provider (default: auto-detect)")
    p_pull.add_argument("--quant", choices=["int4", "fp16"], default="int4", help="Quantization level (default: int4)")
    p_pull.add_argument("--backend", choices=["auto", "onnx", "ollama"], default="auto", help="Inference backend (default: auto)")
    p_pull.set_defaults(func=cmd_pull)

    # run
    p_run = subparsers.add_parser("run", parents=[device_parent], help="Run model completion or streaming generation")
    p_run.add_argument("model", help="Model name or alias")
    p_run.add_argument("prompt", nargs="?", default=None, help="Prompt text (optional, defaults to chat)")
    p_run.add_argument("--max-tokens", type=int, default=512, help="Max generation tokens")
    p_run.set_defaults(func=cmd_run)

    # chat
    p_chat = subparsers.add_parser("chat", parents=[device_parent], help="Start an interactive streaming terminal chat session")
    p_chat.add_argument("model", help="Model name or alias")
    p_chat.set_defaults(func=cmd_chat)

    # serve
    p_serve = subparsers.add_parser("serve", parents=[device_parent], help="Launch OpenAI-compatible REST server")
    p_serve.add_argument("--port", type=int, default=5272, help="Port to listen on (default: 5272)")
    p_serve.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1; use 0.0.0.0 to expose on the network, ideally with --api-key)")
    p_serve.add_argument("--api-key", default=None, help="Require 'Authorization: Bearer <key>' (default: $PRISM_API_KEY)")
    p_serve.add_argument("--cors-origin", action="append", default=[], metavar="ORIGIN", help="Allow a browser origin (repeatable, or '*'); CORS is off by default")
    p_serve.set_defaults(func=cmd_serve)

    # benchmark
    p_bench = subparsers.add_parser("benchmark", parents=[device_parent], help="Run automated micro-benchmark (TTFT, tok/s, VRAM)")
    p_bench.add_argument("model", help="Model to benchmark (e.g. Phi-4-mini-instruct-cuda-gpu)")
    p_bench.set_defaults(func=cmd_benchmark)

    # mcp
    p_mcp = subparsers.add_parser("mcp", help="Run Prism as a stdio Model Context Protocol (MCP) server")
    p_mcp.set_defaults(func=cmd_mcp)

    # connect
    p_conn = subparsers.add_parser("connect", help="Configure IDEs, extensions, and MCP clients")
    conn_sub = p_conn.add_subparsers(dest="connect_target", help="Connector target")

    # connect cursor
    p_conn_cursor = conn_sub.add_parser("cursor", help="Generate Cursor IDE configuration and .cursorrules")
    p_conn_cursor.add_argument("--model", help="Preferred model ID")
    p_conn_cursor.add_argument("--export-rules", action="store_true", help="Write .cursorrules file in current directory")
    p_conn_cursor.add_argument("--export-mcp", action="store_true", help="Write .cursor/mcp.json in current directory")
    p_conn_cursor.add_argument("--test", action="store_true", help="Test reachability of Prism server")
    p_conn_cursor.set_defaults(func=cmd_connect)

    # connect cline
    p_conn_cline = conn_sub.add_parser("cline", help="Generate Cline extension OpenAI & MCP settings")
    p_conn_cline.add_argument("--model", help="Preferred model ID")
    p_conn_cline.add_argument("--export-mcp", action="store_true", help="Write cline_mcp_settings.json in current directory")
    p_conn_cline.add_argument("--test", action="store_true", help="Test reachability of Prism server")
    p_conn_cline.set_defaults(func=cmd_connect)

    # connect mcp
    p_conn_mcp = conn_sub.add_parser("mcp", help="Configure MCP clients (Antigravity, Cursor, Cline, Claude)")
    p_conn_mcp.add_argument("--target", choices=["all", "antigravity", "cursor", "cline", "claude"], default="all", help="Target client")
    p_conn_mcp.add_argument("--write", action="store_true", help="Write or update the target MCP configuration file")
    p_conn_mcp.add_argument("--test", action="store_true", help="Run JSON-RPC protocol handshake test")
    p_conn_mcp.set_defaults(func=cmd_connect)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)
    if getattr(args, "device", None):
        os.environ["PRISM_DEVICE"] = args.device
    try:
        args.func(args)
    except (AmbiguousModelError, ModelLoadError) as ex:
        print(f"❌ {ex}")
        sys.exit(1)


if __name__ == "__main__":
    main()
