#!/usr/bin/env python3
"""
foundry_wsl.cli: Command-Line Interface for the Foundry WSL2 Toolkit.
"""

import argparse
import json
import sys
from pathlib import Path
from foundry_wsl.doctor import run_doctor_report
from foundry_wsl.proxy import run_proxy, resolve_daemon_url
from foundry_wsl.injector import configure_cuda_genai_config

def cmd_doctor(args):
    report = run_doctor_report()
    print("\n🩺 FOUNDRY WSL2 HEALTH REPORT")
    print("=" * 60)
    
    # NVML Check
    nvml = report["nvml"]
    if nvml["status"] == "ok":
        print(f"✅ NVML Driver: Connected ({nvml['device_count']} GPU(s) found)")
        for dev in nvml["devices"]:
            print(f"   • {dev['name']} (Compute Cap: {dev['compute_capability']})")
    else:
        print(f"❌ NVML Driver: {nvml.get('message', 'Error')}")

    # CUDA Libs Check
    cuda = report["cuda_libraries"]
    if cuda["status"] == "ok":
        print(f"✅ CUDA Providers: All {cuda['resolved_count']} libraries resolved.")
    elif cuda["status"] == "error":
        print(f"❌ CUDA Providers: Missing dynamic libraries:")
        for m in cuda.get("missing_libraries", []):
            print(f"   • {m}")
    else:
        print(f"⚠️ CUDA Providers: {cuda.get('message', 'Unknown')}")

    # Daemon Check
    daemon = report["daemon"]
    if daemon["status"] == "running":
        print(f"✅ Foundry Daemon: Running (PID: {daemon['pid']})")
        print(f"   Active URLs: {daemon['urls']}")
    else:
        print(f"ℹ️ Foundry Daemon: {daemon['status'].capitalize()} ({daemon.get('message', '')})")
    print("=" * 60)

def cmd_proxy(args):
    run_proxy(port=args.port, host=args.host)

def cmd_status(args):
    daemon_url = resolve_daemon_url()
    print("\n📊 FOUNDRY WSL2 STATUS")
    print("=" * 60)
    print(f"Active Daemon URL: {daemon_url or 'Not running'}")
    cache_info = Path.home() / ".foundry/cache/models/foundry.modelinfo.json"
    if cache_info.exists():
        try:
            data = json.loads(cache_info.read_text())
            cached_models = [m.get("id") or m.get("alias") for m in data.get("models", []) if m.get("cached")]
            print(f"Cached Models ({len(cached_models)}):")
            for m in cached_models:
                print(f"   • {m}")
        except Exception:
            pass
    print("=" * 60)

def cmd_inject(args):
    target = Path(args.config_path)
    if not target.exists():
        print(f"❌ File not found: {target}")
        sys.exit(1)
    if configure_cuda_genai_config(target, device_id=args.device_id):
        print(f"✅ Successfully injected CUDA provider options into {target}")
    else:
        print(f"❌ Failed to inject CUDA options into {target}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Foundry WSL2 Management & Diagnostics Toolkit")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # doctor
    p_doc = subparsers.add_parser("doctor", help="Run comprehensive WSL2 hardware & runtime diagnostic")
    p_doc.set_defaults(func=cmd_doctor)

    # proxy
    p_proxy = subparsers.add_parser("proxy", help="Run stable reverse proxy forwarding to active daemon")
    p_proxy.add_argument("--port", type=int, default=5272, help="Port to listen on (default: 5272)")
    p_proxy.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    p_proxy.set_defaults(func=cmd_proxy)

    # status
    p_stat = subparsers.add_parser("status", help="Display active daemon and cache status")
    p_stat.set_defaults(func=cmd_status)

    # inject
    p_inj = subparsers.add_parser("inject", help="Patch genai_config.json with CUDA session options")
    p_inj.add_argument("config_path", help="Path to genai_config.json")
    p_inj.add_argument("--device-id", type=int, default=0, help="NVIDIA CUDA device ID (default: 0)")
    p_inj.set_defaults(func=cmd_inject)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)
    args.func(args)

if __name__ == "__main__":
    main()
