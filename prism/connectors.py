"""
prism.connectors: Integrated Connectors & Configuration Generators for Cursor, Cline, and MCP Clients.
Enables instant zero-friction integration of Prism with developer IDEs, AI extensions, and agents.
"""

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from prism.catalog import ModelCatalog


def _get_prism_bin_path() -> str:
    """Resolves the absolute path to the bin/prism executable."""
    script_dir = Path(__file__).resolve().parent.parent
    bin_path = script_dir / "bin" / "prism"
    if bin_path.exists():
        return str(bin_path)
    which_prism = shutil.which("prism")
    if which_prism:
        return which_prism
    return str(bin_path)


def merge_prism_mcp_entry(cfg_path: Path, prism_entry: Dict[str, Any]) -> None:
    """
    Sets `mcpServers.prism` in an MCP client config, keeping every other server. An existing file is
    copied to `<name>.bak` first. Raises ValueError (and writes nothing) if the existing file is not valid JSON.
    """
    data: Dict[str, Any] = {}
    if cfg_path.exists():
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{cfg_path} does not contain a JSON object")
        shutil.copy2(cfg_path, cfg_path.with_name(cfg_path.name + ".bak"))
    servers = data.setdefault("mcpServers", {})
    if not isinstance(servers, dict):
        raise ValueError(f"'mcpServers' in {cfg_path} is not an object")
    servers["prism"] = prism_entry
    cfg_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def test_prism_server_connection(base_url: str = "http://localhost:5272/v1") -> bool:
    """Tests if the Prism OpenAI REST server is responding."""
    try:
        headers = {"Authorization": f"Bearer {os.environ['PRISM_API_KEY']}"} if os.environ.get("PRISM_API_KEY") else {}
        req = urllib.request.Request(f"{base_url.rstrip('/')}/models", headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            return resp.status == 200
    except Exception:
        return False


def test_mcp_protocol(bin_path: Optional[str] = None) -> Dict[str, Any]:
    """Runs a live JSON-RPC 2.0 handshake against prism mcp."""
    prism_bin = bin_path or _get_prism_bin_path()
    init_payload = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "prism-tester", "version": "1.0.0"},
        },
    }) + "\n"

    list_payload = json.dumps({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {},
    }) + "\n"

    proc = subprocess.Popen(
        [prism_bin, "mcp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        stdout_init, _ = proc.communicate(input=init_payload + list_payload, timeout=5)
        lines = [line.strip() for line in stdout_init.strip().split("\n") if line.strip()]
        responses = [json.loads(line) for line in lines]

        init_res = next((r for r in responses if r.get("id") == 1), {})
        tools_res = next((r for r in responses if r.get("id") == 2), {})
        tools_list = tools_res.get("result", {}).get("tools", [])

        return {
            "success": bool(init_res.get("result") and tools_list),
            "protocol_version": init_res.get("result", {}).get("protocolVersion"),
            "server_name": init_res.get("result", {}).get("serverInfo", {}).get("name"),
            "tools_count": len(tools_list),
            "tools": [t.get("name") for t in tools_list],
        }
    except Exception as ex:
        return {"success": False, "error": str(ex)}
    finally:
        if proc.poll() is None:
            proc.kill()


# ============================================================================
# Cursor Connector
# ============================================================================

def connect_cursor(
    model: Optional[str] = None,
    export_rules: bool = False,
    export_mcp: bool = False,
    test: bool = False,
) -> None:
    catalog = ModelCatalog()
    all_models = catalog.list_all_models(include_ollama=True)
    chosen_model = model or (all_models[0]["id"] if all_models else "Phi-4-mini-instruct-cuda-gpu")
    base_url = "http://localhost:5272/v1"
    prism_bin = _get_prism_bin_path()

    print("\n" + "=" * 68)
    print(" 🎯 PRISM CONNECTOR: CURSOR IDE")
    print("=" * 68)

    print("\n1. Configure Custom OpenAI Model in Cursor:")
    print("   • Open Cursor -> Cursor Settings -> Models")
    print("   • Click 'Add Custom Model'")
    print(f"   • Model Name:     {chosen_model}")
    print(f"   • Base URL:       {base_url}")
    print("   • API Key:        prism (or any non-empty string)")
    print("   • Toggle ON your newly added model")

    if test:
        print("\n2. Diagnostics & Server Connectivity:")
        is_up = test_prism_server_connection(base_url)
        if is_up:
            print(f"   ✅ Prism server is ACTIVE and reachable at {base_url}")
        else:
            print(f"   ⚠️ Prism server is OFFLINE at {base_url}")
            print(f"      Launch it in a terminal: prism serve --port 5272")

    if export_rules:
        rules_path = Path(".cursorrules")
        rules_content = (
            "# Prism Local AI Rules for Cursor\n\n"
            "You are connected to a high-performance local AI runtime powered by Prism.\n"
            f"- Model: {chosen_model}\n"
            f"- Endpoint: {base_url}\n"
            "- Characteristics: Sub-50ms TTFT, zero cloud token costs, high privacy.\n"
            "- Focus: Write clear, efficient, bug-free, modern code.\n"
        )
        rules_path.write_text(rules_content, encoding="utf-8")
        print(f"\n✅ Exported Cursor rules to: {rules_path.resolve()}")

    if export_mcp:
        mcp_dir = Path(".cursor")
        mcp_dir.mkdir(parents=True, exist_ok=True)
        mcp_file = mcp_dir / "mcp.json"
        mcp_data: Dict[str, Any] = {"mcpServers": {}}
        if mcp_file.exists():
            try:
                mcp_data = json.loads(mcp_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        mcp_data.setdefault("mcpServers", {})["prism"] = {
            "command": prism_bin,
            "args": ["mcp"],
            "env": {"PRISM_BASE_URL": base_url},
        }
        mcp_file.write_text(json.dumps(mcp_data, indent=2), encoding="utf-8")
        print(f"✅ Exported Cursor MCP configuration to: {mcp_file.resolve()}")

    print("=" * 68 + "\n")


# ============================================================================
# Cline Connector
# ============================================================================

def connect_cline(
    model: Optional[str] = None,
    export_mcp: bool = False,
    test: bool = False,
) -> None:
    catalog = ModelCatalog()
    all_models = catalog.list_all_models(include_ollama=True)
    chosen_model = model or (all_models[0]["id"] if all_models else "Phi-4-mini-instruct-cuda-gpu")
    base_url = "http://localhost:5272/v1"
    prism_bin = _get_prism_bin_path()

    print("\n" + "=" * 68)
    print(" 🤖 PRISM CONNECTOR: CLINE (VS CODE / CURSOR EXTENSION)")
    print("=" * 68)

    print("\n1. Configure OpenAI Compatible Provider in Cline UI:")
    print("   • Click Settings (Gear Icon in Cline tab)")
    print("   • API Provider:    OpenAI Compatible")
    print(f"   • Base URL:        {base_url}")
    print("   • API Key:         prism")
    print(f"   • Model ID:        {chosen_model}")

    cline_snippet = {
        "mcpServers": {
            "prism": {
                "command": prism_bin,
                "args": ["mcp"],
                "env": {"PRISM_BASE_URL": base_url},
                "disabled": False,
                "autoApprove": [
                    "prism_ask_coder",
                    "prism_code_review",
                    "prism_list_models",
                    "prism_get_status",
                    "prism_benchmark",
                ],
            }
        }
    }

    print("\n2. MCP Tool Integration Configuration:")
    print(json.dumps(cline_snippet, indent=2))

    if test:
        print("\n3. Diagnostics & Server Connectivity:")
        is_up = test_prism_server_connection(base_url)
        if is_up:
            print(f"   ✅ Prism server is ACTIVE and reachable at {base_url}")
        else:
            print(f"   ⚠️ Prism server is OFFLINE at {base_url}")
            print(f"      Launch it in a terminal: prism serve --port 5272")

    if export_mcp:
        target_file = Path("cline_mcp_settings.json")
        try:
            merge_prism_mcp_entry(target_file, cline_snippet["mcpServers"]["prism"])
            print(f"\n✅ Exported Cline MCP configuration to: {target_file.resolve()}")
        except ValueError as ex:
            print(f"\n❌ Not overwriting {target_file}: {ex}")

    print("=" * 68 + "\n")


# ============================================================================
# Universal MCP Connector
# ============================================================================

def connect_mcp(
    target: str = "all",
    write: bool = False,
    test: bool = False,
) -> None:
    prism_bin = _get_prism_bin_path()
    base_url = "http://localhost:5272/v1"
    target = target.lower()

    print("\n" + "=" * 68)
    print(" 🔌 PRISM CONNECTOR: MODEL CONTEXT PROTOCOL (MCP)")
    print("=" * 68)

    prism_server_config = {
        "command": prism_bin,
        "args": ["mcp"],
        "env": {"PRISM_BASE_URL": base_url},
    }

    targets = [target] if target != "all" else ["antigravity", "cursor", "cline", "claude"]

    for t in targets:
        print(f"\n📋 Target: {t.upper()}")

        if t == "antigravity":
            cfg_path = Path.home() / ".gemini" / "config" / "mcp_config.json"
            print(f"   Config Path: {cfg_path}")
            if write:
                try:
                    cfg_path.parent.mkdir(parents=True, exist_ok=True)
                    existing: Dict[str, Any] = {"mcpServers": {}}
                    if cfg_path.exists():
                        # Backup
                        bak = cfg_path.with_suffix(".json.bak")
                        shutil.copy2(cfg_path, bak)
                        print(f"   📦 Backup created at: {bak}")
                        existing = json.loads(cfg_path.read_text(encoding="utf-8"))
                    existing.setdefault("mcpServers", {})["prism"] = prism_server_config
                    cfg_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
                    print(f"   ✅ Auto-wired Prism into Antigravity MCP config!")
                except Exception as ex:
                    print(f"   ❌ Failed to write Antigravity config: {ex}")
            else:
                snippet = {"mcpServers": {"prism": prism_server_config}}
                print(json.dumps(snippet, indent=2))

        elif t == "cursor":
            cfg_path = Path(".cursor") / "mcp.json"
            print(f"   Config Path: {cfg_path}")
            if write:
                try:
                    cfg_path.parent.mkdir(parents=True, exist_ok=True)
                    existing: Dict[str, Any] = {"mcpServers": {}}
                    if cfg_path.exists():
                        bak = cfg_path.with_suffix(".json.bak")
                        shutil.copy2(cfg_path, bak)
                        existing = json.loads(cfg_path.read_text(encoding="utf-8"))
                    existing.setdefault("mcpServers", {})["prism"] = prism_server_config
                    cfg_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
                    print(f"   ✅ Auto-wired Prism into Cursor MCP config: {cfg_path}")
                except Exception as ex:
                    print(f"   ❌ Failed to write Cursor config: {ex}")
            else:
                snippet = {"mcpServers": {"prism": prism_server_config}}
                print(json.dumps(snippet, indent=2))

        elif t == "cline":
            cfg_path = Path("cline_mcp_settings.json")
            print(f"   Config Path: {cfg_path}")
            if write:
                try:
                    merge_prism_mcp_entry(cfg_path, prism_server_config)
                    print(f"   ✅ Wrote Cline MCP config to: {cfg_path}")
                except Exception as ex:
                    print(f"   ❌ Failed to write Cline config: {ex}")
            else:
                snippet = {"mcpServers": {"prism": prism_server_config}}
                print(json.dumps(snippet, indent=2))

        elif t == "claude":
            cfg_path = Path.home() / ".config" / "Claude" / "claude_desktop_config.json"
            print(f"   Config Path: {cfg_path}")
            if write:
                try:
                    cfg_path.parent.mkdir(parents=True, exist_ok=True)
                    existing: Dict[str, Any] = {"mcpServers": {}}
                    if cfg_path.exists():
                        bak = cfg_path.with_suffix(".json.bak")
                        shutil.copy2(cfg_path, bak)
                        existing = json.loads(cfg_path.read_text(encoding="utf-8"))
                    existing.setdefault("mcpServers", {})["prism"] = prism_server_config
                    cfg_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
                    print(f"   ✅ Auto-wired Prism into Claude Desktop MCP config!")
                except Exception as ex:
                    print(f"   ❌ Failed to write Claude Desktop config: {ex}")
            else:
                snippet = {"mcpServers": {"prism": prism_server_config}}
                print(json.dumps(snippet, indent=2))

    if test:
        print("\n🧪 Testing MCP Protocol & Handshake:")
        res = test_mcp_protocol(prism_bin)
        if res.get("success"):
            print(f"   ✅ Handshake Successful!")
            print(f"   • Protocol Version: {res.get('protocol_version')}")
            print(f"   • Server Name:      {res.get('server_name')}")
            print(f"   • Available Tools ({res.get('tools_count')}): {', '.join(res.get('tools', []))}")
        else:
            print(f"   ❌ MCP Handshake Failed: {res.get('error')}")

    print("\n" + "=" * 68 + "\n")
