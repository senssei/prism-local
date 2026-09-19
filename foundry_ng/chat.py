"""
foundry_ng.chat: Interactive Streaming Terminal Chat.
Displays real-time generation speed, token metrics, and multi-turn context.
"""

import sys
import time
from typing import Dict, List, Optional
from foundry_ng.catalog import ModelCatalog
from foundry_ng.engine import OnnxGenAiEngine, format_prompt
from foundry_ng.ollama_bridge import stream_ollama_chat


def run_interactive_chat(model_id_or_alias: str):
    catalog = ModelCatalog()
    resolved = catalog.resolve_model(model_id_or_alias)
    if not resolved:
        print(f"❌ Model '{model_id_or_alias}' not found.")
        print("Run 'foundry-ng list' to inspect available models.")
        return

    is_ollama = resolved.get("backend") == "ollama"
    engine: Optional[OnnxGenAiEngine] = None

    if not is_ollama:
        print(f"🔄 Initializing ONNX GenAI CUDA Engine with: {resolved['name']}...")
        engine = OnnxGenAiEngine(resolved["path"])
        print(f"✅ Model loaded successfully onto GPU.")
    else:
        print(f"🦙 Routing to Ollama model: {resolved['name']}")

    print("\n" + "=" * 60)
    print(f"  💬 foundry-ng Chat: {resolved['name']}")
    print("  Type '/exit' or Ctrl+C to quit. Type '/clear' to reset context.")
    print("=" * 60 + "\n")

    history: List[Dict[str, str]] = []

    try:
        while True:
            try:
                user_input = input("You > ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting chat.")
                break

            if not user_input:
                continue

            if user_input.lower() in ["/exit", "exit", "quit"]:
                break
            if user_input.lower() in ["/clear", "clear"]:
                history.clear()
                print("🧹 Context cleared.\n")
                continue

            history.append({"role": "user", "content": user_input})
            print(f"\n{resolved['name']} > ", end="", flush=True)

            t0 = time.perf_counter()
            token_count = 0
            full_response = ""

            if is_ollama:
                for chunk in stream_ollama_chat(resolved["id"], history):
                    print(chunk, end="", flush=True)
                    full_response += chunk
                    token_count += 1
            else:
                prompt = format_prompt(history)
                for token, is_first, speed in engine.stream_generate(prompt=prompt, max_tokens=1024):
                    print(token, end="", flush=True)
                    full_response += token
                    token_count += 1

            elapsed = max(time.perf_counter() - t0, 1e-6)
            decode_speed = token_count / elapsed
            print(f"\n\n[⚡ {token_count} tokens | {decode_speed:.1f} tok/s | {elapsed:.2f}s]\n")
            history.append({"role": "assistant", "content": full_response})

    finally:
        if engine:
            print("Releasing model from GPU memory...")
            engine.unload()
            print("✅ VRAM cleared.")
